"""
Enhanced Anonymization Sampling Script for Chest X-ray Anomaly Detection.

This script provides sampling capabilities with enhanced anonymization schemes:
1. Standard anomaly detection (original functionality)
2. Enhanced image anonymization using multiple transformation schemes
3. Image de-anonymization followed by anomaly detection

Enhanced Transformation Schemes (from distribution transformation research):
- Rademacher: Original sign flipping (k ⊙ z) - baseline
- Signed Permutation: Sign flipping + position shuffling - recommended balance
- Randomized Hadamard Transform: Global diffusion via Hadamard - high security
- Block Rotation: Block-wise orthogonal rotation - local mixing
- Hybrid: Multi-layer combined transformation - maximum security

Based on:
- "Secure and Reversible Face Anonymization with Diffusion Models" by Labarbarie et al.
- Distribution Transformation Research Report (分布轉換研究)
"""
import matplotlib.pyplot as plt
import argparse
import os
import datetime
from pathlib import Path

try:
    from visdom import Visdom
    viz = Visdom(port=8850, use_incoming_socket=False)
except:
    viz = None
    print("Visdom not available")

import sys
sys.path.append("..")
sys.path.append(".")

from guided_diffusion.bratsloader import BRATSDataset, ChexpertDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    create_anonymizer,
    create_enhanced_key,
    add_dict_to_argparser,
    args_to_dict,
    TransformationType,
)
from guided_diffusion.anonymization_enhanced import (
    EnhancedKey,
    EnhancedDiffusionAnonymizer,
    verify_reversibility,
    verify_security,
    verify_distribution_preservation,
)

from PIL import Image

try:
    from torchinfo import summary
except ImportError:
    summary = None


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    args.result_dir = f'{args.result_dir}_{datetime.date.today().strftime("%Y_%m_%d")}'

    dist_util.setup_dist()
    logger.configure(dir=args.result_dir)

    logger.log("=" * 60)
    logger.log("Enhanced Anonymization Sampling Script")
    logger.log("=" * 60)
    logger.log("creating model and diffusion...")
    
    # Determine if we need anonymization
    use_anonymization = args.anonymization_mode is not None
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    if use_anonymization:
        # Create enhanced anonymizer
        in_channels = 4 if args.dataset == 'brats' else 1
        anonymizer = create_anonymizer(
            image_size=args.image_size,
            in_channels=in_channels,
            scaling_factor=1,  # No scaling in pixel space
            anonymization_mask_type=args.anonymization_mask_type,
            anonymization_margin=args.anonymization_margin,
            anonymization_transform_type=args.anonymization_transform_type,
            anonymization_block_size=args.anonymization_block_size,
            anonymization_num_hadamard_layers=args.anonymization_num_hadamard_layers,
            use_enhanced=True,
        )
        
        # Setup enhanced key
        key = setup_enhanced_key(args, anonymizer)
        
        logger.log(f"Anonymization mode: {args.anonymization_mode}")
        logger.log(f"Transform type: {args.anonymization_transform_type}")
        logger.log(f"Mask type: {args.anonymization_mask_type}")
        logger.log(f"Mask margin: {args.anonymization_margin}")
        logger.log(f"Block size: {args.anonymization_block_size}")
        logger.log(f"Key transform: {key.transform_type.value}")
        
        # Verify key properties
        logger.log("Verifying key properties...")
        is_reversible, max_error = verify_reversibility(anonymizer, key, num_samples=5)
        logger.log(f"  Reversibility: {'PASS' if is_reversible else 'FAIL'} (max error: {max_error:.2e})")
        
        # Verify distribution preservation
        dist_results = verify_distribution_preservation(anonymizer, key, num_samples=100)
        logger.log(f"  Distribution preserved: {'PASS' if dist_results['distribution_preserved'] else 'FAIL'}")
        logger.log(f"    Original: mean={dist_results['original_mean']:.4f}, std={dist_results['original_std']:.4f}")
        logger.log(f"    Transformed: mean={dist_results['transformed_mean']:.4f}, std={dist_results['transformed_std']:.4f}")
    else:
        anonymizer = None
        key = None

    # Load dataset
    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=True)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False)

    elif args.dataset == 'chexpert':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            deterministic=True
        )
        datal = iter(data)

    # Load model weights
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    logger.log(f'Model parameters: {p1:,}')

    def model_fn(x, t, y=None, p_uncond=-1, null=False, clf_free=True):
        assert y is not None
        return model(x, t, y if args.class_cond else None, p_uncond, null, clf_free)

    logger.log("sampling...")
    all_orgs = []
    all_images = []
    all_org_labels = []
    all_tgt_labels = []
    all_names = []
    all_results = []
    all_anonymized = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        img = next(datal)
        print('img', img[0].shape, img[1])
        
        if args.dataset == 'brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            number = img[4][0]
            if img[2] == 0:
                continue
            if viz:
                viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input 0"))
        else:
            number = img[1]["name"]
            org_label = img[1]["y"].to(dist_util.dev())
            if viz:
                viz.image(visualize(img[0][0, ...]), 
                         opts=dict(caption=f"img {'diseased' if img[1]['y'] else 'healthy'} {number[0]}"))

        if args.class_cond:
            classes = th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int)
            model_kwargs = dict(
                y=classes,
                p_uncond=-1,
                clf_free=True
            )

        # Choose sampling function based on mode
        if args.anonymization_mode == 'anonymize':
            # Enhanced anonymization mode
            sample, x_T, org = diffusion.ddim_sample_loop_enhanced_anonymization(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                anonymizer=anonymizer,
                key=key,
                org=img,
                mode='anonymize',
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                noise_level=args.noise_level,
                guidance_scale=-1,
            )
            
        elif args.anonymization_mode == 'deanonymize':
            # De-anonymization followed by anomaly detection
            sample, x_noisy, x_rec = diffusion.ddim_sample_loop_enhanced_anomaly_detection_with_deanonymization(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                anonymizer=anonymizer,
                key=key,
                org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                noise_level=args.noise_level,
                guidance_scale=args.guidance_scale,
            )
            org = x_rec  # Use recovered image as reference
            
        else:
            # Standard anomaly detection (no anonymization)
            sample, x_noisy = diffusion.ddim_sample_loop_known(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                noise_level=args.noise_level,
                w=args.guidance_scale,
            )
            org = img[0].to(dist_util.dev())

        # Process results for saving
        for i in range(sample.shape[0]):
            if args.dataset == 'brats':
                sampled_img = ((sample[i, 0, ...] + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
                original_img = ((org[i, 0, ...] + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
            else:
                sampled_img = ((sample[i, 0, ...] + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
                original_img = ((org[i, 0, ...] + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
            
            # Calculate heatmap (anomaly map)
            diff = np.abs(sampled_img.astype(np.float32) - original_img.astype(np.float32))
            heatmap_img = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
            
            # Save individual images
            if args.save_images:
                save_results(args, img, sampled_img, original_img, heatmap_img, i)

        all_names.append(img[1]["name"][0] if isinstance(img[1]["name"], (list, tuple)) else img[1]["name"])

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_orgs = [th.zeros_like(org) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_orgs, org)
        all_orgs.extend([org.cpu().numpy() for org in gathered_orgs])

        if args.class_cond:
            gathered_org_labels = [
                th.zeros_like(org_label) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_org_labels, org_label)
            all_org_labels.extend([labels.cpu().numpy() for labels in gathered_org_labels])

            gathered_tgt_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_tgt_labels, classes)
            all_tgt_labels.extend([labels.cpu().numpy() for labels in gathered_tgt_labels])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]

    org_arr = np.concatenate(all_orgs, axis=0)
    org_arr = org_arr[:args.num_samples]

    if args.class_cond:
        org_label_arr = np.concatenate(all_org_labels, axis=0)
        org_label_arr = org_label_arr[:args.num_samples]

        tgt_label_arr = np.concatenate(all_tgt_labels, axis=0)
        tgt_label_arr = tgt_label_arr[:args.num_samples]

        name_arr = np.array(all_names)
        name_arr = name_arr[:args.num_samples]

    if dist.get_rank() == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        shape_str = "x".join([str(x) for x in arr.shape])
        
        mode_str = f"_{args.anonymization_mode}" if args.anonymization_mode else ""
        transform_str = f"_{args.anonymization_transform_type}" if args.anonymization_mode else ""
        out_path = os.path.join(logger.get_dir(), f"samples{mode_str}{transform_str}_{shape_str}.npz")
        
        logger.log(f"saving to {out_path}")
        np.savez(out_path, samples=arr, org_labels=org_label_arr, tgt_labels=tgt_label_arr, 
                names=name_arr, orgs=org_arr)

        # Save key info if using anonymization
        if args.anonymization_mode and key is not None:
            save_key_info(args, key)

    dist.barrier()
    logger.log("sampling complete")


def save_results(args, img, sampled_img, original_img, heatmap_img, idx):
    """Save result images to disk."""
    img_name = img[1]["name"][0] if isinstance(img[1]["name"], (list, tuple)) else img[1]["name"]
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    
    if args.anonymization_mode == 'anonymize':
        ano_dir = os.path.join(args.result_dir, 'anonymized', args.anonymization_transform_type)
        orig_dir = os.path.join(args.result_dir, 'original')
        os.makedirs(ano_dir, exist_ok=True)
        os.makedirs(orig_dir, exist_ok=True)
        
        ano_path = os.path.join(ano_dir, f'{base_name}_anonymized.png')
        Image.fromarray(sampled_img).save(ano_path, 'PNG')
        
        orig_path = os.path.join(orig_dir, f'{base_name}_original.png')
        Image.fromarray(original_img).save(orig_path, 'PNG')
        
        logger.log(f"Saved: {ano_path}")
        
    elif args.anonymization_mode == 'deanonymize':
        rec_dir = os.path.join(args.result_dir, 'recovered', args.anonymization_transform_type)
        healthy_dir = os.path.join(args.result_dir, 'healthy')
        heatmap_dir = os.path.join(args.result_dir, 'heatmaps')
        os.makedirs(rec_dir, exist_ok=True)
        os.makedirs(healthy_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        
        rec_path = os.path.join(rec_dir, f'{base_name}_recovered.png')
        Image.fromarray(original_img).save(rec_path, 'PNG')
        
        healthy_path = os.path.join(healthy_dir, f'{base_name}_healthy.png')
        Image.fromarray(sampled_img).save(healthy_path, 'PNG')
        
        heatmap_path = os.path.join(heatmap_dir, f'{base_name}_heatmap.png')
        Image.fromarray(heatmap_img).save(heatmap_path, 'PNG')
        
        logger.log(f"Saved: {rec_path}")
        
    else:
        output_dir = os.path.join(args.result_dir, 'outputs')
        heatmap_dir = os.path.join(args.result_dir, 'heatmaps')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        
        out_path = os.path.join(output_dir, f'{base_name}_output.png')
        Image.fromarray(sampled_img).save(out_path, 'PNG')
        
        heatmap_path = os.path.join(heatmap_dir, f'{base_name}_heatmap.png')
        Image.fromarray(heatmap_img).save(heatmap_path, 'PNG')
        
        logger.log(f"Saved: {out_path}")


def save_key_info(args, key):
    """Save key information to file."""
    key_info_path = os.path.join(args.result_dir, 'key_info.txt')
    with open(key_info_path, 'w') as f:
        f.write("Enhanced Anonymization Key Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Transform type: {key.transform_type.value}\n")
        f.write(f"Key shape: {key.shape}\n")
        f.write(f"Seed: {key.seed}\n")
        f.write(f"Block size: {key.block_size}\n")
        f.write(f"Num Hadamard layers: {key.num_hadamard_layers}\n")
        f.write(f"Key path: {args.anonymization_key_path}\n")
        f.write(f"Password: {'<set>' if args.anonymization_key_password else '<not set>'}\n")
        f.write("\nSecurity Properties:\n")
        f.write("-" * 30 + "\n")
        if key.transform_type == TransformationType.RADEMACHER:
            f.write("- Diffusion: None (point-to-point)\n")
            f.write("- Key space: 2^d\n")
            f.write("- Spatial correlation: Preserved\n")
        elif key.transform_type == TransformationType.SIGNED_PERMUTATION:
            f.write("- Diffusion: Low (position change)\n")
            f.write("- Key space: 2^d × d!\n")
            f.write("- Spatial correlation: Destroyed\n")
        elif key.transform_type == TransformationType.HADAMARD:
            f.write("- Diffusion: High (global mixing)\n")
            f.write("- Key space: 2^d\n")
            f.write("- Frequency whitening: Yes\n")
        elif key.transform_type == TransformationType.BLOCK_ROTATION:
            f.write("- Diffusion: Local (per block)\n")
            f.write("- Key space: O(d) matrices\n")
            f.write("- Local mixing: Full\n")
        elif key.transform_type == TransformationType.HYBRID:
            f.write("- Diffusion: Very high (multi-layer)\n")
            f.write("- Key space: Maximum\n")
            f.write("- Security: Maximum\n")
    
    # Save key file if path provided
    if args.anonymization_key_path:
        key.save(args.anonymization_key_path)
        logger.log(f"Key saved to: {args.anonymization_key_path}")


def setup_enhanced_key(args, anonymizer):
    """
    Setup or load the enhanced anonymization key.
    """
    key_path = args.anonymization_key_path
    key_seed = args.anonymization_key_seed
    key_password = args.anonymization_key_password
    transform_type = args.anonymization_transform_type
    
    # Determine the correct shape for pixel-space key
    in_channels = 4 if args.dataset == 'brats' else 1
    pixel_key_shape = (in_channels, args.image_size, args.image_size)
    
    if key_path and os.path.exists(key_path):
        # Load existing key
        logger.log(f"Loading enhanced key from {key_path}")
        key = EnhancedKey.load(key_path)
        
        # Check if loaded key matches expected configuration
        if key.shape != pixel_key_shape:
            logger.log(f"Warning: Loaded key shape {key.shape} doesn't match expected {pixel_key_shape}")
            logger.log(f"Regenerating key with correct shape...")
            key = create_new_enhanced_key(args, pixel_key_shape, transform_type, key_seed, key_password)
            if key_path:
                key.save(key_path)
        elif key.transform_type.value != transform_type:
            logger.log(f"Warning: Loaded key transform {key.transform_type.value} doesn't match {transform_type}")
            logger.log(f"Using loaded key's transform type...")
    else:
        # Generate new enhanced key
        logger.log(f"Generating enhanced key (shape={pixel_key_shape}, transform={transform_type})")
        key = create_new_enhanced_key(args, pixel_key_shape, transform_type, key_seed, key_password)
        
        # Save key if path provided
        if key_path:
            os.makedirs(os.path.dirname(key_path) if os.path.dirname(key_path) else '.', exist_ok=True)
            key.save(key_path)
            logger.log(f"Saved enhanced key to {key_path}")
    
    return key


def create_new_enhanced_key(args, shape, transform_type, seed, password):
    """Create a new enhanced key."""
    return EnhancedKey(
        shape=shape,
        transform_type=TransformationType(transform_type),
        seed=seed,
        password=password,
        block_size=args.anonymization_block_size,
        num_hadamard_layers=args.anonymization_num_hadamard_layers,
    )


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        noise_level=500,
        dataset='chexpert',
        num_classes=2,
        result_dir='./results_enhanced',
        guidance_scale=2.0,
        class_cond=True,
        unet_version='v1',
        save_images=True,
        # Anonymization options
        anonymization_mode=None,  # None, 'anonymize', or 'deanonymize'
        anonymization_key_path=None,
        anonymization_key_seed=None,
        anonymization_key_password=None,
        anonymization_mask_type='full',  # 'full', 'center', 'margin', 'lung_region'
        anonymization_margin=0,
        # Enhanced anonymization options
        anonymization_transform_type='signed_permutation',  # rademacher, signed_permutation, hadamard, block_rotation, hybrid
        anonymization_block_size=8,
        anonymization_num_hadamard_layers=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
