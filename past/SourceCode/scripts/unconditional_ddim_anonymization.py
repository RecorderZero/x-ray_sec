"""
Unconditional DDIM sampling with anonymization support.

This script is designed for comparing encryption/decryption algorithm effectiveness
WITHOUT any guidance (no CFG, no Classifier Guidance).

Purpose:
- Provide a baseline for comparing anonymization effectiveness
- Demonstrate that encryption/decryption works independently of guidance methods
- Validate distribution preservation and reversibility

Comparison with other methods:
1. CFG (cfg_image_sample_anonymization.py): Uses classifier-free guidance
2. Classifier (classifier_sample_known_anonymization.py): Uses classifier gradient
3. This script: Pure DDIM without any guidance (baseline)

Key insight: The anonymization operates at x_T level, which is independent of
the sampling guidance method. This script proves that the encryption effectiveness
is not dependent on the choice of guidance.
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
    create_model_and_diffusion,
    create_anonymizer,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.anonymization import (
    RademacherKey, 
    SignedPermutationKey,
    AnonymizationManager
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

    # Setup distributed
    dist_util.setup_dist()
    
    # Create result directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d")
    if args.anonymization_mode:
        result_subdir = f"unconditional_{args.anonymization_mode}_{timestamp}"
    else:
        result_subdir = f"unconditional_standard_{timestamp}"
    args.result_dir = os.path.join(args.result_dir, result_subdir)
    
    logger.configure(dir=args.result_dir)
    logger.log(f"=== Unconditional DDIM Anonymization ===")
    logger.log(f"Mode: {args.anonymization_mode}")
    logger.log(f"Results will be saved to: {args.result_dir}")

    # Determine if we need anonymization
    use_anonymization = args.anonymization_mode is not None

    # Create model and diffusion (no classifier needed)
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if use_anonymization:
        # Create anonymizer for pixel-space operation
        in_channels = 4 if args.dataset == 'brats' else 1
        anonymizer = create_anonymizer(
            image_size=args.image_size,
            in_channels=in_channels,
            scaling_factor=1,
            anonymization_mask_type=args.anonymization_mask_type,
            anonymization_margin=args.anonymization_margin,
            use_signed_permutation=args.use_signed_permutation,
        )
        
        # Setup key
        key = setup_anonymization_key(args, anonymizer)
        
        logger.log(f"Anonymization mode: {args.anonymization_mode}")
        logger.log(f"Mask type: {args.anonymization_mask_type}")
        logger.log(f"Key type: {'SignedPermutationKey' if isinstance(key, SignedPermutationKey) else 'RademacherKey'}")
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
        )
        datal = iter(data)

    # Load model weights
    logger.log(f"Loading model from {args.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Model function (unconditional - no guidance)
    def model_fn(x, t, y=None, **kwargs):
        """
        Unconditional model function.
        Even if class_cond is True, we don't use guidance.
        """
        if args.class_cond:
            # Pass class label but don't use for guidance
            return model(x, t, y, p_uncond=-1, null=False, clf_free=True)
        else:
            return model(x, t)

    logger.log("Sampling...")
    all_images = []
    all_orgs = []
    all_names = []
    all_results = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        img = next(datal)
        
        if args.dataset == 'brats':
            number = img[4][0]
            if img[2] == 0:
                continue  # take only diseased images
            if viz:
                viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="input"))
        else:
            number = img[1]["name"]
            if viz:
                viz.image(visualize(img[0][0, ...]), 
                         opts=dict(caption=f"input {number[0]}"))

        # For unconditional sampling, we still need to pass y if model expects it
        if args.class_cond:
            classes = th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int)
            model_kwargs["y"] = classes

        # Select sampling mode
        if args.anonymization_mode == 'anonymize':
            # Anonymization mode (no guidance)
            sample, x_T, org = diffusion.ddim_sample_loop_anonymization(
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
                guidance_scale=-1,  # No CFG guidance
            )
            
        elif args.anonymization_mode == 'deanonymize':
            # De-anonymization (no guidance, just reconstruct)
            sample, x_T, org = diffusion.ddim_sample_loop_anonymization(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                anonymizer=anonymizer,
                key=key,
                org=img,
                mode='deanonymize',
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                noise_level=args.noise_level,
                guidance_scale=-1,  # No CFG guidance
            )
            
        else:
            # Standard unconditional reconstruction (baseline)
            sample, x_noisy, org = diffusion.ddim_sample_loop_known(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,  # No classifier guidance
                device=dist_util.dev(),
                noise_level=args.noise_level,
                guidance_scale=-1,  # No CFG guidance
            )

        # Visualization and saving
        if args.dataset == 'brats':
            if viz:
                viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="output"))
            if args.anonymization_mode != 'anonymize':
                difftot = abs(org[0, :4, ...] - sample[0, ...]).sum(dim=0)
                if viz:
                    viz.heatmap(visualize(difftot), opts=dict(caption="diff"))
                
        elif args.dataset == 'chexpert':
            if viz:
                viz.image(visualize(sample[0, ...]), 
                         opts=dict(caption=f'output {img[1]["name"][0]}'))
            
            # Compute diff only for non-anonymize modes
            if args.anonymization_mode != 'anonymize':
                diff = abs(org[0, 0, ...] - sample[0, 0, ...])
                diff = np.array(diff.cpu())
                cm = plt.get_cmap('jet')
                colored_diff = cm(visualize(diff))[:, :, :3]
                if viz:
                    viz.image(colored_diff.transpose(2, 0, 1), 
                             opts=dict(caption=f'diff {img[1]["name"][0]}'))
                heatmap_img = (colored_diff * 255).astype(np.uint8)
            else:
                heatmap_img = None

            # Prepare images
            original_img = (np.concatenate((np.array(visualize(org[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)
            sampled_img = (np.concatenate((np.array(visualize(sample[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)

            # Create result based on mode
            if args.anonymization_mode == 'anonymize':
                result = np.hstack([original_img, sampled_img])
            else:
                result = np.hstack([original_img, sampled_img, heatmap_img])
            all_results.append(result)

            # Save individual images
            if dist.get_rank() == 0:
                img_name = img[1]["name"][0] if isinstance(img[1]["name"], (list, tuple)) else img[1]["name"]
                base_name = os.path.splitext(os.path.basename(img_name))[0]
                
                if args.anonymization_mode == 'anonymize':
                    ano_dir = os.path.join(args.result_dir, 'anonymized')
                    orig_dir = os.path.join(args.result_dir, 'original')
                    os.makedirs(ano_dir, exist_ok=True)
                    os.makedirs(orig_dir, exist_ok=True)
                    
                    Image.fromarray(sampled_img).save(os.path.join(ano_dir, f'{base_name}_anonymized.png'), 'PNG')
                    Image.fromarray(original_img).save(os.path.join(orig_dir, f'{base_name}_original.png'), 'PNG')
                    logger.log(f"Saved: {base_name}_anonymized.png")
                    
                elif args.anonymization_mode == 'deanonymize':
                    rec_dir = os.path.join(args.result_dir, 'recovered')
                    diff_dir = os.path.join(args.result_dir, 'diff')
                    os.makedirs(rec_dir, exist_ok=True)
                    os.makedirs(diff_dir, exist_ok=True)
                    
                    Image.fromarray(sampled_img).save(os.path.join(rec_dir, f'{base_name}_recovered.png'), 'PNG')
                    if heatmap_img is not None:
                        Image.fromarray(heatmap_img).save(os.path.join(diff_dir, f'{base_name}_diff.png'), 'PNG')
                    logger.log(f"Saved: {base_name}_recovered.png")
                    
                else:
                    output_dir = os.path.join(args.result_dir, 'outputs')
                    diff_dir = os.path.join(args.result_dir, 'diff')
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(diff_dir, exist_ok=True)
                    
                    Image.fromarray(sampled_img).save(os.path.join(output_dir, f'{base_name}_output.png'), 'PNG')
                    if heatmap_img is not None:
                        Image.fromarray(heatmap_img).save(os.path.join(diff_dir, f'{base_name}_diff.png'), 'PNG')
                    logger.log(f"Saved: {base_name}_output.png")

        all_names.append(img[1]["name"][0] if args.dataset == 'chexpert' else number)
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([s.cpu().numpy() for s in gathered_samples])

        logger.log(f"Created {len(all_images) * args.batch_size} samples")

    # Save final results
    arr = np.concatenate(all_images, axis=0)[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(args.result_dir, f"samples_{shape_str}.npz")
        logger.log(f"Saving to {out_path}")
        np.savez(out_path, arr)
        
        # Save combined results
        if all_results:
            combined = np.vstack(all_results)
            combined_path = os.path.join(args.result_dir, f"combined_results.png")
            Image.fromarray(combined).save(combined_path, 'PNG')
            logger.log(f"Saved combined results to {combined_path}")

    dist.barrier()
    logger.log("Sampling complete")


def setup_anonymization_key(args, anonymizer):
    """
    Setup anonymization key based on arguments.
    """
    key_path = args.anonymization_key_path
    key_seed = args.anonymization_key_seed
    key_password = args.anonymization_key_password
    use_signed_permutation = getattr(args, 'use_signed_permutation', False)
    
    in_channels = 4 if args.dataset == 'brats' else 1
    pixel_key_shape = (in_channels, args.image_size, args.image_size)
    
    if key_path and os.path.exists(key_path):
        logger.log(f"Loading key from {key_path}")
        try:
            if use_signed_permutation:
                key = SignedPermutationKey.load(key_path)
            else:
                key = RademacherKey.load(key_path)
        except Exception as e:
            logger.log(f"Error loading key: {e}, generating new...")
            if use_signed_permutation:
                key = SignedPermutationKey(shape=pixel_key_shape, seed=key_seed, password=key_password)
            else:
                key = RademacherKey(shape=pixel_key_shape, seed=key_seed, password=key_password)
            if key_path:
                key.save(key_path)
    else:
        if use_signed_permutation:
            logger.log(f"Generating SignedPermutationKey (shape={pixel_key_shape})")
            key = SignedPermutationKey(shape=pixel_key_shape, seed=key_seed, password=key_password)
        else:
            logger.log(f"Generating RademacherKey (shape={pixel_key_shape})")
            key = RademacherKey(shape=pixel_key_shape, seed=key_seed, password=key_password)
        
        if key_path:
            os.makedirs(os.path.dirname(key_path) if os.path.dirname(key_path) else '.', exist_ok=True)
            key.save(key_path)
            logger.log(f"Saved key to {key_path}")
    
    return key


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=True,  # Always use DDIM for this script
        model_path="",
        noise_level=500,
        dataset='chexpert',
        num_classes=2,
        result_dir='./results',
        class_cond=True,
        unet_version='v1',  # Match the trained model version
        # Anonymization options
        anonymization_mode=None,
        anonymization_key_path=None,
        anonymization_key_seed=None,
        anonymization_key_password=None,
        anonymization_mask_type='full',
        anonymization_margin=0,
        use_signed_permutation=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
