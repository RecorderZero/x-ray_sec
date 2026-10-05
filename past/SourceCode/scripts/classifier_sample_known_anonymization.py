"""
Classifier-guided sampling with anonymization support.

This script supports three modes:
1. Standard anomaly detection with classifier guidance
2. Image anonymization using Rademacher key (basic) or SignedPermutationKey (enhanced)
3. Image de-anonymization followed by anomaly detection with classifier guidance

Based on the paper "Secure and Reversible Face Anonymization with Diffusion Models"
by Pol Labarbarie, Vincent Itier and William Puech.

Enhanced with Signed Permutation Transformation for stronger encryption:
- Key space expands from 2^d to 2^d × d!
- Breaks spatial correlations in latent space
- Provides defense against magnitude-based attacks
"""
import matplotlib.pyplot as plt
import argparse, os, datetime
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
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.anonymization import RademacherKey, SignedPermutationKey, AnonymizationMask

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

    logger.log("creating model and diffusion...")
    
    # Determine if we need anonymization
    use_anonymization = args.anonymization_mode is not None
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    if use_anonymization:
        # Create anonymizer for pixel-space operation
        in_channels = 4 if args.dataset == 'brats' else 1
        anonymizer = create_anonymizer(
            image_size=args.image_size,
            in_channels=in_channels,
            scaling_factor=1,  # No scaling in pixel space
            anonymization_mask_type=args.anonymization_mask_type,
            anonymization_margin=args.anonymization_margin,
        )
        
        # Setup key (in pixel space)
        key = setup_anonymization_key(args, anonymizer)
        
        logger.log(f"Anonymization mode: {args.anonymization_mode}")
        logger.log(f"Mask type: {args.anonymization_mask_type}")
        logger.log(f"Mask margin: {args.anonymization_margin}")
        logger.log(f"Key shape: {key.shape}")
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
        if args.data_filter is True:
            ds = ChexpertDataset(args.data_dir, class_cond=True, test_flag=True, data_filter="frontal_only")
            data = th.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False)
            ds.summarize()
        else:
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

    # Load classifier
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path)
    )
    print('loaded classifier')

    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)
    logger.log('pmodel', p1, 'pclass', p2)

    if summary is not None:
        summary(model, input_data={
            'x': th.randn(size=(args.batch_size, 1, 256, 256), device=dist_util.dev()),
            'timesteps': diffusion._scale_timesteps(th.randint(0, 1, (args.batch_size,), device=dist_util.dev()).long()),
            'y': th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int),
            'p_uncond': -1,
            'clf_free': False
        })

        summary(classifier, input_data={
            'x': th.randn(size=(args.batch_size, 1, 256, 256), device=dist_util.dev()),
            'timesteps': diffusion._scale_timesteps(th.randint(0, 1, (args.batch_size,), device=dist_util.dev()).long()),
        })

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.long().view(-1)]
            a = th.autograd.grad(selected.sum(), x_in)[0]
            return a, a * args.classifier_scale

    def model_fn(x, t, y=None, p_uncond=-1, null=False, clf_free=False):
        assert y is not None
        return model(x, t, y if args.class_cond else None, p_uncond=-1, null=False, clf_free=False)

    logger.log("sampling...")
    all_orgs = []
    all_images = []
    all_org_labels = []
    all_tgt_labels = []
    all_names = []
    all_results = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        img = next(datal)
        print('img', img[0].shape, img[1])
        
        if args.dataset == 'brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            number = img[4][0]
            if img[2] == 0:
                continue  # take only diseased images as input
                
            viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input 0"))
            viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input 1"))
            viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input 2"))
            viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input 3"))
            viz.image(visualize(img[3][0, ...]), opts=dict(caption="ground truth"))
        else:
            number = img[1]["name"]
            org_label = img[1]["y"].to(dist_util.dev())
            
            viz.image(visualize(img[0][0, ...]), opts=dict(caption=f"img {'diseased' if img[1]['y'] else 'healthy'} {number[0]}"))
            print('img1', img[1])
            print('number', number)

        if args.class_cond:
            classes = th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int)
            model_kwargs["y"] = classes
            print('target y', model_kwargs["y"])

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()

        # Select sampling mode based on anonymization_mode
        if args.anonymization_mode == 'anonymize':
            # Anonymization mode (no classifier guidance for anonymization)
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
                guidance_scale=-1,  # No guidance for anonymization
            )
            
        elif args.anonymization_mode == 'deanonymize':
            # De-anonymization followed by anomaly detection with classifier guidance
            sample, x_noisy, x_rec = diffusion.ddim_sample_loop_anomaly_detection_with_deanonymization_classifier(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                anonymizer=anonymizer,
                key=key,
                org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                noise_level=args.noise_level,
            )
            org = x_rec  # Use recovered image as reference
            
        else:
            # Standard anomaly detection with classifier guidance
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            print('samplefn', sample_fn)
            
            sample, x_noisy, org = sample_fn(
                model_fn,
                (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size),
                img,
                org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                noise_level=args.noise_level
            )

        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()
        print('time for sampling', start.elapsed_time(end))

        if args.dataset == 'brats':
            viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output0"))
            viz.image(visualize(sample[0, 1, ...]), opts=dict(caption="sampled output1"))
            viz.image(visualize(sample[0, 2, ...]), opts=dict(caption="sampled output2"))
            viz.image(visualize(sample[0, 3, ...]), opts=dict(caption="sampled output3"))
            # Only compute and display heatmap for non-anonymize modes
            if args.anonymization_mode != 'anonymize':
                difftot = abs(org[0, :4, ...] - sample[0, ...]).sum(dim=0)
                viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))
          
        elif args.dataset == 'chexpert':
            viz.image(visualize(sample[0, ...]), opts=dict(caption=f'sampled output {img[1]["name"][0]}'))
            
            # Only compute and display diff/heatmap for non-anonymize modes
            if args.anonymization_mode != 'anonymize':
                diff = abs(visualize(org[0, 0, ...]) - visualize(sample[0, 0, ...]))
                diff = np.array(diff.cpu())
                cm = plt.get_cmap('jet')
                colored_diff = cm(visualize(diff))[:, :, :3]
                viz.image(colored_diff.transpose(2, 0, 1), opts=dict(caption=f'diff {img[1]["name"][0]}'))
                heatmap_img = (colored_diff * 255).astype(np.uint8)
            else:
                # For anonymize mode, no heatmap needed
                heatmap_img = None

            # Prepare images for saving
            original_img = (np.concatenate((np.array(visualize(org[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)
            sampled_img = (np.concatenate((np.array(visualize(sample[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)

            # Save individual images based on mode
            if dist.get_rank() == 0:
                if args.anonymization_mode == 'anonymize':
                    # Save anonymized images
                    ano_dir = os.path.join(args.result_dir, 'anonymized')
                    orig_dir = os.path.join(args.result_dir, 'original')
                    os.makedirs(ano_dir, exist_ok=True)
                    os.makedirs(orig_dir, exist_ok=True)
                    
                    img_name = img[1]["name"][0] if isinstance(img[1]["name"], (list, tuple)) else img[1]["name"]
                    base_name = os.path.splitext(os.path.basename(img_name))[0]
                    
                    # Save anonymized image
                    ano_path = os.path.join(ano_dir, f'{base_name}_anonymized.png')
                    Image.fromarray(sampled_img).save(ano_path, 'PNG')
                    
                    # Save original image for reference
                    orig_path = os.path.join(orig_dir, f'{base_name}_original.png')
                    Image.fromarray(original_img).save(orig_path, 'PNG')
                    
                    logger.log(f"Saved: {ano_path}")
                    
                elif args.anonymization_mode == 'deanonymize':
                    # Save de-anonymized (recovered) and healthy reconstruction
                    rec_dir = os.path.join(args.result_dir, 'recovered')
                    healthy_dir = os.path.join(args.result_dir, 'healthy')
                    heatmap_dir = os.path.join(args.result_dir, 'heatmaps')
                    os.makedirs(rec_dir, exist_ok=True)
                    os.makedirs(healthy_dir, exist_ok=True)
                    os.makedirs(heatmap_dir, exist_ok=True)
                    
                    img_name = img[1]["name"][0] if isinstance(img[1]["name"], (list, tuple)) else img[1]["name"]
                    base_name = os.path.splitext(os.path.basename(img_name))[0]
                    
                    # Save recovered (de-anonymized) image
                    rec_path = os.path.join(rec_dir, f'{base_name}_recovered.png')
                    Image.fromarray(original_img).save(rec_path, 'PNG')
                    
                    # Save healthy reconstruction
                    healthy_path = os.path.join(healthy_dir, f'{base_name}_healthy.png')
                    Image.fromarray(sampled_img).save(healthy_path, 'PNG')
                    
                    # Save heatmap
                    heatmap_path = os.path.join(heatmap_dir, f'{base_name}_heatmap.png')
                    Image.fromarray(heatmap_img).save(heatmap_path, 'PNG')
                    
                    logger.log(f"Saved: {rec_path}")
                    
                else:
                    # Standard anomaly detection mode
                    output_dir = os.path.join(args.result_dir, 'outputs')
                    heatmap_dir = os.path.join(args.result_dir, 'heatmaps')
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(heatmap_dir, exist_ok=True)
                    
                    img_name = img[1]["name"][0] if isinstance(img[1]["name"], (list, tuple)) else img[1]["name"]
                    base_name = os.path.splitext(os.path.basename(img_name))[0]
                    
                    # Save output
                    out_path = os.path.join(output_dir, f'{base_name}_output.png')
                    Image.fromarray(sampled_img).save(out_path, 'PNG')
                    
                    # Save heatmap
                    heatmap_path = os.path.join(heatmap_dir, f'{base_name}_heatmap.png')
                    Image.fromarray(heatmap_img).save(heatmap_path, 'PNG')
                    
                    logger.log(f"Saved: {out_path}")

            # Create result based on mode
            if args.anonymization_mode == 'anonymize':
                # For anonymization, only show original -> anonymized
                result = np.hstack([original_img, sampled_img])
            else:
                # For standard/deanonymize, show original -> sampled -> heatmap
                result = np.hstack([original_img, sampled_img, heatmap_img])
            all_results.append(result)

        all_names.append(img[1]["name"][0])

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
    arr = arr[: args.num_samples]

    org_arr = np.concatenate(all_orgs, axis=0)
    org_arr = org_arr[: args.num_samples]

    if args.class_cond:
        org_label_arr = np.concatenate(all_org_labels, axis=0)
        org_label_arr = org_label_arr[: args.num_samples]

        tgt_label_arr = np.concatenate(all_tgt_labels, axis=0)
        tgt_label_arr = tgt_label_arr[: args.num_samples]
        
        name_arr = np.array(all_names)
        name_arr = name_arr[: args.num_samples]

    if dist.get_rank() == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        shape_str = "x".join([str(x) for x in arr.shape])
        
        # Determine output filename based on mode
        mode_str = args.anonymization_mode if args.anonymization_mode else "standard"
        out_path = os.path.join(args.result_dir, f"samples_{mode_str}_{shape_str}.npz")
        
        logger.log(f"saving to {out_path}")
        np.savez(out_path, samples=arr, org_labels=org_label_arr, tgt_labels=tgt_label_arr, 
                 names=name_arr, orgs=org_arr)
        
        if all_results:
            final_samples_image = Image.fromarray(np.vstack(all_results))
            logger.log(f"saving generated sample images to {args.result_dir}")
            final_samples_image.save(os.path.join(
                args.result_dir, 
                f'latest_run_{mode_str}_{os.path.splitext(os.path.basename(args.model_path))[0]}.png'
            ))

    dist.barrier()
    logger.log("sampling complete")


def setup_anonymization_key(args, anonymizer):
    """
    Setup anonymization key based on arguments.
    
    Priority:
    1. Load from file if path exists
    2. Generate from password if provided
    3. Generate from seed if provided
    4. Generate random key (and save if path provided)
    
    Supports two key types:
    - RademacherKey: Basic sign flipping (default)
    - SignedPermutationKey: Sign flipping + position shuffling (stronger encryption)
    """
    key_path = args.anonymization_key_path
    key_seed = args.anonymization_key_seed
    key_password = args.anonymization_key_password
    use_signed_permutation = getattr(args, 'use_signed_permutation', False)
    
    # Determine key shape for pixel space
    in_channels = 4 if args.dataset == 'brats' else 1
    pixel_key_shape = (in_channels, args.image_size, args.image_size)
    
    if key_path and os.path.exists(key_path):
        # Load existing key
        logger.log(f"Loading key from {key_path}")
        try:
            if use_signed_permutation:
                key = SignedPermutationKey.load(key_path)
                logger.log(f"Loaded SignedPermutationKey")
            else:
                key = RademacherKey.load(key_path)
                logger.log(f"Loaded RademacherKey")
        except Exception as e:
            logger.log(f"Error loading key: {e}, generating new key...")
            if use_signed_permutation:
                key = SignedPermutationKey(
                    shape=pixel_key_shape,
                    seed=key_seed,
                    password=key_password,
                )
            else:
                key = RademacherKey(
                    shape=pixel_key_shape,
                    seed=key_seed,
                    password=key_password,
                )
            if key_path:
                key.save(key_path)
    else:
        # Generate new key with pixel-space shape
        if use_signed_permutation:
            logger.log(f"Generating SignedPermutationKey (shape={pixel_key_shape}, seed={key_seed})")
            logger.log(f"Enhanced security: key space = 2^d × d!")
            key = SignedPermutationKey(
                shape=pixel_key_shape,
                seed=key_seed,
                password=key_password,
            )
        else:
            logger.log(f"Generating RademacherKey (shape={pixel_key_shape}, seed={key_seed})")
            key = RademacherKey(
                shape=pixel_key_shape,
                seed=key_seed,
                password=key_password,
            )
        
        # Save key if path provided
        if key_path:
            os.makedirs(os.path.dirname(key_path) if os.path.dirname(key_path) else '.', exist_ok=True)
            key.save(key_path)
            logger.log(f"Saved key to {key_path}")
    
    key_type = "SignedPermutationKey" if isinstance(key, SignedPermutationKey) else "RademacherKey"
    logger.log(f"Key type: {key_type}, shape: {key.shape}")
    return key


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        result_dir='results/',
        clf_free=False,
        guidance_scale=-1,
        p_uncond=-1,
        null=False,
        data_filter=False,
        # Anonymization options
        anonymization_mode=None,  # None, 'anonymize', or 'deanonymize'
        anonymization_key_path=None,
        anonymization_key_seed=None,
        anonymization_key_password=None,
        anonymization_mask_type='full',  # 'full', 'center', or 'margin'
        anonymization_margin=0,  # Margin size for mask (0 = auto)
        # Enhanced encryption with signed permutation
        use_signed_permutation=False,  # Use SignedPermutationKey for stronger encryption
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
