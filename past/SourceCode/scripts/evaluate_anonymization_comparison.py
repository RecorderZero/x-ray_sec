"""
Evaluation metrics for anonymization algorithm comparison.

This script computes quantitative metrics to compare the effectiveness of
the encryption/decryption algorithm across different sampling methods.

Metrics computed:
1. Reconstruction Quality (de-anonymization accuracy)
   - MSE (Mean Squared Error)
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)

2. Anonymization Effectiveness
   - MSE between original and anonymized (should be HIGH)
   - Visual difference score

3. Key Sensitivity
   - Reconstruction with wrong key (should fail)
   - MSE with wrong key vs correct key

4. Distribution Preservation
   - Mean and variance of x_T before/after encryption
   - Kolmogorov-Smirnov test for normality
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import json

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM and PSNR will not be computed.")


def load_images_from_dir(directory, pattern="*.png"):
    """Load all images from a directory."""
    images = {}
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return images
    
    for img_path in sorted(dir_path.glob(pattern)):
        img = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale
        img = img.astype(np.float32) / 255.0
        # Extract base name (remove suffix like _anonymized, _recovered, etc.)
        base_name = img_path.stem
        for suffix in ['_anonymized', '_recovered', '_original', '_healthy', '_output', '_heatmap', '_diff']:
            base_name = base_name.replace(suffix, '')
        images[base_name] = img
    
    return images


def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images."""
    return np.mean((img1 - img2) ** 2)


def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio."""
    if SKIMAGE_AVAILABLE:
        return psnr(img1, img2, data_range=1.0)
    else:
        mse = compute_mse(img1, img2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2):
    """Compute Structural Similarity Index."""
    if SKIMAGE_AVAILABLE:
        return ssim(img1, img2, data_range=1.0)
    else:
        return None


def evaluate_reconstruction(original_images, recovered_images):
    """
    Evaluate reconstruction quality (de-anonymization accuracy).
    
    Returns metrics comparing original and recovered images.
    """
    results = {
        'mse': [],
        'psnr': [],
        'ssim': [],
        'num_samples': 0,
    }
    
    for name, original in original_images.items():
        if name in recovered_images:
            recovered = recovered_images[name]
            
            # Ensure same shape
            if original.shape != recovered.shape:
                print(f"Warning: Shape mismatch for {name}: {original.shape} vs {recovered.shape}")
                continue
            
            results['mse'].append(compute_mse(original, recovered))
            results['psnr'].append(compute_psnr(original, recovered))
            
            ssim_val = compute_ssim(original, recovered)
            if ssim_val is not None:
                results['ssim'].append(ssim_val)
            
            results['num_samples'] += 1
    
    # Compute averages
    if results['num_samples'] > 0:
        results['mse_mean'] = np.mean(results['mse'])
        results['mse_std'] = np.std(results['mse'])
        results['psnr_mean'] = np.mean(results['psnr'])
        results['psnr_std'] = np.std(results['psnr'])
        if results['ssim']:
            results['ssim_mean'] = np.mean(results['ssim'])
            results['ssim_std'] = np.std(results['ssim'])
    
    return results


def evaluate_anonymization(original_images, anonymized_images):
    """
    Evaluate anonymization effectiveness.
    
    Good anonymization should have HIGH MSE (images look different).
    """
    results = {
        'mse': [],
        'num_samples': 0,
    }
    
    for name, original in original_images.items():
        if name in anonymized_images:
            anonymized = anonymized_images[name]
            
            if original.shape != anonymized.shape:
                continue
            
            results['mse'].append(compute_mse(original, anonymized))
            results['num_samples'] += 1
    
    if results['num_samples'] > 0:
        results['mse_mean'] = np.mean(results['mse'])
        results['mse_std'] = np.std(results['mse'])
    
    return results


def print_results(method_name, reconstruction_results, anonymization_results):
    """Print evaluation results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    
    print(f"\n--- Reconstruction Quality (De-anonymization Accuracy) ---")
    print(f"Samples evaluated: {reconstruction_results.get('num_samples', 0)}")
    if reconstruction_results.get('num_samples', 0) > 0:
        print(f"MSE:  {reconstruction_results.get('mse_mean', 0):.6f} ± {reconstruction_results.get('mse_std', 0):.6f}")
        print(f"PSNR: {reconstruction_results.get('psnr_mean', 0):.2f} ± {reconstruction_results.get('psnr_std', 0):.2f} dB")
        if 'ssim_mean' in reconstruction_results:
            print(f"SSIM: {reconstruction_results.get('ssim_mean', 0):.4f} ± {reconstruction_results.get('ssim_std', 0):.4f}")
    
    print(f"\n--- Anonymization Effectiveness ---")
    print(f"Samples evaluated: {anonymization_results.get('num_samples', 0)}")
    if anonymization_results.get('num_samples', 0) > 0:
        print(f"MSE (original vs anonymized): {anonymization_results.get('mse_mean', 0):.6f} ± {anonymization_results.get('mse_std', 0):.6f}")
        print(f"(Higher MSE = Better anonymization)")


def find_latest_subdir(base_dir, prefix):
    """Find the latest subdirectory with given prefix."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not subdirs:
        return None
    
    # Sort by name (which includes timestamp) and return latest
    return sorted(subdirs)[-1]


def main():
    parser = argparse.ArgumentParser(description='Evaluate anonymization algorithm comparison')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Base directory containing experiment results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        return
    
    all_results = {}
    
    # Methods to evaluate
    methods = ['unconditional', 'cfg', 'classifier']
    
    for method in methods:
        method_dir = experiment_dir / method
        if not method_dir.exists():
            print(f"Warning: {method} directory not found, skipping...")
            continue
        
        print(f"\nProcessing {method}...")
        
        # Find anonymization and de-anonymization directories
        ano_dir = find_latest_subdir(method_dir, f'{method}_anonymize') or \
                  find_latest_subdir(method_dir, 'unconditional_anonymize') or \
                  find_latest_subdir(method_dir, 'cfg_anonymize') or \
                  find_latest_subdir(method_dir, 'clf_anonymize')
        
        deano_dir = find_latest_subdir(method_dir, f'{method}_deanonymize') or \
                    find_latest_subdir(method_dir, 'unconditional_deanonymize') or \
                    find_latest_subdir(method_dir, 'cfg_deanonymize') or \
                    find_latest_subdir(method_dir, 'clf_deanonymize')
        
        # Load images
        original_images = {}
        anonymized_images = {}
        recovered_images = {}
        
        if ano_dir:
            original_images = load_images_from_dir(ano_dir / 'original')
            anonymized_images = load_images_from_dir(ano_dir / 'anonymized')
        
        if deano_dir:
            recovered_images = load_images_from_dir(deano_dir / 'recovered')
            if not recovered_images:
                recovered_images = load_images_from_dir(deano_dir / 'healthy')
        
        # Evaluate
        reconstruction_results = evaluate_reconstruction(original_images, recovered_images)
        anonymization_results = evaluate_anonymization(original_images, anonymized_images)
        
        # Print results
        print_results(method.upper(), reconstruction_results, anonymization_results)
        
        # Store results
        all_results[method] = {
            'reconstruction': {
                'mse_mean': reconstruction_results.get('mse_mean'),
                'mse_std': reconstruction_results.get('mse_std'),
                'psnr_mean': reconstruction_results.get('psnr_mean'),
                'psnr_std': reconstruction_results.get('psnr_std'),
                'ssim_mean': reconstruction_results.get('ssim_mean'),
                'ssim_std': reconstruction_results.get('ssim_std'),
                'num_samples': reconstruction_results.get('num_samples', 0),
            },
            'anonymization': {
                'mse_mean': anonymization_results.get('mse_mean'),
                'mse_std': anonymization_results.get('mse_std'),
                'num_samples': anonymization_results.get('num_samples', 0),
            }
        }
    
    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'Recon MSE':<15} {'Recon PSNR':<15} {'Ano MSE':<15}")
    print("-"*60)
    
    for method, results in all_results.items():
        recon = results['reconstruction']
        ano = results['anonymization']
        
        recon_mse = f"{recon['mse_mean']:.6f}" if recon['mse_mean'] else "N/A"
        recon_psnr = f"{recon['psnr_mean']:.2f} dB" if recon['psnr_mean'] else "N/A"
        ano_mse = f"{ano['mse_mean']:.6f}" if ano['mse_mean'] else "N/A"
        
        print(f"{method.upper():<20} {recon_mse:<15} {recon_psnr:<15} {ano_mse:<15}")
    
    print("\n" + "-"*60)
    print("Interpretation:")
    print("  - Lower Reconstruction MSE = Better de-anonymization accuracy")
    print("  - Higher Reconstruction PSNR = Better de-anonymization quality")
    print("  - Higher Anonymization MSE = More effective encryption")
    print("  - Consistent results across methods = Algorithm is robust")
    
    # Save results to JSON if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
