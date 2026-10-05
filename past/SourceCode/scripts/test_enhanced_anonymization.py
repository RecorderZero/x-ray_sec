"""
Test Script for Enhanced Anonymization Module.

This script tests the enhanced anonymization transformations:
1. Rademacher (baseline)
2. Signed Permutation
3. Randomized Hadamard Transform
4. Block Orthogonal Rotation
5. Hybrid Multi-Layer

Tests include:
- Distribution preservation (N(0, I_d) invariance)
- Perfect reversibility
- Security (wrong key produces different output)
- Performance benchmarking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from typing import Dict, Tuple

from guided_diffusion.anonymization_enhanced import (
    EnhancedKey,
    EnhancedDiffusionAnonymizer,
    TransformationType,
    verify_distribution_preservation,
    verify_reversibility,
    verify_security,
)


def test_all_transforms(
    shape: Tuple[int, int, int] = (1, 64, 64),
    num_samples: int = 10,
    num_distribution_samples: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Test all transformation types.
    
    Args:
        shape: Shape of test data (C, H, W)
        num_samples: Number of samples for reversibility test
        num_distribution_samples: Number of samples for distribution test
        verbose: Print detailed results
        
    Returns:
        Dictionary with test results for each transformation type
    """
    results = {}
    
    transform_types = [
        TransformationType.RADEMACHER,
        TransformationType.SIGNED_PERMUTATION,
        TransformationType.HADAMARD,
        TransformationType.BLOCK_ROTATION,
        TransformationType.HYBRID,
    ]
    
    for transform_type in transform_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {transform_type.value}")
            print('='*60)
        
        try:
            # Create anonymizer
            anonymizer = EnhancedDiffusionAnonymizer(
                latent_shape=shape,
                transform_type=transform_type,
                mask_type='full',
                block_size=8,
                num_hadamard_layers=1,
            )
            
            # Generate key with fixed seed for reproducibility
            key = EnhancedKey(
                shape=shape,
                transform_type=transform_type,
                seed=42,
                block_size=8,
                num_hadamard_layers=1,
            )
            
            # Test 1: Reversibility
            is_reversible, max_error = verify_reversibility(
                anonymizer, key, num_samples=num_samples
            )
            
            if verbose:
                print(f"\n1. Reversibility Test:")
                print(f"   Status: {'✓ PASS' if is_reversible else '✗ FAIL'}")
                print(f"   Max reconstruction error: {max_error:.2e}")
            
            # Test 2: Distribution Preservation
            dist_results = verify_distribution_preservation(
                anonymizer, key, num_samples=num_distribution_samples
            )
            
            if verbose:
                print(f"\n2. Distribution Preservation Test:")
                print(f"   Status: {'✓ PASS' if dist_results['distribution_preserved'] else '✗ FAIL'}")
                print(f"   Original:    mean={dist_results['original_mean']:.4f}, std={dist_results['original_std']:.4f}")
                print(f"   Transformed: mean={dist_results['transformed_mean']:.4f}, std={dist_results['transformed_std']:.4f}")
            
            # Test 3: Security (wrong key)
            wrong_key = EnhancedKey(
                shape=shape,
                transform_type=transform_type,
                seed=123,  # Different seed
                block_size=8,
                num_hadamard_layers=1,
            )
            
            correct_error, wrong_error = verify_security(
                anonymizer, key, wrong_key, num_samples=num_samples
            )
            
            security_passed = correct_error < 1e-5 and wrong_error > 0.1
            
            if verbose:
                print(f"\n3. Security Test:")
                print(f"   Status: {'✓ PASS' if security_passed else '✗ FAIL'}")
                print(f"   Correct key error: {correct_error:.2e}")
                print(f"   Wrong key error: {wrong_error:.4f}")
            
            # Test 4: Performance Benchmark
            test_data = torch.randn(1, *shape)
            
            start_time = time.time()
            for _ in range(100):
                _ = anonymizer.anonymize(test_data, key)
            forward_time = (time.time() - start_time) / 100 * 1000
            
            start_time = time.time()
            anonymized = anonymizer.anonymize(test_data, key)
            for _ in range(100):
                _ = anonymizer.deanonymize(anonymized, key)
            backward_time = (time.time() - start_time) / 100 * 1000
            
            if verbose:
                print(f"\n4. Performance (per operation):")
                print(f"   Forward (anonymize): {forward_time:.3f} ms")
                print(f"   Backward (de-anonymize): {backward_time:.3f} ms")
            
            # Store results
            results[transform_type.value] = {
                'reversibility': {
                    'passed': is_reversible,
                    'max_error': max_error,
                },
                'distribution': {
                    'passed': dist_results['distribution_preserved'],
                    'original_mean': dist_results['original_mean'],
                    'original_std': dist_results['original_std'],
                    'transformed_mean': dist_results['transformed_mean'],
                    'transformed_std': dist_results['transformed_std'],
                },
                'security': {
                    'passed': security_passed,
                    'correct_key_error': correct_error,
                    'wrong_key_error': wrong_error,
                },
                'performance': {
                    'forward_ms': forward_time,
                    'backward_ms': backward_time,
                },
                'overall_passed': is_reversible and dist_results['distribution_preserved'] and security_passed,
            }
            
        except Exception as e:
            if verbose:
                print(f"   ERROR: {str(e)}")
            results[transform_type.value] = {
                'error': str(e),
                'overall_passed': False,
            }
    
    return results


def test_key_persistence(shape: Tuple[int, int, int] = (1, 64, 64)):
    """Test key save and load functionality."""
    print("\n" + "="*60)
    print("Testing Key Persistence")
    print("="*60)
    
    import tempfile
    
    for transform_type in TransformationType:
        print(f"\n{transform_type.value}:")
        
        # Create and save key
        original_key = EnhancedKey(
            shape=shape,
            transform_type=transform_type,
            seed=42,
            password="test_password",
            block_size=8,
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            key_path = f.name
        
        try:
            original_key.save(key_path)
            
            # Load key
            loaded_key = EnhancedKey.load(key_path)
            
            # Verify key properties match
            shape_match = original_key.shape == loaded_key.shape
            transform_match = original_key.transform_type == loaded_key.transform_type
            rademacher_match = torch.allclose(
                original_key.rademacher_key, 
                loaded_key.rademacher_key
            )
            
            all_passed = shape_match and transform_match and rademacher_match
            
            print(f"   Shape match: {'✓' if shape_match else '✗'}")
            print(f"   Transform match: {'✓' if transform_match else '✗'}")
            print(f"   Rademacher key match: {'✓' if rademacher_match else '✗'}")
            print(f"   Overall: {'✓ PASS' if all_passed else '✗ FAIL'}")
            
        finally:
            os.unlink(key_path)


def test_mask_types(shape: Tuple[int, int, int] = (1, 256, 256)):
    """Test different mask types."""
    print("\n" + "="*60)
    print("Testing Mask Types")
    print("="*60)
    
    mask_types = ['full', 'center', 'margin', 'lung_region']
    
    for mask_type in mask_types:
        print(f"\n{mask_type}:")
        
        anonymizer = EnhancedDiffusionAnonymizer(
            latent_shape=shape,
            transform_type=TransformationType.SIGNED_PERMUTATION,
            mask_type=mask_type,
            margin=16 if mask_type in ['center', 'margin'] else 0,
        )
        
        key = EnhancedKey(
            shape=shape,
            transform_type=TransformationType.SIGNED_PERMUTATION,
            seed=42,
        )
        
        test_data = torch.randn(1, *shape)
        anonymized = anonymizer.anonymize(test_data, key)
        recovered = anonymizer.deanonymize(anonymized, key)
        
        max_error = (test_data - recovered).abs().max().item()
        is_reversible = max_error < 1e-5
        
        # Calculate mask coverage
        mask = anonymizer.default_mask
        coverage = mask.sum().item() / mask.numel() * 100
        
        print(f"   Mask coverage: {coverage:.1f}%")
        print(f"   Reversibility: {'✓ PASS' if is_reversible else '✗ FAIL'} (error: {max_error:.2e})")


def print_summary(results: Dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"\n{'Transform Type':<25} {'Reversible':<12} {'Dist. Pres.':<12} {'Security':<12} {'Overall':<10}")
    print("-"*75)
    
    for transform, data in results.items():
        if 'error' in data:
            print(f"{transform:<25} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'FAIL':<10}")
        else:
            rev = '✓' if data['reversibility']['passed'] else '✗'
            dist = '✓' if data['distribution']['passed'] else '✗'
            sec = '✓' if data['security']['passed'] else '✗'
            overall = '✓ PASS' if data['overall_passed'] else '✗ FAIL'
            print(f"{transform:<25} {rev:<12} {dist:<12} {sec:<12} {overall:<10}")
    
    # Performance comparison
    print("\n" + "-"*60)
    print("Performance Comparison (ms per operation):")
    print(f"{'Transform Type':<25} {'Forward':<15} {'Backward':<15}")
    print("-"*55)
    
    for transform, data in results.items():
        if 'performance' in data:
            fwd = f"{data['performance']['forward_ms']:.3f}"
            bwd = f"{data['performance']['backward_ms']:.3f}"
            print(f"{transform:<25} {fwd:<15} {bwd:<15}")
    
    # Count passed tests
    passed = sum(1 for data in results.values() if data.get('overall_passed', False))
    total = len(results)
    
    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} transformations passed all tests")
    print("="*60)


def main():
    print("="*60)
    print("Enhanced Anonymization Module Test Suite")
    print("="*60)
    
    # Run all transformation tests
    results = test_all_transforms(
        shape=(1, 64, 64),
        num_samples=10,
        num_distribution_samples=100,
        verbose=True,
    )
    
    # Test key persistence
    test_key_persistence(shape=(1, 64, 64))
    
    # Test mask types
    test_mask_types(shape=(1, 256, 256))
    
    # Print summary
    print_summary(results)
    
    # Return success status
    all_passed = all(data.get('overall_passed', False) for data in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
