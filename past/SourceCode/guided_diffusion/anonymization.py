"""
Secure and Reversible Anonymization Module.
Based on the paper "Secure and Reversible Face Anonymization with Diffusion Models"
by Pol Labarbarie, Vincent Itier and William Puech.

Adapted for chest X-ray medical images.

Enhanced with Signed Permutation Transformation for stronger encryption.

Key concepts:
1. Rademacher key: k ∈ {-1, +1}^d for element-wise sign flipping
2. Permutation: P is a random permutation matrix for position shuffling
3. Signed Permutation: z_ano_T = P(k ⊙ x_T) - combines sign flip and permutation
4. Anonymization: z_ano_T = Mz ⊙ (k ⊙ zT) + (1 - Mz) ⊙ zT (basic)
                  z_ano_T = P(Mz ⊙ (k ⊙ zT) + (1 - Mz) ⊙ zT) (with permutation)
5. De-anonymization: Apply inverse permutation P^(-1), then apply key again
6. Perfect reversibility: k ⊙ k = 1 and P^(-1) P = I

Security improvements with Signed Permutation:
- Key space expands from 2^d to 2^d × d!
- Breaks spatial correlation in latent space
- Provides defense against residual information leakage
"""

import os
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, List


class RademacherKey:
    """
    Rademacher key for secure anonymization.
    A Rademacher distribution produces values in {-1, +1} with equal probability.
    
    The key can be:
    - Generated randomly
    - Derived from a password/seed
    - Loaded from file
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        key: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize a Rademacher key.
        
        Args:
            shape: Shape of the key (must match latent space shape)
            key: Pre-existing key tensor
            seed: Random seed for reproducible key generation
            password: String password to derive key from
        """
        self.shape = shape
        
        # Convert seed to int if provided as string
        if seed is not None:
            seed = int(seed)
        
        if key is not None:
            # Use provided key
            assert key.shape == shape, f"Key shape {key.shape} doesn't match {shape}"
            self.key = key
        elif password is not None:
            # Derive key from password
            self.key = self._derive_key_from_password(password, shape)
        elif seed is not None:
            # Generate key with seed
            self.key = self._generate_key(shape, seed)
        else:
            # Generate random key
            self.key = self._generate_key(shape)
    
    @staticmethod
    def _generate_key(shape: Tuple[int, ...], seed: Optional[int] = None) -> torch.Tensor:
        """Generate a random Rademacher key."""
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            # Bernoulli(0.5) -> {0, 1}, then convert to {-1, +1}
            binary = torch.bernoulli(torch.ones(shape) * 0.5, generator=generator)
        else:
            binary = torch.bernoulli(torch.ones(shape) * 0.5)
        
        key = 2 * binary - 1  # Convert {0, 1} to {-1, +1}
        return key
    
    @staticmethod
    def _derive_key_from_password(password: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """Derive a deterministic key from a password string."""
        # Hash the password to get a seed
        password_bytes = password.encode('utf-8')
        hash_digest = hashlib.sha256(password_bytes).digest()
        seed = int.from_bytes(hash_digest[:4], byteorder='big')
        
        return RademacherKey._generate_key(shape, seed)
    
    def to(self, device: torch.device) -> 'RademacherKey':
        """Move key to device."""
        self.key = self.key.to(device)
        return self
    
    def save(self, path: str):
        """Save key to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'key': self.key, 'shape': self.shape}, path)
    
    @classmethod
    def load(cls, path: str) -> 'RademacherKey':
        """Load key from file."""
        data = torch.load(path, map_location='cpu')
        return cls(shape=data['shape'], key=data['key'])


class PermutationKey:
    """
    Permutation key for enhanced anonymization security.
    
    A permutation randomly shuffles the positions of elements in the latent space,
    breaking spatial correlations and significantly expanding the key space.
    
    Mathematical property:
    - If x ~ N(0, I), then P @ x ~ N(0, I) for any permutation matrix P
    - This is because permutation matrices are orthogonal: P @ P.T = I
    
    The key can be:
    - Generated randomly
    - Derived from a password/seed
    - Loaded from file
    """
    
    def __init__(
        self,
        size: int,
        permutation: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize a Permutation key.
        
        Args:
            size: Total number of elements to permute (d = C × H × W)
            permutation: Pre-existing permutation indices
            seed: Random seed for reproducible permutation generation
            password: String password to derive permutation from
        """
        self.size = size
        
        # Convert seed to int if provided as string
        if seed is not None:
            seed = int(seed)
        
        if permutation is not None:
            assert len(permutation) == size, f"Permutation size {len(permutation)} doesn't match {size}"
            self.permutation = permutation
            self.inverse_permutation = self._compute_inverse(permutation)
        elif password is not None:
            self.permutation = self._derive_permutation_from_password(password, size)
            self.inverse_permutation = self._compute_inverse(self.permutation)
        elif seed is not None:
            self.permutation = self._generate_permutation(size, seed)
            self.inverse_permutation = self._compute_inverse(self.permutation)
        else:
            self.permutation = self._generate_permutation(size)
            self.inverse_permutation = self._compute_inverse(self.permutation)
    
    @staticmethod
    def _generate_permutation(size: int, seed: Optional[int] = None) -> torch.Tensor:
        """Generate a random permutation."""
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            permutation = torch.randperm(size, generator=generator)
        else:
            permutation = torch.randperm(size)
        return permutation
    
    @staticmethod
    def _derive_permutation_from_password(password: str, size: int) -> torch.Tensor:
        """Derive a deterministic permutation from a password string."""
        # Use a different hash suffix to get different seed from RademacherKey
        password_bytes = (password + "_permutation").encode('utf-8')
        hash_digest = hashlib.sha256(password_bytes).digest()
        seed = int.from_bytes(hash_digest[:4], byteorder='big')
        return PermutationKey._generate_permutation(size, seed)
    
    @staticmethod
    def _compute_inverse(permutation: torch.Tensor) -> torch.Tensor:
        """Compute the inverse permutation."""
        inverse = torch.zeros_like(permutation)
        inverse[permutation] = torch.arange(len(permutation))
        return inverse
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply permutation to tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            Permuted tensor with same shape
        """
        has_batch = x.dim() == 4
        if not has_batch:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, -1)
        
        # Ensure permutation is on correct device
        perm = self.permutation.to(x.device)
        
        # Apply permutation
        x_permuted = x_flat[:, perm]
        
        # Reshape back
        x_permuted = x_permuted.view(original_shape)
        
        if not has_batch:
            x_permuted = x_permuted.squeeze(0)
        
        return x_permuted
    
    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse permutation to tensor.
        
        Args:
            x: Permuted tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            Original tensor with same shape
        """
        has_batch = x.dim() == 4
        if not has_batch:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, -1)
        
        # Ensure inverse permutation is on correct device
        inv_perm = self.inverse_permutation.to(x.device)
        
        # Apply inverse permutation
        x_unpermuted = x_flat[:, inv_perm]
        
        # Reshape back
        x_unpermuted = x_unpermuted.view(original_shape)
        
        if not has_batch:
            x_unpermuted = x_unpermuted.squeeze(0)
        
        return x_unpermuted
    
    def to(self, device: torch.device) -> 'PermutationKey':
        """Move permutation to device."""
        self.permutation = self.permutation.to(device)
        self.inverse_permutation = self.inverse_permutation.to(device)
        return self
    
    def save(self, path: str):
        """Save permutation key to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'permutation': self.permutation,
            'inverse_permutation': self.inverse_permutation,
            'size': self.size
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PermutationKey':
        """Load permutation key from file."""
        data = torch.load(path, map_location='cpu')
        key = cls(size=data['size'], permutation=data['permutation'])
        key.inverse_permutation = data['inverse_permutation']
        return key


class SignedPermutationKey:
    """
    Signed Permutation key combining Rademacher sign flipping and permutation.
    
    This implements the Signed Permutation Transformation from the Hyperoctahedral Group B_d:
        z_ano = P(k ⊙ x_T)
    
    Where:
    - k is a Rademacher key ∈ {-1, +1}^d for sign flipping
    - P is a permutation matrix for position shuffling
    
    Security improvements:
    1. Key space expands from 2^d to 2^d × d! (astronomically larger)
    2. Breaks spatial correlations that pure sign flipping preserves
    3. Provides defense against attacks exploiting magnitude preservation
    4. Each element's position AND sign are both randomized
    
    Mathematical property:
    - The transformation preserves N(0, I) distribution
    - The transformation is perfectly reversible: x = k ⊙ P^(-1)(z_ano)
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        rademacher_key: Optional[RademacherKey] = None,
        permutation_key: Optional[PermutationKey] = None,
        seed: Optional[int] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize a Signed Permutation key.
        
        Args:
            shape: Shape of the data (C, H, W)
            rademacher_key: Pre-existing Rademacher key
            permutation_key: Pre-existing Permutation key
            seed: Random seed for reproducible key generation
            password: String password to derive keys from
        """
        self.shape = shape
        self.size = int(np.prod(shape))
        
        # Convert seed to int if it's a string
        if seed is not None:
            seed = int(seed)
        
        # Generate or use provided Rademacher key
        if rademacher_key is not None:
            self.rademacher_key = rademacher_key
        else:
            self.rademacher_key = RademacherKey(
                shape=shape,
                seed=seed,
                password=password
            )
        
        # Generate or use provided Permutation key
        if permutation_key is not None:
            self.permutation_key = permutation_key
        else:
            # Use different seed derivation for Permutation
            perm_seed = None
            if seed is not None:
                # Derive a different seed for permutation
                perm_seed = seed + 1000000  # Offset to ensure different permutation
            self.permutation_key = PermutationKey(
                size=self.size,
                seed=perm_seed,
                password=password
            )
    
    def apply(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply signed permutation transformation: z_ano = P(k ⊙ x)
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (C, H, W)
            mask: Optional mask for selective application
            
        Returns:
            Transformed tensor with same shape
        """
        has_batch = x.dim() == 4
        if not has_batch:
            x = x.unsqueeze(0)
        
        # Get Rademacher key
        key = self.rademacher_key.key.to(x.device)
        
        # Expand key dimensions if needed
        if key.dim() < x.dim():
            key = key.unsqueeze(0).expand_as(x)
        
        # Step 1: Apply Rademacher sign flip
        if mask is not None:
            mask = mask.to(x.device)
            if mask.dim() < x.dim():
                mask = mask.unsqueeze(0).expand_as(x)
            x_signed = mask * (key * x) + (1 - mask) * x
        else:
            x_signed = key * x
        
        # Step 2: Apply permutation
        x_permuted = self.permutation_key.apply(x_signed)
        
        if not has_batch:
            x_permuted = x_permuted.squeeze(0)
        
        return x_permuted
    
    def apply_inverse(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply inverse signed permutation: x = k ⊙ P^(-1)(z_ano)
        
        Args:
            x: Transformed tensor of shape (B, C, H, W) or (C, H, W)
            mask: Optional mask (same as used in apply)
            
        Returns:
            Original tensor with same shape
        """
        has_batch = x.dim() == 4
        if not has_batch:
            x = x.unsqueeze(0)
        
        # Step 1: Apply inverse permutation
        x_unpermuted = self.permutation_key.apply_inverse(x)
        
        # Get Rademacher key
        key = self.rademacher_key.key.to(x.device)
        
        # Expand key dimensions if needed
        if key.dim() < x_unpermuted.dim():
            key = key.unsqueeze(0).expand_as(x_unpermuted)
        
        # Step 2: Apply Rademacher sign flip (self-inverse)
        if mask is not None:
            mask = mask.to(x.device)
            if mask.dim() < x_unpermuted.dim():
                mask = mask.unsqueeze(0).expand_as(x_unpermuted)
            x_original = mask * (key * x_unpermuted) + (1 - mask) * x_unpermuted
        else:
            x_original = key * x_unpermuted
        
        if not has_batch:
            x_original = x_original.squeeze(0)
        
        return x_original
    
    def to(self, device: torch.device) -> 'SignedPermutationKey':
        """Move keys to device."""
        self.rademacher_key.to(device)
        self.permutation_key.to(device)
        return self
    
    def save(self, path: str):
        """Save signed permutation key to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'shape': self.shape,
            'size': self.size,
            'rademacher_key': self.rademacher_key.key,
            'permutation': self.permutation_key.permutation,
            'inverse_permutation': self.permutation_key.inverse_permutation,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'SignedPermutationKey':
        """Load signed permutation key from file."""
        data = torch.load(path, map_location='cpu')
        
        rademacher_key = RademacherKey(shape=data['shape'], key=data['rademacher_key'])
        permutation_key = PermutationKey(size=data['size'], permutation=data['permutation'])
        permutation_key.inverse_permutation = data['inverse_permutation']
        
        return cls(
            shape=data['shape'],
            rademacher_key=rademacher_key,
            permutation_key=permutation_key
        )
    
    @property
    def key(self) -> torch.Tensor:
        """Return Rademacher key tensor for backward compatibility."""
        return self.rademacher_key.key


class AnonymizationMask:
    """
    Mask for selective anonymization.
    
    For medical images like chest X-rays, we might want to:
    - Anonymize the entire image (mask = all ones)
    - Preserve certain regions (e.g., corners with patient info)
    - Customize mask based on image content
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        mask: Optional[torch.Tensor] = None,
        mask_type: str = 'full',
        margin: int = 0,
    ):
        """
        Initialize anonymization mask.
        
        Args:
            shape: Shape of the mask (H, W) or (C, H, W)
            mask: Pre-existing mask tensor
            mask_type: Type of mask ('full', 'center', 'margin')
            margin: Margin size for margin-type mask
        """
        self.shape = shape
        
        if mask is not None:
            self.mask = mask
        else:
            self.mask = self._create_mask(shape, mask_type, margin)
    
    @staticmethod
    def _create_mask(
        shape: Tuple[int, ...],
        mask_type: str,
        margin: int = 0
    ) -> torch.Tensor:
        """Create mask based on type."""
        if mask_type == 'full':
            # Anonymize everything
            return torch.ones(shape)
        
        elif mask_type == 'center':
            # Only anonymize center region
            mask = torch.zeros(shape)
            if len(shape) == 2:
                h, w = shape
                m = margin or min(h, w) // 8
                mask[m:-m, m:-m] = 1.0
            elif len(shape) >= 3:
                h, w = shape[-2], shape[-1]
                m = margin or min(h, w) // 8
                mask[..., m:-m, m:-m] = 1.0
            return mask
        
        elif mask_type == 'margin':
            # Preserve margins, anonymize center
            mask = torch.ones(shape)
            if len(shape) == 2:
                h, w = shape
                m = margin or min(h, w) // 16
                mask[:m, :] = 0.0
                mask[-m:, :] = 0.0
                mask[:, :m] = 0.0
                mask[:, -m:] = 0.0
            elif len(shape) >= 3:
                h, w = shape[-2], shape[-1]
                m = margin or min(h, w) // 16
                mask[..., :m, :] = 0.0
                mask[..., -m:, :] = 0.0
                mask[..., :, :m] = 0.0
                mask[..., :, -m:] = 0.0
            return mask
        
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
    
    def to(self, device: torch.device) -> 'AnonymizationMask':
        """Move mask to device."""
        self.mask = self.mask.to(device)
        return self


class DiffusionAnonymizer(nn.Module):
    """
    Secure and reversible anonymization module for diffusion models.
    
    This module performs anonymization and de-anonymization in the latent space
    of a diffusion model, following the method described in the paper.
    
    Supports two modes:
    1. Basic Rademacher (use_permutation=False):
       z_ano_T = M ⊙ (k ⊙ zT) + (1-M) ⊙ zT
       
    2. Signed Permutation (use_permutation=True):
       z_ano_T = P(M ⊙ (k ⊙ zT) + (1-M) ⊙ zT)
       
       This provides stronger security by:
       - Expanding key space from 2^d to 2^d × d!
       - Breaking spatial correlations
       - Defending against magnitude-based attacks
    
    De-anonymization is the inverse process with the same key(s).
    """
    
    def __init__(
        self,
        latent_shape: Tuple[int, ...],
        mask_type: str = 'full',
        margin: int = 0,
        use_permutation: bool = False,
    ):
        """
        Initialize the anonymizer.
        
        Args:
            latent_shape: Shape of latent representation (C, H, W)
            mask_type: Type of mask for selective anonymization
            margin: Margin for mask if applicable
            use_permutation: Whether to use signed permutation (stronger encryption)
        """
        super().__init__()
        self.latent_shape = latent_shape
        self.mask_type = mask_type
        self.margin = margin
        self.use_permutation = use_permutation
        
        # Initialize default mask
        self.register_buffer(
            'default_mask',
            AnonymizationMask(latent_shape, mask_type=mask_type, margin=margin).mask
        )
    
    def apply_key(
        self,
        z_t: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Rademacher key to latent representation.
        
        This operation is self-inverse: apply_key(apply_key(z, k), k) = z
        
        Args:
            z_t: Latent representation at timestep T (can be any shape)
            key: Rademacher key tensor (must match z_t shape or be broadcastable)
            mask: Optional mask for selective application
            
        Returns:
            z_ano_t: Anonymized latent representation
        """
        # Dynamically create mask if not provided
        if mask is None:
            # Get the spatial shape from input (excluding batch dim)
            input_shape = z_t.shape[1:] if z_t.dim() > 3 else z_t.shape
            if input_shape != self.latent_shape:
                # Create mask dynamically for the input shape
                mask = AnonymizationMask(
                    input_shape, 
                    mask_type=self.mask_type, 
                    margin=self.margin
                ).mask
            else:
                mask = self.default_mask
        
        # Ensure key and mask are on correct device
        key = key.to(z_t.device)
        mask = mask.to(z_t.device)
        
        # Expand key dimensions if needed for batch
        if key.dim() < z_t.dim():
            key = key.unsqueeze(0).expand_as(z_t)
        
        # If key shape doesn't match, we need to handle it
        # This can happen when key was generated for latent space but input is pixel space
        if key.shape != z_t.shape:
            # Resize key to match input shape using interpolation
            key_resized = torch.nn.functional.interpolate(
                key.unsqueeze(0) if key.dim() == 3 else key,
                size=z_t.shape[-2:],
                mode='nearest'
            )
            # Adjust channels if needed
            if key_resized.shape[1] != z_t.shape[1]:
                # Repeat or slice channels to match
                if key_resized.shape[1] < z_t.shape[1]:
                    key_resized = key_resized.repeat(1, z_t.shape[1] // key_resized.shape[1] + 1, 1, 1)
                key_resized = key_resized[:, :z_t.shape[1], :, :]
            key = key_resized
            if key.dim() > z_t.dim():
                key = key.squeeze(0)
        
        # Expand mask dimensions if needed for batch
        if mask.dim() < z_t.dim():
            mask = mask.unsqueeze(0).expand_as(z_t)
        
        # Apply key: z_ano = M ⊙ (k ⊙ z) + (1-M) ⊙ z
        z_flipped = key * z_t
        z_ano = mask * z_flipped + (1 - mask) * z_t
        
        return z_ano
    
    def anonymize_latent(
        self,
        z_t: torch.Tensor,
        key: Union[torch.Tensor, RademacherKey, SignedPermutationKey],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Anonymize latent representation.
        
        Args:
            z_t: Latent at timestep T from DDIM forward
            key: Rademacher key, SignedPermutationKey, or key tensor
            mask: Optional anonymization mask
            
        Returns:
            z_ano_t: Anonymized latent
        """
        # Handle SignedPermutationKey for enhanced security
        if isinstance(key, SignedPermutationKey):
            # Use signed permutation transformation
            if mask is None:
                input_shape = z_t.shape[1:] if z_t.dim() > 3 else z_t.shape
                if input_shape != self.latent_shape:
                    mask = AnonymizationMask(
                        input_shape,
                        mask_type=self.mask_type,
                        margin=self.margin
                    ).mask
                else:
                    mask = self.default_mask
            return key.apply(z_t, mask)
        
        # Handle basic Rademacher key
        if isinstance(key, RademacherKey):
            key = key.key
        return self.apply_key(z_t, key, mask)
    
    def deanonymize_latent(
        self,
        z_ano_t: torch.Tensor,
        key: Union[torch.Tensor, RademacherKey, SignedPermutationKey],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        De-anonymize latent representation.
        
        For Rademacher key: operation is self-inverse (k ⊙ k = 1)
        For SignedPermutationKey: apply inverse permutation then sign flip
        
        Args:
            z_ano_t: Anonymized latent
            key: Same key used for anonymization
            mask: Same mask used for anonymization
            
        Returns:
            z_rec_t: Recovered latent (should equal original z_t)
        """
        # Handle SignedPermutationKey
        if isinstance(key, SignedPermutationKey):
            # Use inverse signed permutation transformation
            if mask is None:
                input_shape = z_ano_t.shape[1:] if z_ano_t.dim() > 3 else z_ano_t.shape
                if input_shape != self.latent_shape:
                    mask = AnonymizationMask(
                        input_shape,
                        mask_type=self.mask_type,
                        margin=self.margin
                    ).mask
                else:
                    mask = self.default_mask
            return key.apply_inverse(z_ano_t, mask)
        
        # Handle basic Rademacher key (self-inverse)
        if isinstance(key, RademacherKey):
            key = key.key
        return self.apply_key(z_ano_t, key, mask)
    
    def generate_key(
        self,
        batch_size: int = 1,
        seed: Optional[int] = None,
        password: Optional[str] = None,
        device: torch.device = None,
        use_permutation: Optional[bool] = None,
    ) -> Union[RademacherKey, SignedPermutationKey]:
        """
        Generate a new encryption key.
        
        Args:
            batch_size: Number of keys to generate (for batch processing)
            seed: Optional seed for reproducibility
            password: Optional password to derive key from
            device: Device to place key on
            use_permutation: Whether to use signed permutation (defaults to self.use_permutation)
            
        Returns:
            RademacherKey or SignedPermutationKey object
        """
        if use_permutation is None:
            use_permutation = self.use_permutation
        
        if batch_size == 1:
            shape = self.latent_shape
        else:
            shape = (batch_size,) + self.latent_shape
        
        if use_permutation:
            key = SignedPermutationKey(shape=shape, seed=seed, password=password)
        else:
            key = RademacherKey(shape=shape, seed=seed, password=password)
        
        if device is not None:
            key = key.to(device)
        
        return key


def verify_reversibility(
    anonymizer: DiffusionAnonymizer,
    z_t: torch.Tensor,
    key: Union[RademacherKey, SignedPermutationKey],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, float]:
    """
    Verify that anonymization is perfectly reversible.
    
    Args:
        anonymizer: The anonymizer module
        z_t: Original latent
        key: Rademacher key or SignedPermutationKey
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Tuple of (is_reversible, mse)
    """
    z_ano = anonymizer.anonymize_latent(z_t, key)
    z_rec = anonymizer.deanonymize_latent(z_ano, key)
    
    mse = torch.mean((z_t - z_rec) ** 2).item()
    is_close = torch.allclose(z_t, z_rec, rtol=rtol, atol=atol)
    
    return is_close, mse


def verify_signed_permutation_security(
    shape: Tuple[int, ...],
    num_tests: int = 100,
) -> dict:
    """
    Verify security properties of signed permutation transformation.
    
    Tests:
    1. Distribution preservation: transformed data should still be N(0, I)
    2. Perfect reversibility: x = inverse(transform(x))
    3. Key sensitivity: wrong key should produce garbage
    
    Args:
        shape: Shape of test data (C, H, W)
        num_tests: Number of random tests
        
    Returns:
        Dictionary with test results
    """
    results = {
        'distribution_preserved': True,
        'reversibility_verified': True,
        'key_sensitive': True,
        'mean_mse_recovery': 0.0,
        'mean_wrong_key_mse': 0.0,
    }
    
    mse_recovery_list = []
    mse_wrong_key_list = []
    
    for _ in range(num_tests):
        # Generate random N(0, I) data
        x = torch.randn(1, *shape)
        
        # Generate key
        key = SignedPermutationKey(shape=shape)
        
        # Test transformation
        x_ano = key.apply(x)
        
        # Test distribution preservation (mean and variance)
        mean_diff = abs(x_ano.mean().item())
        var_diff = abs(x_ano.var().item() - 1.0)
        if mean_diff > 0.1 or var_diff > 0.1:
            results['distribution_preserved'] = False
        
        # Test reversibility
        x_rec = key.apply_inverse(x_ano)
        mse = torch.mean((x - x_rec) ** 2).item()
        mse_recovery_list.append(mse)
        if mse > 1e-10:
            results['reversibility_verified'] = False
        
        # Test key sensitivity (wrong key)
        wrong_key = SignedPermutationKey(shape=shape)
        x_wrong = wrong_key.apply_inverse(x_ano)
        wrong_mse = torch.mean((x - x_wrong) ** 2).item()
        mse_wrong_key_list.append(wrong_mse)
    
    results['mean_mse_recovery'] = np.mean(mse_recovery_list)
    results['mean_wrong_key_mse'] = np.mean(mse_wrong_key_list)
    
    # Wrong key should produce high MSE
    if results['mean_wrong_key_mse'] < 0.1:
        results['key_sensitive'] = False
    
    return results


def verify_security(
    anonymizer: DiffusionAnonymizer,
    z_t: torch.Tensor,
    correct_key: RademacherKey,
    wrong_key: RademacherKey,
) -> Tuple[float, float]:
    """
    Verify security by comparing recovery with correct vs wrong key.
    
    Args:
        anonymizer: The anonymizer module
        z_t: Original latent
        correct_key: The correct key
        wrong_key: A different key
        
    Returns:
        Tuple of (correct_mse, wrong_mse)
    """
    z_ano = anonymizer.anonymize_latent(z_t, correct_key)
    
    # Recovery with correct key
    z_rec_correct = anonymizer.deanonymize_latent(z_ano, correct_key)
    correct_mse = torch.mean((z_t - z_rec_correct) ** 2).item()
    
    # Recovery with wrong key
    z_rec_wrong = anonymizer.deanonymize_latent(z_ano, wrong_key)
    wrong_mse = torch.mean((z_t - z_rec_wrong) ** 2).item()
    
    return correct_mse, wrong_mse


class AnonymizationManager:
    """
    High-level manager for anonymization operations.
    Handles key management, saving/loading, and batch operations.
    """
    
    def __init__(
        self,
        anonymizer: DiffusionAnonymizer,
        key_dir: str = './anonymization_keys',
    ):
        """
        Initialize the manager.
        
        Args:
            anonymizer: DiffusionAnonymizer instance
            key_dir: Directory for storing keys
        """
        self.anonymizer = anonymizer
        self.key_dir = key_dir
        os.makedirs(key_dir, exist_ok=True)
        self.keys = {}
    
    def generate_and_save_key(
        self,
        key_id: str,
        seed: Optional[int] = None,
        password: Optional[str] = None,
    ) -> RademacherKey:
        """Generate a key and save it with an ID."""
        key = self.anonymizer.generate_key(seed=seed, password=password)
        key_path = os.path.join(self.key_dir, f'{key_id}.pt')
        key.save(key_path)
        self.keys[key_id] = key
        return key
    
    def load_key(self, key_id: str) -> RademacherKey:
        """Load a key by ID."""
        if key_id in self.keys:
            return self.keys[key_id]
        
        key_path = os.path.join(self.key_dir, f'{key_id}.pt')
        if os.path.exists(key_path):
            key = RademacherKey.load(key_path)
            self.keys[key_id] = key
            return key
        else:
            raise FileNotFoundError(f"Key '{key_id}' not found")
    
    def get_or_create_key(
        self,
        key_id: str,
        seed: Optional[int] = None,
        password: Optional[str] = None,
    ) -> RademacherKey:
        """Get existing key or create new one."""
        try:
            return self.load_key(key_id)
        except FileNotFoundError:
            return self.generate_and_save_key(key_id, seed, password)
