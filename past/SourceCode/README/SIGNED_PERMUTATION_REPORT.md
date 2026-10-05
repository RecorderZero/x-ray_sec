# 符號置換變換方案整合報告

## Signed Permutation Transformation Integration Report

基於研究文檔「擴散模型潛在空間的可逆加密與匿名化：超越 Rademacher 分布的高級變換策略研究報告」，本報告說明如何將符號置換變換方案整合到胸部 X 光影像擴散模型異常檢測系統中。

---

## 1. 背景與動機

### 1.1 原有 Rademacher 方法的局限性

原論文（Labarbarie 等人）使用 Rademacher 分布 k ∈ {-1, +1}^d 進行符號翻轉加密：

```
z_ano = k ⊙ x_T
```

**局限性：**
1. **缺乏擴散性**：加密是按元素獨立進行，空間結構被保留
2. **絕對值保留**：|z_ano_i| = |x_T_i|，容易受到統計攻擊
3. **密鑰空間有限**：僅有 2^d 種可能

### 1.2 符號置換變換的優勢

符號置換變換結合了 Rademacher 符號翻轉和置換矩陣：

```
z_ano = P(k ⊙ x_T)
```

**優勢：**
1. **密鑰空間擴大**：從 2^d 擴大到 2^d × d!（天文數字級增長）
2. **破壞空間相關性**：置換打亂了元素的位置
3. **防禦殘留信息洩露**：即使 x_T 殘留有結構信息，置換也能有效分散

---

## 2. 數學原理

### 2.1 分布保持性

對於標準正態分布 x ~ N(0, I)：

1. **Rademacher 變換**：k ⊙ x ~ N(0, I)（因為正態分布關於原點對稱）
2. **置換變換**：Px ~ N(0, I)（因為置換矩陣是正交矩陣，PP^T = I）
3. **組合變換**：P(k ⊙ x) ~ N(0, I)（兩者的組合仍保持分布）

### 2.2 可逆性

符號置換變換是完全可逆的：

```
加密：z_ano = P(k ⊙ x)
解密：x = k ⊙ P^(-1)(z_ano)
```

因為：
- k ⊙ k = 1（Rademacher 自逆性）
- P^(-1) P = I（置換矩陣可逆性）

### 2.3 密鑰空間計算

對於 256×256 的單通道影像，d = 65,536：

| 方法 | 密鑰空間大小 |
|------|-------------|
| Rademacher | 2^65536 ≈ 10^19728 |
| 置換 | 65536! ≈ 10^287193 |
| 符號置換 | 2^65536 × 65536! ≈ 10^306921 |

---

## 3. 程式碼修改說明

### 3.1 新增類別：`PermutationKey`

**檔案**：`guided_diffusion/anonymization.py`

```python
class PermutationKey:
    """
    置換密鑰類別
    
    生成隨機置換並支援正向/逆向應用
    """
    
    def __init__(
        self,
        size: int,                              # 總元素數量 d = C × H × W
        permutation: Optional[torch.Tensor],    # 預設置換
        seed: Optional[int],                    # 種子
        password: Optional[str],                # 密碼
    ):
        ...
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """應用置換：x -> P(x)"""
        ...
    
    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """應用逆置換：x -> P^(-1)(x)"""
        ...
```

**關鍵方法解釋：**

```python
@staticmethod
def _generate_permutation(size: int, seed: Optional[int] = None) -> torch.Tensor:
    """生成隨機置換"""
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(size, generator=generator)
    else:
        permutation = torch.randperm(size)
    return permutation

@staticmethod
def _compute_inverse(permutation: torch.Tensor) -> torch.Tensor:
    """計算逆置換"""
    inverse = torch.zeros_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    return inverse
```

### 3.2 新增類別：`SignedPermutationKey`

**檔案**：`guided_diffusion/anonymization.py`

```python
class SignedPermutationKey:
    """
    符號置換密鑰類別
    
    結合 Rademacher 符號翻轉和置換矩陣
    實現 z_ano = P(k ⊙ x)
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],                     # 資料形狀 (C, H, W)
        rademacher_key: Optional[RademacherKey],    # Rademacher 密鑰
        permutation_key: Optional[PermutationKey],  # 置換密鑰
        seed: Optional[int],                        # 種子
        password: Optional[str],                    # 密碼
    ):
        ...
    
    def apply(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        應用符號置換變換：z_ano = P(k ⊙ x)
        
        步驟：
        1. 應用 Rademacher 符號翻轉
        2. 應用置換
        """
        ...
    
    def apply_inverse(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        應用逆變換：x = k ⊙ P^(-1)(z_ano)
        
        步驟：
        1. 應用逆置換
        2. 應用 Rademacher 符號翻轉（自逆）
        """
        ...
```

**核心變換邏輯：**

```python
def apply(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """應用符號置換變換"""
    # 步驟 1: 應用 Rademacher 符號翻轉
    key = self.rademacher_key.key.to(x.device)
    if mask is not None:
        x_signed = mask * (key * x) + (1 - mask) * x
    else:
        x_signed = key * x
    
    # 步驟 2: 應用置換
    x_permuted = self.permutation_key.apply(x_signed)
    
    return x_permuted

def apply_inverse(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """應用逆變換"""
    # 步驟 1: 應用逆置換
    x_unpermuted = self.permutation_key.apply_inverse(x)
    
    # 步驟 2: 應用 Rademacher 符號翻轉（自逆）
    key = self.rademacher_key.key.to(x.device)
    if mask is not None:
        x_original = mask * (key * x_unpermuted) + (1 - mask) * x_unpermuted
    else:
        x_original = key * x_unpermuted
    
    return x_original
```

### 3.3 修改 `DiffusionAnonymizer` 類別

**新增參數**：`use_permutation`

```python
class DiffusionAnonymizer(nn.Module):
    def __init__(
        self,
        latent_shape: Tuple[int, ...],
        mask_type: str = 'full',
        margin: int = 0,
        use_permutation: bool = False,  # 新增：是否使用符號置換
    ):
        ...
```

**修改 `anonymize_latent` 方法**：

```python
def anonymize_latent(
    self,
    z_t: torch.Tensor,
    key: Union[torch.Tensor, RademacherKey, SignedPermutationKey],  # 支援新密鑰類型
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 處理 SignedPermutationKey
    if isinstance(key, SignedPermutationKey):
        return key.apply(z_t, mask)
    
    # 處理基本 RademacherKey
    if isinstance(key, RademacherKey):
        key = key.key
    return self.apply_key(z_t, key, mask)
```

**修改 `deanonymize_latent` 方法**：

```python
def deanonymize_latent(
    self,
    z_ano_t: torch.Tensor,
    key: Union[torch.Tensor, RademacherKey, SignedPermutationKey],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 處理 SignedPermutationKey
    if isinstance(key, SignedPermutationKey):
        return key.apply_inverse(z_ano_t, mask)
    
    # 處理基本 RademacherKey（自逆）
    if isinstance(key, RademacherKey):
        key = key.key
    return self.apply_key(z_ano_t, key, mask)
```

### 3.4 修改 `script_util.py`

**更新 import**：

```python
from .anonymization import (
    DiffusionAnonymizer, 
    RademacherKey, 
    PermutationKey,
    SignedPermutationKey,
    AnonymizationManager
)
```

**更新 `anonymization_defaults()`**：

```python
def anonymization_defaults():
    return dict(
        anonymization_mode=None,
        anonymization_mask_type='full',
        anonymization_margin=0,
        anonymization_key_path=None,
        anonymization_key_seed=None,
        anonymization_key_password=None,
        use_signed_permutation=False,  # 新增
    )
```

**更新 `create_anonymizer()`**：

```python
def create_anonymizer(
    image_size,
    in_channels=1,
    scaling_factor=1,
    anonymization_mask_type='full',
    anonymization_margin=0,
    use_signed_permutation=False,  # 新增
):
    return DiffusionAnonymizer(
        latent_shape=target_shape,
        mask_type=anonymization_mask_type,
        margin=anonymization_margin,
        use_permutation=use_signed_permutation,  # 新增
    )
```

**更新 `create_model_and_diffusion()` 函數簽名**：

```python
def create_model_and_diffusion(
    ...
    # Anonymization parameters (not used in model/diffusion, but for arg compatibility)
    anonymization_mode=None,
    anonymization_mask_type='full',
    anonymization_margin=0,
    anonymization_key_path=None,
    anonymization_key_seed=None,
    anonymization_key_password=None,
    use_signed_permutation=False,  # 新增：避免 unexpected keyword argument 錯誤
):
```

> **注意**：此修改是為了確保當 `args_to_dict()` 傳遞所有參數時，`create_model_and_diffusion()` 不會因為收到未預期的 `use_signed_permutation` 參數而報錯。

### 3.5 修改採樣腳本

**更新 `setup_anonymization_key()` 函數**：

```python
def setup_anonymization_key(args, anonymizer):
    use_signed_permutation = getattr(args, 'use_signed_permutation', False)
    
    if use_signed_permutation:
        logger.log(f"Generating SignedPermutationKey")
        logger.log(f"Enhanced security: key space = 2^d × d!")
        key = SignedPermutationKey(
            shape=pixel_key_shape,
            seed=key_seed,
            password=key_password,
        )
    else:
        logger.log(f"Generating RademacherKey")
        key = RademacherKey(
            shape=pixel_key_shape,
            seed=key_seed,
            password=key_password,
        )
    
    return key
```

**新增命令列參數**：

```python
defaults = dict(
    ...
    use_signed_permutation=False,  # 使用 SignedPermutationKey 進行更強加密
)
```

---

## 4. 使用方式

### 4.1 基本 Rademacher 加密（原有方式）

```bash
python scripts/cfg_image_sample_anonymization.py \
    --data_dir data/tif_pleural/testing \
    --model_path results/ddim_base/model.pt \
    --result_dir results/anonymized \
    --anonymization_mode anonymize \
    --anonymization_key_seed 42 \
    --anonymization_key_path keys/basic_key.pt \
    ...
```

### 4.2 符號置換加密（增強方式）

```bash
python scripts/cfg_image_sample_anonymization.py \
    --data_dir data/tif_pleural/testing \
    --model_path results/ddim_base/model.pt \
    --result_dir results/anonymized \
    --anonymization_mode anonymize \
    --anonymization_key_seed 42 \
    --anonymization_key_path keys/signed_perm_key.pt \
    --use_signed_permutation True \
    ...
```

### 4.3 使用密碼的符號置換加密

```bash
python scripts/cfg_image_sample_anonymization.py \
    --data_dir data/tif_pleural/testing \
    --model_path results/ddim_base/model.pt \
    --result_dir results/anonymized \
    --anonymization_mode anonymize \
    --anonymization_key_password "my_secure_password_123" \
    --anonymization_key_path keys/signed_perm_key.pt \
    --use_signed_permutation True \
    ...
```

---

## 5. 流程圖

### 5.1 符號置換加密流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        符號置換加密流程                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  輸入影像 x_0     │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ DDIM Forward (加噪)         │
                    │ x_0 → x_T                   │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    符號置換變換 z_ano = P(k ⊙ x_T)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 步驟 1: Rademacher 符號翻轉                                          │   │
│  │                                                                      │   │
│  │  k ∈ {-1, +1}^d                                                     │   │
│  │  x_signed = k ⊙ x_T                                                 │   │
│  │                                                                      │   │
│  │  例如: x_T = [0.5, -0.3, 0.8]                                       │   │
│  │        k   = [-1,   1,  -1]                                         │   │
│  │        x_signed = [-0.5, -0.3, -0.8]                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 步驟 2: 置換                                                         │   │
│  │                                                                      │   │
│  │  P = 隨機置換矩陣                                                    │   │
│  │  z_ano = P(x_signed)                                                │   │
│  │                                                                      │   │
│  │  例如: 置換 = [2, 0, 1]                                             │   │
│  │        x_signed = [-0.5, -0.3, -0.8]                                │   │
│  │        z_ano = [-0.8, -0.5, -0.3]  (位置打亂)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ DDIM Backward (去噪)        │
                    │ z_ano → x_ano              │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  匿名化影像 x_ano │
                         └──────────────────┘
```

### 5.2 符號置換解密流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        符號置換解密流程                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  匿名化影像 x_ano │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ DDIM Forward (加噪)         │
                    │ x_ano → z_ano_T            │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                逆符號置換變換 x_T = k ⊙ P^(-1)(z_ano_T)                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 步驟 1: 逆置換                                                       │   │
│  │                                                                      │   │
│  │  P^(-1) = 逆置換矩陣                                                 │   │
│  │  x_signed = P^(-1)(z_ano_T)                                         │   │
│  │                                                                      │   │
│  │  例如: 逆置換 = [1, 2, 0]                                           │   │
│  │        z_ano_T = [-0.8, -0.5, -0.3]                                 │   │
│  │        x_signed = [-0.5, -0.3, -0.8]  (恢復原位置)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 步驟 2: Rademacher 符號翻轉（自逆）                                  │   │
│  │                                                                      │   │
│  │  x_T = k ⊙ x_signed                                                 │   │
│  │                                                                      │   │
│  │  例如: k = [-1, 1, -1]                                              │   │
│  │        x_signed = [-0.5, -0.3, -0.8]                                │   │
│  │        x_T = [0.5, -0.3, 0.8]  (恢復原始值)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ DDIM Backward (去噪)        │
                    │ x_T → x_rec                │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  恢復影像 x_rec   │
                         └──────────────────┘
```

---

## 6. 安全性比較

| 特性 | Rademacher | 符號置換 |
|------|------------|----------|
| 密鑰空間 | 2^d | 2^d × d! |
| 空間相關性 | 保留 | 破壞 |
| 絕對值 | 保留 | 保留但位置打亂 |
| 計算複雜度 | O(d) | O(d) |
| 記憶體開銷 | O(d) | O(2d) |
| 抗統計攻擊 | 弱 | 強 |
| 向後相容 | - | ✓ |

---

## 7. 修改的檔案列表

| 檔案 | 修改內容 |
|------|---------|
| `guided_diffusion/anonymization.py` | 新增 `PermutationKey`、`SignedPermutationKey` 類別；更新 `DiffusionAnonymizer` |
| `guided_diffusion/script_util.py` | 更新 import、新增 `use_signed_permutation` 參數 |
| `scripts/cfg_image_sample_anonymization.py` | 更新密鑰設置邏輯、新增命令列參數 |
| `scripts/classifier_sample_known_anonymization.py` | 更新密鑰設置邏輯、新增命令列參數 |

---

## 8. 結論

符號置換變換方案成功整合到胸部 X 光影像擴散模型異常檢測系統中：

1. **密鑰空間大幅擴展**：從 2^d 增加到 2^d × d!，使暴力破解在理論上不可能
2. **破壞空間相關性**：置換操作打亂了元素位置，增強了安全性
3. **保持分布不變性**：變換後的資料仍服從 N(0, I)，不影響擴散模型的生成質量
4. **完全可逆**：k ⊙ k = 1 和 P^(-1) P = I 保證了完美的可逆性
5. **向後相容**：預設使用原有的 Rademacher 方法，可選擇啟用符號置換
6. **低計算開銷**：置換操作的時間複雜度為 O(d)，與原方法相當
