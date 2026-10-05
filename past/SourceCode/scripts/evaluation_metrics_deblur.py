import sys
sys.path.append("..")
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import cv2, os, csv, argparse

from sklearn.metrics import roc_curve, auc
from matplotlib import gridspec
from skimage.filters import threshold_otsu
from PIL import Image, ImageDraw
# from visdom import Visdom
# viz = Visdom(port=8850)

from skimage import segmentation
# from guided_diffusion.script_util import add_dict_to_argparser

# Evaluation metrics
from torcheval.metrics import PeakSignalNoiseRatio
# from torcheval.metrics import StructuralSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure as StructuralSimilarity
from torcheval.metrics import FrechetInceptionDistance

'''
#We define the anomaly map as the absolute difference between the original image and the generated healthy reconstruction. We sum up over all channels (4 different MR sequences):

difftot=abs(original-healthyreconstruction).sum(dim=0)

#We compute the Otsu threshold for the anomaly map:

diff = np.array(difftot)
thresh = threshold_otsu(diff)
mask = th.where(th.tensor(diff) > thresh, 1, 0)  #this is our predicted binary segmentation
viz.image(visualize(mask[ 0,...]), opts=dict(caption="mask"))

#We load the ground truth segmetation mask and put all the different tumor labels to 1:

Labelmask_GT = th.where(groundtruth_segmentation > 0, 1, 0)

pixel_wise_cls = visualize(np.array(th.tensor(diff).view(1, -1))[0, :])
pixel_wise_gt = visualize(np.array(th.tensor(Labelmask_GT).view(1, -1))[0, :])

#Then we compute the Dice and AUROC scores

DSC=dice_score(mask.cpu(), Labelmask_GT.cpu()) #predicted Dice score
auc = roc_auc_score(pixel_wise_gt, pixel_wise_cls)
'''

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    # pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


def natural_sort_indices(arr):
    """
    自然排序：正確處理含數字的字串
    例如：CASE 1, CASE 2, ..., CASE 10, CASE 11
    而非：CASE 1, CASE 10, CASE 11, CASE 2, ...
    """
    import re
    
    def natural_key(s):
        """將字串分解為文字和數字部分，數字部分轉為整數用於比較"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', str(s))]
    
    # 獲取排序後的索引
    indexed_arr = [(i, natural_key(name)) for i, name in enumerate(arr)]
    indexed_arr.sort(key=lambda x: x[1])
    return np.array([i for i, _ in indexed_arr])


def load_samples(npz_path):
    samples_npz = np.load(npz_path, allow_pickle=True)
    # 使用自然排序（正確處理 CASE 1, CASE 2, ..., CASE 10, CASE 11）
    names = samples_npz.f.names
    sorting_indices = natural_sort_indices(names)

    normalized_originals = np.array([visualize(img) for img in samples_npz.f.orgs[sorting_indices]])
    normalized_samples = np.array([visualize(img) for img in samples_npz.f.samples[sorting_indices]])

    data = {
        'originals': normalized_originals,
        'samples': normalized_samples,
        'org_labels': samples_npz.f.org_labels[sorting_indices],
        'tgt_labels': samples_npz.f.tgt_labels[sorting_indices],
        'names': names[sorting_indices]
    }

    return data


#===============================================================================
# 加解密可逆性評估函數
# 用於比較 Base 模型輸出與解密後輸出的差異
#===============================================================================

def load_samples_raw(npz_path):
    """
    載入 NPZ 檔案，不進行正規化（用於可逆性精確比較）
    使用自然排序處理含數字的名稱
    """
    samples_npz = np.load(npz_path, allow_pickle=True)
    
    # 使用自然排序（正確處理 CASE 1, CASE 2, ..., CASE 10, CASE 11）
    names = samples_npz.f.names
    sorting_indices = natural_sort_indices(names)
    
    data = {
        'originals': samples_npz.f.orgs[sorting_indices],
        'samples': samples_npz.f.samples[sorting_indices],
        'org_labels': samples_npz.f.org_labels[sorting_indices],
        'tgt_labels': samples_npz.f.tgt_labels[sorting_indices],
        'names': names[sorting_indices]
    }
    return data


def calculate_reversibility_metrics(base_samples, decrypted_samples):
    """
    計算可逆性評估指標
    
    比較 Base 模型輸出與加解密後輸出的差異
    
    Args:
        base_samples: Base 模型輸出 (N, C, H, W)
        decrypted_samples: 加解密後輸出 (N, C, H, W)
    
    Returns:
        dict: 包含 PSNR, SSIM, FID, MAE, MSE, Cosine Similarity 指標
    """
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarity()
    fid = FrechetInceptionDistance()
    
    # 正規化用於指標計算
    base_norm = np.array([visualize(img) for img in base_samples])
    dec_norm = np.array([visualize(img) for img in decrypted_samples])
    
    psnr.update(th.from_numpy(base_norm), th.from_numpy(dec_norm))
    ssim.update(th.from_numpy(base_norm), th.from_numpy(dec_norm))
    fid.update(th.from_numpy(np.repeat(base_norm, 3, axis=1).clip(0, 1)), True)
    fid.update(th.from_numpy(np.repeat(dec_norm, 3, axis=1).clip(0, 1)), False)
    
    # 計算 MAE 和 MSE（像素級誤差）
    mae = np.mean(np.abs(base_samples - decrypted_samples))
    mse = np.mean((base_samples - decrypted_samples) ** 2)
    
    # 計算最大絕對誤差
    max_abs_error = np.max(np.abs(base_samples - decrypted_samples))
    
    # 計算 Cosine Similarity
    # 將每個樣本展平為向量，計算餘弦相似度，然後取平均
    cosine_similarities = []
    for i in range(len(base_samples)):
        base_flat = base_samples[i].flatten()
        dec_flat = decrypted_samples[i].flatten()
        
        # 計算餘弦相似度: cos(θ) = (a · b) / (||a|| * ||b||)
        dot_product = np.dot(base_flat, dec_flat)
        norm_base = np.linalg.norm(base_flat)
        norm_dec = np.linalg.norm(dec_flat)
        
        if norm_base > 0 and norm_dec > 0:
            cos_sim = dot_product / (norm_base * norm_dec)
        else:
            cos_sim = 0.0
        
        cosine_similarities.append(cos_sim)
    
    avg_cosine_similarity = np.mean(cosine_similarities)
    
    return dict(
        psnr=psnr.compute(),
        ssim=ssim.compute(),
        fid=fid.compute(),
        mae=mae,
        mse=mse,
        max_abs_error=max_abs_error,
        cosine_similarity=avg_cosine_similarity
    )


def generate_reversibility_comparison_image(base_npz_path, decrypted_npz_path, dest_path, scheme_name):
    """
    生成可逆性比較圖像
    
    顯示: Base Output | Decrypted Output | Difference Heatmap
    """
    base_data = load_samples(base_npz_path)
    dec_data = load_samples(decrypted_npz_path)
    
    n_samples = min(len(base_data['samples']), 10)  # 最多顯示 10 張
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    cm = plt.get_cmap('jet')
    
    for i in range(n_samples):
        base_img = base_data['samples'][i, 0]
        dec_img = dec_data['samples'][i, 0]
        diff = np.abs(base_img - dec_img)
        
        # Base Output
        axes[i, 0].imshow(base_img, cmap='gray')
        axes[i, 0].set_title(f'Base Output' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Decrypted Output
        axes[i, 1].imshow(dec_img, cmap='gray')
        axes[i, 1].set_title(f'{scheme_name} Decrypted' if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Difference Heatmap
        colored_diff = cm(visualize(diff))[:, :, :3]
        axes[i, 2].imshow(colored_diff)
        axes[i, 2].set_title('Difference' if i == 0 else '')
        axes[i, 2].axis('off')
        
        # Histogram of differences
        axes[i, 3].hist(diff.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[i, 3].set_title(f'Diff Histogram (max={diff.max():.6f})' if i == 0 else f'max={diff.max():.6f}')
        axes[i, 3].set_xlabel('Pixel Difference')
        axes[i, 3].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, f'reversibility_{scheme_name}_comparison.png'), dpi=150)
    plt.close()


def generate_cross_model_comparison_image(base_npz_paths, encrypted_npz_paths, scheme_name, result_dir, n_images=3):
    """
    生成跨模型的可逆性比較圖
    
    對於指定的加密方案，比較三個模型（CFG-DDIM, CLF-DDIM, Uncond-DDIM）的 Base 與解密結果
    
    Args:
        base_npz_paths: Base 模型 NPZ 路徑列表 [cfg_base, clf_base, uncond_base]
        encrypted_npz_paths: 該加密方案的解密後 NPZ 路徑列表 [cfg_enc, clf_enc, uncond_enc]
        scheme_name: 加密方案名稱（'rademacher' 或 'signed_perm'）
        result_dir: 結果輸出目錄
        n_images: 要顯示的影像數量（預設 3）
    """
    model_names = ['CFG-DDIM', 'CLF-DDIM', 'Uncond-DDIM']
    
    # 載入所有模型的資料並匹配
    all_base_samples = []
    all_enc_samples = []
    all_matched_names = []
    
    for i, (base_path, enc_path) in enumerate(zip(base_npz_paths, encrypted_npz_paths)):
        base_data = load_samples_raw(base_path)
        enc_data = load_samples_raw(enc_path)
        
        base_samples, enc_samples, matched_names = match_samples_by_name(base_data, enc_data)
        
        if len(matched_names) == 0:
            print(f"    警告: {model_names[i]} 沒有匹配的樣本，跳過跨模型比較")
            return
        
        all_base_samples.append(base_samples)
        all_enc_samples.append(enc_samples)
        all_matched_names.append(matched_names)
    
    # 找出所有模型共同的樣本名稱
    common_names = set(all_matched_names[0])
    for names in all_matched_names[1:]:
        common_names &= set(names)
    
    if len(common_names) == 0:
        print(f"    警告: 三個模型之間沒有共同的樣本名稱，跳過跨模型比較")
        return
    
    common_names = sorted(common_names, key=natural_sort_key)[:n_images]
    
    # 為每個共同名稱提取對應的樣本
    selected_base = {model: [] for model in model_names}
    selected_enc = {model: [] for model in model_names}
    
    for name in common_names:
        # 從各模型取 Base 和解密結果
        for i, model_name in enumerate(model_names):
            idx = all_matched_names[i].index(name)
            selected_base[model_name].append(all_base_samples[i][idx])
            selected_enc[model_name].append(all_enc_samples[i][idx])
    
    for model_name in model_names:
        selected_base[model_name] = np.array(selected_base[model_name])
        selected_enc[model_name] = np.array(selected_enc[model_name])
    
    # 正規化用於顯示
    base_norm = {model: np.array([visualize(img) for img in selected_base[model]]) 
                 for model in model_names}
    enc_norm = {model: np.array([visualize(img) for img in selected_enc[model]]) 
                for model in model_names}
    
    # 創建比較圖：每行 6 列 (CFG Base, CFG Dec, CLF Base, CLF Dec, Uncond Base, Uncond Dec)
    fig, axes = plt.subplots(n_images, 6, figsize=(24, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    scheme_display_name = 'Rademacher' if scheme_name == 'rademacher' else 'Signed Permutation'
    
    # 設定列標題
    # col_titles = ['CFG Base', 'CFG Dec', 'CLF Base', 'CLF Dec', 'Uncond Base', 'Uncond Dec']
    col_titles = ['CFG-DDIM Base With Deblur', 'CFG-DDIM Decrypted With Deblur', 'CLF-DDIM Base With Deblur', 'CLF-DDIM Decrypted With Deblur', 'Traditional DDIM Base With Deblur', 'Traditional DDIM Decrypted With Deblur']

    for i in range(n_images):
        for j, model_name in enumerate(model_names):
            base_img = base_norm[model_name][i, 0]
            dec_img = enc_norm[model_name][i, 0]
            
            # Base 列
            col_base = j * 2
            axes[i, col_base].imshow(base_img, cmap='gray')
            axes[i, col_base].set_title(col_titles[col_base] if i == 0 else '')
            axes[i, col_base].axis('off')
            
            # Decrypted 列
            col_dec = j * 2 + 1
            axes[i, col_dec].imshow(dec_img, cmap='gray')
            axes[i, col_dec].set_title(col_titles[col_dec] if i == 0 else '')
            axes[i, col_dec].axis('off')
    
    plt.suptitle(f'{scheme_display_name} Encryption: Cross-Model Comparison\n(Base vs Decrypted Output for Each Model)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(result_dir, f'reversibility_{scheme_name}_cross_model_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  跨模型比較圖已保存: {output_path}")


def generate_cross_model_comparison_with_diff(base_npz_paths, encrypted_npz_paths, scheme_name, result_dir, n_images=3):
    """
    生成跨模型的可逆性比較圖（含個別差異圖）
    
    對於指定的加密方案，比較三個模型（CFG-DDIM, CLF-DDIM, Uncond-DDIM）的 Base 與解密結果
    每張影像顯示：CFG Base | CFG Dec | CFG Diff | CLF Base | CLF Dec | CLF Diff | Uncond Base | Uncond Dec | Uncond Diff
    
    Args:
        base_npz_paths: Base 模型 NPZ 路徑列表 [cfg_base, clf_base, uncond_base]
        encrypted_npz_paths: 該加密方案的解密後 NPZ 路徑列表 [cfg_enc, clf_enc, uncond_enc]
        scheme_name: 加密方案名稱（'rademacher' 或 'signed_perm'）
        result_dir: 結果輸出目錄
        n_images: 要顯示的影像數量（預設 3）
    """
    model_names = ['CFG-DDIM', 'CLF-DDIM', 'Uncond-DDIM']
    
    # 載入所有模型的資料並匹配
    all_base_samples = []
    all_enc_samples = []
    all_matched_names = []
    
    for i, (base_path, enc_path) in enumerate(zip(base_npz_paths, encrypted_npz_paths)):
        base_data = load_samples_raw(base_path)
        enc_data = load_samples_raw(enc_path)
        
        base_samples, enc_samples, matched_names = match_samples_by_name(base_data, enc_data)
        
        if len(matched_names) == 0:
            print(f"    警告: {model_names[i]} 沒有匹配的樣本，跳過跨模型比較")
            return
        
        all_base_samples.append(base_samples)
        all_enc_samples.append(enc_samples)
        all_matched_names.append(matched_names)
    
    # 找出所有模型共同的樣本名稱
    common_names = set(all_matched_names[0])
    for names in all_matched_names[1:]:
        common_names &= set(names)
    
    if len(common_names) == 0:
        print(f"    警告: 三個模型之間沒有共同的樣本名稱，跳過跨模型比較")
        return
    
    common_names = sorted(common_names, key=natural_sort_key)[:n_images]
    
    # 為每個共同名稱提取對應的樣本
    selected_base = {model: [] for model in model_names}
    selected_enc = {model: [] for model in model_names}
    
    for name in common_names:
        for i, model_name in enumerate(model_names):
            idx = all_matched_names[i].index(name)
            selected_base[model_name].append(all_base_samples[i][idx])
            selected_enc[model_name].append(all_enc_samples[i][idx])
    
    for model_name in model_names:
        selected_base[model_name] = np.array(selected_base[model_name])
        selected_enc[model_name] = np.array(selected_enc[model_name])
    
    # 正規化用於顯示
    base_norm = {model: np.array([visualize(img) for img in selected_base[model]]) 
                 for model in model_names}
    enc_norm = {model: np.array([visualize(img) for img in selected_enc[model]]) 
                for model in model_names}
    
    # 創建比較圖：每行 9 列 (CFG Base, CFG Dec, CFG Diff, CLF Base, CLF Dec, CLF Diff, Uncond Base, Uncond Dec, Uncond Diff)
    fig, axes = plt.subplots(n_images, 9, figsize=(36, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    cm = plt.get_cmap('jet')
    
    scheme_display_name = 'Rademacher' if scheme_name == 'rademacher' else 'Signed Permutation'
    
    # 設定列標題
    # col_titles = ['CFG Base', 'CFG Dec', 'CFG Diff', 
    #               'CLF Base', 'CLF Dec', 'CLF Diff', 
    #               'Uncond Base', 'Uncond Dec', 'Uncond Diff']
    col_titles = ['CFG-DDIM Base With Deblur', 'CFG-DDIM Decrypted With Deblur', 'CFG-DDIM Diff', 
                  'CLF-DDIM Base With Deblur', 'CLF-DDIM Decrypted With Deblur', 'CLF-DDIM Diff', 
                  'Traditional DDIM Base With Deblur', 'Traditional DDIM Decrypted With Deblur', 'Traditional DDIM Diff']

    for i in range(n_images):
        for j, model_name in enumerate(model_names):
            base_img = base_norm[model_name][i, 0]
            dec_img = enc_norm[model_name][i, 0]
            diff = np.abs(base_img - dec_img)
            
            # Base 列
            col_base = j * 3
            axes[i, col_base].imshow(base_img, cmap='gray')
            axes[i, col_base].set_title(col_titles[col_base] if i == 0 else '')
            axes[i, col_base].axis('off')
            
            # Decrypted 列
            col_dec = j * 3 + 1
            axes[i, col_dec].imshow(dec_img, cmap='gray')
            axes[i, col_dec].set_title(col_titles[col_dec] if i == 0 else '')
            axes[i, col_dec].axis('off')
            
            # Diff 列
            col_diff = j * 3 + 2
            if diff.max() > 0:
                colored_diff = cm(visualize(diff))[:, :, :3]
            else:
                colored_diff = cm(diff)[:, :, :3]
            axes[i, col_diff].imshow(colored_diff)
            axes[i, col_diff].set_title(col_titles[col_diff] if i == 0 else f'max={diff.max():.4f}')
            axes[i, col_diff].axis('off')
    
    plt.suptitle(f'{scheme_display_name} Encryption: Cross-Model Comparison with Difference Maps\n(Base vs Decrypted Output for Each Model)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(result_dir, f'reversibility_{scheme_name}_cross_model_with_diff.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  跨模型比較圖（含差異）已保存: {output_path}")


def generate_cross_model_base_first_comparison(base_npz_paths, encrypted_npz_paths, scheme_name, result_dir, n_images=3):
    """
    生成跨模型的可逆性比較圖（Base 優先排列）
    
    對於指定的加密方案，比較三個模型（CFG-DDIM, CLF-DDIM, Uncond-DDIM）的 Base 與解密結果
    順序：先顯示所有 Base，再顯示所有解密結果
    每行顯示：CFG Base | CLF Base | Uncond Base | CFG Dec | CLF Dec | Uncond Dec
    
    Args:
        base_npz_paths: Base 模型 NPZ 路徑列表 [cfg_base, clf_base, uncond_base]
        encrypted_npz_paths: 該加密方案的解密後 NPZ 路徑列表 [cfg_enc, clf_enc, uncond_enc]
        scheme_name: 加密方案名稱（'rademacher' 或 'signed_perm'）
        result_dir: 結果輸出目錄
        n_images: 要顯示的影像數量（預設 3）
    """
    model_names = ['CFG-DDIM', 'CLF-DDIM', 'Uncond-DDIM']
    
    # 載入所有模型的資料並匹配
    all_base_samples = []
    all_enc_samples = []
    all_matched_names = []
    
    for i, (base_path, enc_path) in enumerate(zip(base_npz_paths, encrypted_npz_paths)):
        base_data = load_samples_raw(base_path)
        enc_data = load_samples_raw(enc_path)
        
        base_samples, enc_samples, matched_names = match_samples_by_name(base_data, enc_data)
        
        if len(matched_names) == 0:
            print(f"    警告: {model_names[i]} 沒有匹配的樣本，跳過跨模型比較")
            return
        
        all_base_samples.append(base_samples)
        all_enc_samples.append(enc_samples)
        all_matched_names.append(matched_names)
    
    # 找出所有模型共同的樣本名稱
    common_names = set(all_matched_names[0])
    for names in all_matched_names[1:]:
        common_names &= set(names)
    
    if len(common_names) == 0:
        print(f"    警告: 三個模型之間沒有共同的樣本名稱，跳過跨模型比較")
        return
    
    common_names = sorted(common_names, key=natural_sort_key)[:n_images]
    
    # 為每個共同名稱提取對應的樣本
    selected_base = {model: [] for model in model_names}
    selected_enc = {model: [] for model in model_names}
    
    for name in common_names:
        for i, model_name in enumerate(model_names):
            idx = all_matched_names[i].index(name)
            selected_base[model_name].append(all_base_samples[i][idx])
            selected_enc[model_name].append(all_enc_samples[i][idx])
    
    for model_name in model_names:
        selected_base[model_name] = np.array(selected_base[model_name])
        selected_enc[model_name] = np.array(selected_enc[model_name])
    
    # 正規化用於顯示
    base_norm = {model: np.array([visualize(img) for img in selected_base[model]]) 
                 for model in model_names}
    enc_norm = {model: np.array([visualize(img) for img in selected_enc[model]]) 
                for model in model_names}
    
    # 創建比較圖：每行 6 列 (CFG Base, CLF Base, Uncond Base, CFG Dec, CLF Dec, Uncond Dec)
    fig, axes = plt.subplots(n_images, 6, figsize=(24, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    scheme_display_name = 'Rademacher' if scheme_name == 'rademacher' else 'Signed Permutation'
    
    # 設定列標題：先 Base 後 Dec
    # col_titles = ['CFG Base', 'CLF Base', 'Uncond Base', 'CFG Dec', 'CLF Dec', 'Uncond Dec']
    col_titles = ['CFG-DDIM Base With Deblur', 'CLF-DDIM Base With Deblur', 'Traditional DDIM Base With Deblur', 'CFG-DDIM Decrypted With Deblur', 'CLF-DDIM Decrypted With Deblur', 'Traditional DDIM Decrypted With Deblur']
    
    for i in range(n_images):
        # 先繪製所有 Base（列 0-2）
        for j, model_name in enumerate(model_names):
            base_img = base_norm[model_name][i, 0]
            axes[i, j].imshow(base_img, cmap='gray')
            axes[i, j].set_title(col_titles[j] if i == 0 else '')
            axes[i, j].axis('off')
        
        # 再繪製所有 Decrypted（列 3-5）
        for j, model_name in enumerate(model_names):
            dec_img = enc_norm[model_name][i, 0]
            axes[i, j + 3].imshow(dec_img, cmap='gray')
            axes[i, j + 3].set_title(col_titles[j + 3] if i == 0 else '')
            axes[i, j + 3].axis('off')
    
    plt.suptitle(f'{scheme_display_name} Encryption: Cross-Model Comparison\n(All Bases → All Decrypted Outputs)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(result_dir, f'reversibility_{scheme_name}_base_first_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  跨模型比較圖（Base優先）已保存: {output_path}")


def evaluate_encryption_reversibility(base_npz_paths, encrypted_npz_paths, scheme_names, result_dir):
    """
    評估加解密可逆性
    
    對每個模型，比較 Base 輸出與各加密方案解密後的輸出
    
    Args:
        base_npz_paths: Base 模型 NPZ 路徑列表 [cfg_base, clf_base, uncond_base]
        encrypted_npz_paths: 加密後解密的 NPZ 路徑字典 
                            {'rademacher': [...], 'signed_perm': [...]}
        scheme_names: 加密方案名稱列表 ['Rademacher', 'SignedPerm']
        result_dir: 結果輸出目錄
    """
    os.makedirs(result_dir, exist_ok=True)
    
    model_names = ['CFG-DDIM', 'CLF-DDIM', 'Uncond-DDIM']
    all_results = []
    
    for scheme_name, enc_paths in encrypted_npz_paths.items():
        print(f"\n{'='*60}")
        print(f"評估加密方案: {scheme_name}")
        print(f"{'='*60}")
        
        scheme_results = []
        
        for i, (base_path, enc_path) in enumerate(zip(base_npz_paths, enc_paths)):
            model_name = model_names[i]
            print(f"\n  [{model_name}]")
            print(f"    Base:      {base_path}")
            print(f"    Encrypted: {enc_path}")
            
            # 載入資料
            base_data = load_samples_raw(base_path)
            enc_data = load_samples_raw(enc_path)
            
            # 診斷：顯示名稱資訊
            print(f"    Base names ({len(base_data['names'])}): {base_data['names'][:3]}...")
            print(f"    Enc names  ({len(enc_data['names'])}): {enc_data['names'][:3]}...")
            
            # 使用名稱匹配而非要求完全相同
            base_samples, enc_samples, matched_names = match_samples_by_name(
                base_data, enc_data
            )
            
            if len(matched_names) == 0:
                print(f"    警告: 沒有匹配的樣本名稱！跳過此模型。")
                continue
            
            print(f"    匹配樣本數: {len(matched_names)}")
            
            # 計算可逆性指標（包含 Cosine Similarity）
            metrics = calculate_reversibility_metrics(base_samples, enc_samples)
            
            metrics['model'] = model_name
            metrics['scheme'] = scheme_name
            metrics['n_samples'] = len(matched_names)
            scheme_results.append(metrics)
            
            print(f"    PSNR: {metrics['psnr']:.4f}")
            print(f"    SSIM: {metrics['ssim']:.6f}")
            print(f"    MAE:  {metrics['mae']:.8f}")
            print(f"    MSE:  {metrics['mse']:.10f}")
            print(f"    Max Abs Error: {metrics['max_abs_error']:.8f}")
            print(f"    Cosine Similarity: {metrics['cosine_similarity']:.6f}")
            
            # 生成比較圖像（使用匹配後的數據）
            generate_reversibility_comparison_image_from_data(
                base_samples, enc_samples, matched_names,
                result_dir, f"{model_name}_{scheme_name}"
            )
        
        all_results.extend(scheme_results)
    
    # 輸出 CSV 結果（包含 Cosine Similarity）
    csv_path = os.path.join(result_dir, 'reversibility_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Scheme', 'N_Samples', 'PSNR', 'SSIM', 'FID', 'MAE', 'MSE', 'MaxAbsError', 'Cosine_Sim'])
        for r in all_results:
            psnr_val = r['psnr'].item() if hasattr(r['psnr'], 'item') else r['psnr']
            ssim_val = r['ssim'].item() if hasattr(r['ssim'], 'item') else r['ssim']
            fid_val = r['fid'].item() if hasattr(r['fid'], 'item') else r['fid']
            writer.writerow([
                r['model'], r['scheme'], r['n_samples'],
                f"{psnr_val:.4f}", f"{ssim_val:.6f}", f"{fid_val:.4f}",
                f"{r['mae']:.8f}", f"{r['mse']:.10f}", f"{r['max_abs_error']:.8f}",
                f"{r['cosine_similarity']:.6f}"
            ])
    
    print(f"\n結果已保存至: {csv_path}")
    
    return all_results


def match_samples_by_name(base_data, enc_data):
    """
    根據名稱匹配 base 和 encrypted 的樣本
    
    處理名稱差異情況：
    - Base: patient64544_study1_view1_frontal
    - Enc:  patient64544_study1_view1_frontal_anonymized
    
    Returns:
        base_samples: 匹配後的 base 樣本 (N, C, H, W)
        enc_samples: 匹配後的 encrypted 樣本 (N, C, H, W)
        matched_names: 匹配的名稱列表（使用 base 名稱）
    """
    # 建立名稱到索引的映射
    base_name_to_idx = {name: idx for idx, name in enumerate(base_data['names'])}
    enc_name_to_idx = {name: idx for idx, name in enumerate(enc_data['names'])}
    
    # 首先嘗試直接匹配
    common_names = set(base_data['names']) & set(enc_data['names'])
    
    if len(common_names) == 0:
        # 嘗試處理 _anonymized 後綴差異
        # 建立 enc 名稱的標準化映射（移除 _anonymized 後綴）
        enc_normalized = {}
        for enc_name in enc_data['names']:
            # 移除常見的後綴
            normalized = enc_name
            for suffix in ['_anonymized', '_encrypted', '_decrypted', '_rademacher', '_signed_perm']:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
            enc_normalized[normalized] = enc_name
        
        # 同樣處理 base 名稱（以防 base 也有後綴）
        base_normalized = {}
        for base_name in base_data['names']:
            normalized = base_name
            for suffix in ['_anonymized', '_encrypted', '_decrypted', '_rademacher', '_signed_perm']:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
            base_normalized[normalized] = base_name
        
        # 找出共同的標準化名稱
        common_normalized = set(base_normalized.keys()) & set(enc_normalized.keys())
        
        if len(common_normalized) == 0:
            print("    診斷: 即使移除後綴仍無法匹配...")
            print(f"    Base 標準化名稱範例: {list(base_normalized.keys())[:3]}")
            print(f"    Enc 標準化名稱範例: {list(enc_normalized.keys())[:3]}")
            return np.array([]), np.array([]), []
        
        # 排序以確保一致性
        common_normalized = sorted(common_normalized, key=natural_sort_key)
        
        # 提取匹配的樣本
        base_indices = [base_name_to_idx[base_normalized[norm_name]] for norm_name in common_normalized]
        enc_indices = [enc_name_to_idx[enc_normalized[norm_name]] for norm_name in common_normalized]
        
        base_samples = base_data['samples'][base_indices]
        enc_samples = enc_data['samples'][enc_indices]
        
        # 返回標準化的名稱用於顯示
        return base_samples, enc_samples, common_normalized
    
    # 直接匹配成功
    common_names = sorted(common_names, key=natural_sort_key)
    
    base_indices = [base_name_to_idx[name] for name in common_names]
    enc_indices = [enc_name_to_idx[name] for name in common_names]
    
    base_samples = base_data['samples'][base_indices]
    enc_samples = enc_data['samples'][enc_indices]
    
    return base_samples, enc_samples, list(common_names)


def natural_sort_key(s):
    """
    自然排序的 key 函數
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def generate_reversibility_comparison_image_from_data(base_samples, enc_samples, names, dest_path, scheme_name):
    """
    從已匹配的數據生成可逆性比較圖像
    """
    # 正規化用於顯示
    base_norm = np.array([visualize(img) for img in base_samples])
    enc_norm = np.array([visualize(img) for img in enc_samples])
    
    n_samples = min(len(base_samples), 10)  # 最多顯示 10 張
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    cm = plt.get_cmap('jet')
    
    for i in range(n_samples):
        base_img = base_norm[i, 0]
        dec_img = enc_norm[i, 0]
        diff = np.abs(base_img - dec_img)
        
        # Base Output
        axes[i, 0].imshow(base_img, cmap='gray')
        axes[i, 0].set_title(f'Base Output' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Decrypted Output
        axes[i, 1].imshow(dec_img, cmap='gray')
        axes[i, 1].set_title(f'{scheme_name} Decrypted' if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Difference Heatmap
        colored_diff = cm(visualize(diff) if diff.max() > 0 else diff)[:, :, :3]
        axes[i, 2].imshow(colored_diff)
        axes[i, 2].set_title('Difference' if i == 0 else '')
        axes[i, 2].axis('off')
        
        # Histogram of differences
        axes[i, 3].hist(diff.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[i, 3].set_title(f'Diff Histogram (max={diff.max():.6f})' if i == 0 else f'max={diff.max():.6f}')
        axes[i, 3].set_xlabel('Pixel Difference')
        axes[i, 3].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, f'reversibility_{scheme_name}_comparison.png'), dpi=150)
    plt.close()


def generate_reversibility_summary_plot(results, result_dir):
    """
    生成可逆性評估總結圖表
    """
    models = ['CFG-DDIM', 'CLF-DDIM', 'Uncond-DDIM']
    schemes = list(set(r['scheme'] for r in results))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_to_plot = [
        ('psnr', 'PSNR (dB)', axes[0, 0]),
        ('ssim', 'SSIM', axes[0, 1]),
        ('cosine_similarity', 'Cosine Similarity', axes[0, 2]),
        ('mae', 'MAE', axes[1, 0]),
        ('mse', 'MSE', axes[1, 1]),
        ('max_abs_error', 'Max Absolute Error', axes[1, 2])
    ]
    
    x = np.arange(len(models))
    width = 0.35
    colors = ['steelblue', 'darkorange', 'seagreen']
    
    for metric_key, metric_name, ax in metrics_to_plot:
        for j, scheme in enumerate(schemes):
            values = []
            for model in models:
                for r in results:
                    if r['scheme'] == scheme and r['model'] == model:
                        val = r.get(metric_key, 0)
                        if hasattr(val, 'item'):
                            val = val.item()
                        values.append(val)
                        break
            
            if not values:
                continue
                
            bars = ax.bar(x + j * width, values, width, label=scheme, color=colors[j % len(colors)])
            
            # 添加數值標籤
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if metric_key in ['psnr', 'ssim', 'cosine_similarity']:
                    label_text = f'{val:.4f}'
                else:
                    label_text = f'{val:.2e}'
                ax.annotate(label_text,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}')
        ax.set_xticks(x + width * (len(schemes) - 1) / 2)
        ax.set_xticklabels(models, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Encryption Reversibility Evaluation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'reversibility_summary.png'), dpi=150)
    plt.close()
    
    print(f"總結圖表已保存至: {os.path.join(result_dir, 'reversibility_summary.png')}")


def compute_reversibility_roc_auc(base_samples, decrypted_samples):
    """
    計算可逆性的 ROC/AUC
    
    將像素差異視為「異常分數」，完美可逆時所有像素應該相同（差異=0）
    這裡我們用差異的分布來計算 ROC 曲線
    
    Args:
        base_samples: Base 模型輸出 (N, C, H, W)
        decrypted_samples: 解密後輸出 (N, C, H, W)
    
    Returns:
        fpr, tpr, roc_auc, thresholds
    """
    # 計算像素級差異
    diff = np.abs(base_samples - decrypted_samples)
    
    # 設定閾值：差異大於某個值視為「不可逆」
    # 使用差異的分布來建立二元標籤
    # 這裡使用中位數作為分界：差異小於中位數=可逆(0)，大於=不可逆(1)
    diff_flat = diff.flatten()
    median_diff = np.median(diff_flat)
    
    # 建立二元標籤（基於差異大小）
    # 差異越小越好，所以我們反轉：小差異=正類(1)，大差異=負類(0)
    labels = (diff_flat <= median_diff).astype(int)
    
    # 分數：1 - normalized_diff（差異越小分數越高）
    scores = 1 - visualize(diff_flat)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, thresholds


def compute_anomaly_detection_roc_auc(npz_path, gt_mask_dir):
    """
    計算異常檢測的 pixel-wise ROC/AUC
    
    比較原始影像與重建影像的差異，與 ground truth mask 對比
    
    Args:
        npz_path: NPZ 檔案路徑
        gt_mask_dir: Ground truth mask 目錄
    
    Returns:
        fpr, tpr, roc_auc
    """
    data = load_samples(npz_path)
    
    gt_masks = []
    for i, name in enumerate(data["names"]):
        try:
            gt_mask_img = Image.open(f'{gt_mask_dir}/{name}_Pleural Effusion_mask.png')
        except FileNotFoundError:
            gt_mask_img = Image.new('1', data["originals"][i].squeeze().shape)
        gt_mask = np.array(gt_mask_img, dtype=np.uint8) / 255
        gt_masks.append(gt_mask)
    gt_masks = np.array(gt_masks)
    
    # 計算異常分數（原始與重建的差異）
    diff = np.abs(data["originals"] - data["samples"])
    diff = np.array([visualize(x) for x in diff])
    
    all_scores = diff.reshape(-1)
    all_labels = gt_masks.reshape(-1).astype(int)
    
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


def plot_reversibility_roc_curves(base_npz_paths, encrypted_npz_paths, model_names, result_dir):
    """
    繪製可逆性 ROC 曲線
    
    比較不同模型和加密方案的可逆性
    """
    plt.figure(figsize=(10, 8))
    
    colors = {
        'rademacher': ['#1f77b4', '#2ca02c', '#9467bd'],  # 藍、綠、紫
        'signed_perm': ['#ff7f0e', '#d62728', '#8c564b']   # 橙、紅、棕
    }
    linestyles = {
        'rademacher': '-',
        'signed_perm': '--'
    }
    
    all_aucs = {}
    
    for scheme_name, enc_paths in encrypted_npz_paths.items():
        for i, (base_path, enc_path) in enumerate(zip(base_npz_paths, enc_paths)):
            model_name = model_names[i]
            
            # 載入資料
            base_data = load_samples_raw(base_path)
            enc_data = load_samples_raw(enc_path)
            
            # 使用名稱匹配
            base_samples, enc_samples, matched_names = match_samples_by_name(base_data, enc_data)
            
            if len(matched_names) == 0:
                print(f"    警告: {model_name} + {scheme_name} 沒有匹配樣本，跳過")
                continue
            
            # 計算 ROC
            fpr, tpr, roc_auc, _ = compute_reversibility_roc_auc(base_samples, enc_samples)
            
            label = f"{model_name} + {scheme_name} (AUC={roc_auc:.4f})"
            plt.plot(fpr, tpr, 
                    color=colors[scheme_name][i], 
                    linestyle=linestyles[scheme_name],
                    lw=2, 
                    label=label)
            
            all_aucs[f"{model_name}_{scheme_name}"] = roc_auc
    
    # 繪製對角線（隨機猜測）
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random (AUC=0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Reversibility ROC Curves\n(Base Output vs Decrypted Output)', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(result_dir, 'reversibility_roc_curves.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"ROC 曲線已保存至: {output_path}")
    
    return all_aucs


def plot_anomaly_detection_roc_comparison(base_npz_paths, encrypted_npz_paths, 
                                          model_names, gt_mask_dir, result_dir):
    """
    繪製異常檢測 ROC 曲線比較
    
    比較 Base 模型與加解密後模型的異常檢測性能
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {
        'base': 'black',
        'rademacher': 'steelblue',
        'signed_perm': 'darkorange'
    }
    
    all_aucs = {}
    
    for idx, (model_name, base_path) in enumerate(zip(model_names, base_npz_paths)):
        ax = axes[idx]
        
        # Base 模型 ROC
        fpr_base, tpr_base, auc_base = compute_anomaly_detection_roc_auc(base_path, gt_mask_dir)
        ax.plot(fpr_base, tpr_base, color=colors['base'], lw=2, 
                label=f'Base (AUC={auc_base:.4f})')
        all_aucs[f"{model_name}_base"] = auc_base
        
        # 各加密方案 ROC
        for scheme_name, enc_paths in encrypted_npz_paths.items():
            enc_path = enc_paths[idx]
            fpr, tpr, roc_auc = compute_anomaly_detection_roc_auc(enc_path, gt_mask_dir)
            ax.plot(fpr, tpr, color=colors[scheme_name], lw=2, 
                    linestyle='--' if scheme_name == 'signed_perm' else '-',
                    label=f'{scheme_name} (AUC={roc_auc:.4f})')
            all_aucs[f"{model_name}_{scheme_name}"] = roc_auc
        
        # 對角線
        ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{model_name}\nAnomaly Detection ROC', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Pixel-wise Anomaly Detection: Base vs Encrypted-Decrypted', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(result_dir, 'anomaly_detection_roc_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"異常檢測 ROC 比較圖已保存至: {output_path}")
    
    # 輸出異常檢測 AUROC 到 CSV（與原始 calculate_plot_auroc 格式類似）
    auroc_csv_path = os.path.join(result_dir, 'anomaly_detection_auroc.csv')
    with open(auroc_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Scheme', 'AUROC'])
        for key, val in all_aucs.items():
            parts = key.rsplit('_', 1)
            model = parts[0]
            scheme = parts[1] if len(parts) > 1 else 'base'
            writer.writerow([model, scheme, f'{val:.4f}'])
    
    print(f"異常檢測 AUROC 已保存至: {auroc_csv_path}")
    
    return all_aucs


def plot_combined_roc_summary(base_npz_paths, encrypted_npz_paths, 
                              model_names, gt_mask_dir, result_dir):
    """
    生成綜合 ROC 曲線總結圖
    
    包含：
    1. 可逆性 ROC（Base vs Decrypted 的像素差異）
    2. 異常檢測 ROC（重建差異 vs GT mask）
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 創建子圖佈局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    #==========================================================================
    # 上半部：可逆性 ROC（每個模型一張）
    #==========================================================================
    colors_scheme = {
        'rademacher': 'steelblue',
        'signed_perm': 'darkorange'
    }
    
    reversibility_aucs = {}
    
    for idx, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, idx])
        base_path = base_npz_paths[idx]
        base_data = load_samples_raw(base_path)
        
        for scheme_name, enc_paths in encrypted_npz_paths.items():
            enc_path = enc_paths[idx]
            enc_data = load_samples_raw(enc_path)
            
            # 使用名稱匹配
            base_samples, enc_samples, matched_names = match_samples_by_name(base_data, enc_data)
            
            if len(matched_names) == 0:
                print(f"    警告: {model_name} + {scheme_name} 沒有匹配樣本，跳過")
                continue
            
            fpr, tpr, roc_auc, _ = compute_reversibility_roc_auc(base_samples, enc_samples)
            
            ax.plot(fpr, tpr, color=colors_scheme[scheme_name], lw=2,
                   linestyle='--' if scheme_name == 'signed_perm' else '-',
                   label=f'{scheme_name} (AUC={roc_auc:.4f})')
            
            reversibility_aucs[f"{model_name}_{scheme_name}_rev"] = roc_auc
        
        ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'{model_name}\nReversibility ROC')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    #==========================================================================
    # 下半部：異常檢測 ROC（每個模型一張）
    #==========================================================================
    anomaly_aucs = {}
    
    for idx, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[1, idx])
        base_path = base_npz_paths[idx]
        
        # Base
        fpr, tpr, roc_auc = compute_anomaly_detection_roc_auc(base_path, gt_mask_dir)
        ax.plot(fpr, tpr, color='black', lw=2, label=f'Base (AUC={roc_auc:.4f})')
        anomaly_aucs[f"{model_name}_base_anom"] = roc_auc
        
        # 各加密方案
        for scheme_name, enc_paths in encrypted_npz_paths.items():
            enc_path = enc_paths[idx]
            fpr, tpr, roc_auc = compute_anomaly_detection_roc_auc(enc_path, gt_mask_dir)
            ax.plot(fpr, tpr, color=colors_scheme[scheme_name], lw=2,
                   linestyle='--' if scheme_name == 'signed_perm' else '-',
                   label=f'{scheme_name} (AUC={roc_auc:.4f})')
            anomaly_aucs[f"{model_name}_{scheme_name}_anom"] = roc_auc
        
        ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'{model_name}\nAnomaly Detection ROC')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    # 總標題
    fig.suptitle('ROC Curves Summary: Reversibility (Top) & Anomaly Detection (Bottom)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    output_path = os.path.join(result_dir, 'roc_curves_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC 曲線總結圖已保存至: {output_path}")
    
    # 合併所有 AUC 結果
    all_aucs = {**reversibility_aucs, **anomaly_aucs}
    
    # 保存 AUC 數據到 CSV
    auc_csv_path = os.path.join(result_dir, 'roc_auc_summary.csv')
    with open(auc_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Scheme', 'Type', 'AUC'])
        for key, val in all_aucs.items():
            parts = key.rsplit('_', 2)
            if len(parts) == 3:
                model, scheme, roc_type = parts
                roc_type = 'Reversibility' if roc_type == 'rev' else 'Anomaly Detection'
            else:
                model, scheme, roc_type = parts[0], parts[1], 'Anomaly Detection'
            writer.writerow([model, scheme, roc_type, f'{val:.4f}'])
    
    print(f"AUC 數據已保存至: {auc_csv_path}")
    
    return all_aucs

def rescale_image(image, scale_factor):
    height, width = image.shape[1:]
    image = cv2.resize(image.transpose(1, 2, 0), (np.int(width * scale_factor), np.int(height * scale_factor)), interpolation=cv2.INTER_AREA)
    image = image.transpose(2, 0, 1)
    return image

def calculate_iou(pred_mask, gt_mask, true_pos_only=True):
    """
    Calculate IoU score between two segmentation masks.

    Args:
        pred_mask (np.array): binary segmentation mask
        gt_mask (np.array): binary segmentation mask
    Returns:
        iou_score (np.float64)
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    if true_pos_only:
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(union) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))

    return iou_score

def calculate_precision_recall_specificity(conf_matrix):
    TP = conf_matrix['TP']
    TN = conf_matrix['TN']
    FP = conf_matrix['FP']
    FN = conf_matrix['FN']

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)

    return precision, recall, sensitivity, specificity

def get_conf_matrix(seg_mask, gt_mask):
    TP = np.sum(np.logical_and(seg_mask == 1, gt_mask == 1))
    TN = np.sum(np.logical_and(seg_mask == 0, gt_mask == 0))
    FP = np.sum(np.logical_and(seg_mask == 1, gt_mask == 0))
    FN = np.sum(np.logical_and(seg_mask == 0, gt_mask == 1))

    return dict(TP=TP, TN=TN, FP=FP, FN=FN)

def calculate_metric_scores(input_images, target_images, names, include_iou=False):
    '''
    Inputs:
    - input_images: A Tensor of size (N, C, H, W), C assumed to be 1
    - target_images: A Tensor of size (N, C, H, W), C assumed to be 1
    '''
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarity()
    fid = FrechetInceptionDistance()

    psnr.update(th.from_numpy(input_images), th.from_numpy(target_images))
    ssim.update(th.from_numpy(input_images), th.from_numpy(target_images))
    fid.update(th.from_numpy(np.repeat(input_images, 3, axis=1).clip(0, 1)), True) # FID requires pixel values to be [0,1] and have 3 channels
    fid.update(th.from_numpy(np.repeat(target_images, 3, axis=1).clip(0, 1)), False) # FID requires pixel values to be [0,1] and have 3 channels

    if include_iou:
        pred_masks, gt_masks = create_mask_batch(input_images, target_images, names, args.gt_mask_path, args.pred_mask_threshold)
        ious = []
        dices = []
        conf_matrix = dict(TP=0, TN=0, FP=0, FN=0)

        for pred_mask, gt_mask in zip(list(pred_masks), list(gt_masks)):
            ious.append(calculate_iou(pred_mask, gt_mask))
            dices.append(dice_score(pred_mask, gt_mask))
            curr_matrix = get_conf_matrix(pred_mask, gt_mask)
            for key in conf_matrix.keys():
                conf_matrix[key] += curr_matrix[key]

        ious = np.array(ious)
        dices = np.array(dices)
        _, _, sensitivity, specificity = calculate_precision_recall_specificity(conf_matrix)

        return dict(
            psnr=psnr.compute(),
            ssim=ssim.compute(),
            fid=fid.compute(),
            miou=np.nanmean(ious),
            dice=np.nanmean(dices),
            sensitivity=sensitivity,
            specificity=specificity
            )

    return dict(
            psnr=psnr.compute(),
            ssim=ssim.compute(),
            fid=fid.compute(),
            )

def compute_roc_auc(npz_path, gt_mask_dir):
    data = load_samples(npz_path)

    gt_masks = []
    for i, name in enumerate(data["names"]):
        try:
            gt_mask_img = Image.open(f'{gt_mask_dir}/{name}_Pleural Effusion_mask.png')
        except FileNotFoundError:
            # print('mask not found')
            gt_mask_img = Image.new('1', data["originals"][i].squeeze().shape)
        gt_mask = np.array(gt_mask_img, dtype=np.uint8) / 255
        gt_masks.append(gt_mask)
    gt_masks = np.array(gt_masks)

    diff = np.abs(data["originals"] - data["samples"])
    diff = np.array([visualize(x) for x in diff])

    all_scores = diff.reshape(-1)
    all_labels = gt_masks.reshape(-1).astype(int)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_plot_auroc(npz_paths, model_names, gt_mask_dir):
    fpr_list = []
    tpr_list = []
    auc_list = []
    color_list = ["darkorange", "seagreen", "royalblue"]
    for npz_path in npz_paths:
        fpr, tpr, roc_auc = compute_roc_auc(npz_path, gt_mask_dir)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

    # Step 5: Plot ROC curve
    plt.figure(figsize=(6.5, 5.5))
    
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], color=color_list[i], lw=2, label=f"{model_names[i]} AUROC = {auc_list[i]:.4f}")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pixel-level ROC Curve (CheXpert Validation)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.result_dir}/roc_curve_plot_{args.img_filename}.png")
    plt.close()

def cam_to_segmentation(cam_mask, threshold=np.nan, smoothing=False, k=0):
    """
    Threshold a saliency heatmap to binary segmentation mask.
    Args:
        cam_mask (torch.Tensor): heat map in the original image size (H x W).
            Will squeeze the tensor if there are more than two dimensions.
        threshold (np.float64): threshold to use
        smoothing (bool): if true, smooth the pixelated heatmaps using box filtering
        k (int): size of kernel used for box filter smoothing (int); k must be
                 >= 0; if k is > 0, make sure to set if_smoothing to True,
                 otherwise no smoothing would be performed.

    Returns:
        segmentation (np.ndarray): binary segmentation output
    """
    if (len(cam_mask.shape) > 2):
        cam_mask = cam_mask.squeeze()

    assert len(cam_mask.shape) == 2

    # normalize heatmap
    mask = cam_mask - cam_mask.min()
    mask = mask / mask.max()

    # use Otsu's method to find threshold if no threshold is passed in
    if np.isnan(threshold):
        mask = np.uint8(255 * mask)

        if smoothing:
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            gray_img = cv2.boxFilter(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY),
                                     -1, (k, k))
            # mask = 255 - gray_img
            mask = gray_img

        maxval = np.max(mask)
        thresh = cv2.threshold(mask, 0, maxval, cv2.THRESH_OTSU)[1]

        # draw out contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        polygons = []
        for cnt in cnts:
            if len(cnt) > 1:
                polygons.append([list(pt[0]) for pt in cnt])

        # create segmentation based on contour
        img_dims = (mask.shape[1], mask.shape[0])
        segmentation_output = Image.new('1', img_dims)
        for polygon in polygons:
            coords = [(point[0], point[1]) for point in polygon]
            ImageDraw.Draw(segmentation_output).polygon(coords,
                                                        outline=1,
                                                        fill=1)
        segmentation = np.array(segmentation_output, dtype="int")
    else:
        segmentation = np.array(mask > threshold, dtype="int")
        if smoothing:
            smoothed_mask = segmentation.copy().astype(np.float32)
            smoothed_mask = cv2.boxFilter(smoothed_mask, -1, (k, k))
            
            # Threshold again to get binary mask
            segmentation = (smoothed_mask > threshold).astype(np.uint8)

    return segmentation

def overlay_masks(image, masks, colors=None, alpha=0.5, border_thickness=2, border_color=None, seed=None):
    """
    Overlays multiple segmentation masks on an image with borders around contours.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image as a numpy array of shape (H, W, C) where C is the number of channels.
    masks : list of numpy.ndarray
        List of masks, each as a numpy array of shape (H, W, C) matching the image dimensions.
        Non-zero values in the mask indicate the areas to be colored.
    colors : list of tuple/list or None, optional
        List of RGB colors for the masks, each as a tuple/list of 3 values.
        If None, random colors will be generated for each mask.
    alpha : float, optional
        Transparency of the overlay, between 0 and 1. Default is 0.5.
    border_thickness : int, optional
        Thickness of the contour border in pixels. Default is 2.
    border_color : tuple/list or None, optional
        RGB color for the contour borders. If None, the same color as the mask will be used
        but with increased intensity.
    seed : int or None, optional
        Random seed for reproducible color generation. Default is None.
        
    Returns:
    --------
    numpy.ndarray
        Image with all masks overlaid on it and contour borders drawn.
    """
    # Check if masks is empty
    if not masks:
        return image.copy()
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Make a copy of the input image to avoid modifying the original
    output = image.copy().astype(np.float32)
    
    # Generate random colors if not provided
    if colors is None:
        colors = []
        for _ in range(len(masks)):
            # Generate random RGB values
            color = (np.random.randint(0, 255), 
                     np.random.randint(0, 255), 
                     np.random.randint(0, 255))
            colors.append(color)
    
    # Check that we have enough colors
    if len(colors) < len(masks):
        raise ValueError("Not enough colors provided for all masks")
    
    # Process each mask
    for i, mask in enumerate(masks):
        # Check that image and mask have the same shape
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask {i} must have the same shape. "
                            f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        
        # Create a binary mask for contour detection (sum across channels)
        binary_mask = (np.sum(mask, axis=2) > 0).astype(np.uint8)
        
        # Create a colored mask
        colored_mask = np.zeros_like(image, dtype=np.float32)
        for j in range(min(3, image.shape[2])):
            colored_mask[..., j] = np.where(mask[..., j] > 0, colors[i][j], 0)
        
        # Create a mask for blending (any channel > 0)
        blend_mask = (np.sum(mask, axis=2) > 0)[..., np.newaxis]
        blend_mask = np.repeat(blend_mask, image.shape[2], axis=2)
        
        # Overlay the colored mask on the image
        output = np.where(blend_mask, 
                         (1 - alpha) * output + alpha * colored_mask, 
                         output)
    
    # Ensure output values are within valid range for drawing contours
    output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
    
    # Draw contours for each mask
    for i, mask in enumerate(masks):
        # Create binary mask for contour detection
        binary_mask = (np.sum(mask, axis=2) > 0).astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Determine border color
        if border_color is None:
            # Make the border slightly brighter than the mask color
            # but preserve the hue
            b_color = tuple([min(int(c * 1.5), 255) for c in colors[i]])
        else:
            b_color = border_color
            
        # Draw contours on the output image
        cv2.drawContours(output_uint8, contours, -1, b_color, border_thickness)
    
    return output_uint8

def get_image_with_mask_overlay(input_image, pred_mask, gt_mask, threshold=np.nan):
    org_scan = (np.repeat(input_image.transpose(1, 2, 0), 3, axis=2) * 255.).astype(np.uint8)
    pred_mask_overlay = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
    gt_mask_overlay = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2)

    overlaid_scan = overlay_masks(org_scan, [pred_mask_overlay, gt_mask_overlay], colors=[(255, 0, 255), (0, 0, 255)],border_thickness=1)
    overlaid_scan = overlaid_scan.transpose(2, 0, 1)
    return overlaid_scan

def create_mask_batch(input_images, target_images, names, gt_mask_dir, threshold=np.nan):
    pred_mask_list = []
    gt_mask_list = []
    for i in range(input_images.shape[0]):
        pred_mask, gt_mask = create_mask(input_images[i][0], target_images[i][0], names[i], gt_mask_dir, threshold)
        pred_mask_list.append(pred_mask)
        gt_mask_list.append(gt_mask)
    return np.array(pred_mask_list), np.array(gt_mask_list)

def create_mask(input_image, target_image, name, gt_mask_dir, threshold=np.nan):
    '''
    Args:
    -----
        input_image, target_image (numpy.ndarray): (C x H x W) image
        name: str
        gt_mask_dir: str
        threshold: float or np.nan

    Returns:
    --------
        pred_mask
            predicted mask generated from heatmap
        gt_mask
            ground truth mask loaded from chexlocalize
    '''
    diff = np.array(abs(input_image - target_image))
    diff = visualize(diff)
    pred_mask = cam_to_segmentation(diff, threshold=threshold, smoothing=True, k=7).astype(np.uint8)
    
    try:
        gt_mask_img = Image.open(f'{gt_mask_dir}/{name}_Pleural Effusion_mask.png')
    except FileNotFoundError:
        # print('mask not found')
        gt_mask_img = Image.new('1', input_image.squeeze().shape)
    gt_mask = np.array(gt_mask_img, dtype=np.uint8) / 255

    return pred_mask, gt_mask

def generate_diff_batch(original, reconstructed):
    '''
    Inputs:
    - original - N, C, H, W tensor
    - reconstructed - N, C, H, W tensor

    Returns
    - N, C, H, W tensor with the applied color map
    '''
    diff_list = []
    for i in range(original.shape[0]):
        diff = generate_diff(original[i][0], reconstructed[i][0])
        diff_list.append(diff)
    return np.concatenate(diff_list, axis=1)

def generate_diff(original, reconstructed):
    '''
    Inputs:
    - original - H, W tensor
    - reconstructed - H, W tensor

    Returns
    - C, H, W tensor with the applied color map
    '''
    cm = plt.get_cmap('jet') # Applying this colormap will return an image in RGBA format
    diff = np.array(abs(original - reconstructed))
    diff = visualize(diff)
    colored_diff = cm(diff)[:, :, :3] # Remove alpha channel
    return colored_diff.transpose(2, 0, 1) # HWC -> CHW

def generate_grid_for_mpl(npz_paths, dest_path='', x_labels=None, generate_masks=False):
    master_image = None
    master_samples = []
    master_diffs = []
    master_overlays = []

    img_grid = []
    y_labels = None
    y_label_colors = None
    if generate_masks:
        col_attr_length = len(npz_paths) * 3 + 1
    else:
        col_attr_length = len(npz_paths) * 2 + 1

    for i, path in enumerate(npz_paths):
        sample_data = load_samples(path)

        if i == 0:
            batch_size = len(sample_data['names'])
            y_labels = sample_data['names']
            y_label_colors = sample_data['org_labels']
            if x_labels is None:
                x_labels = range(col_attr_length)
            assert len(x_labels) == (col_attr_length)

            master_image = np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1)
            if master_image.shape[2] == 512:
                master_image = rescale_image(master_image, 0.5)
            img_list = np.array_split(master_image, batch_size, axis=1)
            for img in img_list:
                img_grid.append([img])
        
        master_sample = np.concatenate(np.repeat(sample_data['samples'], 3, axis=1), axis=1)
        if master_sample.shape[2] == 512:
            master_sample = rescale_image(master_sample, 0.5)
        master_samples.append(master_sample)
        
        master_diff = generate_diff_batch(sample_data['originals'], sample_data['samples'])
        if master_diff.shape[2] == 512:
            master_diff = rescale_image(master_diff, 0.5)
        master_diffs.append(master_diff)

        if generate_masks:
            pred_masks, gt_masks = create_mask_batch(sample_data['originals'],
                                                    sample_data['samples'],
                                                    sample_data['names'],
                                                    args.gt_mask_path,
                                                    args.pred_mask_threshold)
            pred_masks = np.repeat(np.concatenate(pred_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            gt_masks = np.repeat(np.concatenate(gt_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            master_org = (np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1).transpose(1, 2, 0) * 255).astype(np.uint8)
            master_overlay = overlay_masks(master_org, [pred_masks, gt_masks], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
            master_overlays.append((master_overlay / 255.).transpose(2, 0, 1))

    for sample_col in master_samples:
        img_list = np.array_split(sample_col, batch_size, axis=1)
        for i, img in enumerate(img_list):
            img_grid[i].append(img)

    for sample_col in master_diffs:
        img_list = np.array_split(sample_col, batch_size, axis=1)
        for i, img in enumerate(img_list):
            img_grid[i].append(img)
    
    if generate_masks:
        for sample_col in master_overlays:
            img_list = np.array_split(sample_col, batch_size, axis=1)
            for i, img in enumerate(img_list):
                img_grid[i].append(img)

    nrow = len(img_grid)
    ncol = col_attr_length

    fig = plt.subplots(figsize=(ncol * 2.56, nrow * 2.56), dpi=100, layout='tight')
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)

    fontdict = {'size': 16}

    for row in range(nrow):
        for col in range(ncol):
            im = img_grid[row][col]
            ax = plt.subplot(gs[row, col])
            ax.imshow(np.transpose(im, [1, 2, 0]), cmap='gray', aspect='auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(top=False, right=False, bottom=False, left=False, pad=0)
            ax.set(frame_on=False)
            if row == 0:
                ax.set_xlabel(x_labels[col], wrap=True, fontdict=fontdict)
                ax.xaxis.set_label_position('top')
            # if col == 0:
            #     ax.set_ylabel(y_labels.tolist()[row].split('_')[0], wrap=True, fontdict=fontdict)
            #     if y_label_colors.tolist()[row] == 1:
            #         ax.yaxis.label.set_color('red')
            #     ax.yaxis.set_label_position('left')
    
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(dest_path, f"{args.img_filename}_mpl.png"), dpi=100)
    plt.close()

def generate_metrics(npz_paths, include_iou=False):
    results = []
    for i, path in enumerate(npz_paths): # This is a column
        sample_data = load_samples(path)
        results.append(calculate_metric_scores(sample_data['originals'], sample_data['samples'], sample_data['names'], include_iou=include_iou))
    return results

def generate_master_image(npz_paths, generate_masks=False):
    master_image = None
    master_samples = []
    master_diffs = []
    master_overlays = []

    for i, path in enumerate(npz_paths): # This is a column
        sample_data = load_samples(path)

        if i == 0:
            master_image = np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1)
            if master_image.shape[2] == 512:
                master_image = rescale_image(master_image, 0.5)
        
        master_sample = np.concatenate(np.repeat(sample_data['samples'], 3, axis=1), axis=1)
        if master_sample.shape[2] == 512:
            master_sample = rescale_image(master_sample, 0.5)
        master_samples.append(master_sample)
        
        master_diff = generate_diff_batch(sample_data['originals'], sample_data['samples'])
        if master_diff.shape[2] == 512:
            master_diff = rescale_image(master_diff, 0.5)
        master_diffs.append(master_diff)

        if generate_masks:
            pred_masks, gt_masks = create_mask_batch(sample_data['originals'],
                                                    sample_data['samples'],
                                                    sample_data['names'],
                                                    args.gt_mask_path,
                                                    args.pred_mask_threshold)
            pred_masks = np.repeat(np.concatenate(pred_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            gt_masks = np.repeat(np.concatenate(gt_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            master_org = (np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1).transpose(1, 2, 0) * 255).astype(np.uint8)
            master_overlay = overlay_masks(master_org, [pred_masks, gt_masks], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
            master_overlays.append((master_overlay / 255.).transpose(2, 0, 1))

    
    master_samples = np.concatenate(np.array(master_samples), axis=2)
    master_diffs = np.concatenate(np.array(master_diffs), axis=2)
    if generate_masks:
        master_overlays = np.concatenate(np.array(master_overlays), axis=2)
        return np.concatenate([master_image, master_samples, master_diffs, master_overlays], axis=2)
    return np.concatenate([master_image, master_samples, master_diffs], axis=2)

def generate_case_comparison_mpl(npz_paths, dest_path='', y_labels=None, generate_masks=False, data_idx=0):
    img_grid = []
    x_labels = ["Input", "Output", "Heatmap"]

    if generate_masks:
        x_labels.append(f"Overlay (th={'Otsu' if args.pred_mask_threshold == np.nan else args.pred_mask_threshold})")
        col_attr_length = 4
    else:
        col_attr_length = 3

    for i, path in enumerate(npz_paths):
        sample_data = load_samples(path)

        data_name = sample_data['names'][data_idx]
        input_image = sample_data['originals'][data_idx]
        output_image = sample_data['samples'][data_idx]
        heatmap = generate_diff(input_image[0], output_image[0])

        if generate_masks:
            pred_mask, gt_mask = create_mask(input_image[0],
                                             output_image[0],
                                             data_name,
                                             args.gt_mask_path,
                                             args.pred_mask_threshold)

        input_image = (np.repeat(input_image, 3, axis=0).transpose(1, 2, 0) * 255).astype(np.uint8)
        output_image = (np.repeat(output_image, 3, axis=0).transpose(1, 2, 0) * 255).astype(np.uint8)
        heatmap = heatmap.transpose(1, 2, 0)
        img_row = [input_image, output_image, heatmap]

        if generate_masks:
            pred_mask = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
            gt_mask = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2)
            mask_overlay = overlay_masks(input_image, [pred_mask, gt_mask], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
            img_row.append(mask_overlay)
        
        img_grid.append(img_row)
    
    nrow = len(img_grid)
    ncol = col_attr_length

    fig = plt.subplots(figsize=(ncol * 2.56, nrow * 2.56), dpi=100, layout='tight')
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)

    fontdict = {'size': 16}

    for row in range(nrow):
        for col in range(ncol):
            im = img_grid[row][col]
            ax = plt.subplot(gs[row, col])
            ax.imshow(im, cmap='gray', aspect='auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(top=False, right=False, bottom=False, left=False, pad=0)
            ax.set(frame_on=False)
            if row == 0:
                ax.set_xlabel(x_labels[col], wrap=True, fontdict=fontdict)
                ax.xaxis.set_label_position('top')
            if col == 0:
                ax.set_ylabel(y_labels[row], wrap=True, fontdict=fontdict)
                ax.yaxis.set_label_position('left')
    
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(dest_path, f"{args.img_filename}_{data_name}_comp_mpl.png"), dpi=100)
    plt.close()

def split_generate_grid_for_mpl(npz_paths, dest_path='', x_labels=None, generate_masks=False, split_size=None):
    all_sample_data = [load_samples(p) for p in npz_paths]
    total_samples = len(all_sample_data[0]['names'])

    if split_size is None:
        split_size = total_samples

    for start in range(0, total_samples, split_size):
        end = min(start + split_size, total_samples)
        cur_size = end - start

        master_image = None
        master_samples = []
        master_diffs = []
        master_overlays = []
        img_grid = []
        y_labels = None
        y_label_colors = None

        col_attr_length = len(npz_paths) * (3 if generate_masks else 2) + 1
        if x_labels is None:
            x_labels = list(range(col_attr_length))
        assert len(x_labels) == col_attr_length

        for i, sample_data in enumerate(all_sample_data):
            cur_names = sample_data['names'][start:end]
            cur_orgs = sample_data['originals'][start:end]
            cur_samples = sample_data['samples'][start:end]
            cur_org_labels = sample_data['org_labels'][start:end]

            if i == 0:
                y_labels = cur_names
                y_label_colors = cur_org_labels

                master_image = np.concatenate(np.repeat(cur_orgs, 3, axis=1), axis=1)
                if master_image.shape[2] == 512:
                    master_image = rescale_image(master_image, 0.5)
                img_list = np.array_split(master_image, cur_size, axis=1)
                for img in img_list:
                    img_grid.append([img])

            sample_img = np.concatenate(np.repeat(cur_samples, 3, axis=1), axis=1)
            if sample_img.shape[2] == 512:
                sample_img = rescale_image(sample_img, 0.5)
            master_samples.append(sample_img)

            diff = generate_diff_batch(cur_orgs, cur_samples)
            if diff.shape[2] == 512:
                diff = rescale_image(diff, 0.5)
            master_diffs.append(diff)

            if generate_masks:
                pred_masks, gt_masks = create_mask_batch(
                    cur_orgs, cur_samples, cur_names,
                    args.gt_mask_path, args.pred_mask_threshold)
                pred_masks = np.repeat(np.concatenate(pred_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
                gt_masks = np.repeat(np.concatenate(gt_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
                master_org = (np.concatenate(np.repeat(cur_orgs, 3, axis=1), axis=1).transpose(1, 2, 0) * 255).astype(np.uint8)
                master_overlay = overlay_masks(master_org, [pred_masks, gt_masks], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
                master_overlays.append((master_overlay / 255.).transpose(2, 0, 1))

        for sample_col in master_samples:
            img_list = np.array_split(sample_col, cur_size, axis=1)
            for i, img in enumerate(img_list):
                img_grid[i].append(img)

        for sample_col in master_diffs:
            img_list = np.array_split(sample_col, cur_size, axis=1)
            for i, img in enumerate(img_list):
                img_grid[i].append(img)

        if generate_masks:
            for sample_col in master_overlays:
                img_list = np.array_split(sample_col, cur_size, axis=1)
                for i, img in enumerate(img_list):
                    img_grid[i].append(img)

        # Plotting section
        nrow = len(img_grid)
        ncol = col_attr_length
        fig = plt.subplots(figsize=(ncol * 2.56, nrow * 2.56), dpi=100, layout='tight')
        gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)
        fontdict = {'size': 16}

        for row in range(nrow):
            for col in range(ncol):
                im = img_grid[row][col]
                ax = plt.subplot(gs[row, col])
                ax.imshow(np.transpose(im, [1, 2, 0]), cmap='gray', aspect='auto')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(top=False, right=False, bottom=False, left=False, pad=0)
                ax.set(frame_on=False)
                if row == 0:
                    ax.set_xlabel(x_labels[col], wrap=True, fontdict=fontdict)
                    ax.xaxis.set_label_position('top')
                if col == 0:
                    ax.set_ylabel(y_labels.tolist()[row].split('_')[0], wrap=True, fontdict=fontdict)
                    if y_label_colors.tolist()[row] == 1:
                        ax.yaxis.label.set_color('red')
                    ax.yaxis.set_label_position('left')

        plt.tight_layout(pad=1)
        out_filename = os.path.join(dest_path, f"{args.img_filename}_mpl_part_{start:03d}_{end:03d}.png")
        plt.savefig(out_filename, dpi=100)
        plt.close()

def split_generate_master_image(npz_paths, generate_masks=False, split_size=None):
    all_sample_data = [load_samples(p) for p in npz_paths]
    total_samples = len(all_sample_data[0]['names'])

    if split_size is None:
        split_size = total_samples

    master_images = []

    for start in range(0, total_samples, split_size):
        end = min(start + split_size, total_samples)
        cur_size = end - start

        master_image = None
        master_samples = []
        master_diffs = []
        master_overlays = []

        for i, sample_data in enumerate(all_sample_data):
            cur_orgs = sample_data['originals'][start:end]
            cur_samples = sample_data['samples'][start:end]

            if i == 0:
                master_image = np.concatenate(np.repeat(cur_orgs, 3, axis=1), axis=1)
                if master_image.shape[2] == 512:
                    master_image = rescale_image(master_image, 0.5)

            sample_img = np.concatenate(np.repeat(cur_samples, 3, axis=1), axis=1)
            if sample_img.shape[2] == 512:
                sample_img = rescale_image(sample_img, 0.5)
            master_samples.append(sample_img)

            diff = generate_diff_batch(cur_orgs, cur_samples)
            if diff.shape[2] == 512:
                diff = rescale_image(diff, 0.5)
            master_diffs.append(diff)

            if generate_masks:
                pred_masks, gt_masks = create_mask_batch(
                    cur_orgs,
                    cur_samples,
                    sample_data['names'][start:end],
                    args.gt_mask_path,
                    args.pred_mask_threshold
                )
                pred_masks = np.repeat(np.concatenate(pred_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
                gt_masks = np.repeat(np.concatenate(gt_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
                master_org = (np.concatenate(np.repeat(cur_orgs, 3, axis=1), axis=1).transpose(1, 2, 0) * 255).astype(np.uint8)
                overlay = overlay_masks(master_org, [pred_masks, gt_masks], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
                master_overlays.append((overlay / 255.).transpose(2, 0, 1))

        # Combine columns
        master_samples_img = visualize(np.concatenate(master_samples, axis=2))
        master_diffs_img = visualize(np.concatenate(master_diffs, axis=2))
        
        if generate_masks:
            master_overlays_img = visualize(np.concatenate(master_overlays, axis=2))
            combined = np.concatenate([master_image, master_samples_img, master_diffs_img, master_overlays_img], axis=2)
        else:
            combined = np.concatenate([master_image, master_samples_img, master_diffs_img], axis=2)

        master_images.append(combined)

    return master_images  # list of np.ndarray, each a master image

def main():
    os.makedirs(args.result_dir, exist_ok=True)
    # x_labels = ['Original']
    # scenarios = args.x_labels
    # x_labels.extend(scenarios)
    # x_labels.extend([f'Heatmap ({x})' for x in scenarios])
    # if args.include_miou:
    #     x_labels.extend([f'Overlay ({x})' for x in scenarios])

    large_grid_labels = ['Original']
    scenarios = args.model_labels
    large_grid_labels.extend(scenarios)
    large_grid_labels.extend([f'Heatmap\n({x})' for x in scenarios])

    if args.include_miou:
        large_grid_labels.extend([f"Overlay (th={'Otsu' if args.pred_mask_threshold == np.nan else args.pred_mask_threshold})\n({x})" for x in scenarios])
    
    if args.save_image:
        # generate_case_comparison_mpl(args.npz_paths, dest_path=args.result_dir, y_labels=args.model_labels, generate_masks=args.include_miou, data_idx=args.data_idx)

        if args.split_size == 0:
            generate_grid_for_mpl(args.npz_paths, dest_path=args.result_dir, x_labels=large_grid_labels, generate_masks=args.include_miou)
        else:
            assert args.split_size > 0
            split_generate_grid_for_mpl(args.npz_paths, dest_path=args.result_dir, x_labels=large_grid_labels, generate_masks=args.include_miou, split_size=args.split_size)

        if args.split_size == 0:
            final_image = generate_master_image(args.npz_paths, generate_masks=args.include_miou)
            final_image = (final_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(final_image).save(os.path.join(args.result_dir, f"{args.img_filename}.png"))
        else:
            final_images = split_generate_master_image(args.npz_paths, generate_masks=args.include_miou, split_size=args.split_size)
            for i, image in enumerate(final_images):
                image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(image).save(os.path.join(args.result_dir, f"{args.img_filename}_{i:02d}.png"))

    final_metrics = generate_metrics(args.npz_paths, args.include_miou)

    calculate_plot_auroc(args.npz_paths, args.model_labels, args.gt_mask_path)

    if args.include_miou:
        print(f'Model,PSNR,SSIM,FID,mIoU,Sensitivity,Specificity')
        for scenario, metric in zip(scenarios, final_metrics):
            print(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f},{metric['miou'].item():02.3f},{metric['sensitivity'].item():02.3f},{metric['specificity'].item():02.3f}")

        with open(os.path.join(args.result_dir, f"{args.csv_filename}.csv"), 'w') as csvfile:
            csvfile.write(f'Model,PSNR,SSIM,FID,mIoU,Sensitivity,Specificity\n')
            for scenario, metric in zip(scenarios, final_metrics):
                csvfile.write(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f},{metric['miou'].item():02.3f},{metric['sensitivity'].item():02.3f},{metric['specificity'].item():02.3f}\n")
    else:
        print(f'Model,PSNR,SSIM,FID')
        for scenario, metric in zip(scenarios, final_metrics):
            print(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f}")

        with open(os.path.join(args.result_dir, f"{args.csv_filename}.csv"), 'w') as csvfile:
            csvfile.write(f'Model,PSNR,SSIM,FID\n')
            for scenario, metric in zip(scenarios, final_metrics):
                csvfile.write(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f}\n")

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--result_dir", required=False, default=os.path.join('results', 'latest'), type=str)
    parser.add_argument(f"--npz_paths", required=False, type=str, nargs='+')
    parser.add_argument(f"--model_labels", required=False, default=None, type=str, nargs='+')
    parser.add_argument(f"--include_miou", required=False, default=False, type=str2bool)
    parser.add_argument(f"--save_image", required=False, default=True, type=str2bool)
    parser.add_argument(f"--gt_mask_path", type=str, default='/workspace/CheXlocalize/256_segmentations_pleural_effusion')
    parser.add_argument(f"--pred_mask_threshold", type=float, default=np.nan)
    parser.add_argument(f"--img_filename", type=str, default="compiled_master_image")
    parser.add_argument(f"--csv_filename", type=str, default="metrics.csv")
    parser.add_argument(f"--data_idx", type=int, required=False)
    parser.add_argument(f"--split_size", type=int, default=0)
    
    # 可逆性評估參數
    parser.add_argument(f"--eval_mode", type=str, default="standard", 
                        choices=["standard", "reversibility"],
                        help="評估模式: standard=原始vs重建, reversibility=Base vs 加解密後")
    parser.add_argument(f"--base_npz_paths", type=str, nargs='+',
                        help="可逆性評估: Base 模型 NPZ 路徑 [cfg, clf, uncond]")
    parser.add_argument(f"--rademacher_npz_paths", type=str, nargs='+',
                        help="可逆性評估: Rademacher 解密後 NPZ 路徑")
    parser.add_argument(f"--signed_perm_npz_paths", type=str, nargs='+',
                        help="可逆性評估: Signed Permutation 解密後 NPZ 路徑")
    parser.add_argument(f"--include_roc", type=str2bool, default=True,
                        help="是否生成 ROC/AUC 曲線圖")
    parser.add_argument(f"--include_anomaly_roc", type=str2bool, default=True,
                        help="是否生成異常檢測 ROC 比較圖（需要 gt_mask_path）")
    return parser

if __name__ == '__main__':
    args = create_argparser().parse_args()
    
    if args.eval_mode == "reversibility":
        # 可逆性評估模式
        print("="*60)
        print("加解密可逆性評估模式")
        print("="*60)
        
        encrypted_npz_paths = {}
        
        if args.rademacher_npz_paths:
            encrypted_npz_paths['rademacher'] = args.rademacher_npz_paths
        
        if args.signed_perm_npz_paths:
            encrypted_npz_paths['signed_perm'] = args.signed_perm_npz_paths
        
        if not encrypted_npz_paths:
            print("錯誤: 請至少指定一種加密方案的 NPZ 路徑")
            print("  --rademacher_npz_paths 或 --signed_perm_npz_paths")
            exit(1)
        
        if not args.base_npz_paths:
            print("錯誤: 請指定 Base 模型的 NPZ 路徑")
            print("  --base_npz_paths cfg.npz clf.npz uncond.npz")
            exit(1)
        
        model_names = ['CFG-DDIM', 'CLF-DDIM', 'Uncond-DDIM']
        
        # 基本可逆性指標評估
        results = evaluate_encryption_reversibility(
            args.base_npz_paths,
            encrypted_npz_paths,
            list(encrypted_npz_paths.keys()),
            args.result_dir
        )
        
        # 生成跨模型比較圖
        print("\n" + "="*60)
        print("生成跨模型比較圖")
        print("="*60)
        for scheme_name, enc_paths in encrypted_npz_paths.items():
            print(f"\n  處理加密方案: {scheme_name}")
            # 簡潔版本（6 列：各模型 Base + Dec 交錯排列）
            generate_cross_model_comparison_image(
                args.base_npz_paths,
                enc_paths,
                scheme_name,
                args.result_dir,
                n_images=3
            )
            # 完整版本（含差異圖，9 列）
            generate_cross_model_comparison_with_diff(
                args.base_npz_paths,
                enc_paths,
                scheme_name,
                args.result_dir,
                n_images=3
            )
            # Base 優先版本（6 列：所有 Base → 所有 Dec）
            generate_cross_model_base_first_comparison(
                args.base_npz_paths,
                enc_paths,
                scheme_name,
                args.result_dir,
                n_images=3
            )
        
        # 生成可逆性指標總結圖
        generate_reversibility_summary_plot(results, args.result_dir)
        
        # 生成 ROC/AUC 曲線
        if args.include_roc:
            print("\n" + "="*60)
            print("生成 ROC/AUC 曲線")
            print("="*60)
            
            # 可逆性 ROC 曲線
            reversibility_aucs = plot_reversibility_roc_curves(
                args.base_npz_paths,
                encrypted_npz_paths,
                model_names,
                args.result_dir
            )
            
            # 異常檢測 ROC 比較（如果有 GT mask）
            if args.include_anomaly_roc and os.path.exists(args.gt_mask_path):
                print("\n生成異常檢測 ROC 比較圖...")
                anomaly_aucs = plot_anomaly_detection_roc_comparison(
                    args.base_npz_paths,
                    encrypted_npz_paths,
                    model_names,
                    args.gt_mask_path,
                    args.result_dir
                )
                
                # 生成綜合 ROC 總結圖
                print("\n生成綜合 ROC 總結圖...")
                all_aucs = plot_combined_roc_summary(
                    args.base_npz_paths,
                    encrypted_npz_paths,
                    model_names,
                    args.gt_mask_path,
                    args.result_dir
                )
            else:
                if args.include_anomaly_roc:
                    print(f"\n警告: GT mask 路徑不存在: {args.gt_mask_path}")
                    print("跳過異常檢測 ROC 比較圖生成")
        
        print("\n" + "="*60)
        print("評估完成！")
        print("="*60)
        print(f"\n結果目錄: {args.result_dir}")
        
    else:
        # 標準評估模式
        if not args.npz_paths:
            print("錯誤: 標準模式需要指定 --npz_paths")
            exit(1)
        main()