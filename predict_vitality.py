"""
predict_vitality.py
===================
魚隻活力分類器 (ViT) 的 window-level 推論與評估腳本。

對預先切好的 test set (格式: cluster_X/*.npy) 做分類，輸出：
    - 整體 window-level 準確率
    - 每類別 (High / Medium / Low) 的 breakdown
    - sklearn classification_report (precision / recall / f1)
    - Confusion matrix 圖檔 (.png) 與對應的文字 report
"""

import os
import glob

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ViT 分類器定義與訓練腳本共用
from train_vitality_classifier import ViTActionClassifier


# ==========================================
# 1. Label mapping — 需與訓練腳本一致
# ==========================================
NUM_CLASSES = 3

# 將 GMM 分群編號 (gmm_label) 映射為模型訓練標籤 (0=High, 1=Medium, 2=Low)。
#
# 注意: GMM 為無監督分群演算法，cluster 編號本身無語意、且每次執行可能不同
# (受 random seed、場域資料分布影響)。此 mapping 需依該場域實際 GMM 結果調整:
# 跑完 build_pseudo_labeled_dataset.py 的 STAGE 4 後，查看各 cluster 的平均速度、
# 語意變化量等特徵，判定每個 cluster 對應高/中/低活力後填入。
#
# 以下預設值對應 ph2021 場域的訓練設定；部署到其他場域時需重新評估。
LABEL_MAP = {
    2: 0,  # cluster 2 對應 High   (高活力)
    0: 1,  # cluster 0 對應 Medium (中活力)
    1: 2,  # cluster 1 對應 Low    (低活力)
}

# 模型輸出標籤 (0, 1, 2) 對應的顯示名稱
CLASS_NAMES = ['High', 'Medium', 'Low']


# ==========================================
# 2. 掃 test dir → 每個 npy 一個 sample
# ==========================================
def collect_samples(test_npy_root):
    """
    掃 test_npy_root/cluster_X/*.npy, 回傳 [(npy_path, model_label), ...].
    """
    samples = []
    for gmm_label in range(NUM_CLASSES):
        cluster_dir = os.path.join(test_npy_root, f'cluster_{gmm_label}')
        if not os.path.isdir(cluster_dir):
            print(f"找不到 cluster dir: {cluster_dir}")
            continue
        
        # 將資料夾的 GMM 編號轉換為模型使用的訓練標籤 (詳見 LABEL_MAP 定義)
        model_label = LABEL_MAP[gmm_label]
        
        cluster_files = glob.glob(os.path.join(cluster_dir, "*.npy"))
        for npy_path in cluster_files:
            samples.append((npy_path, model_label))
            
        print(f"   cluster_{gmm_label} (對應模型 Label {model_label}: {CLASS_NAMES[model_label]}): "
              f"{len(cluster_files):,} windows")
    return samples


# ==========================================
# 3. Window-level batch inference
# ==========================================
def predict_windows(samples, model, device, batch_size=64):
    """
    對 samples 中所有 npy 做 window-level forward.

    Args:
        samples: [(npy_path, true_label), ...]

    Returns:
        all_true: (N,) np.int
        all_pred: (N,) np.int
    """
    all_true, all_pred = [], []
    n = len(samples)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = samples[i:i + batch_size]
            # 全部 window shape 應該一致 (T 固定), 直接 stack
            feats = np.stack([np.load(p) for p, _ in batch])   # (B, T, 904)
            labels = [lab for _, lab in batch]

            x = torch.tensor(feats, dtype=torch.float32).to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()

            all_true.extend(labels)
            all_pred.extend(pred.tolist())

            # 簡易進度
            if (i // batch_size) % 20 == 0:
                done = min(i + batch_size, n)
                print(f"   [{done:>6d}/{n}] processed", end='\r')

    print(f"   [{n:>6d}/{n}] processed         ")
    return np.array(all_true), np.array(all_pred)


# ==========================================
# 4. 評估
# ==========================================
def evaluate_test_set(test_npy_root, weight_path, output_dir, place,
                      input_dim=904, num_layers=4,
                      vit_name='vit_small_patch16_224', batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Place:       {place}")
    print(f"Test root:   {test_npy_root}")
    print(f"Weight:      {weight_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Label map:   {LABEL_MAP}\n")

    if not os.path.isdir(test_npy_root):
        raise SystemExit(f"Test dir not found: {test_npy_root}")
    if not os.path.exists(weight_path):
        raise SystemExit(f"Weight file not found: {weight_path}")

    # ---- Model ----
    model = ViTActionClassifier(
        input_dim=input_dim, num_classes=NUM_CLASSES,
        vit_name=vit_name, pretrained=False,
        num_layers=num_layers,
    ).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing keys: {missing}")
    if unexpected:
        print(f"   Unexpected keys: {unexpected}")
    model.eval()
    print(f"載入權重完成\n")

    # ---- Collect ----
    print("掃 test dir...")
    samples = collect_samples(test_npy_root)
    if not samples:
        raise SystemExit(f"沒掃到任何 npy in {test_npy_root}")
    print(f"共 {len(samples):,} 個 windows 待推論\n")

    # ---- Predict ----
    print("推論中 (window-level)...")
    all_true, all_pred = predict_windows(samples, model, device, batch_size)

    # ---- Per-class breakdown ----
    print("\n" + "-" * 70)
    # 建立反向映射以利在 Log 中對照原始 cluster 資料夾
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    for c in range(NUM_CLASSES):
        mask = all_true == c
        n_c = int(mask.sum())
        acc = 100.0 * (all_pred[mask] == c).mean() if n_c > 0 else 0.0
        print(f"[{CLASS_NAMES[c]} (cluster_{INV_LABEL_MAP[c]})]  "
              f"windows: {n_c:6,} → acc {acc:5.2f}%")

    overall_acc = 100.0 * (all_true == all_pred).mean()
    print("\n" + "=" * 70)
    print(f"【Overall window accuracy】: {overall_acc:.2f}%   "
          f"(N={len(all_true):,})")
    print("=" * 70 + "\n")

    # ---- Classification report ----
    labels = list(range(NUM_CLASSES))
    print("Window-level classification report:")
    print(classification_report(all_true, all_pred,
                                labels=labels, target_names=CLASS_NAMES,
                                digits=3, zero_division=0))

    # ---- Confusion matrix (count + %) ----
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm * 100.0, row_sum,
                       out=np.zeros_like(cm, dtype=float),
                       where=row_sum > 0)
    annot = np.array([[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
                       for j in range(NUM_CLASSES)]
                      for i in range(NUM_CLASSES)])

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=True, annot_kws={"size": 12})
    plt.title(f"Fish Vitality — Confusion Matrix (place={place}, window-level)\n"
              f"overall acc: {overall_acc:.2f}%")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(output_dir,
                           f"confusion_matrix_{place}_window.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"\nConfusion matrix: {cm_path}")

    # ---- 文字 report ----
    report_path = os.path.join(output_dir,
                               f"classification_report_{place}_window.txt")
    with open(report_path, "w") as f:
        f.write(f"Fish Vitality Test Results (window-level) — place={place}\n")
        f.write(f"Weight: {weight_path}\n")
        f.write(f"Label map: {LABEL_MAP}\n\n")
        f.write(f"Overall window accuracy: {overall_acc:.2f}%  "
                f"(N={len(all_true):,})\n\n")
        f.write("Per-class breakdown:\n")
        for c in range(NUM_CLASSES):
            mask = all_true == c
            n_c = int(mask.sum())
            acc = 100.0 * (all_pred[mask] == c).mean() if n_c > 0 else 0.0
            f.write(f"   [{CLASS_NAMES[c]} (cluster_{INV_LABEL_MAP[c]})]: {n_c:,} windows, "
                    f"acc={acc:.2f}%\n")
        f.write("\nWindow-level classification report:\n")
        f.write(classification_report(all_true, all_pred,
                                      labels=labels, target_names=CLASS_NAMES,
                                      digits=3, zero_division=0))
        f.write("\nConfusion matrix (row=true, col=pred):\n")
        header = "        " + "  ".join(
            f"{n:>7}" for n in CLASS_NAMES) + "   ← pred\n"
        f.write(header)
        for i, name in enumerate(CLASS_NAMES):
            row = f"   {name:>4} " + "  ".join(
                f"{cm[i, j]:>7d}" for j in range(NUM_CLASSES)
            )
            f.write(row + "\n")
    print(f"文字報告: {report_path}")

    return overall_acc


# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    PLACE = 'ph2021'    # 'pd2021' / 'ph2021' / 'ntou2022'

    ROOT          = f"output/vitality_dataset/{PLACE}"
    TEST_NPY_ROOT = os.path.join(ROOT, "features", "test")
    WEIGHT_FILE   = "weights/vitality_classifier.pth"
    OUTPUT_DIR    = os.path.join(ROOT, "test_results")

    evaluate_test_set(
        test_npy_root=TEST_NPY_ROOT,
        weight_path=WEIGHT_FILE,
        output_dir=OUTPUT_DIR,
        place=PLACE,
        input_dim=904,
        num_layers=4,
    )