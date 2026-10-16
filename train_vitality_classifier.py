"""
train_vitality_classifier.py
============================
訓練魚隻活力分類器 (四路投影 ViT-S，904 維輸入)。

Feature layout (904 維，由 build_pseudo_labeled_dataset.py 產出):
  [0:384]    global_emb   DINOv2 backbone CLS token
  [384:640]  object_emb   RF-DETR decoder query embedding
  [640:896]  query_diff   curr_query - prev_query (語意運動特徵)
  [896:900]  bbox         (x, y, w, h) normalized
  [900:904]  bbox_diff    bbox_t - bbox_{t-1}

四路投影 (d_model = 384, ViT-S):
  global  (384) → 128
  object  (256) → 128
  qdiff   (256) →  96
  geom    (  8) →  32     [bbox(4) + bbox_diff(4)]

訓練策略:
  - Mixup (α=0.2) 搭配 label smoothing 0.1
  - WeightedRandomSampler (1/sqrt(count)) 處理類別不均衡 (可透過
    USE_WEIGHTED_SAMPLER 切換；pd2021 場域「弱」類僅佔 14%)
  - Cosine LR schedule with warmup
  - Pretrained ViT blocks 與新 heads 使用不同 learning rate (前者較低)
  - 每 epoch 印出三類 individual accuracy，避免整體 top-1 acc 掩蓋
    少數類別學習不佳的問題

資料路徑設定:
  預設讀取 output/vitality_dataset/{place}/features_csv/ 下的 train.csv 與
  val.csv。路徑定義於下方 __main__ 區塊，部署其他場域時直接修改即可。

本腳本假設 window 長度固定 (MIN_TRACK_LEN=40，無 padding mask)，
因此模型 forward 不處理 attention mask。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import csv
import os
import math
from collections import Counter
import matplotlib.pyplot as plt
import timm


# ==========================================
# 1. Mixup
# ==========================================
def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==========================================
# 2. ViT Model — 四路投影分類器
# ==========================================
class ViTActionClassifier(nn.Module):
    """
    輸入: (B, T, 904)，T 預設 60

    特徵結構 (904 維，與 build_pseudo_labeled_dataset.py 的輸出對齊):
      [0:384]    global_emb   DINOv2 backbone CLS
      [384:640]  object_emb   RF-DETR decoder query embedding
      [640:896]  query_diff   curr_query - prev_query (語意運動特徵)
      [896:900]  bbox
      [900:904]  bbox_diff
    """
    # Layout indices (constants for clarity)
    IDX_GLOBAL     = (0, 384)
    IDX_OBJECT     = (384, 640)
    IDX_QUERY_DIFF = (640, 896)
    IDX_GEOM       = (896, 904)   # bbox + bbox_diff

    GLOBAL_DIM     = 384
    OBJECT_DIM     = 256
    QUERY_DIFF_DIM = 256
    GEOM_DIM       = 8            # bbox(4) + bbox_diff(4)

    def __init__(self, input_dim=904, num_classes=101,
                 vit_name='vit_small_patch16_224', pretrained=True,
                 num_layers=4, dropout=0.1, drop_path_rate=0.0, max_seq_len=62):
        super().__init__()
        assert input_dim == 904, f"expected input_dim=904, got {input_dim}"

        vit = timm.create_model(vit_name, pretrained=pretrained,
                                drop_path_rate=drop_path_rate)
        d_model = vit.embed_dim  # 384

        # 四路投影預算 (d_model = 384 切成 4 路, 對應各 source 維度跟重要性)
        global_out = 128
        object_out = 128
        qdiff_out  = 96
        geom_out   = d_model - global_out - object_out - qdiff_out   # 32
        assert geom_out > 0, "d_model 太小, 4 路分配不夠"

        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, global_out),
            nn.LayerNorm(global_out),
        )
        self.object_proj = nn.Sequential(
            nn.Linear(self.OBJECT_DIM, object_out),
            nn.LayerNorm(object_out),
        )
        self.qdiff_proj = nn.Sequential(
            nn.Linear(self.QUERY_DIFF_DIM, qdiff_out),
            nn.LayerNorm(qdiff_out),
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(self.GEOM_DIM, geom_out),
            nn.LayerNorm(geom_out),
        )
        self.proj_norm = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(vit.cls_token.clone().detach())
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.global_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(list(vit.blocks)[:num_layers])
        self.norm = vit.norm

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        del vit

    def forward(self, x):
        B, T, _ = x.shape

        # 切 4 路 (沿用 IDX_* constants 提高可讀性)
        global_emb  = x[..., self.IDX_GLOBAL[0]:self.IDX_GLOBAL[1]]
        object_emb  = x[..., self.IDX_OBJECT[0]:self.IDX_OBJECT[1]]
        query_diff  = x[..., self.IDX_QUERY_DIFF[0]:self.IDX_QUERY_DIFF[1]]
        geom        = x[..., self.IDX_GEOM[0]:self.IDX_GEOM[1]]

        global_feat = self.global_proj(global_emb)
        object_feat = self.object_proj(object_emb)
        qdiff_feat  = self.qdiff_proj(query_diff)
        geom_feat   = self.geom_proj(geom)

        x_proj = self.proj_norm(
            torch.cat([global_feat, object_feat, qdiff_feat, geom_feat], dim=-1)
        )

        global_motion = x_proj.mean(dim=1, keepdim=True) + self.global_token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, global_motion, x_proj], dim=1)
        x = x + self.pos_embed[:, :T + 2, :]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return self.head(x[:, 0, :] + x[:, 1, :])


# ==========================================
# 3. Dataset
# ==========================================
class FeatureDataset(Dataset):
    def __init__(self, csv_file, is_train=False, noise_std=0.02):
        self.is_train = is_train
        self.noise_std = noise_std
        self.data_list = []

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    gmm_label = int(row[1].strip())
                    new_label = LABEL_MAP[gmm_label] # 進行轉換
                    self.data_list.append((row[0].strip(), new_label))

        mode = "train+noise" if is_train else "eval"
        print(f"載入 {len(self.data_list)} 筆特徵 [{mode}]")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        npy_path, label = self.data_list[idx]
        feat = np.load(npy_path)
        feat_tensor = torch.tensor(feat, dtype=torch.float32)
        if self.is_train:
            feat_tensor = feat_tensor + torch.randn_like(feat_tensor) * self.noise_std
        return feat_tensor, torch.tensor(label, dtype=torch.long)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":

    # -- Paths --
    TRAIN_CSV = "output/vitality_dataset/ph2021/features_csv/train.csv"
    VAL_CSV   = "output/vitality_dataset/ph2021/features_csv/val.csv"
    CKPT_DIR  = "output/vitality_dataset/ph2021/ckpt"

    # 從既有 ckpt 續訓；設為 None 則從零訓練。
    # 續訓的 ckpt 必須與本腳本的 ViT 架構相容 (904 維輸入、四路投影、相同 num_layers)。
    RESUME_FROM_CKPT = None
    FINETUNE_EPOCHS  = 20

    INPUT_DIM   = 904
    NUM_CLASSES = 3                  # High / Medium / Low (對應 K=3 GMM 之 sorted cluster)
    CLASS_NAMES = ['High', 'Medium', 'Low']

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

    VIT_NAME       = 'vit_small_patch16_224'
    NUM_LAYERS     = 4
    DROPOUT        = 0.3
    DROP_PATH_RATE = 0.1
    MIXUP_ALPHA    = 0.2

    BATCH_SIZE    = 64
    SCRATCH_EPOCHS = 80
    WARMUP_EPOCHS = 10
    PATIENCE      = 40

    LR_PRETRAINED_SCRATCH  = 1e-5
    LR_NEW_SCRATCH         = 1e-4
    LR_PRETRAINED_FINETUNE = 5e-6
    LR_NEW_FINETUNE        = 2e-5

    USE_WEIGHTED_SAMPLER = False
    MONITOR_METRIC       = 'val_acc'

    IS_FINETUNE = RESUME_FROM_CKPT is not None
    if IS_FINETUNE:
        EPOCHS = FINETUNE_EPOCHS
        LR_PRETRAINED, LR_NEW = LR_PRETRAINED_FINETUNE, LR_NEW_FINETUNE
        RUN_TAG = "finetune"
    else:
        EPOCHS = SCRATCH_EPOCHS
        LR_PRETRAINED, LR_NEW = LR_PRETRAINED_SCRATCH, LR_NEW_SCRATCH
        RUN_TAG = "scratch"

    print(f"\n=== ViT Vitality Classifier Training (904-d input) ===")
    print(f"Mode: {RUN_TAG}")
    print(f"Resume from: {RESUME_FROM_CKPT if RESUME_FROM_CKPT else 'None (train from scratch)'}")
    print(f"Epochs: {EPOCHS}, LR (pretrained/new): {LR_PRETRAINED:.0e} / {LR_NEW:.0e}")
    print(f"Features: global / object / qdiff / geom (904-d)")
    print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}\n")

    for p in [TRAIN_CSV, VAL_CSV]:
        if not os.path.exists(p):
            print(f"CSV not found: {p}")
            print(f"   請先執行 build_pseudo_labeled_dataset.py 產生訓練資料。")
            exit()
    if IS_FINETUNE and not os.path.exists(RESUME_FROM_CKPT):
        print(f"RESUME_FROM_CKPT not found: {RESUME_FROM_CKPT}")
        exit()

    print("Loading datasets...")
    train_dataset = FeatureDataset(TRAIN_CSV, is_train=True, noise_std=0.02)
    val_dataset   = FeatureDataset(VAL_CSV, is_train=False)

    if USE_WEIGHTED_SAMPLER:
        train_labels = [label for _, label in train_dataset.data_list]
        class_counts = Counter(train_labels)
        class_weights = {c: 1.0 / math.sqrt(n) for c, n in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(f"WeightedRandomSampler 啟用 (1/sqrt(count) 軟平衡)")
        print(f"   類別數: {len(class_counts)}, min={min(class_counts.values())}, "
              f"max={max(class_counts.values())}")
        # 3-class 不平衡情境下, 列出每類具體 count 比抽 min/max 直觀
        print(f"   Per-class train counts:")
        for c in sorted(class_counts.keys()):
            cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            print(f"     [{c}] {cname:>4}: {class_counts[c]:>6,} "
                  f"({100*class_counts[c]/len(train_labels):>5.2f}%)")
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                sampler=sampler, num_workers=4, pin_memory=True)
    else:
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nInitializing model...")
    model = ViTActionClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        vit_name=VIT_NAME,
        pretrained=True,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        drop_path_rate=DROP_PATH_RATE,
    ).to(device)

    if IS_FINETUNE:
        print(f"\nResuming from {RESUME_FROM_CKPT}...")
        state_dict = torch.load(RESUME_FROM_CKPT, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   Missing keys: {missing}")
        if unexpected:
            print(f"   Unexpected keys: {unexpected}")
        print(f"   權重載入完成")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable: {total_params:,}")

    pretrained_ids = (
        set(id(p) for p in model.blocks.parameters()) |
        {id(model.cls_token), id(model.norm.weight), id(model.norm.bias)}
    )
    pretrained_params = [p for p in model.parameters() if id(p) in pretrained_ids]
    new_params = [p for p in model.parameters() if id(p) not in pretrained_ids]

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': LR_PRETRAINED, 'weight_decay': 1e-3},
        {'params': new_params, 'lr': LR_NEW, 'weight_decay': 5e-3},
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)

    os.makedirs(CKPT_DIR, exist_ok=True)
    version = 1
    while True:
        save_path = os.path.join(CKPT_DIR, f"vitality_classifier_{RUN_TAG}_v{version}.pth")
        plot_path = os.path.join(CKPT_DIR, f"vitality_classifier_{RUN_TAG}_curve_v{version}.png")
        if not os.path.exists(save_path) and not os.path.exists(plot_path):
            break
        version += 1

    best_val_loss = float("inf")
    best_val_acc  = -1.0
    patience_counter = 0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_per_class_acc": []}

    print(f"\nTraining ({INPUT_DIM}d, {NUM_LAYERS} blocks, {EPOCHS} epochs, "
          f"monitor={MONITOR_METRIC}, patience={PATIENCE})...\n" + "-" * 70)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(
                inputs, labels, alpha=MIXUP_ALPHA, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (lam * (predicted == labels_a).sum().item()
                        + (1 - lam) * (predicted == labels_b).sum().item())

        scheduler.step()
        train_loss = running_loss / len(dataloader)
        train_acc = 100.0 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        # Per-class tracking (fish 3-class 不平衡情境必要的 diagnostic)
        per_class_correct = [0] * NUM_CLASSES
        per_class_total   = [0] * NUM_CLASSES
        # Optional: 簡易 confusion matrix (預測 × 真實)
        confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                # Per-class
                for c in range(NUM_CLASSES):
                    mask = (labels == c)
                    per_class_total[c]   += mask.sum().item()
                    per_class_correct[c] += ((predicted == c) & mask).sum().item()
                # Confusion matrix
                for t, p in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                    confusion[t][p] += 1

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        per_class_acc = [
            (100.0 * per_class_correct[c] / per_class_total[c]
             if per_class_total[c] > 0 else 0.0)
            for c in range(NUM_CLASSES)
        ]
        current_lr = optimizer.param_groups[1]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_per_class_acc"].append(per_class_acc)

        improved = False
        if MONITOR_METRIC == 'val_acc':
            if val_acc > best_val_acc:
                improved = True
                best_val_acc = val_acc
                best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                improved = True
                best_val_loss = val_loss
                best_val_acc = val_acc

        marker = " <-- best" if improved else ""

        per_class_str = "  ".join(
            f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else c}={per_class_acc[c]:.1f}%"
            for c in range(NUM_CLASSES)
        )
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"PerClass: {per_class_str} | "
              f"LR: {current_lr:.2e}{marker}")

        if improved:
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # 印 confusion matrix (best epoch only, 避免每 epoch 太冗)
            print(f"   Confusion (row=true, col=pred):")
            header = "        " + "  ".join(
                f"{(CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)):>5}"
                for c in range(NUM_CLASSES)
            )
            print(header)
            for t in range(NUM_CLASSES):
                row_name = CLASS_NAMES[t] if t < len(CLASS_NAMES) else str(t)
                row = f"   {row_name:>4} " + "  ".join(
                    f"{confusion[t][p]:>5d}" for p in range(NUM_CLASSES)
                )
                print(row)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}. Best: epoch {best_epoch} "
                      f"({MONITOR_METRIC}={best_val_acc:.2f}% / loss={best_val_loss:.4f})")
                break

    print(f"\nBest checkpoint: {save_path}")

    actual_epochs = len(history["train_loss"])
    x = range(1, actual_epochs + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x, history["train_loss"], color="#E24B4A", linewidth=1.5, label="Train")
    plt.plot(x, history["val_loss"], color="#378ADD", linewidth=1.5, label="Val")
    plt.axvline(x=best_epoch, color="#1D9E75", linestyle="--", linewidth=0.8,
                label=f"best (ep {best_epoch})")
    plt.title(f"Loss ({RUN_TAG})"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x, history["train_acc"], color="#E24B4A", linewidth=1.5, label="Train")
    plt.plot(x, history["val_acc"], color="#378ADD", linewidth=1.5, label="Val")
    plt.axvline(x=best_epoch, color="#1D9E75", linestyle="--", linewidth=0.8,
                label=f"best (ep {best_epoch})")
    plt.title(f"Overall Accuracy ({RUN_TAG})"); plt.xlabel("Epoch"); plt.legend()

    # Per-class accuracy 曲線, 看「弱」這類是否真的學起來
    plt.subplot(1, 3, 3)
    pc_arr = np.array(history["val_per_class_acc"])   # (epochs, NUM_CLASSES)
    palette = ["#E24B4A", "#378ADD", "#1D9E75", "#F2A65A", "#9B59B6"]
    for c in range(NUM_CLASSES):
        cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        plt.plot(x, pc_arr[:, c],
                 color=palette[c % len(palette)],
                 linewidth=1.5, label=cname)
    plt.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"Per-class Val Acc ({RUN_TAG})"); plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)"); plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Curve saved: {plot_path}")