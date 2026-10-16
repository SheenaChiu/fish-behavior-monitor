"""
train_association_head.py
=========================
訓練 AssociationHead 的腳本。以 SAM 3 產生的高信心度追蹤軌跡作為偽標籤，
透過對比式學習將 RF-DETR 的 256 維 decoder query embedding 投影到
128 維 metric space，供下游 tracker 進行個體識別配對。

流程:
  1. 對訓練資料跑 RF-DETR，取得每個 detection 的 256 維 query embedding。
  2. 對每個 SAM 3 track 的每個 frame:
       (a) 從 SAM 3 csv 讀取該 frame 的 bbox；
       (b) 跑 RF-DETR 同幀 forward，找出與該 bbox IoU 最高的 detection；
       (c) 以該 detection 的 query embedding 作為此 (track_id, frame_idx)
           的個體特徵。
  3. 以同一條 track 內的不同 frame 組成 positive pair。
  4. 以 InfoNCE loss 訓練 head，batch 內其他 positive 自動成為 negatives
     (in-batch shuffled negatives)。

Usage (Fish dataset):
  python train_association_head.py \
    --rfdetr_ckpt weights/detector.pth \
    --sam3_csv_dir output/sam3_tracks \
    --frame_dir data/fish \
    --output_dir output/association_head \
    --domain fish \
    --dataset_mode fish

  其中:
    --sam3_csv_dir 指到 parent 目錄 (本腳本會自動掃描 ph2021_*_en_fixed/ 子目錄)
    --frame_dir    指到 parent 目錄 (本腳本會自動對應到 ph2021_X_en/clip_NNNN.mp4)

  訓練完的權重會存為 {output_dir}/association_head_fish.pth，使用前可複製
  (或改名為) weights/association_head.pth 以符合推論流程預設路徑。

本腳本亦支援 UCF101 的 cross-domain 訓練 (使用 --domain ucf101 --dataset_mode ucf101)，
此時 --frame_dir 應指到 UCF101 的 frame 資料夾結構 (action subdirs + jpg frames)。

SAM 3 csv 格式:
  frame_idx, obj_id, center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, mask_area
"""

import os
import argparse
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import cv2
from tqdm import tqdm

from association_head import AssociationHead, InfoNCELoss


# ==========================================
# 1. SAM 3 csv 讀取
# ==========================================
def load_sam3_tracks(csv_path: str) -> dict:
    """
    讀 SAM 3 csv. 格式:
      frame_idx, obj_id, center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, mask_area

    Returns:
        {obj_id: [(frame_idx, (bx, by, bw, bh))]} sorted by frame_idx
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    tracks = defaultdict(list)
    for _, row in df.iterrows():
        obj_id = int(row['obj_id'])
        fid    = int(row['frame_idx'])
        bbox   = (float(row['bbox_x']), float(row['bbox_y']),
                  float(row['bbox_w']), float(row['bbox_h']))
        tracks[obj_id].append((fid, bbox))

    for k in tracks:
        tracks[k].sort(key=lambda x: x[0])
    return dict(tracks)


def load_frame_from_dir(frame_dir: str, frame_idx: int):
    """從 frame dir 內讀 jpg (UCF101 style)."""
    for idx in [frame_idx + 1, frame_idx]:
        for fmt in [f"{idx:04d}.jpg", f"{idx:06d}.jpg", f"{idx}.jpg",
                    f"frame_{idx:04d}.jpg", f"frame_{idx:06d}.jpg"]:
            path = os.path.join(frame_dir, fmt)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


class VideoFrameReader:
    """從 mp4 讀 frame (fish dataset 用), cache 已讀的避免重複 seek."""
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open {video_path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cache = {}
        self.last_idx = -1

    def read(self, frame_idx: int):
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        if frame_idx < 0 or frame_idx >= self.total:
            return None
        # sequential read 比 seek 快; 如果要的 idx 在 last+1, 直接 read
        if frame_idx == self.last_idx + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
        if not ret:
            return None
        self.last_idx = frame_idx
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cache[frame_idx] = rgb
        return rgb

    def close(self):
        self.cap.release()
        self.cache.clear()


def bbox_iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


# ==========================================
# 2. Embedding extractor (與 tracker 共用同一組 RF-DETR forward 邏輯)
# ==========================================
class EmbeddingExtractor:
    def __init__(self, rfdetr_ckpt: str, input_size: int = 378,
                 num_windows: int = 1, det_conf: float = 0.3, device: str = 'cuda'):
        from rfdetr import RFDETRNano

        self.device = device
        self.input_size = input_size
        self.det_conf   = det_conf

        print(f"Loading RF-DETR from {rfdetr_ckpt}...")
        m = RFDETRNano(
            num_classes=1, resolution=input_size, patch_size=14,
            num_windows=num_windows, positional_encoding_size=input_size // 14,
            encoder='dinov2_small', pretrain_weights=rfdetr_ckpt,
            freeze_encoder=True,
        )
        self.lwdetr = m.model.model.to(device).eval()
        self.postprocess = m.model.postprocess
        print(f"  Loaded. input={input_size}, det_conf={det_conf}")

        self.preprocess = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            boxes_xywh: (N, 4) numpy float32, original image coords
            qfeats:     (N, 256) numpy float32, raw decoder query features
        """
        H0, W0 = img_rgb.shape[:2]
        x = self.preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        raw = self.lwdetr(x)
        target_sizes = torch.tensor([[H0, W0]], device=self.device)
        results = self.postprocess(raw, target_sizes=target_sizes)
        r = results[0]
        keep = r["scores"] >= self.det_conf
        if keep.sum() == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, 256), dtype=np.float32))

        boxes_xyxy = r["boxes"][keep].cpu().numpy()
        boxes_xywh = np.zeros_like(boxes_xyxy)
        boxes_xywh[:, 0] = boxes_xyxy[:, 0]
        boxes_xywh[:, 1] = boxes_xyxy[:, 1]
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        qfeats = r["query_features"][keep].cpu().numpy()
        return boxes_xywh.astype(np.float32), qfeats.astype(np.float32)


# ==========================================
# 3. Pseudo-label collection
# ==========================================
def collect_track_embeddings(extractor, sam3_csv_dir, frame_dir,
                              iou_thresh=0.5, dataset_mode='ucf101',
                              max_videos=None, max_frames_per_track=None):
    """
    對每個 SAM 3 track 的每個 frame, 抽 RF-DETR query_features.

    dataset_mode:
      'ucf101': sam3_csv_dir / {action} / {video}.csv,  frame_dir / {action} / {video} / *.jpg
      'fish':   sam3_csv_dir 是 parent (e.g. output/database),
                                  內含 ph2021_X_en_fixed/ph2021_X_NNNN_fixed.csv
                frame_dir 是 parent (e.g. data),
                                  內含 ph2021_X_en/clip_NNNN.mp4

    Returns:
        {global_track_id (str): [(frame_idx, 256-d np.array)]}
    """
    track_embeddings = defaultdict(list)

    # 列出 csv files 跟對應的 video source
    # csv_files 元素: (action_or_prefix, video_base, csv_path, video_source_path, source_is_mp4)
    csv_files = []
    if dataset_mode == 'ucf101':
        for action in sorted(os.listdir(sam3_csv_dir)):
            adir = os.path.join(sam3_csv_dir, action)
            if not os.path.isdir(adir):
                continue
            for f in sorted(os.listdir(adir)):
                if f.endswith('.csv'):
                    video_base = f.replace('.csv', '')
                    csv_path = os.path.join(adir, f)
                    src_path = os.path.join(frame_dir, action, video_base)   # frame dir
                    csv_files.append((action, video_base, csv_path, src_path, False))
    elif dataset_mode == 'fish':
        # sam3_csv_dir = output/database/, 內 ph2021_X_en_fixed/ph2021_X_NNNN_fixed.csv
        # frame_dir    = data/,           內 ph2021_X_en/clip_NNNN.mp4
        import re
        pattern = re.compile(r'ph2021_([A-D])_(\d+)_fixed\.csv$')
        for prefix in sorted(os.listdir(sam3_csv_dir)):
            sub = os.path.join(sam3_csv_dir, prefix)
            if not os.path.isdir(sub):
                continue
            # prefix 應該長 "ph2021_A_en_fixed"
            for f in sorted(os.listdir(sub)):
                m = pattern.match(f)
                if not m:
                    continue
                letter, num = m.group(1), m.group(2)
                csv_path = os.path.join(sub, f)
                mp4_path = os.path.join(frame_dir, f"ph2021_{letter}_en",
                                        f"clip_{num}.mp4")
                if not os.path.exists(mp4_path):
                    print(f"WARN: mp4 not found for {csv_path} -> {mp4_path}")
                    continue
                video_base = f"ph2021_{letter}_{num}"
                csv_files.append((f"ph2021_{letter}", video_base, csv_path,
                                  mp4_path, True))
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    if max_videos is not None:
        csv_files = csv_files[:max_videos]

    print(f"Found {len(csv_files)} csv/video pairs")

    n_match = 0; n_no_det = 0; n_low_iou = 0; n_no_frame = 0

    for action, video_base, csv_path, src_path, is_mp4 in tqdm(csv_files, desc="Videos"):
        if not os.path.exists(src_path):
            continue

        tracks = load_sam3_tracks(csv_path)
        if not tracks:
            continue

        # 重組成 {frame_idx: [(track_id, bbox), ...]}, 一個 frame 跑一次 forward
        frame_to_tracks = defaultdict(list)
        for tid, frame_list in tracks.items():
            sampled = frame_list
            if (max_frames_per_track is not None
                and len(sampled) > max_frames_per_track):
                idxs = np.linspace(0, len(sampled) - 1,
                                   max_frames_per_track).astype(int)
                sampled = [sampled[i] for i in idxs]
            for fid, bbox in sampled:
                frame_to_tracks[fid].append((tid, bbox))

        # 開啟 video reader (mp4) 或留空 (frame dir mode 用 load_frame_from_dir)
        video_reader = None
        if is_mp4:
            try:
                video_reader = VideoFrameReader(src_path)
            except FileNotFoundError:
                continue

        for fid in sorted(frame_to_tracks.keys()):
            if is_mp4:
                img = video_reader.read(fid)
            else:
                img = load_frame_from_dir(src_path, fid)

            if img is None:
                n_no_frame += len(frame_to_tracks[fid])
                continue

            det_boxes, det_qfeats = extractor(img)
            if len(det_boxes) == 0:
                n_no_det += len(frame_to_tracks[fid])
                continue

            for tid, sam3_bbox in frame_to_tracks[fid]:
                # 找跟 SAM 3 bbox IoU 最高的 detection
                best_iou = 0.0; best_i = -1
                for i, db in enumerate(det_boxes):
                    o = bbox_iou_xywh(sam3_bbox, db)
                    if o > best_iou:
                        best_iou = o; best_i = i
                if best_iou >= iou_thresh:
                    global_tid = f"{action}::{video_base}::{tid}"
                    track_embeddings[global_tid].append(
                        (fid, det_qfeats[best_i])
                    )
                    n_match += 1
                else:
                    n_low_iou += 1

        if video_reader is not None:
            video_reader.close()

    # 過濾太短 track (positive pair 至少需要 2 frame)
    track_embeddings = {k: v for k, v in track_embeddings.items() if len(v) >= 2}

    print(f"\nMatching stats:")
    print(f"  Successful matches:    {n_match}")
    print(f"  Frame not loadable:    {n_no_frame}")
    print(f"  No detection in frame: {n_no_det}")
    print(f"  Low IoU (< thresh):    {n_low_iou}")
    print(f"  Valid tracks (>=2 frames): {len(track_embeddings)}")
    print(f"  Total embeddings: {sum(len(v) for v in track_embeddings.values())}")
    return dict(track_embeddings)


# ==========================================
# 4. Positive pair dataset
# ==========================================
class PositivePairDataset(Dataset):
    """
    每次 __getitem__ 從某個 track 隨機抽 2 個 frame 的 embedding 當 positive pair.
    Batch 內所有 (anchor, positive) 對之間互為 negative (InfoNCE shuffle 機制).
    """
    def __init__(self, track_embeddings: dict, samples_per_track: int = 20):
        self.tracks = []
        self.track_ids = []
        for tid, frame_list in track_embeddings.items():
            self.tracks.append(frame_list)
            self.track_ids.append(tid)
        self.samples_per_track = samples_per_track

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def __getitem__(self, idx):
        track_idx = idx % len(self.tracks)
        frames = self.tracks[track_idx]
        i, j = random.sample(range(len(frames)), 2)
        anchor = torch.from_numpy(frames[i][1]).float()
        pos    = torch.from_numpy(frames[j][1]).float()
        return anchor, pos, track_idx


# ==========================================
# 5. Training
# ==========================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: 抽 embedding (有 cache 就 load) ----
    cache_path = os.path.join(args.output_dir, "track_embeddings_cache.pt")
    if os.path.exists(cache_path) and not args.no_cache:
        print(f"Loading cached embeddings from {cache_path}...")
        track_embeddings = torch.load(cache_path, weights_only=False)
        print(f"  {len(track_embeddings)} tracks, "
              f"{sum(len(v) for v in track_embeddings.values())} embeddings")
    else:
        extractor = EmbeddingExtractor(
            rfdetr_ckpt=args.rfdetr_ckpt,
            input_size=378,
            num_windows=1,
            det_conf=args.det_conf,
            device=device,
        )
        track_embeddings = collect_track_embeddings(
            extractor, args.sam3_csv_dir, args.frame_dir,
            iou_thresh=args.iou_thresh,
            dataset_mode=args.dataset_mode,
            max_videos=args.max_videos,
            max_frames_per_track=args.max_frames_per_track,
        )
        torch.save(track_embeddings, cache_path)
        print(f"Cached to {cache_path}")
        del extractor
        torch.cuda.empty_cache()

    if len(track_embeddings) < args.batch_size:
        print(f"ERROR: too few tracks ({len(track_embeddings)}) < batch_size ({args.batch_size})")
        print("       Try lowering --batch_size or use more videos")
        return

    # ---- Step 2: train/val split (by track) ----
    track_ids = list(track_embeddings.keys())
    random.seed(args.seed)
    random.shuffle(track_ids)
    n_val = max(int(len(track_ids) * 0.1), args.batch_size)
    val_ids = set(track_ids[:n_val])
    train_te = {k: v for k, v in track_embeddings.items() if k not in val_ids}
    val_te   = {k: v for k, v in track_embeddings.items() if k in val_ids}
    print(f"\nTrain tracks: {len(train_te)}  Val tracks: {len(val_te)}")

    train_set = PositivePairDataset(train_te, samples_per_track=args.samples_per_track)
    val_set   = PositivePairDataset(val_te,   samples_per_track=5)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                               shuffle=False, num_workers=2, drop_last=True)

    # ---- Step 3: model ----
    head = AssociationHead(in_dim=256, proj_dim=args.proj_dim).to(device)
    loss_fn = InfoNCELoss(temperature=args.temperature).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---- Step 4: train ----
    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"temperature={args.temperature}, lr={args.lr}")
    print("-" * 75)

    best_val_acc = 0.0
    best_val_margin = 0.0
    save_path = os.path.join(args.output_dir, f"association_head_{args.domain}.pth")

    for epoch in range(args.epochs):
        # train
        head.train()
        tot_loss = tot_acc = tot_pos = tot_neg = 0.0
        n_batch = 0
        for anchor, pos, _ in train_loader:
            anchor = anchor.to(device); pos = pos.to(device)
            z_a = head(anchor); z_p = head(pos)
            out = loss_fn(z_a, z_p)

            optim.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()

            tot_loss += out['loss'].item()
            tot_acc  += out['acc'].item()
            tot_pos  += out['pos_sim'].item()
            tot_neg  += out['neg_sim'].item()
            n_batch += 1

        scheduler.step()
        tr_loss = tot_loss / n_batch
        tr_acc  = tot_acc / n_batch
        tr_pos  = tot_pos / n_batch
        tr_neg  = tot_neg / n_batch

        # val
        head.eval()
        v_loss = v_acc = v_pos = v_neg = 0.0; v_n = 0
        with torch.no_grad():
            for anchor, pos, _ in val_loader:
                anchor = anchor.to(device); pos = pos.to(device)
                z_a = head(anchor); z_p = head(pos)
                out = loss_fn(z_a, z_p)
                v_loss += out['loss'].item()
                v_acc  += out['acc'].item()
                v_pos  += out['pos_sim'].item()
                v_neg  += out['neg_sim'].item()
                v_n += 1
        if v_n > 0:
            val_loss = v_loss / v_n; val_acc = v_acc / v_n
            val_pos  = v_pos / v_n; val_neg = v_neg / v_n
            val_margin = val_pos - val_neg
        else:
            val_loss = val_acc = val_pos = val_neg = val_margin = 0.0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_margin = val_margin
            torch.save(head.state_dict(), save_path)
            marker = " <-- best"

        lr_now = optim.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"tr: L={tr_loss:.3f} acc={tr_acc:.3f} pos/neg={tr_pos:.2f}/{tr_neg:.2f} | "
              f"val: L={val_loss:.3f} acc={val_acc:.3f} margin={val_margin:.3f} | "
              f"lr={lr_now:.1e}{marker}")

    print("-" * 75)
    print(f"Best val acc:    {best_val_acc:.4f}")
    print(f"Best val margin: {best_val_margin:.4f} (pos_sim - neg_sim)")
    print(f"Saved to: {save_path}")
    print(f"\nNote: val_margin (pos_sim - neg_sim) 越大，表示 head 將同軌跡與不同軌跡的特徵區分能力越強。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rfdetr_ckpt', required=True)
    p.add_argument('--sam3_csv_dir', required=True)
    p.add_argument('--frame_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--domain', choices=['fish', 'ucf101'], required=True)
    p.add_argument('--dataset_mode', choices=['ucf101', 'fish'], required=True,
                   help='ucf101: action subdirs + frame-jpg; '
                        'fish: ph2021_X_en_fixed dir + mp4 source')

    # Pseudo-label extraction
    p.add_argument('--det_conf', type=float, default=0.3)
    p.add_argument('--iou_thresh', type=float, default=0.5)
    p.add_argument('--max_videos', type=int, default=None)
    p.add_argument('--max_frames_per_track', type=int, default=30,
                   help='Subsample frames per track evenly (default 30)')
    p.add_argument('--no_cache', action='store_true')

    # Head architecture
    p.add_argument('--proj_dim', type=int, default=128)

    # Training
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--samples_per_track', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()