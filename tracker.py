"""
tracker.py — Fully-GPU Tracker with Learnable Association Head + Vector Search
==================================================================================

Phase 1.0e: v6 → v7 改動 (對應教授全 GPU + learnable embedding alignment 要求)

主要改動:
  - 拋棄 scipy.linear_sum_assignment (Hungarian, CPU-only)
  - 改用 iterative greedy assignment (fully GPU, deterministic)
  - 加 AssociationHead (learnable MLP) 投影 query embedding 到 metric space
  - 保留 ByteTrack 兩階段 matching (high-conf + low-conf)
  - 保留 cos_sim + IoU combined cost
  - 保留 EMA track embedding (0.9 momentum)
  - 保留 Kalman filter (8-dim, 跟 v5/v6 一致)

跟 v6 介面比較:
  Detector wrapper: 完全沿用 (DetectorWithQueryFeat from v6 思路)
                    多加一步: query_emb 過 AssociationHead → projected_emb
  Tracker:           完全重寫 (VectorSearchTracker 取代 ByteTracker)
  ReID feature dim:  256 → 128 (因為 head 把 256 投影到 128)
  Kalman / Track:    沿用 v6, 唯一改動: track.emb 維度 256 → 128

要求:
  - 「全 GPU 運算」  → Hungarian 拿掉, iterative greedy + matmul on GPU
  - 「對齊 embedding 的神經網路」→ AssociationHead 投影到 metric space
  - 「vector searching」→ Inference 用 GPU matmul + argmax 做 nearest neighbor
  - 「shuffle」      → 訓練階段 InfoNCE 用 in-batch shuffled negatives
                       (詳見 train_association_head.py)

訓練資料來源 (Decision D6): SAM 3 跑出的 high-confidence tracks 當 GT
訓練 dataset (Decision Q2): fish 跟 UCF101 各訓一個 head

Usage:
  python tracker.py \
    --input data/Fish/ph2021/ph2021_A/clip_0001.mp4 \
    --rfdetr_ckpt output/rfdetr_fish_dinov2s/checkpoint_best_total.pth \
    --association_head_ckpt output/association_head_fish/association_head.pth \
    --output output/tracking/clip_0001_v7 \
    --save_video --save_stats --draw_trail
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from association_head import AssociationHead


# ==========================================
# 全局設定 (跟 v6 一致)
# ==========================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 378
PATCH_SIZE = 14
GRID_SIZE  = INPUT_SIZE // PATCH_SIZE   # 27

# v7: ReID dim 改成 128 (AssociationHead 投影後)
REID_DIM_RAW = 256
REID_DIM     = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


# ==========================================
# Input loaders (跟 v6 完全一致)
# ==========================================
class FrameDirLoader:
    def __init__(self, dir_path: str):
        p = Path(dir_path)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        if not files:
            raise FileNotFoundError(f"{dir_path} 沒有圖檔")
        self.files = files
        first = cv2.imread(str(files[0]))
        self.H0, self.W0 = first.shape[:2]
        self.fps = 30

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for f in self.files:
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img, f.stem


class VideoLoader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"無法開啟 {video_path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.W0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H0 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return self.n

    def __iter__(self):
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, f"{idx:06d}"
            idx += 1
        self.cap.release()


def build_loader(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        return FrameDirLoader(input_path)
    elif p.suffix.lower() in VIDEO_EXTS:
        return VideoLoader(input_path)
    else:
        raise ValueError(f"不支援的 input: {input_path}")


_preprocess_tf = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


# ==========================================
# Detector + AssociationHead (v7 改動)
# ==========================================
class DetectorWithAssocHead:
    """
    v7: RF-DETR + AssociationHead, one forward, ReID feat 從 256 投影到 128.

    Pipeline per frame:
      1. preprocess img_rgb → (1, 3, 378, 378)
      2. LWDETR forward → {pred_logits, pred_boxes, query_features}
      3. PostProcess → list of {scores, labels, boxes, query_features (300, 256)}
      4. 過濾 conf_thresh
      5. AssociationHead 投影 → projected embeddings (N, 128, L2-normalized)
    """
    def __init__(self, rfdetr_model, association_head, conf_thresh: float = 0.3):
        self.lwdetr      = rfdetr_model.model.model
        self.postprocess = rfdetr_model.model.postprocess
        self.head        = association_head
        self.conf_thresh = conf_thresh

        self.lwdetr.to(DEVICE).eval()
        self.head.to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, img_rgb, conf_thresh=None):
        """
        Returns:
            dets: (N, 5) torch.Tensor [x1,y1,x2,y2,conf] in ORIGINAL coords, on GPU
            projs: (N, 128) torch.Tensor projected embeddings, L2-normalized, on GPU
        """
        thresh = conf_thresh if conf_thresh is not None else self.conf_thresh
        H0, W0 = img_rgb.shape[:2]

        x = _preprocess_tf(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        raw = self.lwdetr(x)
        target_sizes = torch.tensor([[H0, W0]], device=DEVICE)
        results = self.postprocess(raw, target_sizes=target_sizes)
        r = results[0]

        scores = r["scores"]                  # (300,)
        boxes  = r["boxes"]                   # (300, 4)
        qfeats = r["query_features"]          # (300, 256)

        keep = scores >= thresh
        if keep.sum() == 0:
            return (torch.zeros((0, 5), device=DEVICE),
                    torch.zeros((0, REID_DIM), device=DEVICE))

        scores_k = scores[keep]               # (N,)
        boxes_k  = boxes[keep]                # (N, 4)
        qfeats_k = qfeats[keep]               # (N, 256)

        # v7: 過 AssociationHead 投影
        projs_k = self.head(qfeats_k)         # (N, 128), L2-normalized

        dets = torch.cat([boxes_k, scores_k.unsqueeze(1)], dim=1)
        return dets, projs_k


# ==========================================
# Kalman 8-dim (跟 v5/v6 完全一致, 不動)
# ==========================================
class KalmanBoxFilter:
    def __init__(self):
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = 1.0
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.Q[4:, 4:] *= 0.001
        self.R = np.eye(4, dtype=np.float32) * 1.0
        self.x = None
        self.P = None

    def init(self, bbox_xywh):
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = bbox_xywh
        self.P = np.eye(8, dtype=np.float32) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, bbox_xywh):
        z = bbox_xywh.astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P


# ==========================================
# Track (v7: emb 維度 128 instead of 256, 邏輯不變)
# ==========================================
class Track:
    next_id = 1

    def __init__(self, bbox_xywh, conf, emb, frame_idx):
        """
        emb: (128,) torch.Tensor on GPU, already L2-normalized
        """
        self.id = Track.next_id
        Track.next_id += 1
        self.kf = KalmanBoxFilter()
        self.kf.init(bbox_xywh)
        self.emb = emb.clone()           # (128,) GPU tensor
        self.conf = float(conf)
        self.state = 'tentative'
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.last_det_xyxy = xywh_to_xyxy(bbox_xywh).astype(np.float32)
        self.trajectory = [(frame_idx, float(bbox_xywh[0]), float(bbox_xywh[1]))]

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return pred

    def update_full(self, bbox_xywh, conf, emb, frame_idx, emb_momentum=0.9):
        """
        EMA emb update (Decision D7): 0.9 × old + 0.1 × new, re-normalize.
        """
        self.kf.update(bbox_xywh)
        self.conf = float(conf)
        new_emb = emb_momentum * self.emb + (1 - emb_momentum) * emb
        self.emb = torch.nn.functional.normalize(new_emb, p=2, dim=0)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.last_det_xyxy = xywh_to_xyxy(bbox_xywh).astype(np.float32)
        self.trajectory.append((frame_idx, float(bbox_xywh[0]), float(bbox_xywh[1])))

    def update_motion_only(self, bbox_xywh, conf, frame_idx):
        self.kf.update(bbox_xywh)
        self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.last_det_xyxy = xywh_to_xyxy(bbox_xywh).astype(np.float32)
        self.trajectory.append((frame_idx, float(bbox_xywh[0]), float(bbox_xywh[1])))

    @property
    def bbox_xywh(self):
        return self.kf.x[:4].copy()


# ==========================================
# 工具 (跟 v5/v6 一致)
# ==========================================
def xywh_to_xyxy(b):
    out = np.zeros_like(b)
    out[..., 0] = b[..., 0] - b[..., 2] / 2
    out[..., 1] = b[..., 1] - b[..., 3] / 2
    out[..., 2] = b[..., 0] + b[..., 2] / 2
    out[..., 3] = b[..., 1] + b[..., 3] / 2
    return out


def xyxy_to_xywh(b):
    out = np.zeros_like(b)
    out[..., 0] = (b[..., 0] + b[..., 2]) / 2
    out[..., 1] = (b[..., 1] + b[..., 3]) / 2
    out[..., 2] = b[..., 2] - b[..., 0]
    out[..., 3] = b[..., 3] - b[..., 1]
    return out


def iou_matrix_torch(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    v7: GPU IoU matrix (跟 v6 numpy 版邏輯相同, 跑 GPU 上).
    boxes_a: (M, 4) xyxy
    boxes_b: (N, 4) xyxy
    returns: (M, N) IoU
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.shape[0], boxes_b.shape[0], device=boxes_a.device)
    ax1, ay1, ax2, ay2 = boxes_a.unbind(-1)
    bx1, by1, bx2, by2 = boxes_b.unbind(-1)

    inter_x1 = torch.maximum(ax1.unsqueeze(1), bx1.unsqueeze(0))
    inter_y1 = torch.maximum(ay1.unsqueeze(1), by1.unsqueeze(0))
    inter_x2 = torch.minimum(ax2.unsqueeze(1), bx2.unsqueeze(0))
    inter_y2 = torch.minimum(ay2.unsqueeze(1), by2.unsqueeze(0))
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter + 1e-6
    return inter / union


def iterative_greedy_match(
    cost: torch.Tensor,
    cost_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Decision D1: Iterative greedy assignment (GPU-friendly替代 Hungarian).

    從 cost 矩陣中, 從小到大排序所有 (i, j) pair, 依序配對 (一個 i 一個 j 最多用一次).
    cost > cost_thresh 的 pair 不配.

    Args:
        cost: (M, N) torch.Tensor on GPU
        cost_thresh: float, pair with cost > thresh 不配
    Returns:
        matches: list of (row_idx, col_idx)
        unmatched_rows: list of row indices not matched
        unmatched_cols: list of col indices not matched
    """
    M, N = cost.shape
    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))

    # Flatten 跟 sort (GPU 上做, 然後 .cpu() 一次性轉回來)
    flat_cost = cost.flatten()
    sorted_idx = torch.argsort(flat_cost)
    sorted_costs = flat_cost[sorted_idx]

    # 只保留 cost <= thresh 的 candidate (大部分 noise 直接丟)
    valid_mask = sorted_costs <= cost_thresh
    candidate_indices = sorted_idx[valid_mask].cpu().numpy()

    matched_rows = set()
    matched_cols = set()
    matches = []

    for idx in candidate_indices:
        r = idx // N
        c = idx % N
        r, c = int(r), int(c)
        if r in matched_rows or c in matched_cols:
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)
        if len(matched_rows) == M or len(matched_cols) == N:
            break

    unmatched_rows = [r for r in range(M) if r not in matched_rows]
    unmatched_cols = [c for c in range(N) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ==========================================
# VectorSearchTracker (v7 取代 ByteTracker)
# ==========================================
class VectorSearchTracker:
    """
    v7: Fully-GPU tracker with vector search + iterative greedy.
    
    保留 v6 ByteTrack 的所有經驗證邏輯:
      - Two-stage matching (high + low conf, Decision D3)
      - cos_sim + IoU combined cost (Decision D2)
      - Kalman predict + EMA emb update (Decision D7)
      - tentative / confirmed / lost / removed states
    
    取代部分:
      - scipy linear_sum_assignment → iterative_greedy_match (Decision D1)
      - raw 256-d cos_sim → projected 128-d cos_sim (用 AssociationHead 投影過)
    """
    def __init__(self, conf_high=0.5, conf_low=0.05,
                 match_cost=0.7, match_iou=0.7,
                 reid_weight=0.7, max_lost=60, min_hits=2,
                 tentative_max_lost=5):
        self.conf_high   = conf_high
        self.conf_low    = conf_low
        self.match_cost  = match_cost
        self.match_iou   = match_iou
        self.reid_weight = reid_weight
        self.max_lost    = max_lost
        self.min_hits    = min_hits
        self.tentative_max_lost = tentative_max_lost
        self.tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

    def update(self, dets: torch.Tensor, projs: torch.Tensor, frame_idx: int):
        """
        Args:
            dets:  (N, 5) torch.Tensor on GPU, [x1,y1,x2,y2,conf]
            projs: (N, 128) torch.Tensor on GPU, projected embeddings (L2-normed)
            frame_idx: int
        """
        # 1. 拆 high/low conf
        if dets.shape[0] > 0:
            scores = dets[:, 4]
            high_mask = scores >= self.conf_high
            low_mask  = (scores >= self.conf_low) & (~high_mask)
        else:
            high_mask = torch.zeros(0, dtype=torch.bool, device=DEVICE)
            low_mask  = torch.zeros(0, dtype=torch.bool, device=DEVICE)

        dets_high  = dets[high_mask]
        projs_high = projs[high_mask] if projs.shape[0] > 0 else projs
        dets_low   = dets[low_mask]

        # 2. predict all
        for tr in self.tracks:
            tr.predict()
        active = self.tracks[:]

        # 3. Stage 1: high-conf vs all tracks (combined cost, iterative greedy, GPU)
        if len(active) > 0 and dets_high.shape[0] > 0:
            tr_xywh = np.array([tr.bbox_xywh for tr in active])
            tr_xyxy_np = xywh_to_xyxy(tr_xywh)
            tr_xyxy = torch.from_numpy(tr_xyxy_np).to(DEVICE)

            iou = iou_matrix_torch(tr_xyxy, dets_high[:, :4])   # (M, N) on GPU

            tr_embs = torch.stack([tr.emb for tr in active])     # (M, 128) on GPU
            cos_sim = tr_embs @ projs_high.T                     # (M, N) on GPU

            cost = (self.reid_weight * (1 - iou) +
                    (1 - self.reid_weight) * (1 - cos_sim))      # (M, N) on GPU

            matches, ut, ud = iterative_greedy_match(cost, self.match_cost)
        else:
            matches, ut = [], list(range(len(active)))
            ud = list(range(dets_high.shape[0]))

        for tr_i, d_i in matches:
            tr = active[tr_i]
            box_np = dets_high[d_i, :4].cpu().numpy()
            bb = xyxy_to_xywh(box_np)
            conf = float(dets_high[d_i, 4].item())
            tr.update_full(bb, conf, projs_high[d_i], frame_idx)
            if tr.state in ('tentative', 'lost'):
                if tr.state == 'lost' or tr.hits >= self.min_hits:
                    tr.state = 'confirmed'

        # 4. Stage 2: unmatched tracks vs low-conf det (IoU only)
        um_tracks = [active[i] for i in ut]
        if len(um_tracks) > 0 and dets_low.shape[0] > 0:
            tr_xywh = np.array([tr.bbox_xywh for tr in um_tracks])
            tr_xyxy_np = xywh_to_xyxy(tr_xywh)
            tr_xyxy = torch.from_numpy(tr_xyxy_np).to(DEVICE)
            iou = iou_matrix_torch(tr_xyxy, dets_low[:, :4])
            cost = 1 - iou
            matches2, ut2, _ = iterative_greedy_match(cost, 1 - self.match_iou)
        else:
            matches2 = []

        for tr_i, d_i in matches2:
            tr = um_tracks[tr_i]
            box_np = dets_low[d_i, :4].cpu().numpy()
            bb = xyxy_to_xywh(box_np)
            conf = float(dets_low[d_i, 4].item())
            tr.update_motion_only(bb, conf, frame_idx)
            if tr.state in ('tentative', 'lost'):
                if tr.state == 'lost' or tr.hits >= self.min_hits:
                    tr.state = 'confirmed'

        # 5. 處理沒 match 的 track (lost / removed)
        new_tracks = []
        for tr in self.tracks:
            if tr.time_since_update == 0:
                new_tracks.append(tr)
            elif tr.state == 'tentative':
                if tr.time_since_update > self.tentative_max_lost:
                    continue
                new_tracks.append(tr)
            elif tr.time_since_update > self.max_lost:
                self.removed_tracks.append(tr)
            else:
                tr.state = 'lost'
                new_tracks.append(tr)
        self.tracks = new_tracks

        # 6. 開新 track (沒配上的 high-conf det)
        for d_i in ud:
            box_np = dets_high[d_i, :4].cpu().numpy()
            bb = xyxy_to_xywh(box_np)
            conf = float(dets_high[d_i, 4].item())
            tr = Track(bb, conf, projs_high[d_i], frame_idx)
            self.tracks.append(tr)

        # 7. 回傳 confirmed + 當前 frame 有 update 的 track
        out = []
        for tr in self.tracks:
            if tr.state == 'confirmed' and tr.time_since_update == 0:
                out.append((tr.id, tr.last_det_xyxy.copy(), tr.conf, tr.trajectory))
        return out


# ==========================================
# 視覺化 / CSV / Stats (跟 v6 一致)
# ==========================================
def get_color(track_id):
    np.random.seed(track_id * 37 + 17)
    h = np.random.randint(0, 180)
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(img_bgr, tracks, draw_trail=False, trail_length=0):
    out = img_bgr.copy()
    for tid, xyxy, conf, trajectory in tracks:
        color = get_color(tid)
        if draw_trail and len(trajectory) >= 2:
            pts = trajectory if trail_length <= 0 else trajectory[-trail_length:]
            poly = np.array([[int(x), int(y)] for _, x, y in pts], dtype=np.int32)
            if len(poly) >= 2:
                cv2.polylines(out, [poly], isClosed=False, color=color,
                              thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(out, tuple(poly[-1]), 4, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {conf:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"{len(tracks)} active", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


CSV_HEADER = ['obj_id', 'center_x', 'center_y',
              'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
              'mask_area', 'raw_obj_id', 'frame_idx']


def tracks_to_rows(tracks, frame_idx):
    rows = []
    for tid, xyxy, _, _ in tracks:
        x1, y1, x2, y2 = xyxy
        w = x2 - x1; h = y2 - y1
        cx = x1 + w / 2; cy = y1 + h / 2
        rows.append([
            tid,
            f"{cx:.2f}", f"{cy:.2f}",
            f"{x1:.2f}", f"{y1:.2f}",
            f"{w:.2f}",  f"{h:.2f}",
            f"{w*h:.2f}",
            tid,
            frame_idx,
        ])
    return rows


def compute_stats(tracker, args, n_frames, total_dets, active_per_frame):
    all_tracks = tracker.removed_tracks + [
        t for t in tracker.tracks if t.state == 'confirmed']
    lengths = [t.last_frame - t.start_frame + 1 for t in all_tracks]

    def bucket(lens):
        b = {'1-10': 0, '11-30': 0, '31-100': 0, '101+': 0}
        for L in lens:
            if L <= 10:    b['1-10']   += 1
            elif L <= 30:  b['11-30']  += 1
            elif L <= 100: b['31-100'] += 1
            else:          b['101+']   += 1
        return b

    return {
        'input': args.input,
        'rfdetr_ckpt': args.rfdetr_ckpt,
        'association_head_ckpt': args.association_head_ckpt,
        'reid_source': 'rfdetr_decoder_query_via_association_head',
        'reid_dim': REID_DIM,
        'tracker_type': 'VectorSearchTracker (iterative greedy, fully GPU)',
        'total_frames': n_frames,
        'total_dets':   int(total_dets),
        'total_tracks_confirmed': len(all_tracks),
        'avg_dets_per_frame':   total_dets / max(n_frames, 1),
        'avg_active_tracks_per_frame': float(np.mean(active_per_frame)) if active_per_frame else 0,
        'track_length_stats': {
            'mean':   float(np.mean(lengths))   if lengths else 0,
            'median': float(np.median(lengths)) if lengths else 0,
            'max':    int(np.max(lengths))      if lengths else 0,
            'min':    int(np.min(lengths))      if lengths else 0,
        },
        'tracks_by_length_bucket': bucket(lengths),
        'params': {
            'conf_high':   args.conf_high,
            'conf_low':    args.conf_low,
            'match_cost':  args.match_cost,
            'match_iou':   args.match_iou,
            'reid_weight': args.reid_weight,
            'max_lost':    args.max_lost,
            'min_hits':    args.min_hits,
            'tentative_max_lost': args.tentative_max_lost,
        }
    }


# ==========================================
# Main
# ==========================================
def run(args):
    print(f"[v7] Fully-GPU Tracker with Learnable Association Head + Vector Search")
    print(f"[v7] INPUT_SIZE: {INPUT_SIZE} (grid {GRID_SIZE}×{GRID_SIZE})")
    print(f"[v7] ReID: projected {REID_DIM}-d (via AssociationHead)")
    print(f"[v7] Matching: iterative greedy (no Hungarian, fully GPU)")

    # ===== Build RF-DETR =====
    print(f"\nLoading RF-DETR from {args.rfdetr_ckpt} ...")
    from rfdetr import RFDETRNano
    rfdetr_model = RFDETRNano(
        num_classes=1,
        resolution=378,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=27,
        encoder='dinov2_small',
        pretrain_weights=args.rfdetr_ckpt,
        freeze_encoder=True,
    )

    # ===== Load AssociationHead =====
    print(f"Loading AssociationHead from {args.association_head_ckpt} ...")
    head = AssociationHead(in_dim=256, hidden_dim=256, proj_dim=REID_DIM)
    head.load_state_dict(torch.load(args.association_head_ckpt, map_location=DEVICE))

    detector = DetectorWithAssocHead(rfdetr_model, head, conf_thresh=args.det_conf)

    tracker = VectorSearchTracker(
        conf_high=args.conf_high,
        conf_low=args.conf_low,
        match_cost=args.match_cost,
        match_iou=args.match_iou,
        reid_weight=args.reid_weight,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
        tentative_max_lost=args.tentative_max_lost,
    )
    Track.next_id = 1

    print(f"\nLoading input from {args.input} ...")
    loader = build_loader(args.input)
    print(f"  {len(loader)} frames, {loader.W0}x{loader.H0}, fps={loader.fps:.1f}")

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, 'tracks.csv')
    csv_f = open(csv_path, 'w', newline='')
    csv_w = csv.writer(csv_f)
    csv_w.writerow(CSV_HEADER)

    video_writer = None
    if args.save_video:
        vid_path = os.path.join(args.output, 'vis.mp4')
        out_fps = args.fps if args.fps > 0 else loader.fps
        for codec in ['avc1', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(vid_path, fourcc, out_fps,
                                           (loader.W0, loader.H0))
            if video_writer.isOpened():
                print(f"  Will save video to {vid_path} (codec={codec})")
                break
            else:
                video_writer.release()
                video_writer = None

    print("\nRunning v7 fully-GPU tracking...")
    total_dets = 0
    active_per_frame = []

    for frame_idx, (img_rgb, _) in enumerate(tqdm(loader, total=len(loader))):
        dets, projs = detector(img_rgb)
        total_dets += dets.shape[0]
        active = tracker.update(dets, projs, frame_idx)
        active_per_frame.append(len(active))

        for row in tracks_to_rows(active, frame_idx):
            csv_w.writerow(row)

        if video_writer is not None:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            vis = draw_tracks(img_bgr, active,
                              draw_trail=args.draw_trail,
                              trail_length=args.trail_length)
            video_writer.write(vis)

    csv_f.close()
    if video_writer is not None:
        video_writer.release()

    n_frames = len(loader)
    n_confirmed = (len(tracker.removed_tracks) +
                   len([t for t in tracker.tracks if t.state == 'confirmed']))

    print("\nDone.")
    print(f"  CSV: {csv_path}")
    print(f"  Total detections: {total_dets} (avg {total_dets/max(n_frames,1):.1f}/frame)")
    print(f"  Total confirmed tracks: {n_confirmed}")
    print(f"  Avg active tracks/frame: {np.mean(active_per_frame):.1f}")

    if args.save_stats:
        stats = compute_stats(tracker, args, n_frames, total_dets, active_per_frame)
        stats_path = os.path.join(args.output, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  Stats: {stats_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--rfdetr_ckpt', required=True)
    p.add_argument('--association_head_ckpt', required=True,
                   help='AssociationHead ckpt (head_fish.pth or head_ucf101.pth)')
    p.add_argument('--det_conf', type=float, default=0.3)
    p.add_argument('--output', required=True)
    p.add_argument('--conf_high',   type=float, default=0.5)
    p.add_argument('--conf_low',    type=float, default=0.05)
    p.add_argument('--match_cost',  type=float, default=0.7)
    p.add_argument('--match_iou',   type=float, default=0.7)
    p.add_argument('--reid_weight', type=float, default=0.7)
    p.add_argument('--max_lost',    type=int, default=60)
    p.add_argument('--min_hits',    type=int, default=2)
    p.add_argument('--tentative_max_lost', type=int, default=5)
    p.add_argument('--save_video', action='store_true')
    p.add_argument('--save_stats', action='store_true')
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--draw_trail', action='store_true')
    p.add_argument('--trail_length', type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()