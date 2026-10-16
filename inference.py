"""
inference.py
============
魚隻活力完整推論管線: 影片 → tracking → 904 維特徵 → ViT → 活力預測。

本腳本將整個 pipeline 串成一條龍，分為五個 stage:

  [Stage 1] Tracking
              RF-DETR + tracker (VectorSearchTracker + AssociationHead)
              對影片中的每條魚產生 (obj_id, frame_idx, bbox) 軌跡。

  [Stage 2] 特徵抽取
              對每條 track 滑窗 (預設 WINDOW_SIZE = 30, stride = 1)，
              每個 window 呼叫 RF-DETR 共享骨幹取得:
                [global_CLS(384) | object_query(256) | qdiff(256) |
                 bbox(4) | bbox_diff(4)] = 904 維
              qdiff 與 bbox 以訓練時統計的 mean / std 做 z-score 正規化。

  [Stage 3] ViT 預測
              把 (T=30, 904) 特徵餵給訓練好的 ViT 分類器，輸出
              3 類 logits (High / Medium / Low)。

  [Stage 4] 聚合
              per_window: 每 window 輸出一個標籤 (影片上以重疊 window 多數決);
              per_track:  整條軌跡用多數決成一個標籤。

  [Stage 5] 輸出
              predictions_{stem}_{mode}.csv : 預測結果 (欄位隨 mode 變化)
              {stem}_vitality.mp4           : 每個 bbox 標上活力分類
                                              (綠 = High / 黃 = Medium / 紅 = Low)
              {stem}_minority.mp4           : 選用 (--render_minority)，只渲染
                                              最少數族群的魚隻，突顯潛在異常個體


顯示標籤:
  模型輸出 0 → High
  模型輸出 1 → Medium
  模型輸出 2 → Low

LABEL_MAP 說明:
  模型推論時直接輸出 0/1/2，不需要 LABEL_MAP。
  LABEL_MAP 僅作為對照資訊保留，紀錄訓練 dataloader 如何將資料夾 cluster_X
  映射到訓練標籤。詳細對應關係見程式碼內 LABEL_MAP 定義處的註解。


用法 (CLI):
  python inference.py \
    --video data/ph2021_A/clip_0001.mp4 \
            data/ph2021_A/clip_0002.mp4 \
    --rfdetr_ckpt weights/detector.pth \
    --association_head_ckpt weights/association_head.pth \
    --vit_ckpt weights/vitality_classifier.pth \
    --stats_dir stats \
    --output_dir output/inference \
    --agg_mode per_track
    --render_minority

多個影片:
  python inference.py --video a.mp4 b.mp4 c.mp4 ...

只輸出 CSV、不渲染影片:
  python inference.py --video x.mp4 ... --no_video


依賴:
  - tracker.py 與 association_head.py 需在 PYTHONPATH 內
  - rfdetr 已安裝 (含 RFDETRNano)
  - RF-DETR 內部的 DinoV2 backbone 需暴露 last_cls (修改已內建於 rfdetr/，
    詳見 rfdetr/MODIFICATIONS.md)
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T

# ==========================================
# 外部依賴 import
# ==========================================
try:
    from tracker import (
        DetectorWithAssocHead,
        VectorSearchTracker,
        Track,
        build_loader,
        tracks_to_rows,
        CSV_HEADER,
        DEVICE,
    )
    from association_head import AssociationHead
except ImportError as e:
    print(f"Import failed: {e}", file=sys.stderr)
    print("  tracker.py 跟 association_head.py 要在 PYTHONPATH 內", file=sys.stderr)
    sys.exit(1)

# ViT 分類器定義與訓練腳本共用
from train_vitality_classifier import ViTActionClassifier


# ==========================================
# 1. 標籤 & 顯示設定
# ==========================================
NUM_CLASSES = 3

# 模型輸出 index → 顯示名稱
CLASS_NAMES      = ['High',          'Medium',        'Low']            # 短標籤 (bbox / CSV)
CLASS_NAMES_FULL = ['High vitality', 'Medium vitality','Low vitality']  # 長標籤 (print / log)

# LABEL_MAP: 將 GMM 分群編號 (gmm_label) 映射為模型訓練標籤 (0=High, 1=Medium, 2=Low)。
#
# 注意: GMM 為無監督分群演算法，cluster 編號本身無語意、且每次執行可能不同
# (受 random seed、場域資料分布影響)。此 mapping 需依該場域實際 GMM 結果調整。
# 推論階段不需要查表 (模型直接輸出訓練標籤)，此處保留供對照用。
#
# 以下預設值對應 ph2021 場域的訓練設定；部署到其他場域時需重新評估:
#   cluster 2 對應 High   (高活力)
#   cluster 0 對應 Medium (中活力)
#   cluster 1 對應 Low    (低活力)
LABEL_MAP = {0: 1, 1: 2, 2: 0}

# 影片顯示顏色 (BGR for OpenCV): 綠=High / 黃=Medium / 紅=Low
CLASS_COLORS_BGR = {
    0: (  0, 200,   0),   # High
    1: (  0, 220, 220),   # Medium
    2: (  0,   0, 220),   # Low
}


# ==========================================
# 2. Feature layout (需與 build_pseudo_labeled_dataset.py 一致)
# ==========================================
GLOBAL_DIM      = 384
OBJECT_DIM      = 256
QUERY_DIFF_DIM  = 256
BBOX_DIM        = 4
BBOX_DIFF_DIM   = 4
FEAT_DIM        = GLOBAL_DIM + OBJECT_DIM + QUERY_DIFF_DIM + BBOX_DIM + BBOX_DIFF_DIM  # 904

GLOBAL_S,  GLOBAL_E  = 0,   384
OBJECT_S,  OBJECT_E  = 384, 640
QDIFF_S,   QDIFF_E   = 640, 896
BBOX_S,    BBOX_E    = 896, 900
BBOXD_S,   BBOXD_E   = 900, 904


# ==========================================
# 3. RF-DETR 相關參數 (預設值; CLI 可覆蓋)
# ==========================================
RFDETR_INPUT   = 378
RFDETR_PATCH   = 14
RFDETR_CONF    = 0.1     # 特徵抽取用的 confidence threshold (非 tracker 的 det_conf)
IOU_MATCH_THR  = 0.3     # feature extraction IoU match threshold

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ==========================================
# 4. Frame / bbox 輔助
# ==========================================
def read_mp4_frames(mp4_path, frame_ids_needed):
    """一次抽出指定 frame_id 的 BGR frames 到 dict."""
    if not frame_ids_needed:
        return {}
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {}
    needed = set(int(x) for x in frame_ids_needed)
    max_fid = max(needed)
    out = {}
    idx = 0
    while idx <= max_fid:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            out[idx] = frame
        idx += 1
        if len(out) == len(needed):
            break
    cap.release()
    return out


def frame_to_pil(frame_bgr):
    if frame_bgr is None:
        frame_bgr = np.zeros((RFDETR_INPUT, RFDETR_INPUT, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def compute_iou_batch(box_a, boxes_b):
    xx1 = np.maximum(box_a[0], boxes_b[:, 0])
    yy1 = np.maximum(box_a[1], boxes_b[:, 1])
    xx2 = np.minimum(box_a[2], boxes_b[:, 2])
    yy2 = np.minimum(box_a[3], boxes_b[:, 3])
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


# ==========================================
# 5. STAGE 1: Tracking
# ==========================================
def run_tracking(detector, video_path, tk_args):
    """
    對單一 mp4 跑 tracker, 回傳:
      rows:      list of lists (跟 CSV_HEADER 對齊)
      n_frames:  影片總 frame 數
      fps:       原始 FPS (video render 用)
      W, H:      原始畫面大小
    """
    Track.next_id = 1
    tracker = VectorSearchTracker(
        conf_high=tk_args['conf_high'],
        conf_low=tk_args['conf_low'],
        match_cost=tk_args['match_cost'],
        match_iou=tk_args['match_iou'],
        reid_weight=tk_args['reid_weight'],
        max_lost=tk_args['max_lost'],
        min_hits=tk_args['min_hits'],
        tentative_max_lost=tk_args['tentative_max_lost'],
    )

    loader = build_loader(str(video_path))
    fps = getattr(loader, 'fps', 30) or 30
    W0 = int(loader.W0)
    H0 = int(loader.H0)

    all_rows = []
    n_frames = 0
    for frame_idx, (img_rgb, _) in enumerate(loader):
        dets, projs = detector(img_rgb)
        active = tracker.update(dets, projs, frame_idx)
        rows_this_frame = tracks_to_rows(active, frame_idx)
        all_rows.extend(rows_this_frame)
        n_frames = frame_idx + 1

    return all_rows, n_frames, fps, W0, H0


# ==========================================
# 6. STAGE 2: Feature extractor
# ==========================================
class FeatureExtractor:
    """
    封裝 RF-DETR 的 unified forward: 一次對一批 frames 同時取得
      - 全幀的 CLS token (來自 DinoV2 backbone，384 維)
      - 偵測到的 bounding boxes 與對應的 query features (256 維)

    注意: 需要 RF-DETR 內部的 DinoV2 backbone 暴露出 last_cls，
    此修改已內建於 rfdetr/，詳見 rfdetr/MODIFICATIONS.md。
    """

    def __init__(self, rfdetr_model, device):
        self.device = device
        self.rfdetr_model = rfdetr_model
        self.lwdetr = rfdetr_model.model.model
        self.postprocess = rfdetr_model.model.postprocess
        self.lwdetr.to(device).eval()

        # DinoV2 backbone (for full-frame CLS)
        self._dino_backbone = self.lwdetr.backbone[0].encoder
        if not hasattr(self._dino_backbone, "last_cls"):
            with torch.no_grad():
                _dummy = torch.zeros(1, 3, RFDETR_INPUT, RFDETR_INPUT, device=device)
                _ = self.lwdetr(_dummy)
            if not hasattr(self._dino_backbone, "last_cls") or \
               self._dino_backbone.last_cls is None:
                raise RuntimeError(
                    "RF-DETR 內部 DinoV2 backbone 沒有 last_cls attribute。"
                    "此修改應已內建於 rfdetr/models/backbone/dinov2.py，"
                    "若此錯誤出現，表示 rfdetr/ 可能被覆蓋為原版，"
                    "詳見 rfdetr/MODIFICATIONS.md。"
                )

        self.transform = T.Compose([
            T.Resize((RFDETR_INPUT, RFDETR_INPUT)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def forward_batch(self, pil_frames):
        """
        一次 forward N 個 frame, 回傳:
          all_boxes: list of (K, 4)   xyxy
          all_scores: list of (K,)
          all_qfeats: list of (K, 256)
          global_cls: (N, 384)
        """
        if not pil_frames:
            return [], [], [], np.zeros((0, GLOBAL_DIM), dtype=np.float32)

        sizes = [(img.size[1], img.size[0]) for img in pil_frames]
        tensors = torch.stack([self.transform(img) for img in pil_frames]).to(self.device)

        raw = self.lwdetr(tensors)

        cls_tensor = getattr(self._dino_backbone, "last_cls", None)
        if cls_tensor is None:
            global_cls = np.zeros((len(pil_frames), GLOBAL_DIM), dtype=np.float32)
        else:
            global_cls = cls_tensor.cpu().numpy().astype(np.float32)

        target_sizes = torch.tensor(sizes, device=self.device)
        results = self.postprocess(raw, target_sizes=target_sizes)

        all_boxes, all_scores, all_qfeats = [], [], []
        for r in results:
            scores = r["scores"]
            boxes  = r["boxes"]
            qfeats = r["query_features"]
            keep = scores >= RFDETR_CONF
            all_boxes.append(boxes[keep].cpu().numpy().astype(np.float32))
            all_scores.append(scores[keep].cpu().numpy().astype(np.float32))
            all_qfeats.append(qfeats[keep].cpu().numpy().astype(np.float32))

        return all_boxes, all_scores, all_qfeats, global_cls


def extract_video_features(feat_extractor, mp4_path, all_frame_ids,
                           track_bboxes_per_frame, batch_size=16):
    """
    對 all_frame_ids 中所有 frames 做 RF-DETR forward, 回傳:
      fullframe_cls_dict: {fid → (384,)}
      query_emb_dict:     {(fid, track_id) → (256,) or None}

    query 用 IoU match: 找 detection 跟 track bbox 最接近的那個,
    IoU >= IOU_MATCH_THR 才算 match, 否則存 None.
    """
    fullframe_cls_dict = {}
    query_emb_dict = {}
    if not all_frame_ids:
        return fullframe_cls_dict, query_emb_dict

    bgr_frames = read_mp4_frames(mp4_path, all_frame_ids)

    pil_frames_dict = {}
    for fid in all_frame_ids:
        bgr = bgr_frames.get(fid)
        if bgr is None:
            bgr = np.zeros((RFDETR_INPUT, RFDETR_INPUT, 3), dtype=np.uint8)
        pil_frames_dict[fid] = frame_to_pil(bgr)

    for batch_start in range(0, len(all_frame_ids), batch_size):
        batch_fids = all_frame_ids[batch_start:batch_start + batch_size]
        batch_pils = [pil_frames_dict[fid] for fid in batch_fids]
        all_boxes_b, all_scores_b, all_qfeats_b, global_cls_b = \
            feat_extractor.forward_batch(batch_pils)

        for idx, fid in enumerate(batch_fids):
            fullframe_cls_dict[fid] = global_cls_b[idx]

        for i, fid in enumerate(batch_fids):
            det_boxes  = all_boxes_b[i]
            det_qfeats = all_qfeats_b[i]

            if fid not in track_bboxes_per_frame:
                continue
            track_bbox_list = track_bboxes_per_frame[fid]

            if len(det_boxes) == 0:
                for track_id, _ in track_bbox_list:
                    query_emb_dict[(fid, track_id)] = None
                continue

            for track_id, track_xywh in track_bbox_list:
                track_xyxy = xywh_to_xyxy(track_xywh)
                ious = compute_iou_batch(track_xyxy, det_boxes)
                best_idx = int(np.argmax(ious))
                best_iou = ious[best_idx]
                if best_iou >= IOU_MATCH_THR:
                    query_emb_dict[(fid, track_id)] = det_qfeats[best_idx]
                else:
                    query_emb_dict[(fid, track_id)] = None

    return fullframe_cls_dict, query_emb_dict


def build_raw_904(global_emb_seq, object_emb_seq, bbox_seq):
    """組出 raw 904 維 feature matrix (qdiff 與 bboxd 尚未 normalize)。"""
    T_eff = global_emb_seq.shape[0]
    qdiff_raw = np.zeros((T_eff, QUERY_DIFF_DIM), dtype=np.float32)
    if T_eff >= 2:
        qdiff_raw[1:] = object_emb_seq[1:] - object_emb_seq[:-1]

    bboxd_raw = np.zeros((T_eff, BBOX_DIM), dtype=np.float32)
    if T_eff >= 2:
        bboxd_raw[1:] = bbox_seq[1:] - bbox_seq[:-1]

    return np.concatenate(
        [global_emb_seq, object_emb_seq, qdiff_raw, bbox_seq, bboxd_raw],
        axis=1
    ).astype(np.float32)


def normalize_904(raw_904, qd_mean, qd_std, bb_mean, bb_std):
    """
    對 raw 904 做 z-score 正規化:
      qdiff_norm = (qdiff_raw - qd_mean) / qd_std
      bbox_norm  = (bbox_raw  - bb_mean) / bb_std
      bbox_diff  = diff(bbox_norm),  第 0 幀補 0
    global_emb / object_emb 不做 normalize；qdiff 的第 0 幀保留原始值。
    """
    qd_std_safe = np.where(qd_std > 1e-6, qd_std, 1.0)
    bb_std_safe = np.where(bb_std > 1e-6, bb_std, 1.0)

    qdiff_raw   = raw_904[:, QDIFF_S:QDIFF_E]
    qdiff_norm  = (qdiff_raw - qd_mean) / qd_std_safe

    bbox_raw    = raw_904[:, BBOX_S:BBOX_E]
    bbox_norm   = (bbox_raw - bb_mean) / bb_std_safe

    bbox_diff   = np.zeros_like(bbox_norm)
    bbox_diff[1:] = bbox_norm[1:] - bbox_norm[:-1]

    out = raw_904.copy()
    out[:, QDIFF_S:QDIFF_E] = qdiff_norm
    out[:, BBOX_S:BBOX_E]   = bbox_norm
    out[:, BBOXD_S:BBOXD_E] = bbox_diff
    return out


def build_windows_and_features(track_rows, feat_extractor, mp4_path,
                                qd_mean, qd_std, bb_mean, bb_std,
                                window_size, min_track_len, stride,
                                batch_size=16):
    """
    從 tracking rows 出發, 對每條 track 滑窗, 抽 RF-DETR 特徵, 組成 (T, 904) normalized.

    Returns: list of dicts, 每個 dict:
      {
        'obj_id':      int,
        'window_idx':  int,
        'frame_start': int (window 第一個 detected frame_idx),
        'frame_end':   int (window 最後一個 detected frame_idx),
        'frame_ids':   [int] (長度 = window_size, 是 track 中連續 detected 的 frame_id),
        'feat_904':    np.ndarray (window_size, 904) normalized,
      }
    """
    # ---- 1. Rows → DataFrame → track_data ----
    df = pd.DataFrame(track_rows, columns=CSV_HEADER)
    if df.empty:
        return []
    df = df.astype({'obj_id': int, 'frame_idx': int})
    for c in ('center_x', 'center_y', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'):
        df[c] = df[c].astype(float)
    df = df.sort_values(['obj_id', 'frame_idx']).reset_index(drop=True)

    track_data = {}
    for tid, tg in df.groupby('obj_id'):
        tid = int(tid)
        if len(tg) < min_track_len:
            continue
        fids = tg['frame_idx'].tolist()
        centers = {int(r.frame_idx): (float(r.center_x), float(r.center_y))
                   for r in tg.itertuples()}
        bboxes = {int(r.frame_idx): [float(r.bbox_x), float(r.bbox_y),
                                      float(r.bbox_w), float(r.bbox_h)]
                  for r in tg.itertuples()}
        track_data[tid] = {'fids': fids, 'centers': centers, 'bboxes': bboxes}

    if not track_data:
        return []

    # ---- 2. 決定要切的 windows ----
    window_jobs = []   # (tid, w_idx, wf)  wf = list of frame_ids in this window
    for tid, td in track_data.items():
        fids = td['fids']
        n = len(fids)
        for w_idx, start in enumerate(range(0, n - window_size + 1, stride)):
            wf = fids[start:start + window_size]
            window_jobs.append((tid, w_idx, wf))

    if not window_jobs:
        return []

    # ---- 3. 該影片要 forward 的所有 frame_ids ----
    all_frame_ids = set()
    for _, _, wf in window_jobs:
        all_frame_ids.update(wf)
    all_frame_ids = sorted(all_frame_ids)

    # 對每個 fid 收集所有 (tid, bbox) — feature extraction 的 IoU match 用
    track_bboxes_per_frame = defaultdict(list)
    for fid in all_frame_ids:
        for tid, td in track_data.items():
            if fid in td['bboxes']:
                track_bboxes_per_frame[fid].append((tid, td['bboxes'][fid]))

    # ---- 4. RF-DETR forward (拿 global CLS + query embeddings) ----
    print(f"   → RF-DETR forward on {len(all_frame_ids)} frames "
          f"(要餵 {len(window_jobs)} windows / {len(track_data)} tracks)")
    fullframe_cls_dict, query_emb_dict = extract_video_features(
        feat_extractor, mp4_path, all_frame_ids, track_bboxes_per_frame,
        batch_size=batch_size
    )

    # ---- 5. 組每個 window 的 raw 904 → normalize ----
    windows = []
    n_bad_shape = 0
    for tid, w_idx, wf in window_jobs:
        global_emb_seq = np.array([
            fullframe_cls_dict.get(fid, np.zeros(GLOBAL_DIM, dtype=np.float32))
            for fid in wf
        ], dtype=np.float32)

        object_emb_seq = np.array([
            query_emb_dict.get((fid, tid))
            if query_emb_dict.get((fid, tid)) is not None
            else np.zeros(OBJECT_DIM, dtype=np.float32)
            for fid in wf
        ], dtype=np.float32)

        bbox_seq = np.array([
            track_data[tid]['bboxes'].get(fid, [0, 0, 0, 0]) for fid in wf
        ], dtype=np.float32)

        raw_904 = build_raw_904(global_emb_seq, object_emb_seq, bbox_seq)
        if raw_904.shape != (window_size, FEAT_DIM):
            n_bad_shape += 1
            continue

        norm_904 = normalize_904(raw_904, qd_mean, qd_std, bb_mean, bb_std)

        windows.append({
            'obj_id':      tid,
            'window_idx':  w_idx,
            'frame_start': int(wf[0]),
            'frame_end':   int(wf[-1]),
            'frame_ids':   [int(f) for f in wf],
            'feat_904':    norm_904.astype(np.float32),
        })

    if n_bad_shape > 0:
        print(f"   dropped {n_bad_shape} windows with bad shape")

    return windows


# ==========================================
# 7. STAGE 3: ViT prediction
# ==========================================
def load_vit_model(ckpt_path, device, vit_name='vit_small_patch16_224',
                    num_layers=4):
    model = ViTActionClassifier(
        input_dim=FEAT_DIM, num_classes=NUM_CLASSES,
        vit_name=vit_name, pretrained=False,
        num_layers=num_layers,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing keys: {missing}")
    if unexpected:
        print(f"   Unexpected keys: {unexpected}")
    model.eval()
    print(f"ViT 載入完成 ({ckpt_path})")
    return model


def predict_windows(windows, vit_model, device, batch_size=64):
    """
    對 windows 做 batch inference, 在原 dict 加上 'pred' (int) 和 'probs' (list[3]).
    """
    if not windows:
        return
    n = len(windows)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = windows[i:i + batch_size]
            feats = np.stack([w['feat_904'] for w in batch])
            x = torch.tensor(feats, dtype=torch.float32).to(device)
            logits = vit_model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            for j, w in enumerate(batch):
                w['pred']  = int(preds[j])
                w['probs'] = [float(p) for p in probs[j]]


# ==========================================
# 8. STAGE 4: Aggregation
# ==========================================
def aggregate_per_track(windows):
    """
    整條軌跡多數決:
      每個 obj_id → {
        'pred': 最多票的類別,
        'n_windows': 該 track 有幾個 window,
        'vote_counts': {0: N0, 1: N1, 2: N2},
        'mean_probs': [p0, p1, p2],  各類別平均機率,
        'frame_first', 'frame_last': track 的 frame 範圍,
      }
    """
    by_tid = defaultdict(list)
    for w in windows:
        by_tid[w['obj_id']].append(w)

    track_preds = {}
    for tid, ws in by_tid.items():
        preds = [w['pred'] for w in ws]
        counts = Counter(preds)
        mean_probs = np.mean(np.stack([w['probs'] for w in ws]), axis=0)

        # 主要以多數決; 平手時用該類別平均機率當 tie-break
        top_vote = max(counts.values())
        top_classes = [c for c, n in counts.items() if n == top_vote]
        if len(top_classes) == 1:
            best = top_classes[0]
        else:
            best = max(top_classes, key=lambda c: mean_probs[c])

        frame_first = min(w['frame_start'] for w in ws)
        frame_last  = max(w['frame_end']   for w in ws)

        track_preds[tid] = {
            'pred':        int(best),
            'n_windows':   len(ws),
            'vote_counts': {c: int(counts.get(c, 0)) for c in range(NUM_CLASSES)},
            'mean_probs':  [float(p) for p in mean_probs],
            'frame_first': int(frame_first),
            'frame_last':  int(frame_last),
        }
    return track_preds


def compute_class_summary(windows, track_preds, agg_mode):
    """
    統計整部影片各活力類別的數量，並找出「最少數族群」(minority class)。

    在水產養殖監控情境下，minority class 常常具有異常警示價值:
    例如整群魚大多為 High 時，少數 Low 的魚可能是潛在健康問題個體。

    Args:
        windows:     list of windows (含 'pred')
        track_preds: aggregate_per_track 的輸出
        agg_mode:    'per_window' 或 'per_track'，決定統計維度

    Returns:
        class_counts:   dict {class_name: count}
        minority_class: str or None (若全為 0 則為 None)
        minority_idx:   int or None (minority 對應的 class index)
    """
    counter = Counter()
    if agg_mode == 'per_track':
        for info in track_preds.values():
            counter[info['pred']] += 1
    else:  # per_window
        for w in windows:
            counter[w['pred']] += 1

    class_counts = {CLASS_NAMES[i]: int(counter.get(i, 0))
                    for i in range(NUM_CLASSES)}
    nonzero = {k: v for k, v in class_counts.items() if v > 0}
    if not nonzero:
        return class_counts, None, None

    minority_class = min(nonzero, key=nonzero.get)
    minority_idx = CLASS_NAMES.index(minority_class)
    return class_counts, minority_class, minority_idx


def build_frame_labels(windows, mode, track_preds=None):
    """
    決定每個 (obj_id, frame_idx) 要在影片上顯示的標籤:
      - per_track:  該 track 所有 frame 都用該 track 的多數決標籤
      - per_window: 由於 stride=1 一個 frame 會被多個 window 覆蓋,
                    對重疊的所有 window 做多數決

    Returns: dict {(obj_id, frame_id) → int label}
    """
    frame_labels = {}

    if mode == 'per_track':
        assert track_preds is not None
        for w in windows:
            lab = track_preds[w['obj_id']]['pred']
            for f in w['frame_ids']:
                frame_labels[(w['obj_id'], f)] = lab
    else:  # per_window
        per_frame_votes = defaultdict(list)
        for w in windows:
            for f in w['frame_ids']:
                per_frame_votes[(w['obj_id'], f)].append(w['pred'])
        for key, preds in per_frame_votes.items():
            frame_labels[key] = Counter(preds).most_common(1)[0][0]

    return frame_labels


# ==========================================
# 9. STAGE 5a: CSV output
# ==========================================
def write_predictions_csv(csv_path, windows, mode, track_preds=None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == 'per_window':
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'obj_id', 'window_idx', 'frame_start', 'frame_end',
                'pred_label', 'pred_name',
                'prob_high', 'prob_medium', 'prob_low',
            ])
            for win in windows:
                w.writerow([
                    win['obj_id'], win['window_idx'],
                    win['frame_start'], win['frame_end'],
                    win['pred'], CLASS_NAMES[win['pred']],
                    f"{win['probs'][0]:.4f}",
                    f"{win['probs'][1]:.4f}",
                    f"{win['probs'][2]:.4f}",
                ])
    else:  # per_track
        assert track_preds is not None
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'obj_id', 'pred_label', 'pred_name',
                'n_windows', 'frame_first', 'frame_last',
                'votes_high', 'votes_medium', 'votes_low',
                'mean_prob_high', 'mean_prob_medium', 'mean_prob_low',
            ])
            for tid in sorted(track_preds.keys()):
                info = track_preds[tid]
                w.writerow([
                    tid, info['pred'], CLASS_NAMES[info['pred']],
                    info['n_windows'], info['frame_first'], info['frame_last'],
                    info['vote_counts'][0], info['vote_counts'][1], info['vote_counts'][2],
                    f"{info['mean_probs'][0]:.4f}",
                    f"{info['mean_probs'][1]:.4f}",
                    f"{info['mean_probs'][2]:.4f}",
                ])


# ==========================================
# 10. STAGE 5b: Video rendering
# ==========================================
def _draw_bbox_with_label(frame_bgr, bbox_xywh, label_text, color_bgr):
    """畫一個 bbox + 上方 label (純 OpenCV)."""
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2)

    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.55, 2)
    # 若上方超出畫面, 改畫在 bbox 內側 (避免文字被裁)
    yb = max(y1 - 4, th + 4)
    cv2.rectangle(frame_bgr, (x1, yb - th - 4), (x1 + tw + 6, yb + baseline),
                  color_bgr, -1)
    cv2.putText(frame_bgr, label_text, (x1 + 3, yb),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _open_writer(out_path, W, H, fps):
    """嘗試 mp4v → avc1 fallback."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ['mp4v', 'avc1']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vw = cv2.VideoWriter(str(out_path), fourcc, float(fps),
                             (int(W), int(H)))
        if vw.isOpened():
            return vw
        vw.release()
    return None


def render_video(video_path, out_video_path, track_rows, frame_labels):
    """
    讀原始影片 → 對每個 frame 依 track_rows 畫 bbox + 活力 label → 存新影片.
    Label 格式: "ID{tid} {High|Medium|Low}"
    Color:     綠=High / 黃=Medium / 紅=Low  (見 CLASS_COLORS_BGR)
    """
    # rows → 依 frame_idx group
    df = pd.DataFrame(track_rows, columns=CSV_HEADER)
    if df.empty:
        print("   沒有 tracking rows, 略過影片渲染")
        return
    df = df.astype({'obj_id': int, 'frame_idx': int})
    for c in ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'):
        df[c] = df[c].astype(float)

    rows_by_frame = defaultdict(list)
    for r in df.itertuples(index=False):
        rows_by_frame[int(r.frame_idx)].append(r)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = _open_writer(out_video_path, W, H, fps)
    if writer is None:
        print(f"無法開啟 VideoWriter for {out_video_path}")
        cap.release()
        return

    frame_idx = 0
    n_written = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        rows_here = rows_by_frame.get(frame_idx, [])
        for r in rows_here:
            tid = int(r.obj_id)
            key = (tid, frame_idx)
            if key not in frame_labels:
                # 該 track 沒有預測 (可能太短沒切到 window), 只畫 bbox 不標活力
                cv2.rectangle(frame_bgr,
                              (int(r.bbox_x), int(r.bbox_y)),
                              (int(r.bbox_x + r.bbox_w), int(r.bbox_y + r.bbox_h)),
                              (150, 150, 150), 1)
                continue
            lab = frame_labels[key]
            color = CLASS_COLORS_BGR[lab]
            label_str = f"ID{tid} {CLASS_NAMES[lab]}"
            _draw_bbox_with_label(
                frame_bgr,
                (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h),
                label_str, color,
            )
        writer.write(frame_bgr)
        n_written += 1
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"   影片輸出: {out_video_path} ({n_written} frames)")


def render_minority_video(video_path, out_video_path,
                          track_rows, windows, track_preds,
                          minority_idx, agg_mode):
    """
    渲染「只包含 minority class 魚隻」的影片。

    僅支援 agg_mode='per_track' (per_window 模式下 minority 定義模糊)。
    若無 minority 魚隻 (全部類別皆為 0 或 minority_idx 為 None)，則不渲染。

    Args:
        video_path:     原始影片路徑
        out_video_path: 輸出路徑
        track_rows:     tracker 輸出的所有 rows
        windows:        所有 windows (含 pred)
        track_preds:    aggregate_per_track 的輸出
        minority_idx:   由 compute_class_summary 得出的 minority class index
        agg_mode:       必須為 'per_track'

    Returns:
        True 若有渲染，False 若略過。
    """
    if minority_idx is None:
        print("   無 minority class (所有類別皆為 0)，略過 minority 影片")
        return False

    if agg_mode != 'per_track':
        print(f"   minority 影片僅支援 agg_mode='per_track' "
              f"(目前為 {agg_mode!r})，略過")
        return False

    # 找出所有 pred 為 minority 的 track id
    target_obj_ids = {oid for oid, info in track_preds.items()
                      if info['pred'] == minority_idx}

    if not target_obj_ids:
        print(f"   無 minority 魚隻 (class={CLASS_NAMES[minority_idx]})，略過")
        return False

    # 過濾出只屬於這些 track 的 rows / windows / preds
    filtered_track_rows  = [r for r in track_rows if r[0] in target_obj_ids]
    filtered_windows     = [w for w in windows if w['obj_id'] in target_obj_ids]
    filtered_track_preds = {oid: info for oid, info in track_preds.items()
                            if oid in target_obj_ids}

    # 用過濾後的資料建 frame_labels，再呼叫既有的 render_video
    frame_labels_min = build_frame_labels(
        filtered_windows,
        mode='per_track',
        track_preds=filtered_track_preds,
    )
    print(f"   Minority class = {CLASS_NAMES[minority_idx]}，"
          f"共 {len(target_obj_ids)} 隻魚")
    render_video(video_path, out_video_path, filtered_track_rows, frame_labels_min)
    return True


# ==========================================
# 11. Stats loader
# ==========================================
def load_stats(stats_dir):
    qd_path = os.path.join(stats_dir, 'qdiff_stats.npy')
    bb_path = os.path.join(stats_dir, 'bbox_stats.npy')
    if not (os.path.exists(qd_path) and os.path.exists(bb_path)):
        raise SystemExit(
            f"找不到正規化 stats:\n"
            f"   {qd_path}\n"
            f"   {bb_path}\n"
            f"   請確認 --stats_dir 指向 build_pseudo_labeled_dataset.py 的輸出目錄。"
        )
    qd = np.load(qd_path, allow_pickle=True).item()
    bb = np.load(bb_path, allow_pickle=True).item()
    print(f"載入 stats:")
    print(f"   qd_mean shape={qd['mean'].shape}, "
          f"qd_std shape={qd['std'].shape}")
    print(f"   bb_mean={bb['mean'].round(2)}, bb_std={bb['std'].round(2)}")
    return qd['mean'], qd['std'], bb['mean'], bb['std']


# ==========================================
# 12. 單一影片完整推論
# ==========================================
def infer_video(video_path, output_dir, args,
                detector, feat_extractor, vit_model,
                qd_mean, qd_std, bb_mean, bb_std, device):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    csv_out   = output_dir / f"predictions_{stem}_{args.agg_mode}.csv"
    video_out = output_dir / f"{stem}_vitality.mp4"

    print(f"\n{'='*72}")
    print(f"處理影片: {video_path}")
    print(f"{'='*72}")

    # ------ [1/4] Tracking ------
    print(f"\n[1/4] Tracking (RF-DETR + tracker)...")
    t0 = time.time()
    tk_args = {
        'conf_high': args.conf_high, 'conf_low': args.conf_low,
        'match_cost': args.match_cost, 'match_iou': args.match_iou,
        'reid_weight': args.reid_weight, 'max_lost': args.max_lost,
        'min_hits': args.min_hits, 'tentative_max_lost': args.tentative_max_lost,
    }
    track_rows, n_frames, fps, W, H = run_tracking(detector, video_path, tk_args)
    n_tracks = len({r[0] for r in track_rows})
    print(f"   {n_frames} frames, {n_tracks} tracks, "
          f"{len(track_rows)} rows  ({time.time()-t0:.1f}s)")

    if not track_rows:
        print("   沒有偵測到魚, 略過此影片")
        return

    # ------ [2/4] Feature extraction ------
    print(f"\n[2/4] 提取 904-d 特徵 (WINDOW_SIZE={args.window_size}, "
          f"MIN_TRACK_LEN={args.min_track_len}, stride={args.window_stride})...")
    t0 = time.time()
    windows = build_windows_and_features(
        track_rows, feat_extractor, video_path,
        qd_mean, qd_std, bb_mean, bb_std,
        window_size=args.window_size,
        min_track_len=args.min_track_len,
        stride=args.window_stride,
        batch_size=args.rfdetr_batch,
    )
    print(f"   {len(windows)} 個 windows  ({time.time()-t0:.1f}s)")

    if not windows:
        print(f"   沒有可預測的 window (track 數量或長度不足, "
              f"MIN_TRACK_LEN={args.min_track_len})")
        return

    # ------ [3/4] ViT prediction ------
    print(f"\n[3/4] ViT 預測 (agg_mode={args.agg_mode})...")
    t0 = time.time()
    predict_windows(windows, vit_model, device, batch_size=args.vit_batch)
    track_preds = aggregate_per_track(windows)  # per_track 一定要算 (per_window 也印出來對照)
    print(f"   {time.time()-t0:.1f}s")

    # Summary
    win_counter = Counter(w['pred'] for w in windows)
    print(f"   Window-level: " + "  ".join(
        f"{CLASS_NAMES[c]:>6}={win_counter.get(c, 0):4d}"
        for c in range(NUM_CLASSES)
    ) + f"  (total={len(windows)})")
    track_counter = Counter(info['pred'] for info in track_preds.values())
    print(f"   Track-level:  " + "  ".join(
        f"{CLASS_NAMES[c]:>6}={track_counter.get(c, 0):4d}"
        for c in range(NUM_CLASSES)
    ) + f"  (total={len(track_preds)})")

    # ------ Minority class 分析 ------
    class_counts, minority_class, minority_idx = compute_class_summary(
        windows, track_preds, args.agg_mode
    )
    if minority_class is not None:
        print(f"   最少數族群 (minority class): {minority_class} "
              f"(count={class_counts[minority_class]})")
    else:
        print(f"   無 minority class (所有類別數量皆為 0)")

    # ------ [4/4] Write CSV + render video ------
    print(f"\n[4/4] 寫 CSV + 渲染影片...")
    t0 = time.time()
    write_predictions_csv(csv_out, windows, args.agg_mode,
                          track_preds=track_preds)
    print(f"   CSV: {csv_out}")

    if not args.no_video:
        frame_labels = build_frame_labels(
            windows,
            mode=args.agg_mode,
            track_preds=track_preds if args.agg_mode == 'per_track' else None,
        )
        render_video(video_path, video_out, track_rows, frame_labels)
    else:
        print("   (跳過主影片渲染, --no_video)")

    # ------ Minority-only video (opt-in) ------
    if args.render_minority:
        minority_out = output_dir / f"{stem}_minority.mp4"
        render_minority_video(
            video_path, minority_out,
            track_rows, windows, track_preds,
            minority_idx, args.agg_mode,
        )

    print(f"   {time.time()-t0:.1f}s")


# ==========================================
# 13. Main
# ==========================================
def main():
    ap = argparse.ArgumentParser(
        description="Fish vitality inference: video → tracking → 904-d 特徵 "
                    "→ ViT → per-window / per-track 活力分類",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ==== I/O ====
    ap.add_argument('--video', required=True, nargs='+',
                    help='輸入 mp4 影片路徑 (可多個)')
    ap.add_argument('--output_dir', default='output/inference',
                    help='輸出目錄 (存放預測 CSV 與渲染後的影片)')

    # ==== Model ckpts ====
    ap.add_argument('--rfdetr_ckpt', required=True,
                    help='RF-DETR 魚隻偵測器權重 (例如 weights/detector.pth)')
    ap.add_argument('--association_head_ckpt', required=True,
                    help='AssociationHead 權重 (例如 weights/association_head.pth)')
    ap.add_argument('--vit_ckpt', required=True,
                    help='訓練好的 ViT 活力分類器權重 (904 維輸入)')

    # ==== Stats for normalization ====
    ap.add_argument('--stats_dir', required=True,
                    help='包含 qdiff_stats.npy 與 bbox_stats.npy 的目錄，'
                         '通常是 output/vitality_dataset/{place}/')

    # ==== Aggregation & rendering ====
    ap.add_argument('--agg_mode', choices=['per_window', 'per_track'],
                    default='per_track',
                    help='per_window: 每個 window 一個標籤 (frame 上用重疊 window 多數決); '
                         'per_track: 整條軌跡多數決成一個標籤. 預設 per_track.')
    ap.add_argument('--no_video', action='store_true',
                    help='只出 CSV, 不渲染影片')
    ap.add_argument('--render_minority', action='store_true',
                    help='另外渲染一支只包含 minority class (最少數活力族群) '
                         '魚隻的影片 {stem}_minority.mp4，用於突顯潛在異常個體。'
                         '僅支援 agg_mode=per_track。')

    # ==== Window / track params (必須跟訓練一致!) ====
    ap.add_argument('--window_size', type=int, default=30,
                    help='時序 window 長度 (必須與訓練時使用的設定一致，預設 30)')
    ap.add_argument('--min_track_len', type=int, default=30,
                    help='track 至少要多長才會被切窗 (預設 = window_size)')
    ap.add_argument('--window_stride', type=int, default=1,
                    help='滑窗 stride (預設 1)')

    # ==== RF-DETR init (需與訓練時使用的設定一致) ====
    ap.add_argument('--resolution', type=int, default=378)
    ap.add_argument('--patch_size', type=int, default=14)
    ap.add_argument('--positional_encoding_size', type=int, default=27)

    # ==== Tracker hyper-params (預設值與 build_tracking_data.py 相同) ====
    ap.add_argument('--det_conf',           type=float, default=0.3)
    ap.add_argument('--conf_high',          type=float, default=0.5)
    ap.add_argument('--conf_low',           type=float, default=0.05)
    ap.add_argument('--match_cost',         type=float, default=0.7)
    ap.add_argument('--match_iou',          type=float, default=0.7)
    ap.add_argument('--reid_weight',        type=float, default=0.7)
    ap.add_argument('--max_lost',           type=int,   default=60)
    ap.add_argument('--min_hits',           type=int,   default=2)
    ap.add_argument('--tentative_max_lost', type=int,   default=5)

    # ==== Batch sizes ====
    ap.add_argument('--rfdetr_batch', type=int, default=16,
                    help='RF-DETR forward batch size (feature extraction)')
    ap.add_argument('--vit_batch', type=int, default=64,
                    help='ViT inference batch size')

    # ==== ViT arch (通常不用改) ====
    ap.add_argument('--vit_name', default='vit_small_patch16_224')
    ap.add_argument('--num_layers', type=int, default=4)

    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*72}")
    print(f"Fish Vitality Inference")
    print(f"{'='*72}")
    print(f"Device:      {device}")
    print(f"Agg mode:    {args.agg_mode}")
    print(f"Window:      size={args.window_size}, min_track={args.min_track_len}, "
          f"stride={args.window_stride}")
    print(f"LABEL_MAP:   {LABEL_MAP}   "
          f"(model 0/1/2 = {CLASS_NAMES_FULL})")
    print(f"Output dir:  {args.output_dir}")
    print(f"{'='*72}\n")

    # ---- Load stats (訓練期算好的 mean/std) ----
    qd_mean, qd_std, bb_mean, bb_std = load_stats(args.stats_dir)

    # ---- Load RF-DETR (一次, tracking + feature extraction 共用) ----
    print(f"\nLoading RF-DETR from {args.rfdetr_ckpt}...")
    from rfdetr import RFDETRNano
    rfdetr_model = RFDETRNano(
        num_classes=1,
        resolution=args.resolution,
        patch_size=args.patch_size,
        num_windows=1,
        positional_encoding_size=args.positional_encoding_size,
        encoder='dinov2_small',
        pretrain_weights=args.rfdetr_ckpt,
        freeze_encoder=True,
    )

    # ---- Load AssociationHead (for tracker) ----
    print(f"Loading AssociationHead from {args.association_head_ckpt}...")
    head = AssociationHead(in_dim=256, hidden_dim=256, proj_dim=128)
    head.load_state_dict(torch.load(args.association_head_ckpt,
                                     map_location=DEVICE))

    # ---- Detector for tracker ----
    detector = DetectorWithAssocHead(rfdetr_model, head,
                                      conf_thresh=args.det_conf)

    # ---- Feature extractor (與 tracker 共用同一個 RFDETRNano，節省 VRAM) ----
    feat_extractor = FeatureExtractor(rfdetr_model, device)
    print("FeatureExtractor 準備完成 (與 tracker 共用 RF-DETR 權重)\n")

    # ---- Load ViT ----
    vit_model = load_vit_model(args.vit_ckpt, device,
                                vit_name=args.vit_name,
                                num_layers=args.num_layers)

    # ---- 對每個 video 跑一次完整 pipeline ----
    t_grand = time.time()
    for video_path in args.video:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}, 略過")
            continue
        try:
            infer_video(video_path, args.output_dir, args,
                        detector, feat_extractor, vit_model,
                        qd_mean, qd_std, bb_mean, bb_std, device)
        except Exception as e:
            print(f"\n{video_path}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*72}")
    print(f"全部影片處理完畢, 共 {(time.time()-t_grand)/60:.1f} min")
    print(f"   結果在: {args.output_dir}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()