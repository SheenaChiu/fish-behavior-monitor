"""
build_pseudo_labeled_dataset.py
================================
產生用於訓練時序活力分類器的偽標籤資料集。

本流程從追蹤軌跡與原始影片一路處理到最終按 train / val / test 切分、
依 GMM cluster 分類好的 .npy 檔與對應 CSV，共分為八個 stage：

  STAGE 1: 遍歷 clips，一次 RF-DETR forward 產生 raw 904 npy 存入 _pending/，
           同時累積全域 stats、並從追蹤 CSV 算三項運動學特徵
  STAGE 2: 計算全域的 qdiff / bbox 的 mean 與 std
  STAGE 3: 讀 _pending/*.npy，normalize 後存入 _staging/*.npy (尚未分 split)，
           同時從 raw object_emb 抽 query_diff L2 統計 (qd_mean / qd_std / qd_p95)，
           合併運動特徵成 gmm_features.csv
  STAGE 4: GMM 分群 (含 z-score、特徵權重、silhouette 與 BIC 報告)
  STAGE 5: 依 cluster 切 split (支援 group_stratified 或 random 兩種策略)
  STAGE 6: 依 split 將 _staging/*.npy 搬到 features/{split}/cluster_{gmm}/
  STAGE 7: 產生 features_csv/{train,val,test}.csv
  STAGE 8: (可選) 清除 _pending/ 與 _staging/ 中間目錄

設計重點：
  1. Stats 以全域方式計算 (不侷限於 train)，因為資料產出時 split 尚未決定。
  2. GMM 特徵為三項運動學特徵加上 qd_mean / qd_std / qd_p95，共六維。
  3. npy 檔走兩階段暫存: _pending/ (raw) → _staging/ (normalized) → 最終 split 目錄。
  4. 支援中斷續跑: 若 _pending/ 或 _staging/ 內已存在對應 npy，
     可透過 RESUME=True 跳過該 window 的重新計算。

一次跑完的 Config:
  PLACE = 'ph2021', MODE = 'full'
"""

import os
import shutil
import random
import numpy as np
import collections
import csv
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==========================================
# 1. 參數設定區
# ==========================================
PLACE = 'ph2021'                # 'pd2021' / 'ph2021' / 'ntou2022'
MODE  = 'full'                  # 'full' / 'smoke'
RESUME = True                   # True: _pending/ 已存在的 npy 不重跑 forward

# Roots
DATA_ROOT          = 'data/fish'
TRACKING_CSV_ROOT  = 'output/tracking'
OUTPUT_ROOT        = 'output/vitality_dataset'

WINDOW_SIZE           = 30
CLUSTER_WINDOW_STRIDE = 1
MIN_TRACK_LEN         = WINDOW_SIZE   # 短於此的整條 track 丟棄

# Split
SPLIT_RATIOS   = (0.70, 0.15, 0.15)
SPLIT_STRATEGY = 'group_stratified'   # 'group_stratified' / 'random'
SEED           = 42

# GMM 分群
NUM_CLUSTERS = 3
GMM_FEATURES = [
    'avg_speed_norm', 'straightness_norm', 'avg_angle_change_norm',
    'qd_mean',        'qd_std',            'qd_p95',
]
GMM_WEIGHTS = {
    'avg_speed_norm':        1.0,
    'straightness_norm':     1.0,
    'avg_angle_change_norm': 1.0,
    'qd_mean':               1.0,
    'qd_std':                1.0,
    'qd_p95':                1.0,
}
GMM_N_INIT = 20

# Cleanup: 跑完後刪除中間目錄
CLEANUP_PENDING = False
CLEANUP_STAGING = True

# Smoke
SMOKE_MAX_CLIPS = 2

# === RF-DETR (Fish ckpt) ===
RFDETR_CKPT     = "output/rfdetr_fish_dinov2s/checkpoint_best_total.pth"
RFDETR_INPUT    = 378
RFDETR_PATCH    = 14
RFDETR_GRID     = RFDETR_INPUT // RFDETR_PATCH
RFDETR_CONF     = 0.1
IOU_MATCH_THR   = 0.3

BATCH_SIZE      = 16
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# === Feature layout ===
GLOBAL_DIM      = 384
OBJECT_DIM      = 256
QUERY_DIFF_DIM  = 256
BBOX_DIM        = 4
BBOX_DIFF_DIM   = 4
FEAT_DIM        = GLOBAL_DIM + OBJECT_DIM + QUERY_DIFF_DIM + BBOX_DIM + BBOX_DIFF_DIM  # 904

# Raw npy 存放 offsets (STAGE 1 產生, 904 維, 但 [640:896]/[896:900]/[900:904] 是 raw)
GLOBAL_S,  GLOBAL_E  = 0,   384
OBJECT_S,  OBJECT_E  = 384, 640
QDIFF_S,   QDIFF_E   = 640, 896
BBOX_S,    BBOX_E    = 896, 900
BBOXD_S,   BBOXD_E   = 900, 904


# ==========================================
# 2. Path helpers
# ==========================================
def place_output_root(place):
    return os.path.join(OUTPUT_ROOT, place)


def pending_dir(place):
    """Raw 904 npy 暫存 (未 normalize)."""
    return os.path.join(place_output_root(place), '_pending')


def staging_dir(place):
    """Normalized 904 npy 暫存 (未分 split)."""
    return os.path.join(place_output_root(place), '_staging')


def stats_paths_for_place(place):
    root = place_output_root(place)
    return (os.path.join(root, 'qdiff_stats.npy'),
            os.path.join(root, 'bbox_stats.npy'))


def windows_meta_csv_for(place):
    return os.path.join(place_output_root(place), 'windows_meta.csv')


def gmm_features_csv_for(place):
    return os.path.join(place_output_root(place), 'gmm_features.csv')


def cluster_csv_for(place):
    return os.path.join(place_output_root(place), f'{place}_clusters.csv')


def split_assignment_csv_for(place):
    return os.path.join(place_output_root(place), 'split_assignment.csv')


def final_npy_dir_for(place, mode, gmm_label):
    return os.path.join(place_output_root(place),
                        'features', mode, f'cluster_{gmm_label}')


def features_csv_path_for(place, mode):
    return os.path.join(place_output_root(place),
                        'features_csv', f'{mode}.csv')


def mp4_path_for(place, class_dir, clip_stem):
    return os.path.join(DATA_ROOT, place, class_dir, f'{clip_stem}.mp4')


def tracking_csv_path_for(place, class_dir, clip_stem):
    return os.path.join(TRACKING_CSV_ROOT, place, class_dir, f'{clip_stem}.csv')


def npy_stem_for(class_dir, clip_stem, obj_id, w_idx):
    return f"{class_dir}_{clip_stem}_track{obj_id}_win{w_idx}"


# ==========================================
# 3. RF-DETR 初始化
# ==========================================
print(f"載入 RF-DETR 從 {RFDETR_CKPT} ...")
from rfdetr import RFDETRNano
rfdetr_model = RFDETRNano(
    num_classes=1,
    resolution=RFDETR_INPUT,
    patch_size=RFDETR_PATCH,
    num_windows=1,
    positional_encoding_size=RFDETR_GRID,
    encoder='dinov2_small',
    pretrain_weights=RFDETR_CKPT,
    freeze_encoder=True,
)
lwdetr = rfdetr_model.model.model
rfdetr_post = rfdetr_model.model.postprocess
lwdetr.to(DEVICE).eval()
print("RF-DETR (fish) 載入完成 (內部 backbone: DINOv2-S/14, frozen)")


_dino_backbone_module = lwdetr.backbone[0].encoder
if not hasattr(_dino_backbone_module, "last_cls"):
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, RFDETR_INPUT, RFDETR_INPUT, device=DEVICE)
        _ = lwdetr(_dummy)
    if not hasattr(_dino_backbone_module, "last_cls") or _dino_backbone_module.last_cls is None:
        raise RuntimeError(
            "RF-DETR 內部 DinoV2 backbone 沒有 last_cls attribute。"
            "此修改應已內建於 rfdetr/models/backbone/dinov2.py，"
            "若此錯誤出現，表示 rfdetr/ 可能被覆蓋為原版，"
            "詳見 rfdetr/MODIFICATIONS.md。"
        )
print(f"DinoV2 backbone.last_cls 可用\n")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

rfdetr_transform = T.Compose([
    T.Resize((RFDETR_INPUT, RFDETR_INPUT)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# 4. Frame / bbox 輔助
# ==========================================
def read_mp4_frames(mp4_path, frame_ids_needed):
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
# 5. RF-DETR unified forward
# ==========================================
@torch.no_grad()
def rfdetr_forward_batch(pil_frames):
    if not pil_frames:
        return [], [], [], np.zeros((0, GLOBAL_DIM), dtype=np.float32)

    sizes = [(img.size[1], img.size[0]) for img in pil_frames]
    tensors = torch.stack([rfdetr_transform(img) for img in pil_frames]).to(DEVICE)

    raw = lwdetr(tensors)

    cls_tensor = getattr(_dino_backbone_module, "last_cls", None)
    if cls_tensor is None:
        global_cls = np.zeros((len(pil_frames), GLOBAL_DIM), dtype=np.float32)
    else:
        global_cls = cls_tensor.cpu().numpy().astype(np.float32)

    target_sizes = torch.tensor(sizes, device=DEVICE)
    results = rfdetr_post(raw, target_sizes=target_sizes)

    all_boxes, all_scores, all_qfeats = [], [], []
    for r in results:
        scores = r["scores"]
        boxes  = r["boxes"]
        qfeats = r["query_features"]

        keep = scores >= RFDETR_CONF
        boxes_k  = boxes[keep].cpu().numpy().astype(np.float32)
        scores_k = scores[keep].cpu().numpy().astype(np.float32)
        qfeats_k = qfeats[keep].cpu().numpy().astype(np.float32)
        all_boxes.append(boxes_k)
        all_scores.append(scores_k)
        all_qfeats.append(qfeats_k)

    return all_boxes, all_scores, all_qfeats, global_cls


def extract_video_query_embs(mp4_path, all_frame_ids, sam3_bboxes_per_frame):
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

    for batch_start in range(0, len(all_frame_ids), BATCH_SIZE):
        batch_fids = all_frame_ids[batch_start:batch_start + BATCH_SIZE]
        batch_pils = [pil_frames_dict[fid] for fid in batch_fids]
        all_boxes_b, all_scores_b, all_qfeats_b, global_cls_b = rfdetr_forward_batch(batch_pils)

        for idx, fid in enumerate(batch_fids):
            fullframe_cls_dict[fid] = global_cls_b[idx]

        for i, fid in enumerate(batch_fids):
            det_boxes  = all_boxes_b[i]
            det_qfeats = all_qfeats_b[i]

            if fid not in sam3_bboxes_per_frame:
                continue
            track_bbox_list = sam3_bboxes_per_frame[fid]

            if len(det_boxes) == 0:
                for track_id, _ in track_bbox_list:
                    query_emb_dict[(fid, track_id)] = None
                continue

            for track_id, sam3_xywh in track_bbox_list:
                sam3_xyxy = xywh_to_xyxy(sam3_xywh)
                ious = compute_iou_batch(sam3_xyxy, det_boxes)
                best_idx = int(np.argmax(ious))
                best_iou = ious[best_idx]
                if best_iou >= IOU_MATCH_THR:
                    query_emb_dict[(fid, track_id)] = det_qfeats[best_idx]
                else:
                    query_emb_dict[(fid, track_id)] = None

    return fullframe_cls_dict, query_emb_dict


# ==========================================
# 6. Motion features (運動學特徵)
# ==========================================
def compute_motion_features(xs, ys):
    """xs, ys: 1-D arrays of length WINDOW_SIZE (center coords)."""
    dx, dy = np.diff(xs), np.diff(ys)
    dists  = np.sqrt(dx**2 + dy**2)
    avg_speed   = float(np.mean(dists))
    path_length = float(np.sum(dists))

    net_dx = xs[-1] - xs[0]
    net_dy = ys[-1] - ys[0]
    net_displacement = float(np.sqrt(net_dx**2 + net_dy**2))
    straightness = net_displacement / (path_length + 1e-6)

    angles = np.arctan2(dy, dx)
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    avg_angle_change = float(np.mean(np.abs(angle_diffs)))

    return avg_speed, straightness, avg_angle_change


def bbox_center(bbox_xywh):
    """[x, y, w, h] → (cx, cy)."""
    x, y, w, h = bbox_xywh
    return x + w / 2.0, y + h / 2.0


# ==========================================
# 7. Raw 904 feature 組裝 (未 normalize)
# ==========================================
def build_raw_904(global_emb_seq, object_emb_seq, bbox_seq):
    """
    產生 raw 904-d matrix (未 normalize):
      [0:384]    global_emb        raw
      [384:640]  object_emb        raw
      [640:896]  query_diff_raw    = diff(object_emb), [0] = 0
      [896:900]  bbox              raw
      [900:904]  bbox_diff_raw     = diff(bbox_raw),  [0] = 0
                 (STAGE 3 會用 normalized bbox 重新計算)
    """
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


# ==========================================
# 8. STAGE 1: 遍歷 clips → raw npy + 運動特徵
# ==========================================
def enumerate_clips(place):
    """回傳所有 (class_dir, clip_stem) - 從 TRACKING_CSV_ROOT/{place}/ 掃出."""
    root = Path(TRACKING_CSV_ROOT) / place
    if not root.is_dir():
        raise FileNotFoundError(f"tracking root 不存在: {root}")

    clip_list = []
    for class_subdir in sorted(root.iterdir()):
        if not class_subdir.is_dir():
            continue
        class_dir = class_subdir.name
        for csv_path in sorted(class_subdir.glob('clip_*.csv')):
            clip_stem = csv_path.stem  # 'clip_0000'
            clip_list.append((class_dir, clip_stem))
    return clip_list


def stage1_forward_and_raw_npy(place, clip_list):
    """
    對每個 clip:
      1. 讀 tracking CSV, 決定要切的 (obj_id, window_idx, window_frames)
      2. 一次 RF-DETR forward 拿所有 fid 的 global_cls + query_emb
      3. 組 raw 904 存到 _pending/*.npy
      4. 從 tracking CSV 算三項運動特徵
      5. 累積 qdiff_raw / bbox_raw for 全域 stats

    Returns:
      windows_meta_list: [
        {video_name, class_dir, clip_stem, obj_id, window_idx, npy_path,
         avg_speed, straightness, avg_angle_change}, ...
      ]
      qdiff_running: (mean_accumulator, sq_accumulator, count)
      bbox_running:  (mean_accumulator, sq_accumulator, count)
    """
    os.makedirs(pending_dir(place), exist_ok=True)

    windows_meta = []
    # 用 Welford-style running stats (省記憶體, 避免 concat 大 array)
    qd_sum = np.zeros(QUERY_DIFF_DIM, dtype=np.float64)
    qd_sq  = np.zeros(QUERY_DIFF_DIM, dtype=np.float64)
    qd_n   = 0
    bb_sum = np.zeros(BBOX_DIM, dtype=np.float64)
    bb_sq  = np.zeros(BBOX_DIM, dtype=np.float64)
    bb_n   = 0

    stats = collections.Counter()
    n_clips = len(clip_list)

    print(f"\nSTAGE 1: forward + raw npy + 運動特徵 ({n_clips} clips)")
    print("-" * 75)

    for i, (class_dir, clip_stem) in enumerate(clip_list):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"   [{i+1:5d}/{n_clips}] {class_dir}/{clip_stem}")

        tracking_csv = tracking_csv_path_for(place, class_dir, clip_stem)
        mp4_path     = mp4_path_for(place, class_dir, clip_stem)
        if not (os.path.exists(tracking_csv) and os.path.exists(mp4_path)):
            stats['missing_files'] += 1
            continue

        try:
            df_tr = pd.read_csv(tracking_csv)
        except Exception:
            stats['bad_csv'] += 1
            continue
        if df_tr.empty:
            stats['empty_csv'] += 1
            continue

        required = {'obj_id', 'center_x', 'center_y', 'frame_idx',
                    'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}
        if not required.issubset(df_tr.columns):
            stats['bad_columns'] += 1
            continue

        # 建 track 結構
        df_tr = df_tr.sort_values(['obj_id', 'frame_idx'])
        track_data = {}   # tid → {'fids': [...], 'centers': {fid: (cx, cy)}, 'bboxes': {fid: [x,y,w,h]}}
        for tid, tg in df_tr.groupby('obj_id'):
            tid = int(tid)
            tg = tg.reset_index(drop=True)
            if len(tg) < MIN_TRACK_LEN:
                stats['track_too_short'] += 1
                continue
            fids = tg['frame_idx'].astype(int).tolist()
            centers = {int(r['frame_idx']): (float(r['center_x']), float(r['center_y']))
                       for _, r in tg.iterrows()}
            bboxes = {int(r['frame_idx']): [float(r['bbox_x']), float(r['bbox_y']),
                                             float(r['bbox_w']), float(r['bbox_h'])]
                      for _, r in tg.iterrows()}
            track_data[tid] = {'fids': fids, 'centers': centers, 'bboxes': bboxes}

        if not track_data:
            continue

        # 決定要切的 windows: 每個 track 按 CLUSTER_WINDOW_STRIDE 滑窗
        window_jobs = []   # (obj_id, w_idx, window_frames, xs, ys)
        for tid, td in track_data.items():
            sorted_fids = td['fids']
            n_frames = len(sorted_fids)
            for w_idx, start in enumerate(range(0, n_frames - WINDOW_SIZE + 1,
                                                 CLUSTER_WINDOW_STRIDE)):
                wf = sorted_fids[start : start + WINDOW_SIZE]
                xs = np.array([td['centers'][f][0] for f in wf], dtype=np.float32)
                ys = np.array([td['centers'][f][1] for f in wf], dtype=np.float32)
                window_jobs.append((tid, w_idx, wf, xs, ys))

        if not window_jobs:
            continue

        # 檢查哪些 windows 已在 _pending/ (RESUME)
        pending_root = pending_dir(place)
        video_name = f"{class_dir}_{clip_stem.replace('clip_', '')}"
        remaining_jobs = []
        for job in window_jobs:
            tid, w_idx, wf, xs, ys = job
            stem = npy_stem_for(class_dir, clip_stem, tid, w_idx)
            npy_path = os.path.join(pending_root, stem + '.npy')
            if RESUME and os.path.exists(npy_path):
                # 已存在, 但仍要累積 stats + 抽 meta
                try:
                    raw = np.load(npy_path)
                except Exception:
                    remaining_jobs.append(job)
                    continue
                qd = raw[:, QDIFF_S:QDIFF_E]
                bb = raw[:, BBOX_S:BBOX_E]
                # 只累積 diff 存在的部分 (排除 [0]=0)
                if qd.shape[0] >= 2:
                    qd_valid = qd[1:]
                    qd_sum += qd_valid.sum(axis=0)
                    qd_sq  += (qd_valid.astype(np.float64) ** 2).sum(axis=0)
                    qd_n   += qd_valid.shape[0]
                bb_sum += bb.sum(axis=0).astype(np.float64)
                bb_sq  += (bb.astype(np.float64) ** 2).sum(axis=0)
                bb_n   += bb.shape[0]

                av_sp, st_r, av_ang = compute_motion_features(xs, ys)
                windows_meta.append({
                    'video_name':       video_name,
                    'class_dir':        class_dir,
                    'clip_stem':        clip_stem,
                    'obj_id':           tid,
                    'window_idx':       w_idx,
                    'npy_path':         npy_path,
                    'avg_speed':        av_sp,
                    'straightness':     st_r,
                    'avg_angle_change': av_ang,
                })
                stats['resumed'] += 1
            else:
                remaining_jobs.append(job)

        if not remaining_jobs:
            continue

        # 需要 forward 的 frame_ids
        all_frame_ids = set()
        for _, _, wf, _, _ in remaining_jobs:
            all_frame_ids.update(wf)
        all_frame_ids = sorted(all_frame_ids)

        # sam3_bboxes: 該 clip 內所有 obj 在 all_frame_ids 的 bbox (for IoU 配對)
        sam3_bboxes_per_frame = collections.defaultdict(list)
        for fid in all_frame_ids:
            for tid, td in track_data.items():
                if fid in td['bboxes']:
                    sam3_bboxes_per_frame[fid].append((tid, td['bboxes'][fid]))

        # RF-DETR forward
        fullframe_cls_dict, query_emb_dict = extract_video_query_embs(
            mp4_path, all_frame_ids, sam3_bboxes_per_frame
        )

        # per window: 組 raw 904 + 存 + 累積 stats + 抽運動特徵
        for tid, w_idx, wf, xs, ys in remaining_jobs:
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
            if raw_904.shape != (WINDOW_SIZE, FEAT_DIM):
                stats['bad_shape'] += 1
                continue

            # 存 raw
            stem = npy_stem_for(class_dir, clip_stem, tid, w_idx)
            npy_path = os.path.join(pending_root, stem + '.npy')
            np.save(npy_path, raw_904)
            stats['saved_raw'] += 1

            # 累積 stats
            qd = raw_904[:, QDIFF_S:QDIFF_E]
            bb = raw_904[:, BBOX_S:BBOX_E]
            if qd.shape[0] >= 2:
                qd_valid = qd[1:]
                qd_sum += qd_valid.sum(axis=0).astype(np.float64)
                qd_sq  += (qd_valid.astype(np.float64) ** 2).sum(axis=0)
                qd_n   += qd_valid.shape[0]
            bb_sum += bb.sum(axis=0).astype(np.float64)
            bb_sq  += (bb.astype(np.float64) ** 2).sum(axis=0)
            bb_n   += bb.shape[0]

            # 運動特徵
            av_sp, st_r, av_ang = compute_motion_features(xs, ys)
            windows_meta.append({
                'video_name':       video_name,
                'class_dir':        class_dir,
                'clip_stem':        clip_stem,
                'obj_id':           tid,
                'window_idx':       w_idx,
                'npy_path':         npy_path,
                'avg_speed':        av_sp,
                'straightness':     st_r,
                'avg_angle_change': av_ang,
            })

    print(f"\n   stats: {dict(stats)}")
    print(f"   total windows: {len(windows_meta)}")

    # 存 windows_meta.csv
    meta_df = pd.DataFrame(windows_meta)
    meta_path = windows_meta_csv_for(place)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    meta_df.to_csv(meta_path, index=False)
    print(f"   windows meta → {meta_path}")

    return meta_df, (qd_sum, qd_sq, qd_n), (bb_sum, bb_sq, bb_n)


# ==========================================
# 9. STAGE 2: 算全域 stats
# ==========================================
def stage2_compute_global_stats(place, qd_running, bb_running):
    qd_sum, qd_sq, qd_n = qd_running
    bb_sum, bb_sq, bb_n = bb_running

    if qd_n > 0:
        qd_mean = (qd_sum / qd_n).astype(np.float32)
        qd_var  = (qd_sq / qd_n) - (qd_sum / qd_n) ** 2
        qd_std  = np.sqrt(np.maximum(qd_var, 0)).astype(np.float32)
    else:
        qd_mean = np.zeros(QUERY_DIFF_DIM, dtype=np.float32)
        qd_std  = np.ones(QUERY_DIFF_DIM, dtype=np.float32)

    if bb_n > 0:
        bb_mean = (bb_sum / bb_n).astype(np.float32)
        bb_var  = (bb_sq / bb_n) - (bb_sum / bb_n) ** 2
        bb_std  = np.sqrt(np.maximum(bb_var, 0)).astype(np.float32)
    else:
        bb_mean = np.zeros(BBOX_DIM, dtype=np.float32)
        bb_std  = np.ones(BBOX_DIM, dtype=np.float32)

    qdiff_path, bbox_path = stats_paths_for_place(place)
    os.makedirs(os.path.dirname(qdiff_path), exist_ok=True)
    np.save(qdiff_path, {'mean': qd_mean, 'std': qd_std})
    np.save(bbox_path,  {'mean': bb_mean, 'std': bb_std})

    print(f"\nSTAGE 2: 全域 stats")
    print(f"   qdiff n={qd_n}, mean[:5]={qd_mean[:5].round(3)}, std[:5]={qd_std[:5].round(3)}")
    print(f"   bbox  n={bb_n}, mean={bb_mean.round(2)}, std={bb_std.round(2)}")
    print(f"   {qdiff_path}")
    print(f"   {bbox_path}")

    return qd_mean, qd_std, bb_mean, bb_std


# ==========================================
# 10. STAGE 3: normalize + 抽 query_diff L2 統計
# ==========================================
def stage3_normalize_and_extract_qd_stats(place, meta_df,
                                           qd_mean, qd_std, bb_mean, bb_std):
    """
    對每個 _pending/*.npy:
      1. 讀 raw
      2. normalize qdiff (raw diff → z-score)
      3. normalize bbox
      4. 重算 bbox_diff = diff(normalized bbox), [0] = 0
      5. 存到 _staging/*.npy (normalized 904)
      6. 從 raw object_emb [384:640] 算 query_diff L2 統計 (qd_mean/qd_std/qd_p95)
    Returns: gmm_features_df (meta + qd_L2 stats)
    """
    os.makedirs(staging_dir(place), exist_ok=True)
    staging_root = staging_dir(place)

    print(f"\nSTAGE 3: normalize + 抽 query_diff L2 統計 ({len(meta_df)} windows)")

    qd_stats_rows = []
    stats = collections.Counter()

    qd_std_safe = np.where(qd_std > 1e-6, qd_std, 1.0)
    bb_std_safe = np.where(bb_std > 1e-6, bb_std, 1.0)

    for i, row in enumerate(meta_df.itertuples(index=False)):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"   [{i+1:6d}/{len(meta_df)}]")

        raw_path = row.npy_path
        try:
            raw = np.load(raw_path)
        except Exception:
            stats['load_fail'] += 1
            qd_stats_rows.append({'qd_mean': np.nan, 'qd_std': np.nan, 'qd_p95': np.nan})
            continue

        if raw.shape != (WINDOW_SIZE, FEAT_DIM):
            stats['bad_shape'] += 1
            qd_stats_rows.append({'qd_mean': np.nan, 'qd_std': np.nan, 'qd_p95': np.nan})
            continue

        # (1) normalize qdiff
        # 注意: 第 0 幀的 raw diff = 0，normalize 後會變成 -qd_mean/qd_std (非 0)，
        # 本流程不強制歸零。若要讓第 0 幀在 normalized 域也維持為 0，取消下行註解即可。
        qdiff_raw   = raw[:, QDIFF_S:QDIFF_E]
        qdiff_norm  = (qdiff_raw - qd_mean) / qd_std_safe
        # qdiff_norm[0] = 0.0   # (可選) 若要強制第 0 幀為 0

        # (2) normalize bbox
        bbox_raw  = raw[:, BBOX_S:BBOX_E]
        bbox_norm = (bbox_raw - bb_mean) / bb_std_safe

        # (3) 重算 bbox_diff = diff(bbox_norm), [0]=0
        bbox_diff = np.zeros_like(bbox_norm)
        bbox_diff[1:] = bbox_norm[1:] - bbox_norm[:-1]

        # 組 final 904
        final = raw.copy()
        final[:, QDIFF_S:QDIFF_E] = qdiff_norm
        final[:, BBOX_S:BBOX_E]   = bbox_norm
        final[:, BBOXD_S:BBOXD_E] = bbox_diff

        # 存 _staging/
        stem = Path(raw_path).stem
        out_path = os.path.join(staging_root, stem + '.npy')
        np.save(out_path, final.astype(np.float32))
        stats['normalized'] += 1

        # (4) 從 raw object_emb 算 query_diff L2 統計
        # 這裡刻意用 raw (未 normalize) 的 diff, 保持 GMM 特徵含物理意義
        object_emb = raw[:, OBJECT_S:OBJECT_E]
        if object_emb.shape[0] >= 2:
            diffs = object_emb[1:] - object_emb[:-1]
            l2 = np.linalg.norm(diffs, axis=1)
            qd_stats_rows.append({
                'qd_mean': float(l2.mean()),
                'qd_std':  float(l2.std()),
                'qd_p95':  float(np.percentile(l2, 95)),
            })
        else:
            qd_stats_rows.append({'qd_mean': 0.0, 'qd_std': 0.0, 'qd_p95': 0.0})

    print(f"   stats: {dict(stats)}")

    # 合併 meta + qd_stats + 更新 npy_path 到 staging
    qd_df = pd.DataFrame(qd_stats_rows)
    merged = pd.concat([meta_df.reset_index(drop=True), qd_df], axis=1)
    # 更新 npy_path 為 staging 版本 (raw 版本繼續存在)
    merged['pending_npy_path'] = merged['npy_path']
    merged['staging_npy_path'] = merged['npy_path'].apply(
        lambda p: os.path.join(staging_root, Path(p).stem + '.npy')
    )

    return merged


# ==========================================
# 11. STAGE 4: GMM 分群
# ==========================================
def stage4_gmm_clustering(place, gmm_df):
    """
    對 gmm_df 做 z-score → weighted → GMM(n=NUM_CLUSTERS)
    產生 gmm 欄位, 存 cluster_csv, 印分群報告 + silhouette + BIC 掃描.
    """
    print(f"\nSTAGE 4: GMM 分群 (n_components={NUM_CLUSTERS})")

    # z-score normalize (基於原始特徵值)
    # 為了輸出 features CSV 兼容, 額外算 *_norm 欄位
    for feat in ['avg_speed', 'straightness', 'avg_angle_change']:
        m = gmm_df[feat].mean()
        s = gmm_df[feat].std(ddof=0)
        gmm_df[f'{feat}_norm'] = (gmm_df[feat] - m) / s if s > 0 else 0.0

    # 也對 qd_* 做 z-score
    for feat in ['qd_mean', 'qd_std', 'qd_p95']:
        m = gmm_df[feat].mean()
        s = gmm_df[feat].std(ddof=0)
        # qd_* 本身作為 GMM 輸入時已 z-score, 命名不加 _norm 保持一致
        gmm_df[feat + '_z'] = (gmm_df[feat] - m) / s if s > 0 else 0.0

    # 建構 GMM 輸入矩陣 (依 GMM_FEATURES 順序)
    feat_map = {
        'avg_speed_norm':        'avg_speed_norm',
        'straightness_norm':     'straightness_norm',
        'avg_angle_change_norm': 'avg_angle_change_norm',
        'qd_mean':               'qd_mean_z',
        'qd_std':                'qd_std_z',
        'qd_p95':                'qd_p95_z',
    }
    cols = [feat_map[f] for f in GMM_FEATURES]

    # dropna
    before = len(gmm_df)
    gmm_df = gmm_df.dropna(subset=cols).reset_index(drop=True)
    if len(gmm_df) < before:
        print(f"   dropped NaN rows: {before - len(gmm_df)}")

    X = gmm_df[cols].values.astype(np.float64).copy()

    # 套用權重
    for i, feat_name in enumerate(GMM_FEATURES):
        X[:, i] *= GMM_WEIGHTS[feat_name]

    print(f"   Fitting GMM (n_init={GMM_N_INIT}, seed={SEED})...")
    gmm = GaussianMixture(n_components=NUM_CLUSTERS, n_init=GMM_N_INIT,
                           random_state=SEED)
    gmm_df['gmm'] = gmm.fit_predict(X)

    # ============ 分群報告 ============
    total = len(gmm_df)
    print(f"\nGMM 分群結果 ({total} windows)")
    print("=" * 60)
    display_features = ['avg_speed', 'straightness', 'avg_angle_change',
                        'qd_mean', 'qd_std', 'qd_p95']
    cluster_means = gmm_df.groupby('gmm')[display_features].mean()
    for cid in sorted(gmm_df['gmm'].unique()):
        count = int((gmm_df['gmm'] == cid).sum())
        pct = count / total * 100
        row = cluster_means.loc[cid]
        print(f"【群集 {cid}】 {count} 筆 ({pct:.1f}%)")
        for f in display_features:
            print(f"  - {f:20s}: {row[f]:.4f}")
        print("-" * 60)

    # ============ 品質指標 ============
    try:
        sil = silhouette_score(X, gmm_df['gmm'])
        print(f"\nSilhouette Score: {sil:.4f}")
    except Exception as e:
        print(f"\nSilhouette 計算失敗: {e}")
    print(f"   BIC: {gmm.bic(X):.2f}")
    print(f"   AIC: {gmm.aic(X):.2f}")

    print(f"\nBIC 掃描 (n_components 2 ~ 8)")
    for k in range(2, 9):
        g = GaussianMixture(n_components=k, n_init=5, random_state=SEED)
        g.fit(X)
        print(f"   k={k}: BIC={g.bic(X):.2f}")

    # 存 cluster CSV
    out_path = cluster_csv_for(place)
    keep_cols = (['global_id', 'video_name', 'class_dir', 'clip_stem',
                  'obj_id', 'window_idx',
                  'pending_npy_path', 'staging_npy_path',
                  'avg_speed', 'straightness', 'avg_angle_change',
                  'avg_speed_norm', 'straightness_norm', 'avg_angle_change_norm',
                  'qd_mean', 'qd_std', 'qd_p95', 'gmm'])
    # global_id
    gmm_df = gmm_df.reset_index(drop=True)
    gmm_df.insert(0, 'global_id', gmm_df.index + 1)
    keep_cols = [c for c in keep_cols if c in gmm_df.columns]
    gmm_df[keep_cols].to_csv(out_path, index=False)
    print(f"\ncluster CSV → {out_path}")

    return gmm_df


# ==========================================
# 12. STAGE 5: Split (per-window)
# ==========================================
def split_windows_by_cluster(df, ratios=SPLIT_RATIOS, seed=SEED):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)

    train_parts, val_parts, test_parts = [], [], []
    print(f"依 gmm cluster 切 split (ratios={ratios}, seed={seed}):")
    for gmm_label in sorted(df['gmm'].unique()):
        sub = df[df['gmm'] == gmm_label].copy()
        idx = np.arange(len(sub))
        rng.shuffle(idx)
        sub = sub.iloc[idx].reset_index(drop=True)

        n = len(sub)
        n_train = int(round(n * ratios[0]))
        n_val   = int(round(n * ratios[1]))
        train_p = sub.iloc[:n_train]
        val_p   = sub.iloc[n_train:n_train + n_val]
        test_p  = sub.iloc[n_train + n_val:]
        train_parts.append(train_p)
        val_parts.append(val_p)
        test_parts.append(test_p)
        print(f"   cluster_{gmm_label}: total={n:6d} → "
              f"train={len(train_p):6d} / val={len(val_p):5d} / test={len(test_p):5d}")

    return {
        'train': pd.concat(train_parts).reset_index(drop=True),
        'val':   pd.concat(val_parts  ).reset_index(drop=True),
        'test':  pd.concat(test_parts ).reset_index(drop=True),
    }


def _assign_group_windows(n, ratios, rng):
    if n <= 0:
        return []
    if n == 1:
        return [rng.choice(['train', 'val', 'test'], p=list(ratios))]

    exact = np.array([n * r for r in ratios], dtype=float)
    base  = np.floor(exact).astype(int)
    frac  = exact - base
    rem   = n - base.sum()
    if rem > 0:
        noise = rng.random(3) * 1e-9
        order = np.argsort(-(frac + noise))
        for i in range(rem):
            base[order[i]] += 1

    target_nonempty = min(3, n)
    max_iter = 3
    while (base > 0).sum() < target_nonempty and max_iter > 0:
        max_iter -= 1
        donor_cands = np.where(base >= 2)[0]
        if len(donor_cands) == 0:
            break
        surplus = base - exact
        donor = int(donor_cands[np.argmax(surplus[donor_cands])])
        recv_cands = np.where(base == 0)[0]
        if len(recv_cands) == 0:
            break
        recv_probs = np.array([ratios[i] for i in recv_cands], dtype=float)
        recv_probs = recv_probs / recv_probs.sum()
        recv = int(rng.choice(recv_cands, p=recv_probs))
        base[donor] -= 1
        base[recv]  += 1

    labels = ['train'] * int(base[0]) + ['val'] * int(base[1]) + ['test'] * int(base[2])
    rng.shuffle(labels)
    return labels


def split_windows_grouped(df, ratios=SPLIT_RATIOS, seed=SEED):
    rng = np.random.RandomState(seed)

    group_keys = sorted(
        df.groupby(['video_name', 'obj_id', 'gmm']).groups.keys()
    )
    print(f"依 (video, obj, gmm) 組 stratified 分 split "
          f"(ratios={ratios}, seed={seed}):")
    print(f"   total groups: {len(group_keys)}")

    group_labels = {}
    diversity_by_size = collections.defaultdict(lambda: collections.Counter())
    for key in group_keys:
        video, obj, gmm = key
        sub = df[(df['video_name'] == video) &
                 (df['obj_id']     == obj)   &
                 (df['gmm']        == gmm)]
        n = len(sub)
        labels = _assign_group_windows(n, ratios, rng)
        group_labels[key] = labels
        diversity_by_size[n][len(set(labels))] += 1

    label_iter = {k: iter(v) for k, v in group_labels.items()}
    row_splits = []
    for row in df.itertuples(index=False):
        key = (row.video_name, int(row.obj_id), int(row.gmm))
        row_splits.append(next(label_iter[key]))
    df = df.copy()
    df['_split'] = row_splits

    total_ge3   = sum(sum(v.values()) for s, v in diversity_by_size.items() if s >= 3)
    covered_ge3 = sum(v.get(3, 0)     for s, v in diversity_by_size.items() if s >= 3)
    n_size1     = sum(diversity_by_size.get(1, {}).values())
    n_size2     = sum(diversity_by_size.get(2, {}).values())
    covered_eq2 = diversity_by_size.get(2, {}).get(2, 0)
    print(f"   group size=1: {n_size1:6d} groups (無法分散)")
    print(f"   group size=2: {n_size2:6d} groups, {covered_eq2}/{n_size2} 涵蓋 2 splits "
          f"({100*covered_eq2/max(n_size2,1):.1f}%)")
    print(f"   group size>=3: {total_ge3:6d} groups, {covered_ge3}/{total_ge3} 涵蓋 3 splits "
          f"({100*covered_ge3/max(total_ge3,1):.1f}%)")

    splits = {}
    for mode in ('train', 'val', 'test'):
        splits[mode] = (df[df['_split'] == mode]
                        .drop(columns=['_split']).reset_index(drop=True))

    for gmm_label in sorted(df['gmm'].unique()):
        sub_all = df[df['gmm'] == gmm_label]
        n_c = len(sub_all)
        counts = sub_all['_split'].value_counts()
        print(f"   cluster_{gmm_label}: total={n_c:6d} → "
              f"train={counts.get('train',0):6d} ({100*counts.get('train',0)/n_c:.2f}%) / "
              f"val={counts.get('val',0):5d} ({100*counts.get('val',0)/n_c:.2f}%) / "
              f"test={counts.get('test',0):5d} ({100*counts.get('test',0)/n_c:.2f}%)")
    return splits


def stage5_split(place, gmm_df):
    print(f"\nSTAGE 5: split (strategy={SPLIT_STRATEGY})")
    if SPLIT_STRATEGY == 'group_stratified':
        splits = split_windows_grouped(gmm_df, ratios=SPLIT_RATIOS, seed=SEED)
    elif SPLIT_STRATEGY == 'random':
        splits = split_windows_by_cluster(gmm_df, ratios=SPLIT_RATIOS, seed=SEED)
    else:
        raise SystemExit(f"Unknown SPLIT_STRATEGY: {SPLIT_STRATEGY!r}")

    # 存 split_assignment.csv
    rows = []
    for mode, sub_df in splits.items():
        tagged = sub_df.copy()
        tagged['split'] = mode
        rows.append(tagged)
    all_df = pd.concat(rows).reset_index(drop=True)
    out_path = split_assignment_csv_for(place)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"   split assignment → {out_path}")

    return splits


# ==========================================
# 13. STAGE 6: 移動 _staging → 最終 split 目錄
# ==========================================
def stage6_move_to_final(place, splits):
    print(f"\nSTAGE 6: 移動 npy → features/{{split}}/cluster_{{gmm}}/")

    csv_data_by_split = {'train': [], 'val': [], 'test': []}
    stats = collections.Counter()

    for mode, sub_df in splits.items():
        for row in sub_df.itertuples(index=False):
            src = row.staging_npy_path
            gmm = int(row.gmm)
            dst_dir = final_npy_dir_for(place, mode, gmm)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, Path(src).name)

            if not os.path.exists(src):
                stats[f'{mode}_missing_src'] += 1
                continue

            try:
                shutil.move(src, dst)
                stats[f'{mode}_moved'] += 1
                csv_data_by_split[mode].append([os.path.abspath(dst), gmm])
            except Exception as e:
                stats[f'{mode}_move_fail'] += 1
                print(f"      move fail: {src} → {dst}: {e}")

    print(f"   stats: {dict(stats)}")

    # 寫 features_csv/{mode}.csv
    for mode, rows in csv_data_by_split.items():
        csv_path = features_csv_path_for(place, mode)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f"   {mode}: {len(rows)} windows → {csv_path}")

    return csv_data_by_split


# ==========================================
# 14. Cleanup
# ==========================================
def cleanup(place):
    if CLEANUP_PENDING and os.path.exists(pending_dir(place)):
        print(f"\n清除 {pending_dir(place)}")
        shutil.rmtree(pending_dir(place))
    if CLEANUP_STAGING and os.path.exists(staging_dir(place)):
        # 若 STAGE 6 已把檔案全部搬走, staging 會是空的
        print(f"\n清除 {staging_dir(place)}")
        shutil.rmtree(staging_dir(place))


# ==========================================
# 15. Smoke test
# ==========================================
def run_smoke(place):
    print(f"\n{'='*75}")
    print(f"SMOKE TEST [{place}], max {SMOKE_MAX_CLIPS} clips")
    print(f"{'='*75}")
    clip_list = enumerate_clips(place)[:SMOKE_MAX_CLIPS]
    if not clip_list:
        print("   no clips found")
        return
    meta_df, qd_run, bb_run = stage1_forward_and_raw_npy(place, clip_list)
    if meta_df.empty:
        return
    qd_mean, qd_std, bb_mean, bb_std = stage2_compute_global_stats(place, qd_run, bb_run)
    gmm_df = stage3_normalize_and_extract_qd_stats(
        place, meta_df, qd_mean, qd_std, bb_mean, bb_std
    )
    print(f"\nGMM features (smoke, {len(gmm_df)} rows) sample:")
    print(gmm_df[['obj_id', 'window_idx',
                  'avg_speed', 'straightness', 'avg_angle_change',
                  'qd_mean', 'qd_std', 'qd_p95']].head())


# ==========================================
# 16. Main
# ==========================================
random.seed(SEED)
np.random.seed(SEED)

print(f"\n{'='*75}")
print(f"=== Fish 偽標籤資料集產生器 (gen npy → GMM → split → move) ===")
print(f"   PLACE:      {PLACE}")
print(f"   MODE:       {MODE}")
print(f"   SPLIT:      train/val/test = {SPLIT_RATIOS}")
print(f"   STRATEGY:   {SPLIT_STRATEGY}")
print(f"   WINDOW:     size={WINDOW_SIZE}, cluster stride={CLUSTER_WINDOW_STRIDE}")
print(f"   CLUSTERS:   {NUM_CLUSTERS}")
print(f"   GMM FEATS:  {GMM_FEATURES}")
print(f"   SEED:       {SEED}")
print(f"   FEAT_DIM:   {FEAT_DIM}")
print(f"   RESUME:     {RESUME}")
print(f"{'='*75}\n")

if MODE == 'smoke':
    run_smoke(PLACE)
    raise SystemExit(0)

# ---- STAGE 1: 全掃 clips, 產 raw npy + 抽運動特徵 + 累積 stats ----
clip_list = enumerate_clips(PLACE)
print(f"總共 {len(clip_list)} clips 待處理")
meta_df, qd_run, bb_run = stage1_forward_and_raw_npy(PLACE, clip_list)

if meta_df.empty:
    raise SystemExit("STAGE 1 沒有產生任何 window, 中止.")

# ---- STAGE 2: 全域 stats ----
qd_mean, qd_std, bb_mean, bb_std = stage2_compute_global_stats(PLACE, qd_run, bb_run)

# ---- STAGE 3: normalize + query_diff L2 統計 ----
gmm_df = stage3_normalize_and_extract_qd_stats(
    PLACE, meta_df, qd_mean, qd_std, bb_mean, bb_std
)

# 存中間 gmm_features (含運動特徵 + qd_L2 統計, 未 GMM 前)
gmm_feat_path = gmm_features_csv_for(PLACE)
os.makedirs(os.path.dirname(gmm_feat_path), exist_ok=True)
gmm_df.to_csv(gmm_feat_path, index=False)
print(f"   gmm features (pre-clustering) → {gmm_feat_path}")

# ---- STAGE 4: GMM 分群 ----
gmm_df = stage4_gmm_clustering(PLACE, gmm_df)

# ---- STAGE 5: Split ----
splits = stage5_split(PLACE, gmm_df)

# ---- STAGE 6: 移動 npy 到最終 split 目錄 + 寫 features_csv ----
csv_data_by_split = stage6_move_to_final(PLACE, splits)

# ---- Cleanup ----
cleanup(PLACE)

print(f"\n{'='*75}")
print(f"FULL pipeline done for {PLACE}")
for mode in ('train', 'val', 'test'):
    print(f"   {mode:6s}: {len(csv_data_by_split[mode])} windows")
print(f"   Layout: [0:384] global | [384:640] object | [640:896] query_diff | "
      f"[896:900] bbox | [900:904] bbox_diff")
print(f"   Label:  per-window gmm cluster ({NUM_CLUSTERS} classes)")
print(f"{'='*75}")