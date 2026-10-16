"""
build_tracking_data.py
======================
對水下魚群 mp4 影片跑 RF-DETR + tracker (VectorSearchTracker + AssociationHead)
產生追蹤 CSV，並用 unique obj_id 數量過濾品質不佳的 clip。

輸入結構:
  data/fish/{place}/{place}_{class}/clip_NNNN.mp4
    place:    {ntou2022, pd2021, ph2021}
    class:    {A, B, ...}   每個 place 的 class 數與 clip 數不固定

  RF-DETR 權重:           weights/detector.pth
  AssociationHead 權重:   weights/association_head.pth

輸出:

  Phase 1 (tracking，以 place 為單位):
    output/tracking/{place}/{place}_{class}/clip_NNNN.csv
      CSV header 跟 tracker.CSV_HEADER 一致:
        obj_id, center_x, center_y,
        bbox_x, bbox_y, bbox_w, bbox_h,
        mask_area, raw_obj_id, frame_idx
      mask_area = w * h (本流程沒有 mask，以 bbox area 替代)

    output/tracking/{place}/{place}_{class}/clip_NNNN.mp4   [選用: --save_video]
      只畫 bbox 與 ID label 的視覺化影片 (不畫軌跡線)。

    output/tracking/{place}/{place}_{class}/clip_NNNN.npz   [選用: --save_embeddings]
      RF-DETR query embeddings，與 CSV 的 row 一一對應。
      Keys:
        qfeats   : (N, 256) float32  原始 decoder query (motion-only frame 為 NaN)
        projs    : (N, 128) float32  AssociationHead 投影後 (選用 --save_projs)
        obj_id   : (N,)     int32    對齊用
        frame_idx: (N,)     int32    對齊用
        matched  : (N,)     bool     True = 來自 detection 配對；False = motion-only

  Phase 2 (filter):
    [2a] Per-track filter (預設啟用，--no_filter_tracks 停用):
      在每個 csv 內移除兩種常見的誤偵測 track:
        - 小框: track median bbox area < --small_area_ratio * clip 整體 area median
          (視為小魚或漂浮物，顯著小於一般魚隻的偵測框)
        - 表面波光: 靠近水面 (top --surface_y_ratio of frame) 且
                    (軌跡跳躍: median 每 frame 位移 > --surface_jump_ratio * frame_h
                     OR 軌跡過短: n_frames < --surface_short_min_frames)
      被移除的 row 從 csv 刪除，.npz (若存在) 以相同 mask 同步過濾維持對齊。
      詳細 drop 紀錄寫到:
        output/tracking/{place}/_filtered_tracks.txt
      可加 --filter_tracks_dry_run 預覽會移除什麼，但不實際改檔。

    [2b] Per-clip filter:
      output/tracking/{place}/_rejected.txt
      unique obj_id 數 < --min_unique_ids 的 csv 會被列入此檔
      (預設同時刪除原 csv，若不要刪除請加 --keep_rejected)。
      注意: 此計數是 [2a] 之後剩下的 track 數。

效能參考:
  - RF-DETR 與 head 只 load 一次，之後 in-memory reuse
  - 每支 mp4 約 1 ~ 3 秒 (~300 frames，~10 條魚 / frame，RTX 4090)
  - 全部 fish 資料 (估 600 支 clip): 約 10 ~ 30 分鐘，視 GPU 與 clip 長度

注意 (rejected csv 與 --skip_existing 的互動):
  預設行為: rejected csv 會被刪除。若之後想用較低 threshold 重跑:
    (a) 不加 --skip_existing，讓 Phase 1 重新生成所有 csv；或
    (b) 第一次跑就加 --keep_rejected，保留 rejected csv，重跑時再調整 threshold。

執行:
  python build_tracking_data.py \
    --rfdetr_ckpt weights/detector.pth \
    --association_head_ckpt weights/association_head.pth \
    --output_root output/tracking \
    --places pd2021 \
    --save_video --save_embeddings --save_projs
"""

import argparse
import contextlib
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# tracker 必須在 PYTHONPATH 內 (或當前目錄)
try:
    from tracker import (
        DetectorWithAssocHead,
        VectorSearchTracker,
        Track,
        build_loader,
        tracks_to_rows,
        draw_tracks,            # 用來畫 bbox visualization
        CSV_HEADER,
        DEVICE,
        REID_DIM,               # AssociationHead 投影後維度 (128)
        _preprocess_tf,         # RF-DETR 前處理 transform
    )
    from association_head import AssociationHead
except ImportError as e:
    print(f"Import failed: {e}", file=sys.stderr)
    print("  tracker.py 跟 association_head.py 要在 PYTHONPATH 內",
          file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------
# Detector wrapper: 額外提取 raw 256-d query embeddings
# ------------------------------------------------------------------
class DetectorWithQueryCapture(DetectorWithAssocHead):
    """
    在 DetectorWithAssocHead 的基礎上，額外保留每個 frame 的原始 256 維
    query embeddings，供下游任務使用。

    每次呼叫 self(img) 後，self.last_qfeats 會存放該 frame 通過 conf_thresh
    的所有 detection 的原始 query embeddings，shape 為 (N, 256)，位於 GPU 上。
    return 格式 (dets, projs) 與父類別相同，不影響既有呼叫端。

    為什麼額外保留 256 維原始 embeddings:
      AssociationHead 輸出的 128 維特徵是以對比式學習針對個體識別做過最佳化，
      適合追蹤配對時的身分比對；但 256 維原始特徵未被個體識別目標偏置，
      更適合作為下游行為辨識的輸入。.npz 兩者皆保存，供下游任務選用。
    """
    QUERY_DIM_RAW = 256

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_qfeats = torch.zeros((0, self.QUERY_DIM_RAW), device=DEVICE)

    @torch.no_grad()
    def __call__(self, img_rgb, conf_thresh=None):
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
            self.last_qfeats = torch.zeros((0, self.QUERY_DIM_RAW), device=DEVICE)
            return (torch.zeros((0, 5), device=DEVICE),
                    torch.zeros((0, REID_DIM), device=DEVICE))

        scores_k = scores[keep]
        boxes_k  = boxes[keep]
        qfeats_k = qfeats[keep]               # (N, 256) ← capture before projection

        self.last_qfeats = qfeats_k           # 保留原始 query embeddings 供外部存取

        projs_k = self.head(qfeats_k)         # (N, 128), L2-normalized

        dets = torch.cat([boxes_k, scores_k.unsqueeze(1)], dim=1)
        return dets, projs_k


# ------------------------------------------------------------------
# Match tracker output → detection index (for aligning qfeats)
# ------------------------------------------------------------------
def match_tracks_to_dets(track_xyxys, det_xyxys, iou_thresh=0.95):
    """
    用 IoU 反查每條 active track 對應的 detection index。

    tracker 輸出的 bbox 雖源自原始 detection 的 bbox，但經過內部座標格式
    轉換 (xyxy ↔ xywh) 可能有輕微浮點誤差；此外，motion-only update 時
    的框是由卡爾曼濾波器預測，沒有對應的 detection。因此不能直接比對
    bbox 字串，改以 IoU >= iou_thresh 容忍誤差做反查。

    Args:
        track_xyxys: (M, 4) np.ndarray
        det_xyxys:   (N, 4) np.ndarray (from dets[:, :4])
    Returns:
        長度 M 的 list，每個 entry 為該 track 對應的 detection index (int)，
        或 None (motion-only update 的 track，沒有對應 detection)。
    """
    if len(track_xyxys) == 0:
        return []
    if len(det_xyxys) == 0:
        return [None] * len(track_xyxys)

    tx1 = track_xyxys[:, 0:1]; ty1 = track_xyxys[:, 1:2]
    tx2 = track_xyxys[:, 2:3]; ty2 = track_xyxys[:, 3:4]
    dx1 = det_xyxys[:, 0:1].T; dy1 = det_xyxys[:, 1:2].T
    dx2 = det_xyxys[:, 2:3].T; dy2 = det_xyxys[:, 3:4].T

    ix1 = np.maximum(tx1, dx1); iy1 = np.maximum(ty1, dy1)
    ix2 = np.minimum(tx2, dx2); iy2 = np.minimum(ty2, dy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_t = (tx2 - tx1) * (ty2 - ty1)
    area_d = (dx2 - dx1) * (dy2 - dy1)
    union = area_t + area_d - inter + 1e-9
    iou = inter / union                                # (M, N)

    best_d   = iou.argmax(axis=1)
    best_iou = iou.max(axis=1)
    return [int(best_d[i]) if best_iou[i] >= iou_thresh else None
            for i in range(len(track_xyxys))]


# ------------------------------------------------------------------
# Storage: 每個 clip 一個 .npz, row-by-row 對齊 CSV
# ------------------------------------------------------------------
def save_embeddings_npz(npz_path, rows, qfeats_list, projs_list=None):
    """
    存 query embeddings 到 .npz, 跟 CSV 的 row 一一對應 (順序一致).

    .npz schema:
      qfeats:    (N, 256) float32   raw RF-DETR query embedding;
                                     motion-only update → NaN
      projs:     (N, 128) float32   (optional) AssociationHead 投影, L2-norm;
                                     motion-only update → NaN
      frame_idx: (N,)     int32     copy from CSV row, 拿來驗證 alignment
      obj_id:    (N,)     int32     copy from CSV row, 拿來驗證 alignment
      matched:   (N,)     bool      True = qfeat 來自實際 det match;
                                     False = motion-only frame (qfeat 是 NaN)

    Loading 範例:
        import numpy as np
        d = np.load("clip_0001.npz")
        qfeats = d['qfeats']                       # (N, 256)
        # 抽出 obj_id == 5 在所有有 det match 的 frames 的時序 qfeats
        mask = (d['obj_id'] == 5) & d['matched']
        tracklet_qfeats = qfeats[mask]             # (T, 256), 已按 frame 順序
    """
    n = len(rows)
    qfeats_arr = np.full((n, DetectorWithQueryCapture.QUERY_DIM_RAW),
                         np.nan, dtype=np.float32)
    matched = np.zeros(n, dtype=bool)

    for i, qf in enumerate(qfeats_list):
        if qf is not None:
            qfeats_arr[i] = qf
            matched[i] = True

    # row schema (見 tracker.CSV_HEADER):
    #   [0]=obj_id, [-1]=frame_idx
    obj_ids    = np.array([int(r[0])  for r in rows], dtype=np.int32)
    frame_idxs = np.array([int(r[-1]) for r in rows], dtype=np.int32)

    save_dict = {
        'qfeats':    qfeats_arr,
        'frame_idx': frame_idxs,
        'obj_id':    obj_ids,
        'matched':   matched,
    }
    if projs_list is not None:
        projs_arr = np.full((n, REID_DIM), np.nan, dtype=np.float32)
        for i, pj in enumerate(projs_list):
            if pj is not None:
                projs_arr[i] = pj
        save_dict['projs'] = projs_arr

    np.savez_compressed(npz_path, **save_dict)


@contextlib.contextmanager
def _suppress_native_stderr():
    """
    暫時把 OS-level stderr (fd 2) 重導向到 /dev/null.

    用來吞 OpenCV / FFmpeg 在 C 層直接 fprintf(stderr, ...) 的訊息
    (例如試 codec 失敗時的 "Could not find encoder for codec_id=..." spam).
    Python 端的 try/except 跟 sys.stderr redirect 都攔不到 C 層 stderr,
    只有 OS-level dup2 能壓住.

    退出 context 後 fd 2 還原, Python exceptions 等其他 stderr 不受影響.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _setup_video_writer(out_path, W, H, fps):
    """
    嘗試多個 codec (跟 tracker 一樣 avc1 → mp4v fallback), 用 stderr
    suppression 吞掉 codec 試錯時的 noise.

    為何要 suppress: 很多 OpenCV build 的 FFmpeg 沒帶 libx264 (avc1 encoder),
    每次 try avc1 都會 spam 兩行 "Encoder not found" 到 C-level stderr.
    在 per-clip loop 裡跑 N 次就是 2N 行 noise. 包 _suppress_native_stderr
    讓 codec 試錯安靜, 成功的 writer 走出去後一切正常.

    回傳 opened VideoWriter, 或 None (兩個 codec 都開不起來).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        with _suppress_native_stderr():
            vw = cv2.VideoWriter(str(out_path), fourcc, float(fps),
                                 (int(W), int(H)))
        if vw.isOpened():
            return vw
        vw.release()
    return None


# ------------------------------------------------------------------
# 單一 video 跑 tracker
# ------------------------------------------------------------------
def process_video(detector, video_path, tk_args,
                  save_video_path=None, save_npz_path=None,
                  save_projs=True):
    """
    對單一 mp4 跑 tracker，回傳 (rows, n_frames)。
    每個 video 都新建 VectorSearchTracker (避免跨影片狀態殘留)，
    Track.next_id 也重設為 1，所以每個 video 內的 obj_id 都從 1 開始。

    選用輸出:
      save_video_path: Path or None
          若給定，額外輸出 bbox 視覺化影片 (不畫軌跡線，只有 bbox + ID label)。
      save_npz_path: Path or None
          若給定且 detector 是 DetectorWithQueryCapture，額外輸出 .npz
          包含 query embeddings，與 CSV 的 row 一一對應。
      save_projs: bool
          .npz 是否也存 128 維 projected embeddings (AssociationHead 之後)。
    """
    Track.next_id = 1
    tracker = VectorSearchTracker(
        conf_high=tk_args.conf_high,
        conf_low=tk_args.conf_low,
        match_cost=tk_args.match_cost,
        match_iou=tk_args.match_iou,
        reid_weight=tk_args.reid_weight,
        max_lost=tk_args.max_lost,
        min_hits=tk_args.min_hits,
        tentative_max_lost=tk_args.tentative_max_lost,
    )

    loader = build_loader(str(video_path))      # mp4 → VideoLoader (auto-detect)

    # video writer setup
    video_writer = None
    if save_video_path is not None:
        fps_out = getattr(loader, 'fps', 30) or 30
        video_writer = _setup_video_writer(
            save_video_path, loader.W0, loader.H0, fps_out)
        if video_writer is None:
            print(f"  Could not open VideoWriter for {save_video_path}; "
                  "skipping video output for this clip")

    capture_embeddings = save_npz_path is not None

    all_rows = []
    all_qfeats_per_row = []   # parallel to all_rows; each (256,) np.ndarray or None
    all_projs_per_row  = []   # parallel; each (128,) np.ndarray or None
    n_frames = 0

    for frame_idx, (img_rgb, _) in enumerate(loader):
        dets, projs = detector(img_rgb)
        active = tracker.update(dets, projs, frame_idx)
        rows_this_frame = tracks_to_rows(active, frame_idx)
        all_rows.extend(rows_this_frame)

        # ----- 對齊 qfeats 到每個 active track row -----
        if capture_embeddings and len(active) > 0:
            track_xyxys = np.stack([a[1] for a in active], axis=0)  # (M, 4)
            if dets.shape[0] == 0:
                # active 但沒 det → 全部都是 motion-only
                all_qfeats_per_row.extend([None] * len(active))
                all_projs_per_row.extend([None] * len(active))
            else:
                det_xyxys_np = dets[:, :4].detach().cpu().numpy()
                qfeats_np    = detector.last_qfeats.detach().cpu().numpy()
                projs_np     = projs.detach().cpu().numpy() if save_projs else None

                match_idxs = match_tracks_to_dets(track_xyxys, det_xyxys_np)
                for mi in match_idxs:
                    if mi is None:
                        all_qfeats_per_row.append(None)
                        all_projs_per_row.append(None)
                    else:
                        all_qfeats_per_row.append(qfeats_np[mi])
                        all_projs_per_row.append(
                            projs_np[mi] if projs_np is not None else None)

        # ----- 寫 visualization 影片 (不畫軌跡線) -----
        if video_writer is not None:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            vis = draw_tracks(img_bgr, active,
                              draw_trail=False, trail_length=0)
            video_writer.write(vis)

        n_frames = frame_idx + 1

    if video_writer is not None:
        video_writer.release()

    # ----- 寫 .npz -----
    if capture_embeddings and save_npz_path is not None:
        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_embeddings_npz(
            save_npz_path,
            all_rows,
            all_qfeats_per_row,
            projs_list=all_projs_per_row if save_projs else None,
        )

    return all_rows, n_frames


def count_unique_ids(csv_path):
    """數 CSV 內 unique obj_id 個數 (不含 header)."""
    ids = set()
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row['obj_id'])
    return len(ids)


# ------------------------------------------------------------------
# Per-track filter: drop 水面波光誤偵測 + 小魚/漂浮物誤偵測
# ------------------------------------------------------------------
def _get_video_frame_h(mp4_path):
    """
    取得 video frame 高度 (像素). 失敗回 None.
    用 cv2.VideoCapture 只讀 metadata, 不解碼 pixel, ~O(1).
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h if h > 0 else None


def compute_track_stats(csv_rows):
    """
    按 obj_id group, 算每條 track 的統計量, 同時算整 clip 的 detection area median.

    csv_rows: list of dict (csv.DictReader 出來的 row).

    Returns:
      stats: dict {obj_id (int) -> {
          'n_frames':        int,     track 有幾個 detection
          'median_area':     float,   bbox area median (= bbox_w * bbox_h)
          'median_center_y': float,   center_y median, 拿來判斷 "靠近水面"
          'median_step':     float | None,
                                       相鄰 frame 中心位移 median (除以 frame gap);
                                       n_frames < 2 時為 None
      }}
      global_median_area: float
          全 clip 所有 detection rows 的 area median (跨 track), 拿來當小框 filter
          的基準. 用全 detection (非 per-track median 的 median) 比較 robust:
          就算一半 track 是漂浮物, 真魚 detection 數量通常仍主導 median.
    """
    by_id = defaultdict(list)
    for r in csv_rows:
        by_id[int(r['obj_id'])].append(r)

    all_areas = []
    stats = {}
    for oid, rows in by_id.items():
        rows = sorted(rows, key=lambda r: int(r['frame_idx']))
        areas = np.array([float(r['bbox_w']) * float(r['bbox_h']) for r in rows],
                         dtype=np.float64)
        cys   = np.array([float(r['center_y']) for r in rows], dtype=np.float64)
        cxs   = np.array([float(r['center_x']) for r in rows], dtype=np.float64)
        fidx  = np.array([int(r['frame_idx'])   for r in rows], dtype=np.int64)

        all_areas.append(areas)

        if len(rows) >= 2:
            dframes = np.maximum(np.diff(fidx), 1)            # 避免除以 0
            dcx     = np.diff(cxs)
            dcy     = np.diff(cys)
            steps   = np.sqrt(dcx * dcx + dcy * dcy) / dframes
            median_step = float(np.median(steps))
        else:
            median_step = None

        stats[oid] = {
            'n_frames':        len(rows),
            'median_area':     float(np.median(areas)),
            'median_center_y': float(np.median(cys)),
            'median_step':     median_step,
        }

    if all_areas:
        global_median_area = float(np.median(np.concatenate(all_areas)))
    else:
        global_median_area = 0.0

    return stats, global_median_area


def filter_tracks_in_clip(csv_path, mp4_path,
                          small_area_ratio,
                          surface_y_ratio,
                          surface_jump_ratio,
                          surface_short_min_frames,
                          dry_run=False):
    """
    對單一 clip 的 CSV 做 per-track filter, 移除兩類 false positive:

    (1) 小框 (小魚 / 漂浮物):
          track median bbox area < small_area_ratio * clip global area median
        用 global median 當基準, 對「顯著小於平均」的偵測框敏感.

    (2) 表面波光:
          near_surface = (track median_center_y < surface_y_ratio * frame_h)
          jumpy        = (median_step > surface_jump_ratio * frame_h)
          short        = (n_frames < surface_short_min_frames)
          drop if  near_surface AND (jumpy OR short)
        必須 near_surface 才考慮 — 真魚 (不論在哪) 跟表層真魚 (不跳) 都不會被砍.

    若 .npz 存在 (--save_embeddings 跑過), 用同一個 row mask 同步過濾, 維持
    CSV ↔ npz 對齊.

    參數設 0 (或 surface_y_ratio >= 1) 可單獨停用該道 filter.
    mp4 找不到 → frame_h 拿不到 → 跳過 surface filter, 但小框 filter 仍跑.

    Returns:
        {
          'kept':       int  保留的 track 數,
          'dropped':    int  drop 的 track 數,
          'reasons':    dict {obj_id (int) -> reason str},
          'frame_h':    int | None,
        }
    """
    # ----- 讀 csv (整份載入, 後面要 rewrite) -----
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        header = reader.fieldnames
    if not all_rows:
        return {'kept': 0, 'dropped': 0, 'reasons': {}, 'frame_h': None}

    # ----- frame_h (surface filter 用) -----
    frame_h = _get_video_frame_h(mp4_path) if mp4_path.exists() else None

    # ----- 計算 per-track 統計 + 全 clip 基準 -----
    stats, global_median_area = compute_track_stats(all_rows)

    # ----- 判斷哪些 track 要 drop -----
    reasons = {}
    surface_enabled = (frame_h is not None and 0 < surface_y_ratio < 1)
    small_enabled = (small_area_ratio > 0 and global_median_area > 0)

    for oid, s in stats.items():
        rs = []

        # (1) 小框 filter
        if small_enabled:
            if s['median_area'] < small_area_ratio * global_median_area:
                rs.append(
                    f"small_bbox(med_area={s['median_area']:.0f},"
                    f"clip_median={global_median_area:.0f},"
                    f"ratio={s['median_area']/global_median_area:.2f})"
                )

        # (2) 表面波光 filter
        if surface_enabled:
            y_thresh = surface_y_ratio * frame_h
            if s['median_center_y'] < y_thresh:
                step_thresh = surface_jump_ratio * frame_h
                is_short = s['n_frames'] < surface_short_min_frames
                is_jumpy = (s['median_step'] is not None and
                            s['median_step'] > step_thresh)
                if is_short or is_jumpy:
                    step_str = (f"{s['median_step']:.1f}"
                                if s['median_step'] is not None else 'NA')
                    rs.append(
                        f"surface_flicker("
                        f"med_y={s['median_center_y']:.0f},y_thresh={y_thresh:.0f},"
                        f"n_frames={s['n_frames']},short_thresh={surface_short_min_frames},"
                        f"med_step={step_str},step_thresh={step_thresh:.1f})"
                    )

        if rs:
            reasons[oid] = '; '.join(rs)

    drop_ids = set(reasons.keys())
    kept_count = len(stats) - len(drop_ids)

    # dry_run 或沒東西要砍 → 不動檔, 直接 return
    if dry_run or not drop_ids:
        return {
            'kept': kept_count,
            'dropped': len(drop_ids),
            'reasons': reasons,
            'frame_h': frame_h,
        }

    # ----- rewrite csv (保留 row 順序, 跟原本 frame-major 排序一致) -----
    keep_mask = np.array(
        [int(r['obj_id']) not in drop_ids for r in all_rows],
        dtype=bool,
    )
    kept_rows = [r for r, k in zip(all_rows, keep_mask) if k]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(kept_rows)

    # ----- 同步過濾 .npz (若存在) -----
    npz_path = csv_path.with_suffix('.npz')
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            # 所有 array 都是 row-aligned (length N), 用相同 mask
            filtered = {k: data[k][keep_mask] for k in data.files}
            np.savez_compressed(npz_path, **filtered)
        except Exception as e:
            print(f"  npz filter failed for {npz_path}: {e}")

    return {
        'kept': kept_count,
        'dropped': len(drop_ids),
        'reasons': reasons,
        'frame_h': frame_h,
    }


# ------------------------------------------------------------------
# 兩個 Phase
# ------------------------------------------------------------------
def run_phase1_tracking(detector, args, place):
    """Phase 1: 對 place 下所有 mp4 跑 tracker, 寫 csv."""
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    place_dir = data_root / place

    if not place_dir.is_dir():
        return None

    place_output_dir = output_root / place
    place_output_dir.mkdir(parents=True, exist_ok=True)

    # 找所有 class subdir (e.g. ntou2022_A, ntou2022_B, ...)
    # 用 prefix match, 自動排除 _Split 之類 underscore-leading 目錄.
    class_dirs = sorted([d for d in place_dir.iterdir()
                         if d.is_dir() and d.name.startswith(f"{place}_")])
    print(f"  Found {len(class_dirs)} class subdirs: "
          f"{[d.name for d in class_dirs]}")

    all_clips = []   # (class_name, mp4_path, csv_path)
    for class_dir in class_dirs:
        class_name = class_dir.name
        for mp4 in sorted(class_dir.glob("clip_*.mp4")):
            csv_path = place_output_dir / class_name / f"{mp4.stem}.csv"
            all_clips.append((class_name, mp4, csv_path))
    print(f"  Total mp4 clips: {len(all_clips)}")

    if args.limit_per_place is not None:
        all_clips = all_clips[:args.limit_per_place]
        print(f"  Limited to first {len(all_clips)}")

    t_start = time.time()
    n_done = n_skipped = n_failed = n_frames_total = 0

    for class_name, mp4_path, csv_path in tqdm(all_clips, desc=f"{place}"):
        if args.skip_existing and csv_path.exists():
            n_skipped += 1
            continue

        # 同一個 base name, 不同副檔名, 放在 csv 旁邊
        save_video_path = csv_path.with_suffix('.mp4') if args.save_video else None
        save_npz_path   = csv_path.with_suffix('.npz') if args.save_embeddings else None

        try:
            rows, n_frames = process_video(
                detector, mp4_path, args,
                save_video_path=save_video_path,
                save_npz_path=save_npz_path,
                save_projs=args.save_projs,
            )
        except Exception as e:
            print(f"\n{mp4_path}: {type(e).__name__}: {e}")
            n_failed += 1
            continue
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(rows)
        n_done += 1
        n_frames_total += n_frames

    elapsed = time.time() - t_start
    print(f"  Phase 1 done in {elapsed/60:.1f} min: "
          f"{n_done} done, {n_skipped} skipped, {n_failed} failed, "
          f"{n_frames_total} frames")
    return {
        'n_done': n_done, 'n_skipped': n_skipped, 'n_failed': n_failed,
        'n_frames': n_frames_total, 'elapsed': elapsed,
    }


def run_phase2_filter(args, place):
    """
    Phase 2: 過濾品質不佳的 track 與 clip。

    [2a] Per-track filter: 在每個 csv 內移除小框 (小魚 / 漂浮物) 與
         表面波光的誤偵測 track，.npz 同步過濾。
         --no_filter_tracks 可停用；--filter_tracks_dry_run 只紀錄不動檔。
    [2b] Per-clip filter: unique obj_id 數不足的 csv 整支 reject (預設刪除)。
         注意: 此計數是 [2a] 之後剩下的 track 數。
    """
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    place_output_dir = output_root / place

    accepted_by_class = defaultdict(list)
    rejected = []                                          # [(entry, n_ids)]
    delete_flag = not args.keep_rejected
    do_track_filter = not args.no_filter_tracks

    all_csvs = sorted(place_output_dir.rglob("clip_*.csv"))

    # ================================================================
    # [2a] Per-track filter
    # ================================================================
    track_filter_log = []   # [(entry, n_kept, n_dropped, reasons_dict)]
    n_clips_with_drops = 0
    n_tracks_dropped_total = 0
    n_clips_with_no_frame_h = 0

    if do_track_filter:
        print(f"  [2a] per-track filter "
              f"(small_area_ratio={args.small_area_ratio}, "
              f"surface_y_ratio={args.surface_y_ratio}, "
              f"surface_jump_ratio={args.surface_jump_ratio}, "
              f"surface_short_min_frames={args.surface_short_min_frames}"
              f"{', DRY-RUN' if args.filter_tracks_dry_run else ''})")

        for csv_path in tqdm(all_csvs, desc=f"  [2a] {place}"):
            class_name = csv_path.parent.name
            entry = f"{class_name}/{csv_path.stem}"
            mp4_path = data_root / place / class_name / f"{csv_path.stem}.mp4"
            try:
                res = filter_tracks_in_clip(
                    csv_path, mp4_path,
                    small_area_ratio=args.small_area_ratio,
                    surface_y_ratio=args.surface_y_ratio,
                    surface_jump_ratio=args.surface_jump_ratio,
                    surface_short_min_frames=args.surface_short_min_frames,
                    dry_run=args.filter_tracks_dry_run,
                )
            except Exception as e:
                print(f"\n  {csv_path}: track filter failed: "
                      f"{type(e).__name__}: {e}")
                continue
            if res['frame_h'] is None:
                n_clips_with_no_frame_h += 1
            if res['dropped'] > 0:
                track_filter_log.append((entry, res['kept'], res['dropped'],
                                         res['reasons']))
                n_clips_with_drops += 1
                n_tracks_dropped_total += res['dropped']

        print(f"  [2a] dropped {n_tracks_dropped_total} tracks "
              f"across {n_clips_with_drops}/{len(all_csvs)} clips"
              f"{' (dry-run, no file modified)' if args.filter_tracks_dry_run else ''}")
        if n_clips_with_no_frame_h > 0:
            print(f"  [2a] {n_clips_with_no_frame_h} clips: mp4 missing → "
                  f"surface filter skipped (only small-bbox applied)")

        # Log: _filtered_tracks.txt
        track_log_path = place_output_dir / "_filtered_tracks.txt"
        place_output_dir.mkdir(parents=True, exist_ok=True)
        with open(track_log_path, 'w') as f:
            f.write(f"# Per-track filter log\n")
            f.write(f"# small_area_ratio={args.small_area_ratio} "
                    f"surface_y_ratio={args.surface_y_ratio} "
                    f"surface_jump_ratio={args.surface_jump_ratio} "
                    f"surface_short_min_frames={args.surface_short_min_frames}\n")
            f.write(f"# dry_run={args.filter_tracks_dry_run}\n")
            f.write(f"# total_clips_scanned = {len(all_csvs)}\n")
            f.write(f"# clips_with_drops = {n_clips_with_drops}\n")
            f.write(f"# total_tracks_dropped = {n_tracks_dropped_total}\n")
            f.write(f"# format: <entry>  kept=<K>  dropped=<D>\n")
            f.write(f"#         <obj_id>: <reason>\n")
            for entry, kept, dropped, reasons in sorted(track_filter_log):
                f.write(f"{entry}\tkept={kept}\tdropped={dropped}\n")
                for oid, reason in sorted(reasons.items()):
                    f.write(f"\t{oid}: {reason}\n")
        print(f"  [2a] log: {track_log_path}")
    else:
        print(f"  [2a] per-track filter SKIPPED (--no_filter_tracks)")

    # ================================================================
    # [2b] Per-clip filter (unique obj_id 計數)
    # ================================================================
    print(f"  [2b] per-clip filter (min_unique_ids={args.min_unique_ids})")
    for csv_path in all_csvs:
        if not csv_path.exists():
            continue   # 理論上 2a 不會刪 csv, 但 defensive check
        try:
            n_ids = count_unique_ids(csv_path)
        except Exception as e:
            print(f"  {csv_path}: count failed: {e}")
            continue
        class_name = csv_path.parent.name                  # {place}_{class}
        entry = f"{class_name}/{csv_path.stem}"            # {place}_{class}/clip_NNNN
        if n_ids < args.min_unique_ids:
            rejected.append((entry, n_ids))
            if delete_flag:
                csv_path.unlink()
        else:
            accepted_by_class[class_name].append(entry)

    n_accepted = sum(len(v) for v in accepted_by_class.values())
    print(f"  [2b] Accepted: {n_accepted} csvs (across {len(accepted_by_class)} classes)")
    print(f"  [2b] Rejected: {len(rejected)} csvs "
          f"(deleted from disk: {delete_flag})")

    # Log: 寫一份 rejected.txt 到 output_root/{place}/
    log_path = place_output_dir / "_rejected.txt"
    with open(log_path, 'w') as f:
        f.write(f"# Rejected CSVs: unique obj_id < {args.min_unique_ids} "
                f"(after per-track filter 2a)\n")
        f.write(f"# total_rejected = {len(rejected)}\n")
        f.write(f"# deleted_from_disk = {delete_flag}\n")
        f.write(f"# format: <entry>\\tunique_ids=<N>\n")
        for entry, n_ids in sorted(rejected):
            f.write(f"{entry}\tunique_ids={n_ids}\n")
    print(f"  [2b] Rejection log: {log_path}")

    return accepted_by_class


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ===== paths =====
    ap.add_argument('--data_root', default='data/fish',
                    help='Fish 影片根目錄；期待結構 {data_root}/{place}/{place}_{class}/clip_*.mp4')
    ap.add_argument('--output_root', default='output/tracking',
                    help='Tracking csv 輸出根；結構 {output_root}/{place}/{place}_{class}/clip_NNNN.csv')
    ap.add_argument('--places', nargs='+',
                    default=['ntou2022', 'pd2021', 'ph2021'])

    # ===== model ckpts =====
    ap.add_argument('--rfdetr_ckpt', required=True,
                    help='RF-DETR 魚隻偵測器權重 (例如 weights/detector.pth)')
    ap.add_argument('--association_head_ckpt', required=True,
                    help='AssociationHead 權重 (例如 weights/association_head.pth)')

    # ===== RF-DETR backbone settings (需與 train_detector.py 一致) =====
    ap.add_argument('--resolution', type=int, default=378)
    ap.add_argument('--patch_size', type=int, default=14)
    ap.add_argument('--positional_encoding_size', type=int, default=27)
    ap.add_argument('--num_classes', type=int, default=1)

    # ===== detector =====
    ap.add_argument('--det_conf', type=float, default=0.3)

    # ===== tracker hyper-params (預設值與 tracker.py 內的預設值相同) =====
    ap.add_argument('--conf_high',          type=float, default=0.5)
    ap.add_argument('--conf_low',           type=float, default=0.05)
    ap.add_argument('--match_cost',         type=float, default=0.7)
    ap.add_argument('--match_iou',          type=float, default=0.7)
    ap.add_argument('--reid_weight',        type=float, default=0.7)
    ap.add_argument('--max_lost',           type=int,   default=60)
    ap.add_argument('--min_hits',           type=int,   default=2)
    ap.add_argument('--tentative_max_lost', type=int,   default=5)

    # ===== filter (per-clip: unique obj_id 計數) =====
    ap.add_argument('--min_unique_ids', type=int, default=10,
                    help='unique obj_id 少於此數的 csv 被 reject (預設 10). '
                         '注意此計數是 per-track filter (2a) 之後剩下的 track 數.')
    ap.add_argument('--keep_rejected', action='store_true',
                    help='不刪除 rejected csv (預設刪除, 只留在 _rejected.txt 紀錄)')

    # ===== filter (per-track: 表面波光 + 小框 false positive) =====
    ap.add_argument('--no_filter_tracks', action='store_true',
                    help='停用 per-track filter (預設啟用): 不過濾表面波光 + 小框.')
    ap.add_argument('--filter_tracks_dry_run', action='store_true',
                    help='Per-track filter dry run: 只 log 不實際 drop track. '
                         '配合 _filtered_tracks.txt 可預覽會砍什麼再決定門檻.')
    ap.add_argument('--small_area_ratio', type=float, default=0.4,
                    help='Track median bbox area < ratio * clip 整體 area median '
                         '→ drop. 預設 0.4 (顯著小於平均的視為小魚/漂浮物). '
                         '設 0 停用此 filter.')
    ap.add_argument('--surface_y_ratio', type=float, default=0.20,
                    help='Track median center_y < ratio * frame_h 視為 "靠近水面". '
                         '預設 0.20 (top 20%% of frame). 設 0 或 >=1 停用 surface filter.')
    ap.add_argument('--surface_jump_ratio', type=float, default=0.05,
                    help='靠近水面的 track, 若 median 相鄰 frame 位移 > ratio * frame_h '
                         '視為位置跳躍, drop. 預設 0.05 (5%% frame_h per frame).')
    ap.add_argument('--surface_short_min_frames', type=int, default=5,
                    help='靠近水面 且 軌跡長度 < N frames → drop (容易閃的短 track). '
                         '預設 5.')

    # ===== runtime =====
    ap.add_argument('--skip_existing', action='store_true',
                    help='跳過 csv 已存在的 video (resumable)')
    ap.add_argument('--limit_per_place', type=int, default=None,
                    help='只處理前 N 個 clip per place (debug)')
    ap.add_argument('--skip_tracking', action='store_true',
                    help='只跑 Phase 2, 假設 csv 已經生成')

    # ===== 選用輸出 =====
    ap.add_argument('--save_video', action='store_true',
                    help='另存 bbox-only visualization 影片到 clip_NNNN.mp4 '
                         '(不畫軌跡線, 只 bbox + ID)')
    ap.add_argument('--save_embeddings', action='store_true',
                    help='另存 raw 256-d RF-DETR query embeddings 到 clip_NNNN.npz, '
                         'row-by-row 對齊 CSV')
    ap.add_argument('--save_projs', action='store_true',
                    help='搭配 --save_embeddings 使用; .npz 額外存 128-d projected '
                         '(AssociationHead 後) embeddings')

    args = ap.parse_args()

    # ===== Load model (一次性) =====
    detector = None
    if not args.skip_tracking:
        print(f"\nLoading RF-DETR from {args.rfdetr_ckpt}...")
        from rfdetr import RFDETRNano
        rfdetr_model = RFDETRNano(
            num_classes=args.num_classes,
            resolution=args.resolution,
            patch_size=args.patch_size,
            num_windows=1,
            positional_encoding_size=args.positional_encoding_size,
            encoder='dinov2_small',
            pretrain_weights=args.rfdetr_ckpt,
            freeze_encoder=True,
        )

        print(f"Loading AssociationHead from {args.association_head_ckpt}...")
        head = AssociationHead(in_dim=256, hidden_dim=256, proj_dim=128)
        head.load_state_dict(torch.load(args.association_head_ckpt,
                                        map_location=DEVICE))

        detector = DetectorWithQueryCapture(rfdetr_model, head,
                                            conf_thresh=args.det_conf)
    else:
        print("--skip_tracking: Phase 1 skipped, only Phase 2 will run")

    # ===== 兩個 Phase per place =====
    t_grand = time.time()
    grand_done = grand_skipped = grand_failed = grand_frames = 0

    for place in args.places:
        print(f"\n{'='*72}")
        print(f"=== place = {place} ===")
        print(f"{'='*72}")

        # Phase 1: tracking
        if not args.skip_tracking:
            print(f"\n[Phase 1] tracking → {args.output_root}/{place}/")
            stats = run_phase1_tracking(detector, args, place)
            if stats is None:
                print(f"  {args.data_root}/{place} not found, skipping place")
                continue
            grand_done    += stats['n_done']
            grand_skipped += stats['n_skipped']
            grand_failed  += stats['n_failed']
            grand_frames  += stats['n_frames']
        else:
            print(f"\n[Phase 1] skipped (--skip_tracking)")
            # 確認 output dir 有東西可以 filter
            place_output_dir = Path(args.output_root) / place
            if not place_output_dir.is_dir():
                print(f"  {place_output_dir} not found, skipping place")
                continue

        # Phase 2: filter (2a: per-track, 2b: per-clip)
        print(f"\n[Phase 2] filter "
              f"({'per-track + ' if not args.no_filter_tracks else ''}"
              f"unique_ids < {args.min_unique_ids})")
        accepted_by_class = run_phase2_filter(args, place)
        if not accepted_by_class:
            print(f"  no accepted csvs in {place}")
            continue
        n_accepted = sum(len(v) for v in accepted_by_class.values())
        print(f"  Accepted {n_accepted} csvs across {len(accepted_by_class)} classes")

    # ===== Grand summary =====
    elapsed = time.time() - t_grand
    print(f"\n{'='*72}")
    print(f"=== ALL DONE in {elapsed/60:.1f} min ===")
    print(f"{'='*72}")
    if not args.skip_tracking:
        print(f"  Phase 1: {grand_done} done, {grand_skipped} skipped, "
              f"{grand_failed} failed, {grand_frames} total frames")
        if grand_done > 0:
            print(f"           avg {grand_frames / grand_done:.1f} frames/clip, "
                  f"{elapsed / grand_done:.2f} s/clip")


if __name__ == "__main__":
    main()