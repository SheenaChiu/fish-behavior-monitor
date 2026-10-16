"""
peer_anomaly_detection.py
=========================
魚隻同儕比較預警 (Per-clip Peer Ranking)

對每段影片內的所有魚，計算每條魚與同段影片中其他魚的平均相似度，
相似度最低的個體即為該段影片的同儕異常個體 (rank-1 outlier)。

本方法的特性:
    不需要事先建立 baseline、prototype 或任何前置準備步驟，
    新場域部署即可直接運作。

依 rank-1 個體的差異度分三級告警:
    正常     : dissim < mild_thr                影片內個體行為一致
    輕度異常 : mild_thr <= dissim < severe_thr  建議留意
    明顯異常 : dissim >= severe_thr             建議捕撈檢視

輸入格式:
    features/{split}/cluster_{gmm}/
        {class_dir}_{clip_stem}_track{tid}_win{w}.npy

    每個 npy 為 904 維向量，本模組只使用 [384:640] 這段 (RF-DETR
    object embedding，個體外觀表徵)。此輸入來自 build_pseudo_labeled_dataset.py。

執行範例:
    python peer_anomaly_detection.py \
        --features_dir output/vitality_dataset/ph2021/features \
        --output_dir output/peer_anomaly \
        --video_root data/fish \
        --top_k 20

    # 調整分級閾值:
    python peer_anomaly_detection.py \
        --features_dir output/vitality_dataset/ph2021/features \
        --output_dir output/peer_anomaly \
        --mild_thr 0.25 --severe_thr 0.45
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np


# ============================================================
# Loader: 讀 features/ 目錄下的 per-window RF-DETR embedding
# ============================================================
def load_windows(features_dir, splits=None, clusters=None):
    """
    掃 features_dir 底下所有 npy，只讀取 [384:640] 這段 (RF-DETR query embedding)。

    檔名格式: {prefix}_track{tid}_win{w}.npy
    """
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f'features_dir not found: {features_dir}')

    # 決定要掃哪些 split 子目錄
    known_splits = ('train', 'val', 'test')
    if splits is None:
        found = [d.name for d in features_dir.iterdir()
                 if d.is_dir() and d.name in known_splits]
        splits = found if found else ['.']

    # 收集 npy 檔
    files = []
    for split in splits:
        split_dir = features_dir if split == '.' else features_dir / split
        if not split_dir.exists():
            print(f'  split not found, skipped: {split_dir}')
            continue
        cluster_dirs = sorted(d for d in split_dir.iterdir()
                              if d.is_dir() and d.name.startswith('cluster_'))
        if cluster_dirs:
            if clusters is not None:
                wanted = {f'cluster_{c}' for c in clusters}
                cluster_dirs = [d for d in cluster_dirs if d.name in wanted]
            for cd in cluster_dirs:
                files.extend(sorted(cd.glob('*.npy')))
        else:
            files.extend(sorted(split_dir.glob('*.npy')))

    if not files:
        raise RuntimeError(
            f'No .npy files under {features_dir} '
            f'(splits={splits}, clusters={clusters})'
        )

    pat = re.compile(r'^(?P<prefix>.+)_track(?P<tid>\d+)_win(?P<w>\d+)\.npy$')
    feats, sources, tids = [], [], []
    skipped_name, skipped_shape = 0, 0
    first_shape_reported = False
    for fp in files:
        m = pat.match(fp.name)
        if m is None:
            skipped_name += 1
            continue
        arr = np.asarray(np.load(fp))

        # 支援兩種 shape:
        #   1D  (D,)     → 直接用
        #   2D  (T, D)   → 一個 window 內 T frame, 對 frame 軸取 mean
        #   其他         → 跳過
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        elif arr.ndim != 1:
            skipped_shape += 1
            continue

        if arr.shape[0] < 640:
            skipped_shape += 1
            continue

        if not first_shape_reported:
            orig = np.load(fp)
            print(f'  First npy shape: {orig.shape} → after aggregate: '
                  f'{arr.shape} (using [384:640] slice)')
            first_shape_reported = True

        feats.append(arr[384:640].astype(np.float32))
        sources.append(m.group('prefix'))
        tids.append(int(m.group('tid')))

    if skipped_name:
        print(f'  {skipped_name} file(s) skipped: filename pattern mismatch')
    if skipped_shape:
        print(f'  {skipped_shape} file(s) skipped: unexpected npy shape')
    if not feats:
        raise RuntimeError('No usable npy after filtering.')

    feats = np.stack(feats).astype(np.float32)
    print(f'  Loaded {len(feats)} windows from {features_dir}')
    return feats, np.array(sources), np.array(tids, dtype=np.int64)


# ============================================================
# Per-clip peer ranking
# ============================================================
def peer_ranking(features_dir, output_dir, video_root, top_k,
                 min_peers=3, mild_thr=0.3, severe_thr=0.5,
                 splits=None, clusters=None):
    """
    對每段影片內所有魚做同儕比較, 找出 rank-1 outlier, 依差異度分級告警.
    """
    # ---- 讀資料 ----
    obj, sources, tids = load_windows(features_dir, splits, clusters)

    # ---- 每條魚 (source, tid) 取平均 embedding → per-fish 代表向量 ----
    fish_key = np.array([f'{s}||{t}' for s, t in zip(sources, tids)])
    unique_fish, inv = np.unique(fish_key, return_inverse=True)
    F, D = len(unique_fish), obj.shape[1]
    sums = np.zeros((F, D), dtype=np.float64)
    counts = np.zeros(F, dtype=np.int64)
    np.add.at(sums, inv, obj)
    np.add.at(counts, inv, 1)
    fish_feats = (sums / counts[:, None]).astype(np.float32)
    # L2 normalize → cosine 空間
    norms = np.linalg.norm(fish_feats, axis=1, keepdims=True) + 1e-8
    fish_feats_l2 = fish_feats / norms

    fish_source = np.array([k.split('||', 1)[0] for k in unique_fish])
    fish_tid = np.array([int(k.split('||', 1)[1]) for k in unique_fish])
    fish_wcnt = counts

    def _severity(d):
        if d >= severe_thr:
            return 'severe'
        elif d >= mild_thr:
            return 'mild'
        else:
            return 'normal'

    # ---- 依 clip 分組, 對每 clip 內做同儕比較 ----
    rows = []
    skipped_small = 0
    for src in np.unique(fish_source):
        mask = fish_source == src
        idxs = np.where(mask)[0]
        n = len(idxs)
        if n < min_peers:
            skipped_small += 1
            continue
        X = fish_feats_l2[idxs]                     # (n, 256)
        # 兩兩 cos_sim, 對角線設 0 排除自己
        S = X @ X.T
        np.fill_diagonal(S, 0.0)
        mean_sim_to_peers = S.sum(axis=1) / (n - 1)
        peer_dissim = 1.0 - mean_sim_to_peers       # 越大越像 outlier

        # 排名 (1 = 最不像同儕)
        order = np.argsort(-peer_dissim)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(1, n + 1)
        # 該 clip 的分級由 rank-1 fish 的 dissim 決定
        rank1_dissim = float(peer_dissim[order[0]])
        clip_severity = _severity(rank1_dissim)

        for k, gi in enumerate(idxs):
            rows.append({
                'clip':             str(fish_source[gi]),
                'tracklet_id':      int(fish_tid[gi]),
                'n_windows':        int(fish_wcnt[gi]),
                'n_peers_in_clip':  n,
                'peer_dissim':      float(peer_dissim[k]),
                'peer_rank':        int(ranks[k]),
                'is_top1_outlier':  bool(ranks[k] == 1),
                'clip_severity':    clip_severity,
                'rank1_dissim':     rank1_dissim,
            })

    if skipped_small:
        print(f"  {skipped_small} clip(s) skipped: fewer than {min_peers} fish")

    # ---- 排序輸出: 先 clip, 再 rank ----
    rows.sort(key=lambda r: (r['clip'], r['peer_rank']))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'clip_peer_ranking.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = build_markdown(rows, video_root, top_k, mild_thr, severe_thr)
    (out_dir / 'peer_report.md').write_text('\n'.join(md), encoding='utf-8')

    # ---- 統計 ----
    n_clips = len(set(r['clip'] for r in rows))
    top1_rows = [r for r in rows if r['is_top1_outlier']]
    n_severe = sum(1 for r in top1_rows if r['clip_severity'] == 'severe')
    n_mild = sum(1 for r in top1_rows if r['clip_severity'] == 'mild')
    n_normal = sum(1 for r in top1_rows if r['clip_severity'] == 'normal')

    # rank-1 dissim 分布 (幫助決定合理閾值)
    r1_arr = np.array([r['peer_dissim'] for r in top1_rows])
    p50, p75, p90, p95, p99 = np.percentile(r1_arr, [50, 75, 90, 95, 99])

    print(f"\nPeer ranking done.")
    print(f"  Total fish scored:             {len(rows)}")
    print(f"  Clips analyzed:                {n_clips}")
    print(f"\n  Rank-1 dissim distribution (across {n_clips} clips):")
    print(f"    min = {r1_arr.min():.4f}")
    print(f"    P50 = {p50:.4f}")
    print(f"    P75 = {p75:.4f}")
    print(f"    P90 = {p90:.4f}")
    print(f"    P95 = {p95:.4f}")
    print(f"    P99 = {p99:.4f}")
    print(f"    max = {r1_arr.max():.4f}")

    print(f"\n  Per-clip severity distribution:")
    print(f"    正常     (dissim <  {mild_thr}):     "
          f"{n_normal:4d} ({n_normal/n_clips*100:5.1f}%)")
    print(f"    輕度異常 (dissim ∈ [{mild_thr},{severe_thr})): "
          f"{n_mild:4d} ({n_mild/n_clips*100:5.1f}%)")
    print(f"    明顯異常 (dissim >= {severe_thr}):    "
          f"{n_severe:4d} ({n_severe/n_clips*100:5.1f}%)")

    # 閾值合理性檢查, 太不平衡就提示建議值
    if n_severe / n_clips < 0.01 and n_mild / n_clips < 0.05:
        print(f"\n  現有閾值 (mild={mild_thr}, severe={severe_thr}) "
              f"對此資料集偏嚴, 幾乎沒有 clip 被標記異常.")
        print(f"     建議依上方分布重跑 (讓輕度約佔 5-10%, 明顯約佔 1-2%):")
        print(f"       --mild_thr {p90:.3f} --severe_thr {p99:.3f}")
    elif n_severe / n_clips > 0.3:
        print(f"\n  現有閾值偏寬鬆, 超過 30% clip 被標記明顯異常.")
        print(f"     建議調高: --mild_thr {p75:.3f} --severe_thr {p95:.3f}")

    print(f"\n  Output → {out_dir}")
    print(f"    clip_peer_ranking.csv        ({len(rows)} rows)")
    print(f"    peer_report.md               (top-{top_k} outliers)")


# ============================================================
# Markdown report
# ============================================================
def _clip_to_video(clip, video_root):
    stem = clip[:-4] if clip.endswith('.csv') else clip
    name = stem if stem.endswith('.mp4') else stem + '.mp4'
    return (Path(video_root) / name).as_posix() if video_root else name


def build_markdown(rows, video_root, top_k, mild_thr, severe_thr):
    n_fish = len(rows)
    clips = sorted(set(r['clip'] for r in rows))
    n_clips = len(clips)
    top1 = [r for r in rows if r['is_top1_outlier']]
    top1.sort(key=lambda r: -r['peer_dissim'])

    n_severe = sum(1 for r in top1 if r['clip_severity'] == 'severe')
    n_mild = sum(1 for r in top1 if r['clip_severity'] == 'mild')
    n_normal = sum(1 for r in top1 if r['clip_severity'] == 'normal')

    md = [
        '# Fish Peer Comparison Report (Per-clip Outlier)',
        '',
        '## Summary',
        '',
        f'- Total fish scored: **{n_fish}**',
        f'- Clips analyzed: **{n_clips}**',
        '',
        '## Severity distribution (per-clip, by rank-1 fish)',
        '',
        '| 狀態 | 判斷條件 (rank-1 差異度) | Clip 數 | 佔比 |',
        '|---|---|---|---|',
        f'| 正常 | dissim < {mild_thr} | {n_normal} '
        f'| {n_normal/n_clips*100:.1f}% |',
        f'| 輕度異常 | {mild_thr} ≤ dissim < {severe_thr} | {n_mild} '
        f'| {n_mild/n_clips*100:.1f}% |',
        f'| 明顯異常 | dissim ≥ {severe_thr} | {n_severe} '
        f'| {n_severe/n_clips*100:.1f}% |',
        '',
        '## System response',
        '',
        '- **正常** → 影片內個體行為一致, 無預警',
        '- **輕度異常** → 提示: 某個體與同儕略有差異, 建議留意',
        '- **明顯異常** → 預警: 某個體明顯偏離同儕行為, 建議捕撈檢視',
        '',
    ]

    # 明顯異常清單
    severe_rows = [r for r in top1 if r['clip_severity'] == 'severe']
    md.append(f'## 明顯異常清單 (severe, n={len(severe_rows)})')
    md.append('')
    md.append('| Clip | Tracklet | n_peers | dissim | Video |')
    md.append('|---|---|---|---|---|')
    if not severe_rows:
        md.append('| (無) | | | | |')
    for r in severe_rows[:top_k]:
        md.append(
            f"| `{r['clip']}` | {r['tracklet_id']} "
            f"| {r['n_peers_in_clip']} | {r['peer_dissim']:.4f} "
            f"| `{_clip_to_video(r['clip'], video_root)}` |"
        )
    if len(severe_rows) > top_k:
        md.append(f'| ...共 {len(severe_rows)} 筆, 只顯示前 {top_k}, '
                  f'完整見 CSV | | | | |')
    md.append('')

    # 輕度異常清單
    mild_rows = [r for r in top1 if r['clip_severity'] == 'mild']
    md.append(f'## 輕度異常清單 (mild, n={len(mild_rows)})')
    md.append('')
    md.append('| Clip | Tracklet | n_peers | dissim | Video |')
    md.append('|---|---|---|---|---|')
    if not mild_rows:
        md.append('| (無) | | | | |')
    for r in mild_rows[:top_k]:
        md.append(
            f"| `{r['clip']}` | {r['tracklet_id']} "
            f"| {r['n_peers_in_clip']} | {r['peer_dissim']:.4f} "
            f"| `{_clip_to_video(r['clip'], video_root)}` |"
        )
    if len(mild_rows) > top_k:
        md.append(f'| ...共 {len(mild_rows)} 筆, 只顯示前 {top_k}, '
                  f'完整見 CSV | | | | |')
    md.append('')

    md.append('## Full ranking')
    md.append('')
    md.append('CSV `clip_peer_ranking.csv` 有完整資料 — 每 clip 每條魚一列, '
              '含 rank 2/3/... 的次要 outlier + severity 欄位.')
    return md


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features_dir', required=True,
                    help='features/ 根目錄 (build_pseudo_labeled_dataset.py 的輸出)')
    ap.add_argument('--output_dir', required=True,
                    help='輸出 dir (會產生 clip_peer_ranking.csv + peer_report.md)')
    ap.add_argument('--video_root', default=None,
                    help='影片根目錄, 用來組 video 完整路徑')
    ap.add_argument('--top_k', type=int, default=20,
                    help='markdown 每個等級印前 K 名 (預設 20)')

    ap.add_argument('--splits', nargs='+', default=None,
                    help='只讀指定 split (預設: 自動掃 train/val/test 全部)')
    ap.add_argument('--clusters', nargs='+', default=None,
                    help='只讀指定 cluster id (預設: 全部)')

    ap.add_argument('--min_peers', type=int, default=3,
                    help='一段影片至少要 N 條魚才做同儕比較 (預設 3)')
    ap.add_argument('--mild_thr', type=float, default=0.3,
                    help='輕度異常門檻 (預設 0.3)')
    ap.add_argument('--severe_thr', type=float, default=0.5,
                    help='明顯異常門檻 (預設 0.5)')

    args = ap.parse_args()

    peer_ranking(args.features_dir, args.output_dir, args.video_root,
                 args.top_k,
                 min_peers=args.min_peers,
                 mild_thr=args.mild_thr, severe_thr=args.severe_thr,
                 splits=args.splits, clusters=args.clusters)


if __name__ == '__main__':
    main()