"""
association_head.py
===================
用於多物件追蹤的學習式特徵投影模組。

將偵測器輸出的物件特徵投影到新的特徵空間，讓同一條魚在不同幀的特徵
彼此靠近、不同魚的特徵彼此遠離。相較於傳統 ByteTrack 直接用原始特徵
計算相似度，這樣的學習式投影能在高密度魚群中提供更強的個體區辨能力。

架構:
    輸入:     RF-DETR decoder 輸出的 query embedding (256 維)
    輸出:     投影後的特徵 (128 維, 經 L2 normalize)
    訓練損失: InfoNCE，搭配批次內打散的負樣本

訓練監督來源:
    以 SAM 3 產生的高信心度追蹤軌跡作為偽標籤。
    同一條軌跡內的兩幀特徵視為正樣本對；
    同批次內其他軌跡的特徵作為負樣本。

推論時，tracker 將原始的 256 維 query embedding 經此模組投影到 128 維，
在該空間進行最近鄰匹配。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociationHead(nn.Module):
    """
    將 256 維物件特徵投影到 128 維的兩層 MLP。
    輸出會經過 L2 normalize，便於後續以餘弦相似度進行比較。

    Args:
        in_dim:     輸入維度 (256 對應 RF-DETR Nano decoder 的 hidden size)。
        proj_dim:   投影後的輸出維度。
        hidden_dim: 中間層維度，預設與 in_dim 相同。
    """

    def __init__(self, in_dim: int = 256, proj_dim: int = 128, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.in_dim   = in_dim
        self.proj_dim = proj_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) 原始物件特徵。
        Returns:
            (B, proj_dim) 經 L2 normalize 的投影特徵。
        """
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class InfoNCELoss(nn.Module):
    """
    對比式學習使用的 InfoNCE 損失，搭配批次內打散的負樣本。

    每個批次由 (anchor, positive) 配對組成，兩者皆來自同一條軌跡的不同幀。
    對任一 anchor 而言，批次內其他樣本的 positive 皆視為負樣本
    (SimCLR / CLIP 常用做法)。

    損失公式 (對 anchor i):
        L_i = -log [ exp(sim(z_i, z_i+) / tau) / sum_j exp(sim(z_i, z_j) / tau) ]
        其中 j 遍歷整個批次內所有 positive。

    Args:
        temperature: Softmax 溫度參數 (0.07 參照 CLIP / SimCLR 的設定)。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> dict:
        """
        Args:
            z_anchor:   (B, D) 經 L2 normalize 的 anchor 特徵。
            z_positive: (B, D) 經 L2 normalize 的 positive 特徵。
                        z_positive[i] 為 z_anchor[i] 的正樣本配對。
                        z_positive[j] (j != i) 作為批次內打散負樣本。

        Returns:
            dict，包含:
              'loss':    純量損失值
              'acc':     Top-1 配對準確率
              'pos_sim': 正樣本對的平均餘弦相似度
              'neg_sim': 負樣本對的平均餘弦相似度
        """
        B = z_anchor.shape[0]
        # logits[i, j] = sim(z_anchor[i], z_positive[j]) / tau
        logits = (z_anchor @ z_positive.T) / self.temperature   # (B, B)
        labels = torch.arange(B, device=z_anchor.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc  = (pred == labels).float().mean()
            mask = torch.eye(B, dtype=torch.bool, device=z_anchor.device)
            raw_sim = z_anchor @ z_positive.T   # 未經溫度縮放的原始相似度
            pos_sim = raw_sim[mask].mean()
            neg_sim = raw_sim[~mask].mean()

        return {
            'loss':    loss,
            'acc':     acc,
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
        }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== association_head smoke test ===\n")
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    head = AssociationHead(in_dim=256, proj_dim=128).to(device)
    x = torch.randn(8, 256, device=device)
    z = head(x)
    print(f"\nTest 1: forward")
    print(f"  input  shape: {tuple(x.shape)}")
    print(f"  output shape: {tuple(z.shape)}")
    print(f"  output norm:  {z.norm(dim=-1).mean():.4f}  (expect ~1.0)")

    loss_fn = InfoNCELoss(temperature=0.07)
    z_a = head(torch.randn(32, 256, device=device))
    z_p = head(torch.randn(32, 256, device=device))
    out = loss_fn(z_a, z_p)
    print(f"\nTest 2: random pairs (chance-level)")
    print(f"  loss:    {out['loss']:.4f}  (expect ~log(32)={math.log(32):.4f})")
    print(f"  acc:     {out['acc']:.4f}  (expect ~1/32={1/32:.4f})")
    print(f"  pos_sim: {out['pos_sim']:.4f}  neg_sim: {out['neg_sim']:.4f}")

    z_same = head(torch.randn(32, 256, device=device))
    out = loss_fn(z_same, z_same)
    print(f"\nTest 3: identical pairs (should be near-perfect)")
    print(f"  loss:    {out['loss']:.4f}  acc: {out['acc']:.4f}")
    print(f"  pos_sim: {out['pos_sim']:.4f}  neg_sim: {out['neg_sim']:.4f}")

    head.zero_grad()
    out['loss'].backward()
    has_grad = all(p.grad is not None for p in head.parameters())
    print(f"\nTest 4: gradient flows: {has_grad}")

    print("\nAll smoke tests passed")