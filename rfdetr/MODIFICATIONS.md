# Modifications to RF-DETR

This directory contains a modified version of [RF-DETR](https://github.com/roboflow/rf-detr), adapted to serve as the detection backbone of a shared visual perception module for underwater fish behavior recognition.

## Base version

Based on `rfdetr==1.7.1` (PyPI release).

## Summary

The modifications in this project enable **a single DINOv2-S/14 backbone forward pass to serve three downstream tasks simultaneously**: object detection, multi-object tracking (via Re-ID features), and temporal behavior classification. The original RF-DETR is optimized for detection alone; our changes expose intermediate backbone features for the tracking and classification tasks without requiring additional backbone computation.

Two categories of changes are involved:

1. **Backbone configuration** — using vanilla DINOv2-S instead of the RF-DETR Nano default. This uses an existing code path in RF-DETR, selected via constructor arguments; no source code change.
2. **CLS token exposure** — a source code modification to `rfdetr/models/backbone/dinov2.py` that exposes the DINOv2 CLS token as a module attribute for downstream retrieval.

---

## 1. Backbone configuration: vanilla DINOv2-S/14

### What

The original RF-DETR Nano uses `dinov2_windowed_registers_small` as its default backbone, combining DINOv2 with register tokens and windowed self-attention. This project instead uses the vanilla `dinov2_small` option, routing the forward pass through `rfdetr/models/backbone/dinov2.py` with `use_windowed_attn=False`.

### Why

The windowed attention and register tokens in the default backbone are design choices optimized for standalone detection efficiency. In this project, the DINOv2 backbone is a **shared feature source** for:

- Detection (via RF-DETR decoder)
- Re-ID features for multi-object tracking (via a learned Association Head)
- Global scene features for temporal behavior classification (via ViT classifier)

The latter two tasks rely on full global semantic coherence of the backbone output. Using vanilla DINOv2-S preserves this coherence, at the cost of slightly higher detection-only computation — a trade-off this project deliberately accepts for multi-task feature quality.

### Configuration

| Parameter | RF-DETR Nano default | This project |
|---|---|---|
| `encoder` | `dinov2_windowed_registers_small` | `dinov2_small` |
| `patch_size` | 16 | 14 |
| `num_windows` | 2 | 1 |
| `resolution` | 384 | 378 (= 14 × 27) |
| `positional_encoding_size` | 24 | 27 |
| `pretrain_weights` | `rf-detr-nano.pth` | None (see note) |

**Note on pretrained weights**: The RF-DETR Nano pretrained weights are incompatible with the altered patch size (14 vs. 16), so they cannot be loaded. The DINOv2-S backbone weights are instead loaded from Hugging Face Hub (`facebook/dinov2-small`), while the projector, decoder, and detection head are trained from scratch on the fish dataset.

---

## 2. CLS token exposure on DINOv2 backbone

### What

Modified `rfdetr/models/backbone/dinov2.py` so that the DINOv2 backbone module stores its last-layer CLS token as `self.last_cls` during the forward pass. Downstream code can then retrieve the global scene feature after a backbone forward call via:

```python
cls_token = lwdetr.backbone[0].encoder.last_cls  # shape: (B, 384)
```

### Why

The DINOv2 CLS token aggregates a semantic summary of the entire input frame, and is used in this project as the **global scene feature** input to the ViT behavior classifier (see thesis §3.3.1 and §3.5). Without this modification, retrieving the CLS token would require either:

- A second forward pass through the backbone — wasteful, and defeats the shared-backbone design, or
- Rewriting large portions of the RF-DETR forward chain to propagate the CLS token through intermediate return values.

The chosen approach — exposing CLS via a module attribute — adds a single tensor assignment per forward call, with negligible compute and memory overhead.

### Implementation

The modification changes DINOv2's forward call to request hidden states, then extracts and stores the CLS token:

```python
# Before
x = self.encoder(x)

# After
x = self.encoder(x, output_hidden_states=True)
# ... extract CLS token from the last hidden state ...
self.last_cls = x.hidden_states[-1][:, 0, :].detach()  # (B, 384)
```

The modification preserves the original return signature (list of feature maps for the detection decoder), so no other RF-DETR code is affected.

### Files modified

- `rfdetr/models/backbone/dinov2.py` — `DinoV2.forward()`

---

## License

This modified version retains the original Apache License 2.0 of RF-DETR. See `LICENSE` in this directory.

Original project: https://github.com/roboflow/rf-detr