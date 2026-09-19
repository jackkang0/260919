"""
產生驗證用序列：6 frames，模擬 8-bit dither +1，並注入三種已知缺陷
以確認 KPI 引擎能分辨「正常 dither 噪聲」與「真實 flicker」。
"""
import os
import numpy as np
import cv2

np.random.seed(7)
H, W, N = 128, 192, 6
out = "/home/claude/kpi_seq"
os.makedirs(out, exist_ok=True)

# 基底：水平漸層 + 幾個色塊
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.stack([60 + xx * 0.6, 90 + yy * 0.5,
                 120 + (xx + yy) * 0.2], axis=-1)      # RGB

# 缺陷區域定義 (y0,y1,x0,x1)
REGIONS = {
    "A_strong_RB_flicker": (16, 32, 16, 48),    # R/B 大幅雙態跳動 -> 模擬 mode 切換
    "B_mid_G_flicker":     (60, 76, 90, 122),   # G 中幅跳動
    "C_single_spike":      (100, 108, 150, 178),# 只有第 4 幀突跳一次
}

for i in range(N):
    f = base.copy()
    # 全圖 dither +1：0/+1 的隨機擾動（符合 8 bits dither +1）
    f += np.random.randint(0, 2, size=f.shape).astype(np.float32)

    # A: R,B 在 ±60 雙態交錯
    y0, y1, x0, x1 = REGIONS["A_strong_RB_flicker"]
    s = 60.0 if i % 2 == 0 else -60.0
    f[y0:y1, x0:x1, 0] += s
    f[y0:y1, x0:x1, 2] -= s

    # B: G 在 ±20 交錯
    y0, y1, x0, x1 = REGIONS["B_mid_G_flicker"]
    f[y0:y1, x0:x1, 1] += 20.0 if i % 2 == 0 else -20.0

    # C: 只有 frame index 3 出現 +90 單次突跳
    if i == 3:
        y0, y1, x0, x1 = REGIONS["C_single_spike"]
        f[y0:y1, x0:x1, :] += 90.0

    cv2.imwrite(os.path.join(out, f"img_{i+1:04d}.bmp"),
                np.clip(f, 0, 255).astype(np.uint8)[:, :, ::-1])

print(f"已產生 {N} frames ({W}x{H}) 於 {out}")
for k, v in REGIONS.items():
    print(f"  {k}: rows {v[0]}-{v[1]}, cols {v[2]}-{v[3]}")
