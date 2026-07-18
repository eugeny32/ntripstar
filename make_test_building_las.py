#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Синтетическое здание для проверки мультифасадного режима facade2dxf.py:
прямоугольник 20x12 м высотой 9 м, 4 стены с окнами; стена №3 глухая
(имитация фасада, закрытого плакатом). Плюс шум, земля и мусор.
"""
import laspy
import numpy as np

rng = np.random.default_rng(7)

H, DENSITY = 9.0, 1200
ang = np.deg2rad(23)
R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
base = np.array([500.0, 800.0])

# углы здания 20x12 в локальных осях
corners = [np.array(c) for c in [(0, 0), (20, 0), (20, 12), (0, 12)]]

def wall(a, b, blank=False):
    d = (b - a).astype(float)
    L = np.linalg.norm(d)
    d /= L
    n = int(L * H * DENSITY)
    u = rng.uniform(0, L, n)
    v = rng.uniform(0, H, n)
    if not blank:  # окна: 2 этажа, шаг 3 м
        keep = np.ones(n, bool)
        nw = int((L - 2.0) // 3.0)
        for i in range(nw):
            for fv in (1.2, 5.2):
                x = 1.5 + i * 3.0
                keep &= ~((u > x) & (u < x + 1.2) & (v > fv) & (v < fv + 1.5))
        u, v = u[keep], v[keep]
    depth = rng.normal(0, 0.012, len(u))
    nrm = np.array([-d[1], d[0]])
    xy = a + np.outer(u, d) + np.outer(depth, nrm)
    return xy, v

walls_xy, walls_z = [], []
for i, blank in zip(range(4), [False, False, True, False]):
    xy, v = wall(corners[i], corners[(i + 1) % 4], blank)
    walls_xy.append(xy)
    walls_z.append(v)

xy = np.vstack(walls_xy)
z = np.concatenate(walls_z)

# земля вокруг + мусор
m = int(0.3 * len(z))
gxy = np.column_stack([rng.uniform(-8, 28, m), rng.uniform(-8, 20, m)])
gz = np.abs(rng.normal(0, 0.35, m))

XY = np.vstack([xy, gxy]) @ R.T + base
Z = np.concatenate([z, gz]) + 120.0

header = laspy.LasHeader(point_format=0, version="1.2")
header.offsets = [XY[:, 0].min(), XY[:, 1].min(), Z.min()]
header.scales = [0.001, 0.001, 0.001]
las = laspy.LasData(header)
las.x, las.y, las.z = XY[:, 0], XY[:, 1], Z
las.intensity = rng.integers(100, 2000, len(Z), dtype=np.uint16)
las.write("test_building.las")
print(f"test_building.las: {len(Z):,} точек, 4 стены (стена f. 20x12, "
      f"одна глухая — «плакат»), земля и мусор")
