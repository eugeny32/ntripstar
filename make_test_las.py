#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетического LAS для проверки facade2dxf.py:
стена 20x9 м, повёрнутая в плане, 3 этажа по 6 окон 1.2x1.5 м, дверь,
гауссов шум точек, зона окклюзии (неполнота скана) и мусор перед фасадом.
"""
import laspy
import numpy as np

rng = np.random.default_rng(42)

W, H = 20.0, 9.0
DENSITY = 1500  # точек/м²

n = int(W * H * DENSITY)
u = rng.uniform(0, W, n)
v = rng.uniform(0, H, n)

# окна: 3 этажа x 6 окон 1.2x1.5, подоконники на 1.0/4.0/7.0 м
windows = [(1.5 + i * 3.1, floor_v, 1.2, 1.5)
           for i in range(6) for floor_v in (1.0, 4.0, 7.0)]
door = (9.2, 0.0, 1.6, 2.4)
holes = windows + [door]

keep = np.ones(n, bool)
for (x, y, w, h) in holes:
    keep &= ~((u > x) & (u < x + w) & (v > y) & (v < y + h))

# неполнота скана: эллипс окклюзии (дерево закрыло часть стены)
occ = ((u - 16.5) / 1.7) ** 2 + ((v - 2.5) / 2.1) ** 2 < 1.0
keep &= ~occ
u, v = u[keep], v[keep]

# шум стены + разворот в плане на 37°
depth = rng.normal(0, 0.012, len(u))
ang = np.deg2rad(37)
d = np.array([np.cos(ang), np.sin(ang)])
nrm = np.array([-d[1], d[0]])
base = np.array([120.0, 340.0])
xy = base + np.outer(u, d) + np.outer(depth, nrm)
z = 55.0 + v

# мусор перед фасадом: кусты/земля (15% точек)
m = int(0.15 * len(u))
cu = rng.uniform(0, W, m)
cd = rng.uniform(0.5, 4.0, m)
cxy = base + np.outer(cu, d) + np.outer(cd, nrm)
cz = 55.0 + np.abs(rng.normal(0, 1.2, m))

X = np.concatenate([xy[:, 0], cxy[:, 0]])
Y = np.concatenate([xy[:, 1], cxy[:, 1]])
Z = np.concatenate([z, cz])

header = laspy.LasHeader(point_format=0, version="1.2")
header.offsets = [X.min(), Y.min(), Z.min()]
header.scales = [0.001, 0.001, 0.001]
las = laspy.LasData(header)
las.x, las.y, las.z = X, Y, Z
las.intensity = rng.integers(100, 2000, len(X), dtype=np.uint16)
las.write("test_facade.las")
print(f"test_facade.las: {len(X):,} точек "
      f"(стена {len(u):,} + мусор {m:,}), окон {len(windows)}, дверь 1, окклюзия 1")
