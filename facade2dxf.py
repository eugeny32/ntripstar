#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facade2dxf — прототип конвейера «облако точек LAS -> DXF-подоснова фасада».

Что делает:
  1. Читает LAS/LAZ (laspy).
  2. Находит доминирующую вертикальную плоскость (фасад) RANSAC-подгонкой
     прямой по проекции точек на план XY (ось Z считается вертикальной).
  3. Вырезает слой точек вдоль плоскости и проецирует его в 2D-координаты
     фасада: u — вдоль стены, v — высота.
  4. Строит растровую карту плотности (ортоизображение фасада).
  5. Находит проёмы как «дыры» в плотности внутри контура фасада;
     прямоугольные дыры уходят в слой OPENINGS, нерегулярные — в GAPS_REVIEW
     (вероятные пропуски сканирования, требуют проверки человеком).
  6. Пишет DXF (метры): FACADE_OUTLINE, OPENINGS, GAPS_REVIEW,
     а рядом — ортофото PNG, превью с детекцией и JSON с привязкой к миру.

Запуск:
  python3 facade2dxf.py вход.las -o фасад.dxf
Подробнее: python3 facade2dxf.py --help
"""

import argparse
import json
import sys

import cv2
import ezdxf
import laspy
import numpy as np


# ----------------------------------------------------------------------------
# 1. Чтение облака
# ----------------------------------------------------------------------------

def read_las(path, max_points):
    las = laspy.read(path)
    pts = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    try:
        intensity = np.asarray(las.intensity, dtype=np.float64)
    except AttributeError:
        intensity = np.zeros(len(pts))
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, intensity = pts[idx], intensity[idx]
    return pts, intensity


# ----------------------------------------------------------------------------
# 2. Поиск плоскости фасада (RANSAC по прямой в плане)
# ----------------------------------------------------------------------------

def ransac_facade_line(xy, tol, iters=1000, seed=0):
    """Возвращает (origin, direction, inlier_mask) доминирующей прямой в плане."""
    rng = np.random.default_rng(seed)
    n = len(xy)
    sample = xy if n <= 100_000 else xy[rng.choice(n, 100_000, replace=False)]
    best_count, best = -1, None
    for _ in range(iters):
        i, j = rng.choice(len(sample), 2, replace=False)
        d = sample[j] - sample[i]
        norm = np.linalg.norm(d)
        if norm < 0.5:  # слишком близкие точки дают шумную прямую
            continue
        d = d / norm
        normal = np.array([-d[1], d[0]])
        dist = np.abs((sample - sample[i]) @ normal)
        count = int((dist < tol).sum())
        if count > best_count:
            best_count, best = count, (sample[i], d)
    if best is None:
        raise RuntimeError("Не удалось найти плоскость фасада: слишком мало точек")

    origin, direction = best
    normal = np.array([-direction[1], direction[0]])
    # уточнение: МНК по инлайерам полного облака
    dist_all = np.abs((xy - origin) @ normal)
    inl = dist_all < tol
    c = xy[inl].mean(axis=0)
    vt = np.linalg.svd(xy[inl] - c, full_matrices=False)[2]
    direction = vt[0] / np.linalg.norm(vt[0])
    normal = np.array([-direction[1], direction[0]])
    inl = np.abs((xy - c) @ normal) < tol
    return c, direction, inl


# ----------------------------------------------------------------------------
# 3-4. Проекция на фасад и растеризация
# ----------------------------------------------------------------------------

def project_to_facade(pts, origin_xy, direction, slice_tol):
    normal = np.array([-direction[1], direction[0]])
    d = (pts[:, :2] - origin_xy) @ normal
    keep = np.abs(d) < slice_tol
    u = (pts[keep, :2] - origin_xy) @ direction
    v = pts[keep, 2]
    return u, v, keep


def rasterize(u, v, res):
    u0, v0 = u.min(), v.min()
    w = int(np.ceil((u.max() - u0) / res)) + 1
    h = int(np.ceil((v.max() - v0) / res)) + 1
    iu = ((u - u0) / res).astype(np.int32)
    iv = ((v - v0) / res).astype(np.int32)
    density = np.zeros((h, w), dtype=np.float32)
    np.add.at(density, (iv, iu), 1.0)
    return density, (u0, v0)


# ----------------------------------------------------------------------------
# 5. Детекция контура фасада, проёмов и пропусков
# ----------------------------------------------------------------------------

def detect(density, res, min_opening_m2, min_rectangularity):
    occ = (density > 0).astype(np.uint8)
    # мостим разрывы между соседними точками скана
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # оставляем крупнейшую связную компоненту — сам фасад
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(occ)
    if ncomp < 2:
        raise RuntimeError("Пустая карта плотности — проверьте параметры среза")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    occ = (labels == main).astype(np.uint8)

    # дыры = не-точки, не достижимые от границы кадра
    inv = (occ == 0).astype(np.uint8)
    nholes, hlabels, hstats, _ = cv2.connectedComponentsWithStats(inv)
    border = set(np.unique(np.concatenate([
        hlabels[0], hlabels[-1], hlabels[:, 0], hlabels[:, -1]])))

    filled = occ.copy()
    openings, gaps = [], []
    min_area_px = min_opening_m2 / (res * res)
    for k in range(1, nholes):
        if k in border:
            continue
        x, y, w, h, area = hstats[k]
        filled[hlabels == k] = 1
        if area < min_area_px:
            continue
        rect_m = (x * res, y * res, w * res, h * res)  # в координатах фасада (u,v) от (0,0) растра
        # шум точек делает края дыры рваными — замыкаем маску, чтобы
        # честный прямоугольник не терял прямоугольность из-за зазубрин
        hole = (hlabels[y:y + h, x:x + w] == k).astype(np.uint8)
        hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        rectangularity = float(hole.sum()) / float(w * h)
        (openings if rectangularity >= min_rectangularity else gaps).append(rect_m)

    # внешний контур фасада
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_px = max(contours, key=cv2.contourArea)
    eps = max(2.0, 0.03 / res * 0.5)
    outline_px = cv2.approxPolyDP(outline_px, eps, True).reshape(-1, 2)
    outline_m = outline_px.astype(np.float64) * res
    return outline_m, openings, gaps, filled


# ----------------------------------------------------------------------------
# 6. Вывод DXF, ортофото, превью
# ----------------------------------------------------------------------------

def write_dxf(path, outline, openings, gaps, offset_uv):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 6  # метры
    msp = doc.modelspace()
    for name, color in [("FACADE_OUTLINE", 7), ("OPENINGS", 5), ("GAPS_REVIEW", 1)]:
        doc.layers.add(name, color=color)

    u0, v0 = offset_uv
    msp.add_lwpolyline(
        [(u0 + p[0], v0 + p[1]) for p in outline],
        close=True, dxfattribs={"layer": "FACADE_OUTLINE"})
    for layer, rects in [("OPENINGS", openings), ("GAPS_REVIEW", gaps)]:
        for (x, y, w, h) in rects:
            msp.add_lwpolyline(
                [(u0 + x, v0 + y), (u0 + x + w, v0 + y),
                 (u0 + x + w, v0 + y + h), (u0 + x, v0 + y + h)],
                close=True, dxfattribs={"layer": layer})
    doc.saveas(path)


def save_images(density, outline, openings, gaps, res, ortho_path, preview_path):
    img = np.clip(density / max(np.percentile(density[density > 0], 90), 1) * 255,
                  0, 255).astype(np.uint8)
    img = cv2.flip(img, 0)  # v растёт вверх, а строки изображения — вниз
    cv2.imwrite(ortho_path, img)

    prev = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hpx = img.shape[0]

    def to_px(x, y):
        return int(round(x / res)), hpx - 1 - int(round(y / res))

    pts = np.array([to_px(x, y) for x, y in outline], np.int32)
    cv2.polylines(prev, [pts], True, (0, 255, 0), 2)
    for rects, color in [(openings, (255, 128, 0)), (gaps, (0, 0, 255))]:
        for (x, y, w, h) in rects:
            p1, p2 = to_px(x, y), to_px(x + w, y + h)
            cv2.rectangle(prev, p1, p2, color, 2)
    cv2.imwrite(preview_path, prev)


def ensure_writable(out_path, las_path):
    """Проверяет, что выходной файл можно создать.

    Защитник Windows (контролируемый доступ к папкам) и OneDrive блокируют
    запись в Documents с невнятной ошибкой FileNotFoundError. В этом случае
    переносим вывод в папку исходного LAS.
    """
    import os
    try:
        with open(out_path, "w"):
            pass
        os.remove(out_path)
        return out_path
    except OSError:
        fallback = os.path.join(os.path.dirname(os.path.abspath(las_path)),
                                os.path.basename(out_path))
        print(f"  ВНИМАНИЕ: нет записи в '{out_path}' (антивирус/OneDrive "
              f"блокирует папку?) — сохраняю рядом с LAS: {fallback}")
        return fallback


def main():
    ap = argparse.ArgumentParser(description="LAS -> DXF-подоснова фасада")
    ap.add_argument("las", help="входной файл LAS/LAZ")
    ap.add_argument("-o", "--out", default="facade.dxf", help="выходной DXF")
    ap.add_argument("--res", type=float, default=0.02,
                    help="разрешение растра, м/пиксель (по умолчанию 0.02)")
    ap.add_argument("--plane-tol", type=float, default=0.08,
                    help="допуск RANSAC при поиске плоскости фасада, м")
    ap.add_argument("--slice", type=float, default=0.15,
                    help="полутолщина среза вдоль фасада, м")
    ap.add_argument("--min-opening", type=float, default=0.15,
                    help="минимальная площадь проёма, м²")
    ap.add_argument("--rect", type=float, default=0.85,
                    help="порог прямоугольности: выше — проём, ниже — пропуск данных")
    ap.add_argument("--max-points", type=int, default=15_000_000,
                    help="прореживание облака до N точек (по умолчанию 15 млн)")
    args = ap.parse_args()

    print(f"Читаю {args.las} ...")
    pts, _ = read_las(args.las, args.max_points)
    print(f"  точек: {len(pts):,}")

    print("Ищу плоскость фасада (RANSAC)...")
    origin_xy, direction, inl = ransac_facade_line(pts[:, :2], args.plane_tol)
    print(f"  инлайеров: {int(inl.sum()):,} ({inl.mean() * 100:.0f}% облака), "
          f"направление в плане: ({direction[0]:+.3f}, {direction[1]:+.3f})")

    u, v, _ = project_to_facade(pts, origin_xy, direction, args.slice)

    # при редком облаке мелкий растр разваливается на несвязные точки —
    # подбираем шаг так, чтобы на ячейку приходилось ~3 точки стены
    area = (u.max() - u.min()) * (v.max() - v.min())
    res = args.res
    auto_res = float(np.sqrt(3.0 * area / max(len(u), 1)))
    if auto_res > res * 1.3:
        res = round(auto_res, 3)
        print(f"  ВНИМАНИЕ: точек в срезе мало ({len(u):,} на {area:.0f} м²) — "
              f"растр укрупнён до {res} м/px.\n"
              f"  Для детализации дайте больше точек: --max-points 30000000, "
              f"либо вырежьте один фасад из облака (ReCap/CloudCompare).")

    density, (u0, v0) = rasterize(u, v, res)
    print(f"Растр {density.shape[1]}x{density.shape[0]} px @ {res} м/px; "
          f"фасад ~{(u.max() - u.min()):.1f} x {(v.max() - v.min()):.1f} м")

    outline, openings, gaps, _ = detect(density, res,
                                        args.min_opening, args.rect)
    print(f"Найдено проёмов: {len(openings)}, зон на проверку (пропуски): {len(gaps)}")

    out = ensure_writable(args.out, args.las)
    stem = out.rsplit(".", 1)[0]
    write_dxf(out, outline, openings, gaps, (0.0, 0.0))
    save_images(density, outline, openings, gaps, res,
                stem + "_ortho.png", stem + "_preview.png")
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "las": args.las,
            "world_origin_xy": [float(origin_xy[0]) + float(u0) * float(direction[0]),
                                float(origin_xy[1]) + float(u0) * float(direction[1])],
            "facade_direction_xy": [float(direction[0]), float(direction[1])],
            "v_zero_world_z": float(v0),
            "note": "точка DXF (u,v) -> мир: origin + u*direction, Z = v_zero + v",
        }, f, ensure_ascii=False, indent=2)

    print(f"Готово: {out}, {stem}_ortho.png, {stem}_preview.png, {stem}_meta.json")


if __name__ == "__main__":
    sys.exit(main())
