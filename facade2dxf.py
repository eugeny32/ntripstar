#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facade2dxf — прототип конвейера «облако точек LAS -> DXF-подоснова фасадов».

Что делает:
  1. Читает LAS/LAZ (laspy).
  2. Итеративно находит вертикальные плоскости (фасады) RANSAC-подгонкой
     прямых по проекции облака на план XY (ось Z считается вертикальной).
     Каждая плоскость разбивается на отдельные фасады по разрывам в плане
     (разные здания на одной линии не склеиваются).
  3. Каждый фасад: срез точек вдоль плоскости -> проекция в 2D (u — вдоль
     стены, v — высота) -> карта плотности -> детекция проёмов.
     Прямоугольные «дыры» уходят в слой OPENINGS, нерегулярные — в
     GAPS_REVIEW (вероятные пропуски сканирования — проверить вручную).
  4. На каждый фасад пишутся: DXF в метрах (FACADE_OUTLINE / OPENINGS /
     GAPS_REVIEW), ортофото PNG, превью с детекцией, JSON с привязкой к миру.
     Плюс общий план сцены scene_plan.png с номерами найденных фасадов.

Запуск:
  python3 facade2dxf.py вход.las -o фасад.dxf            # все фасады: фасад_f01.dxf, ...
  python3 facade2dxf.py вход.las -o фасад.dxf --facades 1  # только доминирующий
Подробнее: python3 facade2dxf.py --help
"""

import argparse
import json
import os
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

def filter_vertical(pts, cell=0.5, min_extent=2.0):
    """Оставляет точки вертикальных структур (стены), убирая землю и мусор.

    План разбивается на ячейки cell x cell; остаются точки ячеек, где
    перепад высот не меньше min_extent — у земли и кустов он мал.
    """
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(ij, axis=0, return_inverse=True)
    ncell = inv.max() + 1
    zmin = np.full(ncell, np.inf)
    zmax = np.full(ncell, -np.inf)
    np.minimum.at(zmin, inv, pts[:, 2])
    np.maximum.at(zmax, inv, pts[:, 2])
    return (zmax - zmin)[inv] >= min_extent


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


def split_by_gaps(u_values, gap, min_len, min_points):
    """Разбивает точки прямой на отдельные фасады по разрывам вдоль стены.

    Возвращает список (u_min, u_max, n_points) непрерывных участков.
    """
    u_sorted = np.sort(u_values)
    breaks = np.where(np.diff(u_sorted) > gap)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(u_sorted) - 1]])
    segments = []
    for s, e in zip(starts, ends):
        lo, hi, n = u_sorted[s], u_sorted[e], e - s + 1
        if hi - lo >= min_len and n >= min_points:
            segments.append((float(lo), float(hi), int(n)))
    segments.sort(key=lambda t: -t[2])
    return segments


# ----------------------------------------------------------------------------
# 3-4. Растеризация
# ----------------------------------------------------------------------------

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
    main_c = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    occ = (labels == main_c).astype(np.uint8)

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
        rect_m = (x * res, y * res, w * res, h * res)
        # шум точек делает края дыры рваными — замыкаем маску, чтобы
        # честный прямоугольник не терял прямоугольность из-за зазубрин
        hole = (hlabels[y:y + h, x:x + w] == k).astype(np.uint8)
        hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        rectangularity = float(hole.sum()) / float(w * h)
        (openings if rectangularity >= min_rectangularity else gaps).append(rect_m)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_px = max(contours, key=cv2.contourArea)
    eps = max(2.0, 0.03 / res * 0.5)
    outline_px = cv2.approxPolyDP(outline_px, eps, True).reshape(-1, 2)
    outline_m = outline_px.astype(np.float64) * res
    return outline_m, openings, gaps, filled


# ----------------------------------------------------------------------------
# 6. Вывод DXF, ортофото, превью
# ----------------------------------------------------------------------------

def write_dxf(path, outline, openings, gaps):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 6  # метры
    msp = doc.modelspace()
    for name, color in [("FACADE_OUTLINE", 7), ("OPENINGS", 5), ("GAPS_REVIEW", 1)]:
        doc.layers.add(name, color=color)

    msp.add_lwpolyline([(p[0], p[1]) for p in outline],
                       close=True, dxfattribs={"layer": "FACADE_OUTLINE"})
    for layer, rects in [("OPENINGS", openings), ("GAPS_REVIEW", gaps)]:
        for (x, y, w, h) in rects:
            msp.add_lwpolyline(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
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


def save_scene_plan(pts_xy, facades, path, res=0.25):
    """План сцены сверху с осями найденных фасадов и их номерами."""
    x0, y0 = pts_xy.min(axis=0)
    w = int((pts_xy[:, 0].max() - x0) / res) + 2
    h = int((pts_xy[:, 1].max() - y0) / res) + 2
    img = np.zeros((h, w), np.float32)
    ix = ((pts_xy[:, 0] - x0) / res).astype(np.int32)
    iy = ((pts_xy[:, 1] - y0) / res).astype(np.int32)
    np.add.at(img, (iy, ix), 1.0)
    img = np.clip(img / max(np.percentile(img[img > 0], 95), 1) * 255,
                  0, 255).astype(np.uint8)
    plan = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_GRAY2BGR)

    def to_px(p):
        return (int((p[0] - x0) / res), h - 1 - int((p[1] - y0) / res))

    for f in facades:
        a = f["origin"] + f["u_range"][0] * f["direction"]
        b = f["origin"] + f["u_range"][1] * f["direction"]
        cv2.line(plan, to_px(a), to_px(b), (0, 255, 255), 2)
        mid = to_px((a + b) / 2)
        cv2.putText(plan, f"f{f['index']:02d}", (mid[0] + 4, mid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
    cv2.imwrite(path, plan)


def ensure_writable(out_path, las_path):
    """Проверяет, что выходной файл можно создать.

    Защитник Windows (контролируемый доступ к папкам) и OneDrive блокируют
    запись в Documents с невнятной ошибкой FileNotFoundError. В этом случае
    переносим вывод в папку исходного LAS.
    """
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


# ----------------------------------------------------------------------------
# Обработка одного фасада
# ----------------------------------------------------------------------------

def process_facade(u, v, args, out_dxf, world):
    area = (u.max() - u.min()) * (v.max() - v.min())
    res = args.res
    auto_res = float(np.sqrt(3.0 * area / max(len(u), 1)))
    coarse = auto_res > res * 1.3
    if coarse:
        res = round(auto_res, 3)

    density, (u0, v0) = rasterize(u, v, res)
    outline, openings, gaps, _ = detect(density, res, args.min_opening, args.rect)

    stem = out_dxf.rsplit(".", 1)[0]
    write_dxf(out_dxf, outline, openings, gaps)
    save_images(density, outline, openings, gaps, res,
                stem + "_ortho.png", stem + "_preview.png")
    origin, direction = world
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "world_origin_xy": [float(origin[0]) + float(u0) * float(direction[0]),
                                float(origin[1]) + float(u0) * float(direction[1])],
            "facade_direction_xy": [float(direction[0]), float(direction[1])],
            "v_zero_world_z": float(v0),
            "raster_res_m": res,
            "note": "точка DXF (u,v) -> мир: origin + u*direction, Z = v_zero + v",
        }, f, ensure_ascii=False, indent=2)
    return {"size": (u.max() - u.min(), v.max() - v.min()),
            "openings": len(openings), "gaps": len(gaps),
            "res": res, "coarse": coarse}


def main():
    ap = argparse.ArgumentParser(description="LAS -> DXF-подоснова фасадов")
    ap.add_argument("las", help="входной файл LAS/LAZ")
    ap.add_argument("-o", "--out", default="facade.dxf",
                    help="базовое имя выходных DXF (фасады получат суффикс _fNN)")
    ap.add_argument("--facades", type=int, default=8,
                    help="максимум фасадов за запуск (по умолчанию 8)")
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
    ap.add_argument("--gap", type=float, default=3.0,
                    help="разрыв в плане, разделяющий здания на одной линии, м")
    ap.add_argument("--min-facade-len", type=float, default=4.0,
                    help="минимальная длина фасада, м")
    args = ap.parse_args()

    print(f"Читаю {args.las} ...")
    pts, _ = read_las(args.las, args.max_points)
    print(f"  точек: {len(pts):,}")

    out_base = ensure_writable(args.out, args.las)
    stem_base = out_base.rsplit(".", 1)[0]

    vert = filter_vertical(pts)
    print(f"  вертикальных структур: {int(vert.sum()):,} точек "
          f"({vert.mean() * 100:.0f}%), земля и низкий мусор отброшены")
    work = pts[vert]
    min_wall_points = max(15_000, int(0.003 * len(pts)))
    facades, fidx = [], 0
    while fidx < args.facades and len(work) > 2 * min_wall_points:
        try:
            origin, direction, inl = ransac_facade_line(
                work[:, :2], args.plane_tol, seed=fidx)
        except RuntimeError:
            break
        if int(inl.sum()) < min_wall_points:
            break

        normal = np.array([-direction[1], direction[0]])
        dist = (work[:, :2] - origin) @ normal
        u_all = (work[:, :2] - origin) @ direction
        segments = split_by_gaps(u_all[inl], args.gap,
                                 args.min_facade_len, min_wall_points)

        for (lo, hi, n) in segments:
            if fidx >= args.facades:
                break
            sel = (np.abs(dist) < args.slice) & (u_all > lo - 0.5) & (u_all < hi + 0.5)
            u, v = u_all[sel], work[sel, 2]
            fidx += 1
            out_dxf = f"{stem_base}_f{fidx:02d}.dxf"
            try:
                info = process_facade(u, v, args, out_dxf, (origin, direction))
            except (RuntimeError, ValueError) as e:
                print(f"  f{fidx:02d}: пропущен ({e})")
                fidx -= 1
                continue
            note = f" [растр укрупнён до {info['res']} м/px]" if info["coarse"] else ""
            print(f"  f{fidx:02d}: {info['size'][0]:.1f} x {info['size'][1]:.1f} м, "
                  f"проёмов {info['openings']}, на проверку {info['gaps']}"
                  f" -> {out_dxf}{note}")
            facades.append({"index": fidx, "origin": origin,
                            "direction": direction, "u_range": (lo, hi)})

        # убираем обработанную плоскость из рабочего облака
        work = work[np.abs(dist) > 1.5 * args.slice]

    if not facades:
        print("Фасады не найдены — попробуйте увеличить --plane-tol или --slice")
        return 1

    plan_path = stem_base + "_scene_plan.png"
    sample = pts if len(pts) <= 3_000_000 else \
        pts[np.random.default_rng(1).choice(len(pts), 3_000_000, replace=False)]
    save_scene_plan(sample[:, :2], facades, plan_path)
    print(f"Готово: фасадов {len(facades)}; план сцены с номерами: {plan_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
