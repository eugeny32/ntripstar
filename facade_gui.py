#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FacadeStudio — графический интерфейс к facade2dxf.

Возможности:
  - загрузка LAS/LAZ, автопоиск фасадов (все стены зданий);
  - план сцены с осями фасадов: линию плоскости можно нарисовать и
    подвинуть мышью (концы линии — маркеры), толщина среза — параметром;
  - пересчёт выбранного фасада после правки линии/среза;
  - просмотр превью и ортофото, вывод DXF по слоям.

Запуск:  python facade_gui.py
Сборка exe (Windows):  build_exe.bat
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageTk

import facade2dxf as core

PLAN_MAX_W, PLAN_MAX_H = 980, 640


class FacadeStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FacadeStudio — облако точек -> DXF фасада")
        self.geometry("1360x860")

        self.pts = None          # полное облако (после прореживания)
        self.vert = None         # маска вертикальных структур
        self.plan = None         # dict: img (PIL), x0, y0, res, scale
        self.facades = []        # записи фасадов
        self.current_line = None # [p1_world, p2_world] редактируемой линии
        self.drag_target = None
        self.draw_mode = False
        self.log_q = queue.Queue()
        self.ui_q = queue.Queue()   # колбэки из фоновых потоков -> главный цикл
        self.busy = False

        self._build_ui()
        self.after(100, self._poll_log)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Файл LAS:").pack(side="left")
        self.las_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.las_var, width=70).pack(
            side="left", padx=4, fill="x", expand=True)
        ttk.Button(top, text="Обзор...", command=self._browse).pack(side="left")
        self.load_btn = ttk.Button(top, text="Загрузить облако",
                                   command=self._load_cloud)
        self.load_btn.pack(side="left", padx=6)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        # левая панель: параметры и действия
        left = ttk.Frame(body, padding=6)
        left.pack(side="left", fill="y")

        params = ttk.LabelFrame(left, text="Параметры", padding=6)
        params.pack(fill="x")
        self.vars = {}
        for key, label, default in [
                ("max_points", "Точек макс., млн", 15),
                ("facades", "Фасадов макс.", 8),
                ("res", "Растр, м/px", 0.02),
                ("plane_tol", "Допуск плоскости, м", 0.08),
                ("slice", "Полутолщина среза, м", 0.15),
                ("min_opening", "Мин. проём, м²", 0.15),
                ("rect", "Порог прямоугольности", 0.85),
                ("gap", "Разрыв зданий, м", 3.0),
                ("min_facade_len", "Мин. длина фасада, м", 4.0)]:
            row = ttk.Frame(params)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=22).pack(side="left")
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=8).pack(side="right")
            self.vars[key] = var

        actions = ttk.LabelFrame(left, text="Действия", padding=6)
        actions.pack(fill="x", pady=6)
        self.auto_btn = ttk.Button(actions, text="Автопоиск фасадов",
                                   command=self._auto_find)
        self.auto_btn.pack(fill="x", pady=2)
        self.line_btn = ttk.Button(actions, text="Нарисовать линию фасада",
                                   command=self._start_draw)
        self.line_btn.pack(fill="x", pady=2)
        self.proc_btn = ttk.Button(actions, text="Обработать линию/пересчитать",
                                   command=self._process_line)
        self.proc_btn.pack(fill="x", pady=2)
        ttk.Button(actions, text="Открыть папку результатов",
                   command=self._open_folder).pack(fill="x", pady=2)

        lf = ttk.LabelFrame(left, text="Фасады", padding=4)
        lf.pack(fill="both", expand=True)
        self.flist = tk.Listbox(lf, width=34, height=14)
        self.flist.pack(fill="both", expand=True)
        self.flist.bind("<<ListboxSelect>>", self._on_select)

        # центр: вкладки план/фасад
        center = ttk.Frame(body, padding=(0, 6, 6, 6))
        center.pack(side="left", fill="both", expand=True)
        self.nb = ttk.Notebook(center)
        self.nb.pack(fill="both", expand=True)

        self.plan_canvas = tk.Canvas(self.nb, bg="#101010",
                                     width=PLAN_MAX_W, height=PLAN_MAX_H)
        self.nb.add(self.plan_canvas, text="  План сцены  ")
        self.plan_canvas.bind("<Button-1>", self._plan_click)
        self.plan_canvas.bind("<B1-Motion>", self._plan_drag)
        self.plan_canvas.bind("<ButtonRelease-1>", self._plan_release)

        self.prev_canvas = tk.Canvas(self.nb, bg="#101010")
        self.nb.add(self.prev_canvas, text="  Превью фасада  ")

        self.log = tk.Text(center, height=8, bg="#111", fg="#ddd",
                           font=("Consolas", 9))
        self.log.pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------- журнал --
    def _logln(self, msg):
        self.log_q.put(msg)

    def _poll_log(self):
        try:
            while True:
                self.log.insert("end", self.log_q.get_nowait() + "\n")
                self.log.see("end")
        except queue.Empty:
            pass
        try:
            while True:
                self.ui_q.get_nowait()()
        except queue.Empty:
            pass
        self.after(150, self._poll_log)

    def _args(self):
        g = lambda k, cast=float: cast(self.vars[k].get().replace(",", "."))
        return SimpleNamespace(
            res=g("res"), plane_tol=g("plane_tol"), slice=g("slice"),
            min_opening=g("min_opening"), rect=g("rect"), gap=g("gap"),
            min_facade_len=g("min_facade_len"),
            facades=g("facades", int),
            max_points=int(g("max_points") * 1e6))

    def _out_stem(self, idx):
        base = os.path.dirname(os.path.abspath(self.las_path))
        return os.path.join(base, f"fasad_f{idx:02d}")

    # ------------------------------------------------------------ загрузка --
    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Облако точек", "*.las *.laz"), ("Все файлы", "*.*")])
        if p:
            self.las_var.set(p)

    def _load_cloud(self):
        path = self.las_var.get().strip()
        if not os.path.isfile(path):
            messagebox.showerror("Ошибка", "Укажите существующий LAS-файл")
            return
        self.las_path = path
        self._run_bg(self._load_cloud_job, path, self._args())

    def _load_cloud_job(self, path, args):
        self._logln(f"Читаю {path} ...")
        pts, _ = core.read_las(path, args.max_points)
        self._logln(f"  точек: {len(pts):,}")
        vert = core.filter_vertical(pts)
        self._logln(f"  вертикальных структур: {int(vert.sum()):,} "
                    f"({vert.mean() * 100:.0f}%)")
        self.pts, self.vert = pts, vert
        self.facades, self.current_line = [], None
        self.ui_q.put(self._render_plan)
        self.ui_q.put(lambda: self.flist.delete(0, "end"))
        self._logln("Облако загружено. Запустите автопоиск или нарисуйте линию.")

    # ----------------------------------------------------------- план сцены --
    def _render_plan(self):
        if self.pts is None:
            return
        xy = self.pts[self.vert][:, :2] if self.vert.any() else self.pts[:, :2]
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        res = max((x1 - x0) / PLAN_MAX_W, (y1 - y0) / PLAN_MAX_H, 0.02)
        w = int((x1 - x0) / res) + 2
        h = int((y1 - y0) / res) + 2
        img = np.zeros((h, w), np.float32)
        ix = ((xy[:, 0] - x0) / res).astype(np.int32)
        iy = ((xy[:, 1] - y0) / res).astype(np.int32)
        np.add.at(img, (iy, ix), 1.0)
        nz = img[img > 0]
        img = np.clip(img / max(np.percentile(nz, 95), 1) * 255,
                      0, 255).astype(np.uint8)
        img = np.flipud(img)
        self.plan = {"x0": x0, "y0": y0, "res": res, "h": h,
                     "img": Image.fromarray(img)}
        self._redraw_plan()

    def _w2c(self, p):
        pl = self.plan
        return ((p[0] - pl["x0"]) / pl["res"],
                pl["h"] - 1 - (p[1] - pl["y0"]) / pl["res"])

    def _c2w(self, cx, cy):
        pl = self.plan
        return np.array([pl["x0"] + cx * pl["res"],
                         pl["y0"] + (pl["h"] - 1 - cy) * pl["res"]])

    def _redraw_plan(self):
        c = self.plan_canvas
        c.delete("all")
        if self.plan is None:
            return
        self._tkimg = ImageTk.PhotoImage(self.plan["img"])
        c.create_image(0, 0, anchor="nw", image=self._tkimg)

        for f in self.facades:
            a = self._w2c(f["origin"] + f["u_range"][0] * f["direction"])
            b = self._w2c(f["origin"] + f["u_range"][1] * f["direction"])
            c.create_line(*a, *b, fill="#00d0ff", width=2)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            c.create_text(mx + 8, my - 8, text=f"f{f['index']:02d}",
                          fill="#ffb000", font=("Arial", 11, "bold"))

        if self.current_line is not None:
            p1, p2 = self.current_line
            a, b = self._w2c(p1), self._w2c(p2)
            # полоса среза
            args = self._args()
            d = p2 - p1
            n = np.linalg.norm(d)
            if n > 1e-6:
                nrm = np.array([-d[1], d[0]]) / n * args.slice
                for s in (+1, -1):
                    aa = self._w2c(p1 + s * nrm)
                    bb = self._w2c(p2 + s * nrm)
                    c.create_line(*aa, *bb, fill="#00ff66", dash=(4, 4))
            c.create_line(*a, *b, fill="#00ff00", width=2)
            for p in (a, b):
                c.create_oval(p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5,
                              fill="#ff3030", outline="white", tags="handle")

    # ------------------------------------------------- рисование/перетаскивание
    def _start_draw(self):
        if self.plan is None:
            messagebox.showinfo("Нет облака", "Сначала загрузите LAS")
            return
        self.draw_mode = True
        self.current_line = None
        self._redraw_plan()
        self._logln("Кликните две точки на плане — начало и конец фасада.")

    def _plan_click(self, ev):
        if self.plan is None:
            return
        w = self._c2w(ev.x, ev.y)
        if self.draw_mode:
            if self.current_line is None:
                self.current_line = [w, w.copy()]
            else:
                self.current_line[1] = w
                self.draw_mode = False
                self._logln("Линия задана. Концы можно двигать мышью; затем "
                            "«Обработать линию».")
            self._redraw_plan()
            return
        if self.current_line is not None:
            for i, p in enumerate(self.current_line):
                cx, cy = self._w2c(p)
                if abs(cx - ev.x) < 8 and abs(cy - ev.y) < 8:
                    self.drag_target = i
                    return

    def _plan_drag(self, ev):
        if self.drag_target is None and self.draw_mode and self.current_line:
            self.current_line[1] = self._c2w(ev.x, ev.y)
            self._redraw_plan()
        if self.drag_target is not None:
            self.current_line[self.drag_target] = self._c2w(ev.x, ev.y)
            self._redraw_plan()

    def _plan_release(self, _ev):
        self.drag_target = None

    # ------------------------------------------------------------ обработка --
    def _run_bg(self, fn, *a):
        if self.busy:
            messagebox.showinfo("Занят", "Дождитесь завершения операции")
            return
        self.busy = True
        for b in (self.load_btn, self.auto_btn, self.proc_btn):
            b.state(["disabled"])

        def wrap():
            try:
                fn(*a)
            except Exception as e:
                self._logln(f"ОШИБКА: {e}")
            finally:
                self.busy = False
                self.ui_q.put(lambda: [b.state(["!disabled"]) for b in
                                       (self.load_btn, self.auto_btn,
                                        self.proc_btn)])
        threading.Thread(target=wrap, daemon=True).start()

    def _auto_find(self):
        if self.pts is None:
            messagebox.showinfo("Нет облака", "Сначала загрузите LAS")
            return
        self._run_bg(self._auto_find_job, self._args())

    def _auto_find_job(self, args):
        pts = self.pts
        work = pts[self.vert]
        min_wall = max(15_000, int(0.003 * len(pts)))
        self.facades = []
        fidx = 0
        while fidx < args.facades and len(work) > 2 * min_wall:
            try:
                origin, direction, inl = core.ransac_facade_line(
                    work[:, :2], args.plane_tol, seed=fidx)
            except RuntimeError:
                break
            if int(inl.sum()) < min_wall:
                break
            normal = np.array([-direction[1], direction[0]])
            dist = (work[:, :2] - origin) @ normal
            u_all = (work[:, :2] - origin) @ direction
            for (lo, hi, _n) in core.split_by_gaps(
                    u_all[inl], args.gap, args.min_facade_len, min_wall):
                if fidx >= args.facades:
                    break
                sel = (np.abs(dist) < args.slice) & \
                      (u_all > lo - 0.5) & (u_all < hi + 0.5)
                fidx += 1
                rec = self._process(work[sel], origin, direction,
                                    (lo, hi), fidx, args)
                if rec is None:
                    fidx -= 1
            work = work[np.abs(dist) > 1.5 * args.slice]
        self.ui_q.put(self._redraw_plan)
        self._logln(f"Автопоиск завершён: фасадов {len(self.facades)}")

    def _process(self, pts_sel, origin, direction, u_range, idx, args):
        """Обработка среза и запись результатов; возвращает запись фасада."""
        u = (pts_sel[:, :2] - origin) @ direction
        v = pts_sel[:, 2]
        if len(u) < 500:
            self._logln(f"  f{idx:02d}: слишком мало точек в срезе, пропущен")
            return None
        stem = self._out_stem(idx)
        try:
            info = core.process_facade(u, v, args, stem + ".dxf",
                                       (origin, direction))
        except (RuntimeError, ValueError) as e:
            self._logln(f"  f{idx:02d}: пропущен ({e})")
            return None
        rec = {"index": idx, "origin": origin, "direction": direction,
               "u_range": u_range, "stem": stem, "info": info}
        done = [f for f in self.facades if f["index"] != idx]
        self.facades = sorted(done + [rec], key=lambda r: r["index"])
        note = f" [растр {info['res']} м/px]" if info["coarse"] else ""
        self._logln(f"  f{idx:02d}: {info['size'][0]:.1f} x "
                    f"{info['size'][1]:.1f} м, проёмов {info['openings']}, "
                    f"на проверку {info['gaps']}{note}")
        self.ui_q.put(self._refresh_list)
        return rec

    def _process_line(self):
        if self.pts is None or self.current_line is None:
            messagebox.showinfo("Нет линии",
                                "Нарисуйте линию фасада или выберите фасад")
            return
        p1, p2 = self.current_line
        d = p2 - p1
        L = np.linalg.norm(d)
        if L < 1.0:
            messagebox.showinfo("Линия коротка", "Растяните линию хотя бы на 1 м")
            return
        args = self._args()
        direction = d / L
        sel_idx = self.flist.curselection()
        idx = (self.facades[sel_idx[0]]["index"] if sel_idx
               else (max([f["index"] for f in self.facades], default=0) + 1))

        def job():
            normal = np.array([-direction[1], direction[0]])
            dist = (self.pts[:, :2] - p1) @ normal
            u_all = (self.pts[:, :2] - p1) @ direction
            sel = (np.abs(dist) < args.slice) & (u_all > -0.5) & (u_all < L + 0.5)
            self._process(self.pts[sel], p1, direction, (0.0, L), idx, args)
            self.ui_q.put(self._redraw_plan)
        self._run_bg(job)

    # ------------------------------------------------------------- просмотр --
    def _refresh_list(self):
        self.flist.delete(0, "end")
        for f in self.facades:
            i = f["info"]
            self.flist.insert(
                "end", f"f{f['index']:02d}  {i['size'][0]:.1f}x"
                       f"{i['size'][1]:.1f} м  проёмов {i['openings']}"
                       f"  проверка {i['gaps']}")

    def _on_select(self, _ev):
        sel = self.flist.curselection()
        if not sel:
            return
        f = self.facades[sel[0]]
        self.current_line = [
            f["origin"] + f["u_range"][0] * f["direction"],
            f["origin"] + f["u_range"][1] * f["direction"]]
        self._redraw_plan()
        p = f["stem"] + "_preview.png"
        if os.path.isfile(p):
            img = Image.open(p)
            cw = max(self.prev_canvas.winfo_width(), 700)
            ch = max(self.prev_canvas.winfo_height(), 500)
            k = min(cw / img.width, ch / img.height, 1.0)
            img = img.resize((int(img.width * k), int(img.height * k)))
            self._prev_img = ImageTk.PhotoImage(img)
            self.prev_canvas.delete("all")
            self.prev_canvas.create_image(0, 0, anchor="nw",
                                          image=self._prev_img)
            self.nb.select(1)

    def _open_folder(self):
        path = os.path.dirname(os.path.abspath(getattr(self, "las_path", ".")))
        if sys.platform == "win32":
            os.startfile(path)  # noqa: attribute exists on Windows
        else:
            subprocess.Popen(["xdg-open", path])


if __name__ == "__main__":
    app = FacadeStudio()
    app.mainloop()
