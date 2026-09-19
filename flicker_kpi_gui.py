"""
Flicker KPI GUI
===============
使用方式:
    python flicker_kpi_gui.py
需求: numpy, opencv-python, pillow  (tkinter 為 Python 內建)

功能:
  - 選擇資料夾：自動把 <prefix>0001 ~ <prefix>0006 分組成多組序列，左上清單逐組檢視
  - 「批次分析+匯出全部」：每組輸出 <name>_risk_map.png，另輸出全部序列的 CSV 與 summary
  - 20 個 threshold (pixel-based 10 + block-based 10) 皆可即時調整
  - 即時重算超標比例，metric 表格以顏色標示 PASS / FAIL
  - 右側 heatmap: 風險分級總圖 / ratio (value/threshold) / raw / binary fail mask / 綜合超標張數
  - 風險分級總圖: 20 項 metric 取 max(value/threshold)，以 5 色疊在原圖上
  - FAIL 判定: 任一 metric 超標位置數 >= K 即 FAIL (K=1 即「任一 pixel 超標」)
  - Threshold 最佳化: 用「不閃 folder」+「混合 folder（逐組標記閃/不閃）」求一組 threshold 與 K，
                      使漏抓 = 0 且誤報最少，可一鍵套用回主視窗
  - 分格總圖: 以 tile x tile (預設 40x40) 彙整超標點，RED/ORANGE/YELLOW 外框標示，
              點格子可看該格明細（最大 ratio、位置、pixel/block 超標數、主要 metric）
  - 匯出 CSV 報表 + 單張風險分級總圖 (risk_map.png)

效能備註: metric map 只在載入序列或切換 difference mode 時重算一次；
         調 threshold 只做比較運算，不重算 metric，所以拖 slider 不會卡。
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

from flicker_kpi_core import (
    DEFAULT_THRESHOLDS, METRIC_NAMES,
    analyze_sequence, evaluate_kpi, combined_fail_map,
    render_heatmap, upscale,
    DEFAULT_RISK_BOUNDS, RISK_LEVELS, compute_risk, render_risk_map,
    parse_risk_bounds, risk_range_labels, imwrite_unicode,
    discover_sequences, summarize_result, max_pool_fit,
    DEFAULT_TILE, DEFAULT_TILE_RULES, compute_tiles, tile_detail, render_tile_map,
    parse_tile_size, parse_tile_rule, tile_rule_text,
    metric_is_fail, k_for_level,
)
import flicker_kpi_opt as kopt

N_FRAMES = 6
CSV_HEADER = ["sequence", "level", "metric", "threshold", "fail_ratio_%",
              "fail_count", "total", "max", "p99.9", "p99", "mean",
              "max/threshold", "verdict"]
SUMMARY_HEADER = ["sequence", "verdict", "pixel_fail_metrics", "block_fail_metrics",
                  "FAIL+SEVERE_area_%", "worst_level", "max_ratio", "max_metric",
                  "max_x", "max_y", "tile_size", "tiles_red", "tiles_orange",
                  "tiles_yellow", "tiles_fail", "tile_fail_%"]
SEQ_COLS = ("sequence", "verdict", "pix_fail", "blk_fail", "danger%", "worst", "max_ratio",
            "tiles R/O/Y")


def metric_rows(result, min_fail=1):
    rows = []
    bs = result.get("block_size", 2)
    for level in ("pixel", "block"):
        rep = result[level]["report"]
        for n in METRIC_NAMES:
            s = rep[n]
            rows.append([result["name"], level, n, s["threshold"],
                         f"{s['fail_ratio']*100:.6f}", s["fail_count"], s["total"],
                         f"{s['max']:.3f}", f"{s['p999']:.3f}", f"{s['p99']:.3f}",
                         f"{s['mean']:.4f}", f"{s['margin']:.3f}",
                         "FAIL" if metric_is_fail(s, level, min_fail, bs) else "PASS"])
    return rows


def summary_row(sm):
    return [sm["name"], sm["verdict"], sm["pixel_fail_metrics"], sm["block_fail_metrics"],
            f"{sm['danger_pct']:.4f}", sm["worst_level"], f"{sm['max_ratio']:.3f}",
            sm["max_metric"], sm["max_xy"][0], sm["max_xy"][1],
            sm["tile_size"], sm["tiles_red"], sm["tiles_orange"], sm["tiles_yellow"],
            sm["tiles_fail"], f"{sm['tile_fail_pct']:.4f}"]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def _report_errors(title):
    """tkinter callback 的例外只會印到 console（pythonw 下看不到），這裡顯式跳窗"""
    def deco(fn):
        def wrapper(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception:
                import traceback
                tb = traceback.format_exc()
                print(tb)
                messagebox.showerror(title, tb)
                self.status.config(text=title)
        return wrapper
    return deco


class FlickerKPIApp:
    def __init__(self, root):
        self.root = root
        root.title("Flicker KPI Analyzer")
        root.geometry("1500x900")

        self.result = None
        self.folder = None
        self.seqs = []          # discover_sequences 結果
        self.cur_idx = None
        self.thresholds = {
            "pixel": dict(DEFAULT_THRESHOLDS["pixel"]),
            "block": dict(DEFAULT_THRESHOLDS["block"]),
        }
        self.vars = {"pixel": {}, "block": {}}
        self.tree_items = {"pixel": {}, "block": {}}

        self.level = tk.StringVar(value="pixel")
        self.metric = tk.StringVar(value="R_difference")
        self.view_mode = tk.StringVar(value="tile")
        self.tile_text = tk.StringVar(value=str(DEFAULT_TILE))
        self.rule_text = {k: tk.StringVar(value=",".join(f"{x:g}" for x in v))
                          for k, v in DEFAULT_TILE_RULES.items()}
        self.tile_size = DEFAULT_TILE
        self.tile_rules = dict(DEFAULT_TILE_RULES)
        self.sel_tile = None        # (row, col)
        self._tile_view = None      # (tiles, render scale)
        self._geom = None           # 畫面上圖片的 (ox, oy, display scale)
        self.risk_bounds_text = tk.StringVar(
            value=",".join(f"{b:g}" for b in DEFAULT_RISK_BOUNDS))
        self.risk_bounds = DEFAULT_RISK_BOUNDS
        self.diff_mode = tk.StringVar(value="peak_to_peak")
        self.pattern = tk.StringVar(value="*.bmp")
        self.vmax_ratio = tk.DoubleVar(value=2.0)
        self.min_fail_text = tk.StringVar(value="1")
        self.min_fail = 1
        self.opt_win = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")

        ttk.Button(top, text="載入資料夾", command=self.load_folder).pack(side="left")
        self.path_label = ttk.Label(top, text="(未載入)", width=40)
        self.path_label.pack(side="left", padx=8)

        ttk.Label(top, text="檔名 pattern:").pack(side="left")
        ttk.Entry(top, textvariable=self.pattern, width=10).pack(side="left", padx=4)

        ttk.Label(top, text="difference 定義:").pack(side="left", padx=(12, 2))
        diff_cb = ttk.Combobox(top, textvariable=self.diff_mode, width=14, state="readonly",
                               values=["peak_to_peak", "max_adjacent"])
        diff_cb.pack(side="left")
        diff_cb.bind("<<ComboboxSelected>>", lambda e: self.recompute_all())

        ttk.Button(top, text="還原預設 threshold",
                   command=self.reset_thresholds).pack(side="left", padx=12)
        ttk.Button(top, text="匯出目前序列",
                   command=self.export).pack(side="left")
        ttk.Button(top, text="批次分析+匯出全部",
                   command=self.batch_export).pack(side="left", padx=6)

        top2 = ttk.Frame(self.root, padding=(6, 0))
        top2.pack(fill="x")
        ttk.Label(top2, text="FAIL 判定: 任一 metric 超標位置數 >= K，K =").pack(side="left")
        ke = ttk.Entry(top2, textvariable=self.min_fail_text, width=6)
        ke.pack(side="left", padx=4)
        ke.bind("<Return>", lambda e: self.apply_min_fail())
        ttk.Button(top2, text="套用", command=self.apply_min_fail).pack(side="left")
        ttk.Label(top2, text="(pixel 計；block 以 ceil(K/4) 個 block 計)").pack(side="left", padx=6)
        ttk.Button(top2, text="Threshold 最佳化...",
                   command=self.open_optimizer).pack(side="left", padx=20)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(body)
        body.add(left, weight=3)
        right = ttk.Frame(body)
        body.add(right, weight=4)

        # ---- 左上：序列清單 ----
        sf = ttk.LabelFrame(left, text="序列清單 (<prefix>0001~0006)", padding=4)
        sf.pack(fill="x")
        self.seq_tree = ttk.Treeview(sf, columns=SEQ_COLS, show="headings", height=7,
                                     selectmode="browse")
        for c, w in zip(SEQ_COLS, (170, 55, 55, 55, 65, 60, 65, 85)):
            self.seq_tree.heading(c, text=c)
            self.seq_tree.column(c, width=w, anchor="e")
        self.seq_tree.column("sequence", anchor="w")
        vsb = ttk.Scrollbar(sf, orient="vertical", command=self.seq_tree.yview)
        self.seq_tree.configure(yscrollcommand=vsb.set)
        self.seq_tree.pack(side="left", fill="x", expand=True)
        vsb.pack(side="right", fill="y")
        self.seq_tree.tag_configure("fail", background="#ffd6d6")
        self.seq_tree.tag_configure("pass", background="#e3f7e3")
        self.seq_tree.tag_configure("stale", foreground="#999999")
        self.seq_tree.tag_configure("error", background="#ffb0b0")
        self.seq_tree.bind("<<TreeviewSelect>>", lambda e: self._on_seq_select())

        # ---- 左：threshold 控制 + 結果表 ----
        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)
        for level, title in (("pixel", "Pixel-based"), ("block", "Block-based (2x2)")):
            tab = ttk.Frame(nb, padding=4)
            nb.add(tab, text=title)
            self._build_level_tab(tab, level)
        nb.bind("<<NotebookTabChanged>>",
                lambda e: self._on_level_change(nb.index("current")))

        # ---- 右：heatmap ----
        ctrl = ttk.Frame(right, padding=4)
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="顯示 metric:").pack(side="left")
        mcb = ttk.Combobox(ctrl, textvariable=self.metric, width=16,
                           state="readonly", values=METRIC_NAMES)
        mcb.pack(side="left", padx=4)
        mcb.bind("<<ComboboxSelected>>", lambda e: self.refresh_heatmap())

        ttk.Label(ctrl, text="模式:").pack(side="left", padx=(12, 2))
        for text, val in (("分格總圖", "tile"), ("風險分級總圖", "risk"),
                          ("ratio(值/門檻)", "ratio"), ("raw", "raw"),
                          ("fail mask", "binary"), ("綜合超標數", "combined")):
            ttk.Radiobutton(ctrl, text=text, value=val, variable=self.view_mode,
                            command=self.refresh_heatmap).pack(side="left")

        ttk.Label(ctrl, text="色階上限(×門檻):").pack(side="left", padx=(12, 2))
        sc = ttk.Scale(ctrl, from_=1.0, to=5.0, variable=self.vmax_ratio,
                       orient="horizontal", length=100,
                       command=lambda e: self.refresh_heatmap())
        sc.pack(side="left")

        ctrl2 = ttk.Frame(right, padding=(4, 0))
        ctrl2.pack(fill="x")
        ttk.Label(ctrl2, text="風險分界 (×門檻, 4 個遞增值 → SAFE/WATCH/NEAR/FAIL/SEVERE):"
                  ).pack(side="left")
        be = ttk.Entry(ctrl2, textvariable=self.risk_bounds_text, width=18)
        be.pack(side="left", padx=4)
        be.bind("<Return>", lambda e: self.apply_risk_bounds())
        ttk.Button(ctrl2, text="套用", command=self.apply_risk_bounds).pack(side="left")

        ctrl3 = ttk.Frame(right, padding=(4, 2))
        ctrl3.pack(fill="x")
        ttk.Label(ctrl3, text="分格:").pack(side="left")
        tcb = ttk.Combobox(ctrl3, textvariable=self.tile_text, width=5,
                           values=["20", "40", "60", "120"])
        tcb.pack(side="left", padx=(2, 10))
        tcb.bind("<<ComboboxSelected>>", lambda e: self.apply_tile_params())
        tcb.bind("<Return>", lambda e: self.apply_tile_params())
        for key in ("RED", "ORANGE"):
            ttk.Label(ctrl3, text=f"{key} (ratio,px,blk 任一達到):").pack(side="left")
            e = ttk.Entry(ctrl3, textvariable=self.rule_text[key], width=10)
            e.pack(side="left", padx=(2, 10))
            e.bind("<Return>", lambda ev: self.apply_tile_params())
        ttk.Button(ctrl3, text="套用", command=self.apply_tile_params).pack(side="left")

        self.canvas = tk.Canvas(right, bg="#202020")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.info = tk.Text(right, height=11, font=("Consolas", 9))
        self.info.pack(fill="x", padx=4, pady=(0, 4))

        self.status = ttk.Label(self.root, text="就緒", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _build_level_tab(self, parent, level):
        thr_frame = ttk.LabelFrame(parent, text="Threshold (可直接輸入或拖曳)", padding=4)
        thr_frame.pack(fill="x")

        for i, name in enumerate(METRIC_NAMES):
            row = i % 5
            col = i // 5
            ttk.Label(thr_frame, text=name, width=14).grid(
                row=row, column=col * 3, sticky="w", pady=1)
            var = tk.DoubleVar(value=self.thresholds[level][name])
            self.vars[level][name] = var
            ent = ttk.Entry(thr_frame, textvariable=var, width=9)
            ent.grid(row=row, column=col * 3 + 1, padx=(2, 8))
            ent.bind("<Return>", lambda e, l=level: self.apply_thresholds(l))
            ent.bind("<FocusOut>", lambda e, l=level: self.apply_thresholds(l))

        ttk.Button(thr_frame, text="套用",
                   command=lambda l=level: self.apply_thresholds(l)
                   ).grid(row=5, column=0, columnspan=6, pady=4, sticky="we")

        cols = ("metric", "thr", "fail%", "fail_cnt", "max", "p99.9", "mean", "max/thr")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (120, 70, 80, 80, 80, 80, 75, 70)):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="e")
        tree.column("metric", anchor="w")
        tree.pack(fill="both", expand=True, pady=4)
        tree.tag_configure("fail", background="#ffd6d6")
        tree.tag_configure("pass", background="#e3f7e3")
        tree.bind("<<TreeviewSelect>>", lambda e, t=tree: self._on_tree_select(t))
        setattr(self, f"tree_{level}", tree)

    # ------------------------------------------------------------------
    # 行為
    # ------------------------------------------------------------------
    def _on_level_change(self, idx):
        self.level.set("pixel" if idx == 0 else "block")
        self.refresh_heatmap()

    def _on_tree_select(self, tree):
        sel = tree.selection()
        if sel:
            self.metric.set(tree.item(sel[0], "values")[0])
            self.refresh_heatmap()

    @_report_errors("載入失敗")
    def load_folder(self):
        folder = filedialog.askdirectory(title="選擇資料夾（含一組或多組 xxxx0001~0006）")
        if not folder:
            return
        seqs, skipped = discover_sequences(folder, self.pattern.get(), N_FRAMES)
        if not seqs:
            msg = f"{folder} 內找不到完整的 <prefix>0001~{N_FRAMES:04d} ({self.pattern.get()})"
            if skipped:
                msg += "\n\n不完整:\n" + "\n".join(f"  {p}: {r}" for p, r in skipped[:20])
            messagebox.showerror("沒有可分析的序列", msg)
            return
        self.folder, self.seqs, self.cur_idx, self.result = folder, seqs, None, None
        self.seq_tree.delete(*self.seq_tree.get_children())
        for i, sq in enumerate(seqs):
            self.seq_tree.insert("", "end", iid=str(i),
                                 values=(sq["name"], "—", "", "", "", "", "", ""))
        self.path_label.config(text=folder)
        if skipped:
            messagebox.showwarning(
                "部分 prefix 未納入",
                f"找到 {len(seqs)} 組完整序列；以下 {len(skipped)} 組缺幀，已略過:\n"
                + "\n".join(f"  {p}: {r}" for p, r in skipped[:20]))
        self.seq_tree.selection_set("0")     # 觸發 _on_seq_select 分析第一組

    def _on_seq_select(self):
        sel = self.seq_tree.selection()
        if sel and int(sel[0]) != self.cur_idx:
            self._analyze_current(int(sel[0]))

    @_report_errors("分析失敗")
    def _analyze_current(self, idx):
        sq = self.seqs[idx]
        self.status.config(text=f"計算中: {sq['name']} ...")
        self.root.update_idletasks()
        self.result = analyze_sequence(
            self.folder, thresholds=self.thresholds, diff_mode=self.diff_mode.get(),
            paths=sq["paths"], name=sq["name"])
        self.cur_idx = idx
        self.sel_tile = None
        self.update_tables()
        self.refresh_heatmap()
        self._update_current_row()
        h, w = self.result["shape"]
        extra = f"   (忽略多出的 frame: {sq['extra']})" if sq["extra"] else ""
        self.status.config(text=f"[{idx+1}/{len(self.seqs)}] {sq['name']}: "
                                f"{self.result['n_frames']} frames, {w}x{h}{extra}")

    def _set_seq_row(self, idx, sm, tag):
        self.seq_tree.item(str(idx), tags=(tag,), values=(
            sm["name"], sm["verdict"], sm["pixel_fail_metrics"], sm["block_fail_metrics"],
            f"{sm['danger_pct']:.3f}", sm["worst_level"], f"{sm['max_ratio']:.2f}",
            f"{sm['tiles_red']}/{sm['tiles_orange']}/{sm['tiles_yellow']}"))

    def _update_current_row(self):
        if self.result is None or self.cur_idx is None:
            return
        risk = compute_risk(self.result, self.risk_bounds)
        sm = summarize_result(self.result, risk, self._tiles_for(self.result, risk),
                              self.min_fail)
        self._set_seq_row(self.cur_idx, sm, "fail" if sm["verdict"] == "FAIL" else "pass")

    def _mark_others_stale(self):
        """threshold / 定義改了之後，其他序列的摘要已過期（灰字），需重選或重跑批次"""
        for iid in self.seq_tree.get_children():
            if int(iid) != self.cur_idx and self.seq_tree.item(iid, "values")[1] != "—":
                self.seq_tree.item(iid, tags=("stale",))

    def recompute_all(self):
        if self.cur_idx is not None:
            self._analyze_current(self.cur_idx)
            self._mark_others_stale()

    def apply_thresholds(self, level=None):
        levels = [level] if level else ["pixel", "block"]
        for l in levels:
            for name, var in self.vars[l].items():
                try:
                    self.thresholds[l][name] = float(var.get())
                except (tk.TclError, ValueError):
                    pass
        if self.result:
            # 只重算判定，不重算 metric map
            for l in ["pixel", "block"]:
                self.result[l]["report"] = evaluate_kpi(
                    self.result[l]["maps"], self.thresholds[l])
            self.update_tables()
            self.refresh_heatmap()
            self._update_current_row()
            self._mark_others_stale()

    def apply_risk_bounds(self):
        try:
            self.risk_bounds = parse_risk_bounds(self.risk_bounds_text.get())
        except ValueError as e:
            messagebox.showerror("風險分界錯誤", str(e))
            return
        self.view_mode.set("risk")
        self.refresh_heatmap()
        self._update_current_row()
        self._mark_others_stale()

    def apply_min_fail(self):
        try:
            k = int(self.min_fail_text.get())
            if k < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("K 錯誤", "K 需為 >= 1 的整數")
            return
        self.min_fail = k
        if self.result:
            self.update_tables()
            self.refresh_heatmap()
            self._update_current_row()
            self._mark_others_stale()

    def open_optimizer(self):
        if self.opt_win is not None and self.opt_win.win.winfo_exists():
            self.opt_win.win.lift()
            return
        self.opt_win = ThresholdOptWindow(self)

    def set_thresholds_and_k(self, thr, K):
        """由最佳化視窗套用：thr = {"pixel": {...}, "block": {...}}"""
        for lv in ("pixel", "block"):
            for m, v in thr[lv].items():
                self.vars[lv][m].set(float(v))    # 最佳化端已進位到 4 位並驗證，不可再捨入
        self.min_fail_text.set(str(int(K)))
        self.min_fail = int(K)
        self.apply_thresholds()
        if self.result is None:
            return
        self._update_current_row()
        self._mark_others_stale()

    def apply_tile_params(self):
        try:
            bs = self.result.get("block_size", 2) if self.result else 2
            tile = parse_tile_size(self.tile_text.get(), block_size=bs)
            rules = {k: parse_tile_rule(v.get()) for k, v in self.rule_text.items()}
        except ValueError as e:
            messagebox.showerror("分格參數錯誤", str(e))
            return
        if tile != self.tile_size:
            self.sel_tile = None
        self.tile_size, self.tile_rules = tile, rules
        self.view_mode.set("tile")
        self.refresh_heatmap()
        self._update_current_row()
        self._mark_others_stale()

    def _tiles_for(self, result, risk):
        return compute_tiles(result, risk, self.tile_size, self.tile_rules)

    def reset_thresholds(self):
        for l in ("pixel", "block"):
            for name, v in DEFAULT_THRESHOLDS[l].items():
                self.vars[l][name].set(v)
        self.apply_thresholds()

    def update_tables(self):
        for level in ("pixel", "block"):
            tree = getattr(self, f"tree_{level}")
            tree.delete(*tree.get_children())
            rep = self.result[level]["report"]
            bs = self.result.get("block_size", 2)
            for name in METRIC_NAMES:
                s = rep[name]
                tag = "fail" if metric_is_fail(s, level, self.min_fail, bs) else "pass"
                tree.insert("", "end", values=(
                    name, f"{s['threshold']:.1f}", f"{s['fail_ratio']*100:.4f}%",
                    s["fail_count"], f"{s['max']:.1f}", f"{s['p999']:.1f}",
                    f"{s['mean']:.2f}", f"{s['margin']:.2f}"), tags=(tag,))

    # ------------------------------------------------------------------
    # heatmap
    # ------------------------------------------------------------------
    def refresh_heatmap(self):
        if not self.result:
            return
        level = self.level.get()
        rep = self.result[level]["report"]
        maps = self.result[level]["maps"]
        name = self.metric.get()
        mode = self.view_mode.get()

        if mode != "tile":
            self._tile_view = None
        if mode == "tile":
            risk = compute_risk(self.result, self.risk_bounds)
            tiles = self._tiles_for(self.result, risk)
            img, sc = render_tile_map(self.result, tiles, fit=self._canvas_size())
            self._tile_view = (tiles, sc)
            self._show(img)
            self._draw_tile_selection()
            self._update_tile_info(tiles)
            return

        if mode == "risk":
            risk = compute_risk(self.result, self.risk_bounds)
            self._show(render_risk_map(self.result, risk, fit=self._canvas_size()))
            self._update_risk_info(risk)
            return

        # 縮圖一律用 max-pool（先換算回原圖尺度），1x1 超標點才不會在畫面上消失
        H, W = self.result["shape"]
        f = max(1, H // maps[name].shape[0])          # block-based 的放大倍率
        cw, chh = self._canvas_size()
        fit = (max(1, cw // f), max(1, chh // f))

        if mode == "combined":
            acc, _ = max_pool_fit(combined_fail_map(rep), *fit)
            mx = max(int(acc.max()), 1)
            vis = cv2.applyColorMap(
                (acc / mx * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            desc = f"綜合超標數 (0~{mx} 項 metric 同時 fail)"
        else:
            m, _ = max_pool_fit(maps[name], *fit)
            thr = rep[name]["threshold"]
            vis = render_heatmap(m, thr, mode=mode,
                                 vmax_ratio=self.vmax_ratio.get())
            desc = f"{name}  mode={mode}  threshold={thr:.1f}"

        # block-based 放大回與 pixel-based 相同的顯示尺度
        vis = upscale(vis, f)
        self._show(vis)
        self._update_info(level, name, mode, desc)

    def _canvas_size(self):
        self.canvas.update_idletasks()
        return max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)

    def _show(self, bgr):
        cw, chh = self._canvas_size()
        h, w = bgr.shape[:2]
        scale = min(cw / w, chh / h)
        disp = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(disp[:, :, ::-1])
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, chh // 2, image=self._photo)
        dw, dh = disp.shape[1], disp.shape[0]
        self._geom = (cw // 2 - dw // 2, chh // 2 - dh // 2, dw / w)

    def _on_canvas_click(self, ev):
        if self._tile_view is None or self._geom is None:
            return
        tiles, sc = self._tile_view
        ox, oy, ds = self._geom
        x = (ev.x - ox) / ds / sc            # -> 原圖 pixel 座標
        y = (ev.y - oy) / ds / sc
        H, W = tiles["shape"]
        if not (0 <= x < W and 0 <= y < H):
            return
        t = tiles["tile"]
        self.sel_tile = (int(y // t), int(x // t))
        self._draw_tile_selection()
        self._update_tile_info(tiles)

    def _draw_tile_selection(self):
        self.canvas.delete("sel")
        if self._tile_view is None or self._geom is None or self.sel_tile is None:
            return
        tiles, sc = self._tile_view
        ox, oy, ds = self._geom
        H, W = tiles["shape"]
        t = tiles["tile"]
        r, c = self.sel_tile
        k = sc * ds
        self.canvas.create_rectangle(ox + c * t * k, oy + r * t * k,
                                     ox + min((c + 1) * t, W) * k,
                                     oy + min((r + 1) * t, H) * k,
                                     outline="#00ffff", width=2, dash=(4, 2), tags="sel")

    def _update_tile_info(self, tiles):
        c = tiles["counts"]
        txt = [f"分格總圖  tile {tiles['tile']}x{tiles['tile']}   "
               f"超標格 {tiles['n_fail']}/{tiles['n_tiles']} "
               f"({tiles['n_fail']/tiles['n_tiles']*100:.2f}%)",
               f"  RED {c['RED']}   ({tile_rule_text(tiles['rules']['RED'])})",
               f"  ORANGE {c['ORANGE']}   ({tile_rule_text(tiles['rules']['ORANGE'])})",
               f"  YELLOW {c['YELLOW']}   (其餘有超標的格子)", ""]
        if self.sel_tile is None:
            txt.append("點選圖上的格子查看明細")
        else:
            d = tile_detail(tiles, *self.sel_tile)
            txt.append(f"選取格 (col {d['col']}, row {d['row']})  x {d['x0']}~{d['x1']-1}, "
                       f"y {d['y0']}~{d['y1']-1}   等級: {d['level']}")
            if d["max_ratio"] > 1:
                txt.append(f"  最大 ratio : {d['max_ratio']:.2f}x   ({d['max_metric']}) "
                           f"@ x={d['max_x']}, y={d['max_y']}")
                txt.append(f"  pixel 超標 : {d['pixel_fail']} px     "
                           f"block 超標 : {d['block_fail']} blk")
                txt.append(f"  主要 metric: {d['main_metric']}")
            else:
                txt.append(f"  無超標 (最大 ratio {d['max_ratio']:.2f}x)")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    def _update_risk_info(self, risk):
        txt = ["風險分級總圖  risk = max(value/threshold)，pixel 10 項 + block 10 項", ""]
        for (label, _), rng, fr in zip(RISK_LEVELS, risk_range_labels(risk["bounds"]),
                                       risk["frac"]):
            txt.append(f"  {label:<7}{rng:<12}{fr*100:9.4f}%")
        danger = risk["level"] >= 3
        if danger.any():
            cnt = np.bincount(risk["best"][danger].ravel(), minlength=len(risk["names"]))
            top = np.argsort(-cnt)[:4]
            txt.append("")
            txt.append("FAIL/SEVERE 區主要成因 (最大 ratio 來源 metric, 佔危險區 %):")
            txt.append("  " + ",  ".join(
                f"{risk['names'][i]} {cnt[i]/danger.sum()*100:.1f}%"
                for i in top if cnt[i] > 0))
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    def _update_info(self, level, name, mode, desc):
        rep = self.result[level]["report"]
        txt = [desc, ""]
        if mode != "combined":
            s = rep[name]
            txt.append(f"超標比例 : {s['fail_ratio']*100:.4f}%  "
                       f"({s['fail_count']} / {s['total']})")
            txt.append(f"最大值   : {s['max']:.2f}   (= {s['margin']:.2f} x threshold)")
            txt.append(f"p99.9    : {s['p999']:.2f}    p99: {s['p99']:.2f}    "
                       f"mean: {s['mean']:.3f}")
            txt.append("")
        bs = self.result.get("block_size", 2)
        fails = [(n, rep[n]["fail_ratio"]) for n in METRIC_NAMES
                 if metric_is_fail(rep[n], level, self.min_fail, bs)]
        if fails:
            fails.sort(key=lambda x: -x[1])
            txt.append("此層級 FAIL 的 metric (依超標比例排序):")
            txt.append("  " + ",  ".join(f"{n} {r*100:.4f}%" for n, r in fails))
        else:
            txt.append("此層級全部 metric PASS")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    # ------------------------------------------------------------------
    @_report_errors("匯出失敗")
    def export(self):
        """只匯出目前選取的序列"""
        if not self.result:
            messagebox.showwarning("尚未載入", "請先載入資料夾")
            return
        out = filedialog.askdirectory(title="選擇輸出資料夾")
        if not out:
            return
        os.makedirs(out, exist_ok=True)
        name = self.result["name"]
        write_csv(os.path.join(out, f"{name}_flicker_kpi_report.csv"),
                  CSV_HEADER, metric_rows(self.result, self.min_fail))
        risk = compute_risk(self.result, self.risk_bounds)
        imwrite_unicode(os.path.join(out, f"{name}_risk_map.png"),
                        render_risk_map(self.result, risk))
        imwrite_unicode(os.path.join(out, f"{name}_tile_map.png"),
                        render_tile_map(self.result, self._tiles_for(self.result, risk))[0])
        messagebox.showinfo("完成", f"{name} 已輸出到 {out}")
        self.status.config(text=f"已匯出 {name} 至 {out}")

    @_report_errors("批次匯出失敗")
    def batch_export(self):
        """每組序列：分析 -> <name>_risk_map.png + <name>_tile_map.png；全部序列合併 CSV + summary"""
        if not self.seqs:
            messagebox.showwarning("尚未載入", "請先載入資料夾")
            return
        out = filedialog.askdirectory(title="選擇批次輸出資料夾")
        if not out:
            return
        os.makedirs(out, exist_ok=True)

        all_rows, summ_rows, errors = [], [], []
        n = len(self.seqs)
        for i, sq in enumerate(self.seqs):
            self.status.config(text=f"批次 {i+1}/{n}: {sq['name']} ...")
            self.seq_tree.see(str(i))
            self.root.update()
            try:
                r = analyze_sequence(self.folder, thresholds=self.thresholds,
                                     diff_mode=self.diff_mode.get(),
                                     paths=sq["paths"], name=sq["name"])
                risk = compute_risk(r, self.risk_bounds)
                tiles = self._tiles_for(r, risk)
                imwrite_unicode(os.path.join(out, f"{sq['name']}_risk_map.png"),
                                render_risk_map(r, risk))
                imwrite_unicode(os.path.join(out, f"{sq['name']}_tile_map.png"),
                                render_tile_map(r, tiles)[0])
                sm = summarize_result(r, risk, tiles, self.min_fail)
                all_rows += metric_rows(r, self.min_fail)
                summ_rows.append(summary_row(sm))
                self._set_seq_row(i, sm, "fail" if sm["verdict"] == "FAIL" else "pass")
                if i == self.cur_idx:
                    self.result = r
                del r, risk, tiles                      # 逐組釋放，避免大圖序列吃光記憶體
            except Exception as e:
                errors.append(f"{sq['name']}: {e}")
                summ_rows.append([sq["name"], "ERROR"] + [""] * (len(SUMMARY_HEADER) - 2))
                self.seq_tree.item(str(i), tags=("error",),
                                   values=(sq["name"], "ERROR", "", "", "", "", "", ""))

        write_csv(os.path.join(out, "flicker_kpi_report_all.csv"), CSV_HEADER, all_rows)
        write_csv(os.path.join(out, "flicker_kpi_summary.csv"), SUMMARY_HEADER, summ_rows)

        n_fail = sum(1 for r in summ_rows if r[1] == "FAIL")
        msg = (f"{n} 組序列完成: FAIL {n_fail} / PASS {n - n_fail - len(errors)}"
               f" / ERROR {len(errors)}\n輸出: {out}")
        if errors:
            msg += "\n\n錯誤:\n" + "\n".join(errors[:20])
        self.status.config(text=msg.splitlines()[0])
        messagebox.showinfo("批次完成", msg)


# ======================================================================
# Threshold 最佳化視窗
# ======================================================================
LABEL_FILE = "flicker_labels.csv"
LABEL_TXT = {1: "閃", 0: "不閃", None: "未標記"}


class ThresholdOptWindow:
    """
    不閃 folder：全部序列 label = 不閃（固定）
    混合 folder：逐組標記 閃 / 不閃，存於 <混合 folder>/flicker_labels.csv（載入時自動讀取）
    執行：對每個候選 K 求「漏抓 = 0、誤報最少」的 20 個 threshold，結果可套用回主視窗
    """
    SEQ_COLS = ("source", "sequence", "label", "判定", "max S/t", "主要 metric")

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Threshold 最佳化（漏抓 = 0，誤報最少）")
        self.win.geometry("1300x820")
        self.clean_dir = None
        self.mixed_dir = None
        self.rows = []          # [{"source","name","paths","label"}]
        self.topk_cache = {}    # (paths, diff_mode, block, kmax) -> topk dict
        self.results = []
        self.k_text = tk.StringVar(value=",".join(str(k) for k in kopt.DEFAULT_K_LIST))
        self.tol_text = tk.StringVar(value="0")
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        w = self.win
        f1 = ttk.Frame(w, padding=6)
        f1.pack(fill="x")
        ttk.Button(f1, text="選擇「不閃」folder",
                   command=lambda: self._pick("clean")).grid(row=0, column=0, sticky="we")
        self.clean_lbl = ttk.Label(f1, text="(未選擇)")
        self.clean_lbl.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(f1, text="選擇「混合」folder",
                   command=lambda: self._pick("mixed")).grid(row=1, column=0, sticky="we", pady=2)
        self.mixed_lbl = ttk.Label(f1, text="(未選擇)")
        self.mixed_lbl.grid(row=1, column=1, sticky="w", padx=6)

        pw = ttk.PanedWindow(w, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=1)
        pw.add(right, weight=1)

        # ---- 左：序列 + 標記 ----
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Button(bar, text="選取的標為「閃」",
                   command=lambda: self._set_label(1)).pack(side="left")
        ttk.Button(bar, text="選取的標為「不閃」",
                   command=lambda: self._set_label(0)).pack(side="left", padx=4)
        ttk.Button(bar, text="儲存標記", command=self._save_labels).pack(side="left", padx=4)
        ttk.Label(bar, text="(混合 folder 的列可多選；雙擊切換)").pack(side="left", padx=6)

        tf = ttk.Frame(left)
        tf.pack(fill="both", expand=True, pady=4)
        self.tree = ttk.Treeview(tf, columns=self.SEQ_COLS, show="headings",
                                 selectmode="extended")
        for c, wd in zip(self.SEQ_COLS, (50, 150, 50, 50, 60, 150)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=wd, anchor="w")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._toggle_label)
        self.tree.tag_configure("flick", background="#ffe0e0")
        self.tree.tag_configure("clean", background="#e3f7e3")
        self.tree.tag_configure("unl", background="#f0f0f0")
        self.tree.tag_configure("fa", background="#ff9f9f")      # 誤報
        self.tree.tag_configure("miss", background="#ffcf70")    # 漏抓

        # ---- 右：參數 + 結果 ----
        pf = ttk.LabelFrame(right, text="參數", padding=4)
        pf.pack(fill="x")
        ttk.Label(pf, text="候選 K (pixel 數):").grid(row=0, column=0, sticky="w")
        ttk.Entry(pf, textvariable=self.k_text, width=22).grid(row=0, column=1, sticky="w")
        ttk.Label(pf, text="嚴格模式容忍度 (%):").grid(row=1, column=0, sticky="w")
        ttk.Entry(pf, textvariable=self.tol_text, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(pf, text="嚴格: 門檻 = 不閃樣本最大值 x (1+容忍度)，不漏抓優先\n"
                           "平衡: 閃 / 不閃兩側邊界相等\n兩種模式都會算，結果表可直接比較",
                  foreground="#555").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(pf, text="執行最佳化", command=self.run).grid(
            row=3, column=0, columnspan=2, sticky="we", pady=4)

        cols = ("模式", "K", "誤報", "誤報率", "漏抓", "閃最小S/t", "不閃最大S/t", "解法")
        self.rtree = ttk.Treeview(right, columns=cols, show="headings", height=11,
                                  selectmode="browse")
        for c, wd in zip(cols, (45, 40, 40, 55, 40, 70, 75, 45)):
            self.rtree.heading(c, text=c)
            self.rtree.column(c, width=wd, anchor="e")
        self.rtree.pack(fill="x", pady=4)
        self.rtree.bind("<<TreeviewSelect>>", lambda e: self._show_result())

        ttk.Button(right, text="套用選取結果到主視窗 (threshold + K)",
                   command=self._apply).pack(fill="x")
        self.detail = tk.Text(right, font=("Consolas", 9), height=20)
        self.detail.pack(fill="both", expand=True, pady=4)

        self.status = ttk.Label(w, text="選擇兩個 folder，標記混合 folder 後執行",
                                relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    def _pick(self, kind):
        d = filedialog.askdirectory(parent=self.win,
                                    title="選擇「不閃」folder" if kind == "clean"
                                    else "選擇「混合」folder")
        if not d:
            return
        seqs, skipped = discover_sequences(d, self.app.pattern.get(), N_FRAMES)
        if not seqs:
            messagebox.showerror("沒有序列", f"{d} 找不到完整的 <prefix>0001~{N_FRAMES:04d}",
                                 parent=self.win)
            return
        if skipped:
            messagebox.showwarning("部分 prefix 缺幀已略過",
                                   "\n".join(f"{p}: {r}" for p, r in skipped[:20]),
                                   parent=self.win)
        if kind == "clean":
            self.clean_dir = d
            self.clean_lbl.config(text=f"{d}   ({len(seqs)} 組，全部視為不閃)")
            new = [{"source": "不閃", "name": s["name"], "paths": s["paths"], "label": 0}
                   for s in seqs]
        else:
            self.mixed_dir = d
            saved = self._load_labels(d)
            new = [{"source": "混合", "name": s["name"], "paths": s["paths"],
                    "label": saved.get(s["name"])} for s in seqs]
            n_l = sum(r["label"] is not None for r in new)
            self.mixed_lbl.config(text=f"{d}   ({len(seqs)} 組，已標記 {n_l})")
        src = "不閃" if kind == "clean" else "混合"
        self.rows = [r for r in self.rows if r["source"] != src] + new
        self.rows.sort(key=lambda r: (r["source"] != "不閃", r["name"]))
        self.results = []
        self.rtree.delete(*self.rtree.get_children())
        self._fill_tree()

    def _fill_tree(self, res=None):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.rows):
            pred = ratio = metric = ""
            tag = {1: "flick", 0: "clean", None: "unl"}[r["label"]]
            if res is not None and r.get("_idx") is not None:
                k = r["_idx"]
                ratio = f"{res['seq_ratio'][k]:.2f}"
                metric = "{}:{}".format(*kopt.KEYS[int(res["seq_metric"][k])])
                pred = "閃" if res["seq_ratio"][k] > 1 else "不閃"
                if k in res["fa_idx"]:
                    tag = "fa"
                elif k in res["miss_idx"]:
                    tag = "miss"
            self.tree.insert("", "end", iid=str(i), tags=(tag,), values=(
                r["source"], r["name"], LABEL_TXT[r["label"]], pred, ratio, metric))

    def _set_label(self, lab, iids=None):
        iids = iids if iids is not None else self.tree.selection()
        for iid in iids:
            r = self.rows[int(iid)]
            if r["source"] == "混合":
                r["label"] = lab
                self.tree.set(iid, "label", LABEL_TXT[lab])
                self.tree.item(iid, tags=("flick" if lab else "clean",))
        self._update_mixed_count()

    def _toggle_label(self, ev):
        iid = self.tree.identify_row(ev.y)
        if iid and self.rows[int(iid)]["source"] == "混合":
            cur = self.rows[int(iid)]["label"]
            self._set_label(0 if cur == 1 else 1, [iid])

    def _update_mixed_count(self):
        if self.mixed_dir:
            mixed = [r for r in self.rows if r["source"] == "混合"]
            n_l = sum(r["label"] is not None for r in mixed)
            self.mixed_lbl.config(text=f"{self.mixed_dir}   ({len(mixed)} 組，已標記 {n_l})")

    def _load_labels(self, d):
        path = os.path.join(d, LABEL_FILE)
        out = {}
        if os.path.isfile(path):
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        out[row["sequence"]] = int(row["label"])
                    except (KeyError, ValueError):
                        pass
        return out

    @_report_errors("儲存標記失敗")
    def _save_labels(self):
        if not self.mixed_dir:
            messagebox.showwarning("未選擇", "請先選擇混合 folder", parent=self.win)
            return
        rows = [[r["name"], r["label"]] for r in self.rows
                if r["source"] == "混合" and r["label"] is not None]
        path = os.path.join(self.mixed_dir, LABEL_FILE)
        write_csv(path, ["sequence", "label"], rows)
        self.status.config(text=f"已儲存 {len(rows)} 筆標記 -> {path}  (label: 1=閃, 0=不閃)")

    # ------------------------------------------------------------------
    @_report_errors("最佳化失敗")
    def run(self):
        try:
            k_list = sorted({int(x) for x in self.k_text.get().replace(" ", "").split(",")})
            tol = float(self.tol_text.get()) / 100
            if not k_list or k_list[0] < 1 or tol < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("參數錯誤", "K 需為逗號分隔的正整數；容忍度為 >= 0 的數字 (%)",
                                 parent=self.win)
            return
        use = [r for r in self.rows if r["label"] is not None]
        n_unl = len(self.rows) - len(use)
        if not any(r["label"] == 0 for r in use) or not any(r["label"] == 1 for r in use):
            messagebox.showerror("標記不足", "需要至少一組「不閃」與一組「閃」", parent=self.win)
            return
        if n_unl and not messagebox.askyesno(
                "有未標記序列", f"{n_unl} 組未標記，將不納入計算。繼續？", parent=self.win):
            return

        diff_mode = self.app.diff_mode.get()
        bs = 2
        kmax = max(k_list)
        topks = []
        for i, r in enumerate(use):
            key = (tuple(r["paths"]), diff_mode, bs, kmax)
            if key not in self.topk_cache:
                self.status.config(text=f"計算 metric {i+1}/{len(use)}: {r['source']} {r['name']}")
                self.win.update()
                self.topk_cache[key] = kopt.sequence_topk(r["paths"], kmax, diff_mode, bs)
            topks.append(self.topk_cache[key])
        for r in self.rows:
            r["_idx"] = None
        for k, r in enumerate(use):
            r["_idx"] = k
        labels = [r["label"] == 1 for r in use]

        self.status.config(text="最佳化中...")
        self.win.update()
        self.results = kopt.sweep_k(topks, labels, k_list, bs, tol=tol)
        self.used = use
        n_clean = sum(1 for l in labels if not l)
        self.rtree.delete(*self.rtree.get_children())
        for i, res in enumerate(self.results):
            self.rtree.insert("", "end", iid=str(i), values=(
                "嚴格" if res["mode"] == "strict" else "平衡",
                res["K"], len(res["fa_idx"]), f"{len(res['fa_idx'])/n_clean*100:.1f}%",
                len(res["miss_idx"]), f"{res['flick_min']:.3f}", f"{res['clean_max']:.3f}",
                "精確" if res["exact"] else "近似"))
        self.rtree.selection_set("0")                  # 嚴格模式的最佳結果
        best = self.results[0]
        self.status.config(
            text=f"完成：{len(use)} 組 (閃 {sum(labels)} / 不閃 {n_clean})；"
                 f"嚴格模式最佳 K={best['K']}，誤報 {len(best['fa_idx'])}，"
                 f"漏抓 {len(best['miss_idx'])}")

    def _show_result(self):
        sel = self.rtree.selection()
        if not sel or not self.results:
            return
        res = self.results[int(sel[0])]
        self._fill_tree(res)
        use = self.used
        cur = self.app.thresholds
        if res["mode"] == "strict":
            mode_txt = f"嚴格 (容忍度 {res['tol']*100:g}%"
            if res["tol_applied"] is not None and res["tol_applied"] < res["tol"] - 1e-12:
                mode_txt += f"，為避免漏抓自動降為 {res['tol_applied']*100:.2f}%"
            mode_txt += ")"
        else:
            mode_txt = f"平衡 (分離度 {res['separation']:.3f})"
        lines = [f"模式: {mode_txt}",
                 f"K = {res['K']}  (block: {k_for_level(res['K'], 'block')} 個 block)   "
                 f"解法: {'精確 (MILP)' if res['exact'] else '近似 (greedy)'}",
                 f"誤報 {len(res['fa_idx'])}   漏抓 {len(res['miss_idx'])}   "
                 f"閃序列最小 S/t {res['flick_min']:.3f}   "
                 f"不閃序列最大 S/t {res['clean_max']:.3f}", "",
                 f"{'metric':<22}{'目前':>10}{'最佳化':>12}"]
        for j, (lv, m) in enumerate(kopt.KEYS):
            mark = " *" if j in res["lowered"] else ""
            lines.append(f"{lv+':'+m:<22}{cur[lv][m]:>10.2f}{res['thresholds'][j]:>12.2f}{mark}")
        lines.append("  (* = 造成誤報的 metric)")
        lines.append("")
        if res["fa_idx"]:
            lines.append("誤報的不閃序列:")
            lines += [f"  {use[k]['source']} {use[k]['name']}  S/t={res['seq_ratio'][k]:.2f}"
                      for k in res["fa_idx"]]
        ci = [k for k, r in enumerate(use) if r["label"] == 0 and k not in res["fa_idx"]]
        fi = [k for k, r in enumerate(use) if r["label"] == 1]
        if ci:
            k = max(ci, key=lambda k: res["seq_ratio"][k])
            lines.append(f"最接近門檻的不閃序列: {use[k]['name']}  S/t={res['seq_ratio'][k]:.3f}")
        if fi:
            k = min(fi, key=lambda k: res["seq_ratio"][k])
            lines.append(f"最接近門檻的閃序列  : {use[k]['name']}  S/t={res['seq_ratio'][k]:.3f}")
        lines.append("")
        lines.append("S/t > 1 判閃。閃最小 S/t 越大 -> 閃的抓得越穩；不閃最大 S/t 越小 -> 越不易誤報")
        lines.append("S = 該 metric 第 K 大的值 (即超標位置數 >= K 的判定量)")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", "\n".join(lines))

    def _apply(self):
        sel = self.rtree.selection()
        if not sel or not self.results:
            messagebox.showwarning("無結果", "請先執行最佳化並選一個結果", parent=self.win)
            return
        res = self.results[int(sel[0])]
        self.app.set_thresholds_and_k(kopt.thresholds_to_dict(res["thresholds"]), res["K"])
        self.status.config(text=f"已套用 {'嚴格' if res['mode'] == 'strict' else '平衡'}模式 "
                                f"K={res['K']} 與 20 個 threshold 到主視窗")
        self._show_result()


if __name__ == "__main__":
    root = tk.Tk()
    FlickerKPIApp(root)
    root.mainloop()
