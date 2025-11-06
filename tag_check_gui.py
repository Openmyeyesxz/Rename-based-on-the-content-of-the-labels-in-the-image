# -*- coding: utf-8 -*-
"""人工核验（直接改名）：三段式命名；文件名仅在左侧区域居中；CPU/内存；编号可空；方向键/回车快捷"""
from __future__ import annotations

import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps  # EXIF 方向纠正

try:
    import psutil
except Exception:
    psutil = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SANITIZE_RE = re.compile(r"[^A-Za-z0-9\-_]+")


def sanitize_and_upper(s: str) -> str:
    s = (s or "").strip()
    s = SANITIZE_RE.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s.upper()


class ReviewFrame(tk.Frame):
    def __init__(self, parent, on_title=None, on_need_close=None):
        super().__init__(parent)
        self.on_title = on_title
        self.on_need_close = on_need_close

        # 状态
        self.image_root: Path | None = None
        self.files: list[Path] = []
        self.idx = -1
        self._tkimg = None

        # 选项
        self.rotate_upright = tk.BooleanVar(value=True)   # 宽>高时旋转90°
        self.keep_prefix = tk.BooleanVar(value=False)     # 前缀沿用
        self.keep_middle = tk.BooleanVar(value=False)     # 中间项沿用
        self.keep_index  = tk.BooleanVar(value=False)     # 编号沿用
        self.index_custom_mode = tk.BooleanVar(value=False)  # 使用自定义编号

        self._build_ui()
        self.after(700, self._update_resource)

    # ---- UI ----
    def _build_ui(self):
        base_font = ("Microsoft YaHei UI", 11)
        s = ttk.Style(self)
        s.configure("TLabel", font=base_font)
        s.configure("TButton", font=base_font)
        s.configure("TCheckbutton", font=base_font)
        s.configure("TEntry", font=base_font)
        s.configure("TCombobox", font=base_font)

        # 顶栏
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="📂 选择图片根目录", command=self._pick_root).pack(side="left")
        self.var_recur = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="递归子文件夹", variable=self.var_recur, command=self._reload).pack(side="left", padx=(8,0))
        ttk.Button(top, text="刷新列表", command=self._reload).pack(side="left", padx=(8,0))
        self.lbl_total = ttk.Label(top, text="0/0")
        self.lbl_total.pack(side="right")

        # 主区：左图区（文件名+画布） / 右控件
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # ——左侧容器（只在这个区域内居中文件名）——
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.lbl_name = ttk.Label(left, text="", anchor="center",
                                  font=("Microsoft YaHei UI", 16, "bold"))
        self.lbl_name.pack(fill="x", padx=4, pady=(0,4))
        self.canvas = tk.Canvas(left, bg="#f6f6f6")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_current_image())

        # ——右侧功能区——
        right = ttk.Frame(main)
        right.pack(side="right", fill="y", padx=(10,0))

        ttk.Checkbutton(right, text="强制竖直显示（宽>高时旋转90°）",
                        variable=self.rotate_upright, command=self._draw_current_image).pack(anchor="w")

        lf = ttk.LabelFrame(right, text="三段式命名（有主体时自动加 '-'；空段不输出）")
        lf.pack(fill="x", pady=6)

        # 前缀
        r1 = ttk.Frame(lf)
        r1.pack(fill="x", padx=6, pady=4)
        ttk.Label(r1, text="前缀：").pack(side="left")
        self.var_prefix = tk.StringVar(value="")
        self.ent_prefix = ttk.Entry(r1, textvariable=self.var_prefix, width=22)
        self.ent_prefix.pack(side="left", padx=6)
        ttk.Checkbutton(r1, text="沿用", variable=self.keep_prefix).pack(side="left")

        # 中间项
        r2 = ttk.Frame(lf)
        r2.pack(fill="x", padx=6, pady=4)
        ttk.Label(r2, text="中间项：").pack(side="left")
        self.var_middle = tk.StringVar(value="")
        self.ent_middle = ttk.Entry(r2, textvariable=self.var_middle, width=30)
        self.ent_middle.pack(side="left", padx=6)
        ttk.Checkbutton(r2, text="沿用", variable=self.keep_middle).pack(side="left")

        # 编号（含“沿用”）
        r3 = ttk.Frame(lf)
        r3.pack(fill="x", padx=6, pady=4)
        ttk.Label(r3, text="编号：").pack(side="left")
        idx_values = [""] + [str(i) for i in range(1, 101)]  # 可为空
        self.var_index_combo = tk.StringVar(value="")
        self.cb_index = ttk.Combobox(r3, values=idx_values, textvariable=self.var_index_combo,
                                     width=8, state="readonly")
        self.cb_index.pack(side="left", padx=6)

        ttk.Checkbutton(r3, text="自定义", variable=self.index_custom_mode,
                        command=self._toggle_index_mode).pack(side="left", padx=(10,4))
        self.var_index_custom = tk.StringVar(value="")
        self.ent_index_custom = ttk.Entry(r3, textvariable=self.var_index_custom,
                                          width=12, state="disabled")
        self.ent_index_custom.pack(side="left", padx=(0,10))

        # 编号“沿用”开关
        ttk.Checkbutton(r3, text="沿用", variable=self.keep_index).pack(side="left")

        # 预览
        self.var_preview = tk.StringVar(value="预览文件名：")
        ttk.Label(lf, textvariable=self.var_preview).pack(fill="x", padx=6, pady=(6,2))

        # 输入变更实时预览
        for v in (self.var_prefix, self.var_middle, self.var_index_combo, self.var_index_custom):
            v.trace_add("write", lambda *_: self._update_preview())

        # 导航/动作
        nav = ttk.Frame(right)
        nav.pack(fill="x", pady=(6,0))
        ttk.Button(nav, text="← 上一个", command=self.prev_item).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="下一个 →", command=self.next_item).pack(side="left", expand=True, fill="x", padx=(6,0))

        act = ttk.Frame(right)
        act.pack(fill="x", pady=6)
        ttk.Button(act, text="通过（不改）并下一张（方向键）", command=self.pass_and_next).pack(fill="x", pady=4)
        ttk.Button(act, text="保存并下一张（回车）", command=self.save_and_next).pack(fill="x", pady=4)
        ttk.Button(act, text="🧹 清空输入", command=self._clear_inputs).pack(fill="x", pady=4)

        # 底部状态与资源
        self.var_status = tk.StringVar(value="状态：未加载")
        ttk.Label(right, textvariable=self.var_status, foreground="#666").pack(fill="x", pady=(8,0))

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(0,8))
        self.lbl_progress = ttk.Label(bottom, text="进度：0/0")
        self.lbl_progress.pack(side="left")
        self.lbl_usage = ttk.Label(bottom, text="CPU 0% | 内存 0/0")
        self.lbl_usage.pack(side="right")

        # 全局快捷键：方向键只换图不改名
        # 回车保存并下一张
        self.bind_all("<KeyPress-Left>", self._nav_left, add=True)
        self.bind_all("<KeyPress-Right>", self._nav_right, add=True)
        self.bind_all("<Return>", self._hit_enter, add=True)

    # 资源显示
    def _update_resource(self):
        try:
            if psutil:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                self.lbl_usage.config(text=f"CPU {int(cpu)}% | 内存 {mem.used//(1024**2)}M/{mem.total//(1024**2)}M")
        finally:
            self.after(1000, self._update_resource)

    # 文件加载
    def _pick_root(self):
        d = filedialog.askdirectory(title="选择图片根目录", parent=self)
        if not d: return
        self.image_root = Path(d)
        self._reload()

    def _reload(self):
        if not self.image_root:
            self.files = []
            self.idx = -1
            self._refresh_view()
            return
        if self.var_recur.get():
            self.files = sorted([p for p in self.image_root.rglob("*")
                                 if p.is_file() and p.suffix.lower() in IMG_EXTS],
                                key=lambda p: str(p).lower())
        else:
            self.files = sorted([p for p in self.image_root.iterdir()
                                 if p.is_file() and p.suffix.lower() in IMG_EXTS],
                                key=lambda p: str(p).lower())
        self.idx = 0 if self.files else -1
        self._refresh_view()

    # 视图刷新
    def _refresh_view(self):
        total = len(self.files)
        cur = self.idx + 1 if self.idx >= 0 else 0
        self.lbl_total.config(text=f"总数：{total}")
        self.lbl_progress.config(text=f"进度：{cur}/{total}")
        if self.on_title:
            self.on_title(f"人工核验（{cur}/{total}）")

        if self.idx < 0 or not self.files:
            self.lbl_name.config(text="")
            self.var_preview.set("预览文件名：")
            self.var_status.set("状态：未加载")
            self.canvas.delete("all")
            return

        p = self.files[self.idx]
        self.lbl_name.config(text=p.name)             # 左侧居中文件名
        self.var_status.set(f"状态：{p}")

        # 切图后的默认填充：“沿用”勾选
        # 前缀
        if not self.keep_prefix.get():
            self.var_prefix.set("")
        # 中间项
        if not self.keep_middle.get():
            self.var_middle.set("")
        # 编号
        if not self.keep_index.get():
            self.index_custom_mode.set(False)
            self.var_index_combo.set("")
            self.var_index_custom.set("")
            self.ent_index_custom.configure(state="disabled")

        self._draw_current_image()
        self._update_preview()

    def _draw_current_image(self):
        self.canvas.delete("all")
        if self.idx < 0 or not self.files: return
        p = self.files[self.idx]
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)  # EXIF 方向纠正
            if self.rotate_upright.get() and img.width > img.height:
                img = img.rotate(270, expand=True)  # 宽>高时逆时针90°
            cw = max(100, self.canvas.winfo_width() or 900)
            ch = max(100, self.canvas.winfo_height() or 640)
            img.thumbnail((cw-20, ch-20))
            self._tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self._tkimg)
        except Exception as e:
            self.canvas.create_text(10, 10, anchor="nw", text=f"图片加载失败：{e}")

    def _toggle_index_mode(self):
        self.ent_index_custom.configure(state="normal" if self.index_custom_mode.get() else "disabled")
        self._update_preview()

    # ---- 预览名 ----
    def _compose_stem(self) -> str:
        prefix = sanitize_and_upper(self.var_prefix.get())
        middle = sanitize_and_upper(self.var_middle.get())
        idx = sanitize_and_upper(self.var_index_custom.get()) if self.index_custom_mode.get() \
              else sanitize_and_upper(self.var_index_combo.get())
        parts = []
        if prefix: parts.append(prefix)
        if middle: parts.append(middle)
        base = "".join(parts) if parts else "UNNAMED"
        return f"{base}-{idx}" if idx else base

    def _update_preview(self):
        if self.idx < 0 or not self.files:
            self.var_preview.set("预览文件名：")
            return
        ext = self.files[self.idx].suffix or ".jpg"
        self.var_preview.set(f"预览文件名：{self._compose_stem()}{ext}")

    # 全局快捷键（方向键只换图；回车保存）
    def _nav_left(self, e):
        self.prev_item()
        return "break"

    def _nav_right(self, e):
        self.next_item()
        return "break"

    def _hit_enter(self, e):
        self.save_and_next()
        return "break"

    # ---- 导航/操作 ----
    def prev_item(self):
        if not self.files: return
        self.idx = max(0, self.idx - 1)
        self._refresh_view()

    def next_item(self):
        if not self.files: return
        self.idx = min(len(self.files) - 1, self.idx + 1)
        self._refresh_view()

    def pass_and_next(self):
        self.next_item()

    def _clear_inputs(self):
        self.var_prefix.set("")
        self.var_middle.set("")
        self.var_index_combo.set("")
        self.var_index_custom.set("")
        self.index_custom_mode.set(False)
        self.ent_index_custom.configure(state="disabled")
        self._update_preview()

    def save_and_next(self):
        if self.idx < 0 or not self.files: return
        src = self.files[self.idx]
        stem_new = self._compose_stem()
        if not stem_new:
            messagebox.showwarning("提示", "目标文件名为空。", parent=self)
            return
        dst = src.with_name(f"{stem_new}{src.suffix}")
        
        if dst.exists() and dst.resolve() != src.resolve():
            messagebox.showerror("冲突", f"目标已存在：\n{dst}", parent=self)
            return
        try:
            src.rename(dst)
            self.files[self.idx] = dst
            self.var_status.set(f"已改名：{dst.name}")
        except Exception as e:
            messagebox.showerror("失败", f"改名失败：\n{e}", parent=self)
            return
        self.next_item()


def build_frame(parent, on_title=None, on_need_close=None):
    frm = ReviewFrame(parent, on_title=on_title, on_need_close=on_need_close)
    frm.pack(fill="both", expand=True)
    return frm


if __name__ == "__main__":
    root = tk.Tk()
    root.title("人工核验（独立窗口）")
    build_frame(root)
    root.mainloop()

