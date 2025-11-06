#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行版：YOLO 检测白色标签 → 裁剪 → Ark OCR → 批量安全改名
依赖：ultralytics, opencv-python, volcenginesdkarkruntime
默认行为：
- 设备：CPU
- 非递归、非演练（真实改名）
- 保存裁剪图到 INPUT/cropped，并在本次结束后清空
- 运行前清空“改名后输出目录”（如不想清空，命令行加 --no-clean-out）
- 日志仅输出到 stdout（被 Slurm 收集到 slurm.o）；除非显式传 --log-file 才额外落盘
- 重复处理策略通过 --duplicates 指定（必填）：True=编号去重；False=直接用 OCR 为文件名，不做去重
"""

import os, re, csv, sys, uuid, time, shutil, argparse
import cv2
import base64 as _b64
from ultralytics import YOLO
from volcenginesdkarkruntime import Ark
from pathlib import Path
from collections import defaultdict

# ========= 常量（按需调整） =========
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SANITIZE_RE = re.compile(r"[^A-Za-z0-9\-_]+")
##调用你自己的模型和其他模型设置
DEFAULT_ARK_MODEL = "doubao-1-5-thinking-vision-pro-250428"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_CLASS_NAME = "WhiteTag"
# 可在此填默认 Ark Key；留空则需要 --ark-key 或环境变量 ARK_API_KEY
DEFAULT_ARK_KEY = "your key"

# ========= 小工具 =========
def log(msg: str, file=None):
 t = time.strftime("%H:%M:%S")
 s = f"[{t}] {msg}"
 print(s, flush=True)
 if file:
  try:
   file.write(s + "\n"); file.flush()
  except Exception:
   pass

def sanitize_and_upper(s: str) -> str:
 s = (s or "").strip()
 s = SANITIZE_RE.sub("-", s)
 s = re.sub(r"-{2,}", "-", s).strip("-")
 return s.upper() or "UNNAMED"

def _normalize_device_str(device_arg: str):
 if not device_arg: return None
 dev = str(device_arg).strip().lower()
 if dev in ("cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"):
  return dev
 if dev.isdigit(): return f"cuda:{dev}"
 return dev

def plan_final_name(base: str, counts: dict, reserved: set, ext: str, existing: set) -> str:
 IS_WIN = os.name == "nt"
 idx = counts.get(base, 0) + 1
 while True:
  cand = f"{base}-{idx}{ext}"
  key = cand.lower() if IS_WIN else cand
  if (key not in reserved) and (key not in existing):
   counts[base] = idx
   reserved.add(key)
   return cand
  idx += 1

def safe_batch_rename(pairs, dry_run=False, log_fn=print):
 """临时名→目标名；跨盘失败回退 shutil.move。"""
 tmps = []
 try:
  for src, dst in pairs:
   tmp = src.with_name(f"__TMP__{uuid.uuid4().hex}__{src.name}")
   if dry_run: log_fn(f"[演练] 临时改名：{src.name} -> {tmp.name}")
   else: src.rename(tmp)
   tmps.append((tmp, dst))
  for tmp, dst in tmps:
   if dry_run: log_fn(f"[演练] 最终改名：{tmp.name} -> {dst.name}")
   else:
    try:
     tmp.rename(dst)
    except Exception:
     dst.parent.mkdir(parents=True, exist_ok=True)
     shutil.move(str(tmp), str(dst))
 except Exception as e:
  log_fn(f"[错误] 批量改名失败：{e}")
  for tmp, _ in tmps:
   if tmp.exists():
    try:
     orig = tmp.with_name(tmp.name.split("__", 2)[-1]) if "__TMP__" in tmp.name else tmp
     tmp.rename(orig)
    except Exception:
     pass
  raise

# ========= Ark OCR =========
def _ark_client(api_key: str):

 return Ark(base_url=DEFAULT_ARK_BASE_URL, api_key=api_key)

def _ndarray_to_data_url(img_bgr, mime="image/png"):

 ok, buf = cv2.imencode(".png", img_bgr)
 if not ok: raise RuntimeError("图像编码失败")
 b64 = _b64.b64encode(buf.tobytes()).decode("utf-8")
 return f"data:{mime};base64,{b64}"

def _ark_ocr(client, model: str, crop_bgr, prompt: str) -> str:
 data_url = _ndarray_to_data_url(crop_bgr, mime="image/png")
 try:
  resp = client.chat.completions.create(
   model=model,
   messages=[{
    "role": "user",
    "content": [
     {"type": "image_url", "image_url": data_url},
     {"type": "text", "text": prompt},
    ],
   }],
  )
  text = (resp.choices[0].message.content or "").strip()
  lines = [ln for ln in text.splitlines() if ln.strip()]
  return lines[-1] if lines else ""
 except Exception as e:
  return f"[OCR错误]{e}"

# ========= YOLO 裁剪（合并标签框最小外接矩形） =========
def _detect_crop_legacy(yolo_model, image_bgr, target_class_name: str):
 class_names = yolo_model.names
 class_id = None
 for cid, name in class_names.items():
  if str(name).lower() == str(target_class_name).lower():
   class_id = cid; break
 if class_id is None:
  return None

 results = yolo_model(image_bgr)
 min_x, min_y = float('inf'), float('inf')
 max_x, max_y = float('-inf'), float('-inf')
 for r in results:
  boxes = r.boxes.cpu().numpy() if r.boxes is not None else []
  for box in boxes:
   cls_id = int(box.cls[0])
   if cls_id != class_id: continue
   x1, y1, x2, y2 = map(int, box.xyxy[0])
   min_x = min(min_x, x1); min_y = min(min_y, y1)
   max_x = max(max_x, x2); max_y = max(max_y, y2)

 if min_x == float('inf'): return None
 h, w = image_bgr.shape[:2]
 x1 = max(0, min(min_x, w-1)); y1 = max(0, min(min_y, h-1))
 x2 = max(0, min(max_x, w-1)); y2 = max(0, min(max_y, h-1))
 if x2 <= x1: x2 = min(w-1, x1+1)
 if y2 <= y1: y2 = min(h-1, y1+1)
 return image_bgr[y1:y2, x1:x2]

# ========= 遍历 =========
def iter_images(root: Path, recursive: bool):
 it = root.rglob("*") if recursive else root.iterdir()
 for p in it:
  if p.is_file() and p.suffix.lower() in IMG_EXTS:
   yield p

# ========= 主程序 =========
def parse_bool_choice(v: str) -> bool:
 if isinstance(v, bool): return v
 s = str(v).strip().lower()
 if s in ("true", "t", "1", "yes", "y"): return True
 if s in ("false", "f", "0", "no", "n"): return False
 raise argparse.ArgumentTypeError("必须为 True/False")

def main():
 parser = argparse.ArgumentParser(
  description="YOLO_OCR_Rename(CIL)",
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
 )
 # 必填
 parser.add_argument("-i", "--input", required=True, type=Path, help="输入原图目录")
 parser.add_argument("-w", "--weights", required=True, type=Path, help="YOLO 权重 .pt")
 parser.add_argument("-o", "--out-renamed", required=True, type=Path, help="改名后输出目录（不能与输入相同）")
 parser.add_argument("--prompt", required=True, help="OCR 提示词（必填，无默认）")

 # 识别/设备/Ark
 parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME, help="YOLO 类别名")
 parser.add_argument("--ark-key", default=None,
  help="Ark API Key（不提供时用环境变量 ARK_API_KEY；再退回 DEFAULT_ARK_KEY）")
 parser.add_argument("--ark-model", default=DEFAULT_ARK_MODEL, help="Ark 模型版本")
 parser.add_argument("--device", default="cpu", help="设备：cpu / cuda / cuda:0 等（默认 cpu）")

 # 裁剪图：默认保存到 INPUT/cropped，处理完成后默认清空
 parser.add_argument("--save-crops", dest="save_crops", action="store_true", default=True,
  help="保存裁剪图（默认）")
 parser.add_argument("--no-save-crops", dest="save_crops", action="store_false",
  help="不保存裁剪图")
 parser.add_argument("--crops-dir", type=Path, default=Path("cropped"),
  help="裁剪图输出目录（相对路径时放到输入目录下）")
 parser.add_argument("--clean-crops-after", dest="clean_crops_after", action="store_true", default=True,
  help="处理完成后清空裁剪目录（默认）")
 parser.add_argument("--no-clean-crops-after", dest="clean_crops_after", action="store_false",
  help="处理完成后不清空裁剪目录")

 # 其它：默认清空输出目录；可显式关闭。非递归、非演练。
 parser.add_argument("--clean-out", dest="clean_out", action="store_true", default=True,
  help="运行前清空改名输出目录（默认）")
 parser.add_argument("--no-clean-out", dest="clean_out", action="store_false",
  help="不清空改名输出目录")
 parser.add_argument("--recursive", action="store_true", help="递归子文件夹（默认否）")
 parser.add_argument("--dry-run", action="store_true", help="仅演练，不真正改名（默认否）")

 # 日志：默认不落盘，仅 stdout（便于 Slurm 收集到 slurm.o）
 parser.add_argument("--csv", type=Path, default=None, help="映射表 CSV 路径（默认写到输入目录 rename_mapping.csv）")
 parser.add_argument("--log-file", type=Path, default=None, help="可选：另存日志到文件（默认不保存）")

 # ★ 重复处理选项（必填，无默认）
 parser.add_argument("--duplicates", required=True, type=parse_bool_choice,
  help="是否存在重复样本：True=使用编号去重(Base-1/-2/-3…)，False=不做重复检测，直接用OCR结果为文件名")

 args = parser.parse_args()

 # 校验必填/默认
 if not str(args.prompt).strip():
  print("错误：--prompt 不能为空。"); sys.exit(2)

 in_dir: Path = args.input
 if not in_dir.is_dir():
  print(f"错误：输入目录不存在：{in_dir}"); sys.exit(2)
 weights: Path = args.weights
 if not weights.is_file():
  print(f"错误：YOLO 权重不存在：{weights}"); sys.exit(2)
 if in_dir.resolve() == args.out_renamed.resolve():
  print("错误：改名后输出目录不能与输入目录相同。"); sys.exit(2)
 out_dir: Path = args.out_renamed; out_dir.mkdir(parents=True, exist_ok=True)

 # 运行前清空输出目录（默认真）
 if args.clean_out:
  cnt = 0
  for p in out_dir.glob("*"):
   try:
    if p.is_file(): p.unlink()
    elif p.is_dir(): shutil.rmtree(p)
    cnt += 1
   except Exception:
    pass
  print(f"[info] 已清空输出目录 {out_dir}（清理项 {cnt}）")

 # 日志文件：仅当显式给出 --log-file 时才落盘
 log_fp = None
 if args.log_file:
  args.log_file.parent.mkdir(parents=True, exist_ok=True)
  log_fp = open(args.log_file, "w", encoding="utf-8")

 # Ark Key 优先级：命令行 > 环境变量 > 代码常量
 ark_key = (args.ark_key or os.getenv("ARK_API_KEY") or DEFAULT_ARK_KEY).strip()
 if not ark_key:
  print("错误：未提供 Ark API Key（--ark-key 或 ARK_API_KEY，或在脚本 DEFAULT_ARK_KEY 中填写）")
  sys.exit(2)

 # 惰性导入重型依赖



 device = _normalize_device_str(args.device)
 client = _ark_client(ark_key)

 log("加载 YOLO 权重中...", log_fp)
 model = YOLO(str(weights))
 if device and device != "cpu":
  try:
   model.to(device); log(f"使用设备：{device}", log_fp)
  except Exception as e:
   log(f"[warning] 切换设备失败，使用默认设备：{e}", log_fp)
 else:
  log("使用设备：cpu", log_fp)

 # 裁剪输出
 save_crops = args.save_crops
 crops_dir = None
 if save_crops:
  crops_dir = args.crops_dir
  if not crops_dir.is_absolute():
   crops_dir = in_dir / crops_dir
  crops_dir.mkdir(parents=True, exist_ok=True)
  log(f"[info] 裁剪图输出：{crops_dir}", log_fp)

 # CSV 映射表
 out_csv = (args.csv if args.csv else (in_dir / "rename_mapping.csv"))
 out_csv.parent.mkdir(parents=True, exist_ok=True)
 fcsv = open(out_csv, "w", newline="", encoding="utf-8")
 writer = csv.writer(fcsv)
 writer.writerow(["src_dir","old_name","ocr_text","base_sanitized","index","final_name","status"])

 planned = []
 counts_by_dir = defaultdict(dict)
 reserved_by_dir = defaultdict(set)
 existing_cache = {}

 images = sorted(list(iter_images(in_dir, recursive=args.recursive)), key=lambda p: str(p).lower())
 total = len(images)
 if total == 0:
  log("未发现待处理图片。", log_fp); fcsv.close(); sys.exit(0)

 ok = fail = 0
 t0 = time.time()
 for i, img_path in enumerate(images, 1):
  log(f"{i}/{total} 处理：{img_path.name}", log_fp)
  img = cv2.imread(str(img_path))
  if img is None:
   writer.writerow([str(img_path.parent), img_path.name, "", "", "", "", "READ_FAIL"]); fail += 1
   log(f"[跳过] 无法读取：{img_path.name}", log_fp); continue

  crop = _detect_crop_legacy(model, img, args.class_name)
  if crop is None:
   writer.writerow([str(img_path.parent), img_path.name, "", "", "", "", "NO_DET"]); fail += 1
   log(f"[提示] 未检测到 {args.class_name}：{img_path.name}", log_fp); continue

  if save_crops and crops_dir:
   cp = crops_dir / f"{img_path.stem}_cropped{img_path.suffix.lower()}"
   try:
    import cv2 as _cv2
    _cv2.imwrite(str(cp), crop)
    log(f"Saved crop: {cp.name}", log_fp)
   except Exception as e:
    log(f"[warning] 保存裁剪失败：{e}", log_fp)

  # OCR
  ocr_text = _ark_ocr(client, args.ark_model, crop, args.prompt)
  if (not ocr_text) or ocr_text.startswith("[OCR错误]"):
   writer.writerow([str(img_path.parent), img_path.name, ocr_text, "", "", "", "NO_TEXT"]); fail += 1
   log(f"[提示] OCR 无结果/错误：{ocr_text}", log_fp); continue

  base = sanitize_and_upper(ocr_text)
  ext = img_path.suffix.lower()
  target_dir = out_dir

  # 为冲突检查构建“已存在”缓存（大小写无关/有关由平台决定）
  if target_dir not in existing_cache:
   IS_WIN = os.name == "nt"
   existing_cache[target_dir] = {
    (p.name.lower() if IS_WIN else p.name)
    for p in target_dir.iterdir() if p.is_file()
   }

  if args.duplicates:
   # === 有重复：使用编号去重 ===
   final_name = plan_final_name(base, counts_by_dir[target_dir],
    reserved_by_dir[target_dir], ext, existing_cache[target_dir])
   dst = target_dir / final_name
   planned.append((img_path, dst))
   # 解析索引
   try:
    idx_val = int(Path(final_name).stem.split("-")[-1])
   except Exception:
    idx_val = ""
   writer.writerow([str(img_path.parent), img_path.name, ocr_text, base, idx_val, final_name, "OK"])
   log(f"✔ {img_path.name} -> {final_name}", log_fp); ok += 1
  else:
   # === 无重复：直接用 OCR 结果作为文件名，不做编号/去重 ===
   final_name = f"{base}{ext}"
   key = (final_name.lower() if os.name == "nt" else final_name)
   if (key in existing_cache[target_dir]) or (key in reserved_by_dir[target_dir]):
    writer.writerow([str(img_path.parent), img_path.name, ocr_text, base, "", final_name, "NAME_CONFLICT"])
    log(f"[冲突] 目标已存在，跳过：{final_name}", log_fp); fail += 1
    continue
   reserved_by_dir[target_dir].add(key)
   dst = target_dir / final_name
   planned.append((img_path, dst))
   writer.writerow([str(img_path.parent), img_path.name, ocr_text, base, "", final_name, "OK"])
   log(f"✔ {img_path.name} -> {final_name}", log_fp); ok += 1

 # 执行批量改名/移动
 try:
  safe_batch_rename(planned, dry_run=args.dry_run, log_fn=lambda s: log(s, log_fp))
 except Exception as e:
  log(f"[目录级错误] 改名中断：{e}", log_fp)

 fcsv.close()
 log(f"\n完成：成功 {ok}，失败 {fail}，耗时 {time.time()-t0:.1f}s", log_fp)
 log(f"映射表：{out_csv}", log_fp)

 # 处理完成后按默认清空裁剪目录
 if save_crops and crops_dir and args.clean_crops_after and crops_dir.exists():
  removed = 0
  for p in crops_dir.glob("*"):
   try:
    if p.is_file():
     p.unlink()
    else:
     shutil.rmtree(p)
    removed += 1
   except Exception:
    pass
  log(f"[info] 已清空裁剪目录 {crops_dir}（删除 {removed} 项）", log_fp)

 if log_fp:
  log_fp.close()

if __name__ == "__main__":
 main()
