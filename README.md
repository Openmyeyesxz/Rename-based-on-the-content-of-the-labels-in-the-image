# Rename based on the content of the labels in the image
  During field trials, sample names or sample codes are typically displayed on rigid labels of various colours. To reduce labour costs, minimise time expenditure, and enhance operational efficiency during phenotypic data collection, we have developed a streamlined programme based on AI-OCR technology to simplify our workflow.

## Note⚠️: If you have a file containing multiple images, after the renaming process completes, successfully renamed images will be moved from the original file into the newly created folder and will not remain in the original folder.

> **Our process defaults to YOLO-based segmentation and the Doubao AI.**

> **We provide run files that can process multiple subsets in a loop.**

> **Additionally, we provide an executable program for manually reviewing photos, which can change your photo names in real time.**

# 1. Files structure
## Job-root layout (typical)
```
<WORKDIR>/
├─ photo-rename.slurm        # your sbatch script
├─ detect_tags.py            # Python entrypoint (SCRIPT)
├─ slurm.o                   # stdout from all sub-runs
├─ slurm.e                   # stderr from all sub-runs
└─ test/                     # ROOT (processed subfolders live here)
   ├─ aaa/ ...
   ├─ bbb/ ...
   └─ ...
```
**Per-subfolder lifecycle (example: test/aaa/)**
## During run (temporary)

```
test/
├─ aaa/
│  ├─ IMG_0001.png
│  ├─ IMG_0002.png
│  ├─ cropped/                 # created when saving crops (default on)
│  │   ├─ IMG_0001_cropped.png
│  │   └─ IMG_0002_cropped.png
│  └─ ...
└─ aaa_renamed_out/            # created and cleared due to --clean-out
```

## After run (final state, defaults)

```
test/
├─ aaa/
│  ├─ IMG_0001.png             # originals untouched
│  ├─ IMG_0002.png
│  ├─ rename_mapping.csv       # per-folder mapping & status
│  └─ (cropped/ removed)       # deleted by --clean-crops-after (default)
│
└─ aaa_renamed_out/
   ├─ RIL123-1.png             # de-duplicated numbering (when --duplicates True)
   ├─ RIL123-2.png
   ├─ RIL045-1.png
   └─ ...
```

## Whole ROOT after processing multiple subfolders

```
test/
├─ aaa/
│  ├─ IMG_*.jpg|png|tif ...
│  ├─ rename_mapping.csv
│  └─ (cropped/ removed)
├─ aaa_renamed_out/
│  ├─ <OCR_BASE>-1.png
│  ├─ <OCR_BASE>-2.png
│  └─ ...
├─ bbb/
│  ├─ IMG_*.jpg|png|tif ...
│  ├─ rename_mapping.csv
│  └─ (cropped/ removed)
└─ bbb_renamed_out/
   ├─ <OCR_BASE>-1.png
   └─ ...
```

# 2. Command-line Arguments

| Flag                                           | Type / Values      | Required | Default                                   | What it does                                                            | Notes / Example                                                                                            |
| ---------------------------------------------- | ------------------ | -------: | ----------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `-i, --input`                                  | `Path` (directory) |        ✅ | —                                         | Input folder containing source images.                                  | Supports non-recursive by default. Use `--recursive` to descend into subfolders.                           |
| `-w, --weights`                                | `Path` (`.pt`)     |        ✅ | —                                         | Ultralytics YOLO weights file.                                          | Example: `weights/best.pt`.                                                                                |
| `-o, --out-renamed`                            | `Path` (directory) |        ✅ | —                                         | Destination folder for renamed images.                                  | **Must not** be the same as `--input`. Will be cleaned at start unless `--no-clean-out`.                   |
| `--prompt`                                     | `str`              |        ✅ | —                                         | OCR prompt sent to Ark model.                                           | Provide a concise instruction describing the label text to extract.                                        |
| `--duplicates`                                 | `True/False`       |        ✅ | —                                         | Duplicate-handling strategy.                                            | `True`: auto-numbering `BASE-1/-2/...`; `False`: use OCR text directly (conflicts are skipped).            |
| `--class-name`                                 | `str`              |        — | `"WhiteTag"`                              | YOLO class name to crop (merge all detections to one min bounding box). | Case-insensitive match against model’s `names`.                                                            |
| `--ark-key`                                    | `str`              |        — | env `ARK_API_KEY` → code fallback         | Ark API key for OCR.                                                    | Precedence: CLI `--ark-key` > env `ARK_API_KEY` > `DEFAULT_ARK_KEY` in code. **Avoid hardcoding secrets.** |
| `--ark-model`                                  | `str`              |        — | `"doubao-1-5-thinking-vision-pro-250428"` | Ark (Doubao) vision model to use.                                       | Must be a model supported by your Ark endpoint.                                                            |
| `--device`                                     | `str`              |        — | `"cpu"`                                   | Compute device.                                                         | Accepts `cpu`, `cuda`, `cuda:0`, `cuda:1`, or a digit (mapped to `cuda:<n>`).                              |
| `--save-crops / --no-save-crops`               | flag               |        — | **save** (True)                           | Whether to save cropped tag images.                                     | Default **on**. Use `--no-save-crops` to disable.                                                          |
| `--crops-dir`                                  | `Path`             |        — | `cropped`                                 | Where to save crops.                                                    | If **relative**, it is created under `--input`.                                                            |
| `--clean-crops-after / --no-clean-crops-after` | flag               |        — | **clean** (True)                          | Delete the crops folder after the run.                                  | Default **on**. Use `--no-clean-crops-after` to keep crops.                                                |
| `--clean-out / --no-clean-out`                 | flag               |        — | **clean** (True)                          | Clean `--out-renamed` before processing.                                | Default **on**. Use `--no-clean-out` to append instead.                                                    |
| `--recursive`                                  | flag               |        — | `False`                                   | Recursively traverse subfolders of `--input`.                           | Only files with suffix in `IMG_EXTS` are processed.                                                        |
| `--dry-run`                                    | flag               |        — | `False`                                   | Plan and log all renames but **don’t** actually move/rename files.      | Good for verification.                                                                                     |
| `--csv`                                        | `Path`             |        — | `<input>/rename_mapping.csv`              | Where to write the rename mapping CSV.                                  | CSV columns: `src_dir, old_name, ocr_text, base_sanitized, index, final_name, status`.                     |
| `--log-file`                                   | `Path`             |        — | none (stdout only)                        | Additionally write logs to a file.                                      | Stdout remains active; this option **adds** file logging.                                                  |

## Status codes in CSV
- `OK`: planned to rename/move.
- `READ_FAIL`: image cannot be read.
- `NO_DET`: no target class detected by YOLO.
- `NO_TEXT`: OCR returned empty or error.
- `NAME_CONFLICT`: when --duplicates False, target name already exists.
- Plus possible top-level error logs from batch rename.


# 3. How to run scripts using the command line?

## Default run (CPU, non-recursive; crops saved under the **input** folder)
* Requires an explicit output directory via `-o`.
* Saves crops to `<INPUT>/cropped` (default).
* Cleans the output dir before the run and removes the crops folder after the run (defaults).

```bash
python detect_tags.py \
  -i /abs/path/to/raw_imgs \
  -w /abs/path/to/yolo_white_tag.pt \
  -o /abs/path/to/renamed_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --duplicates True
```

## GPU run (use `cuda:0`), scan subfolders, keep crops, do not wipe previous outputs

* Still must provide `-o`.
* `--recursive` scans deeper folders.
* `--no-clean-crops-after` keeps `<INPUT>/cropped`.
* `--no-clean-out` keeps any prior results in the output folder.

```bash
python detect_tags.py \
  -i /abs/path/to/raw_imgs \
  -w /abs/path/to/yolo_white_tag.pt \
  -o /abs/path/to/renamed_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --device cuda:0 \
  --recursive \
  --no-clean-crops-after \
  --no-clean-out \
  --duplicates True
```

## Dry run (simulate only), also export CSV mapping and a log file

* **Dry runs still require** `-o` so targets can be planned.
* `--duplicates False` uses the OCR text directly as the filename (conflicts will be flagged).

```bash
python detect_tags.py \
  -i ./in \
  -w ./yolo.pt \
  -o ./out_dryrun \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --dry-run \
  --duplicates False \
  --csv ./in/rename_mapping.csv \
  --log-file ./in/run.log
```

## Custom destinations (absolute paths for both renamed images and crops)

* By default, crops go to `<INPUT>/cropped`; here we redirect them elsewhere.

```bash
python detect_tags.py \
  -i /data/raw \
  -w /models/white_tag.pt \
  -o /data/renamed_out \
  --crops-dir /data/crops_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --duplicates True
```

## Explicit cleaning flags (these are already the defaults)

```bash
python detect_tags.py \
  -i /abs/path/to/raw_imgs \
  -w /abs/path/to/yolo_white_tag.pt \
  -o /abs/path/to/renamed_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --clean-out \
  --clean-crops-after \
  --duplicates True
```

## Providing the Ark API key

```bash
export ARK_API_KEY="YOUR_ARK_KEY"
python detect_tags.py \
  -i /data/imgs \
  -w /data/weights/white.pt \
  -o /data/renamed_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --duplicates True
```

> Replace `detect_tags.py` with your actual script name if different (e.g., `main.py`).

## Please indicate that this material is sourced from the Center for Wheat Genetics and Genomics Center （WGGC）, College of Agriculture, China Agricultural University.
