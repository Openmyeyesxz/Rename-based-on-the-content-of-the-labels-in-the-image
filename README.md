# <p align="center">基于图像中标签内容的自动重命名</p>
<p align="center">在田间试验中，样本名称或样本编码通常印在不同颜色的硬质标签上。为了降低人工成本、减少数据采集时间并提升表型数据采集的整体效率，我们基于 AI-OCR 技术做了一套轻量化的自动化流程，用于自动识别标签并批量重命名图片。</p>

<p align="center">
  <strong>语言：</strong>
  <a href="README.md">中文</a> ·
  <a href="README_EN.md">English</a>
</p>

<p align="center">
  <strong>崭新的GUI：</strong>
  <a href="https://github.com/Openmyeyesxz/Image-Renamd_By_AI-OCR_app">Tagtool</a>
</p>


---

## 程序功能详情
>- 根据图像中的标签上的内容，基于AI-OCR能够循环重命名某个父文件夹下多个子文件夹的图片。
>- 当前流程默认使用 YOLO 模型来检测分割目标标签，并调用豆包（Doubao）视觉大模型进行OCR。（后续会考虑增加其他模型与AI种类）。

## 注意 ⚠️
当你开始处理前，如果不放心，请尽量备份你的原始数据。因为在重命名完成后，**该程序会将已成功重命名的图片会从原始文件夹移动到输出文件夹中，原文件夹中不会再保留这些对应的原始图片**。

---

# 1. 文件结构说明
## 任务根目录典型布局
```text
<WORKDIR>/
├─ photo-rename.slurm        # 你的 sbatch 提交脚本
├─ detect_tags.py            # Python 入口脚本（主程序）
├─ slurm.o                   # 所有子任务的标准输出
├─ slurm.e                   # 所有子任务的标准错误
└─ test/                     # 根目录（各个待处理的子文件夹放这里）
   ├─ aaa/ ...
   ├─ bbb/ ...
   └─ ...
````

**单个子文件夹的处理流程示例（如：`test/aaa/`）**

## 运行中（临时态）

```text
test/
├─ aaa/
│  ├─ IMG_0001.png
│  ├─ IMG_0002.png
│  ├─ cropped/                 # 检测到标签后保存的裁剪图（默认开启）
│  │   ├─ IMG_0001_cropped.png
│  │   └─ IMG_0002_cropped.png
│  └─ ...
└─ aaa_renamed_out/            # 输出目录，若使用 --clean-out 会在运行前清空
```

## 运行结束（默认最终态）

```text
test/
├─ aaa/
│  ├─ IMG_0001.png             # 原图不做覆盖修改
│  ├─ IMG_0002.png
│  ├─ rename_mapping.csv       # 当前文件夹的重命名映射与状态
│  └─ (cropped/ removed)       # 若使用默认 --clean-crops-after，则裁剪图会被删除
│
└─ aaa_renamed_out/
   ├─ RIL123-1.png             # 当 --duplicates True 时，自动去重并编号
   ├─ RIL123-2.png
   ├─ RIL045-1.png
   └─ ...
```

## 整个根目录在处理多个子文件夹后的样子

```text
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

---

# 2. 命令行参数说明

| 参数（Flag）                                       | 类型 / 取值        | 必填 | 默认值                                       | 功能说明                          | 备注 / 示例                                                                           |
| ---------------------------------------------- | -------------- | -: | ----------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------- |
| `-i, --input`                                  | `Path` (目录)    |  ✅ | —                                         | 含有待处理图片的输入目录                  | 默认不递归；如需处理子目录，使用 `--recursive`                                                    |
| `-w, --weights`                                | `Path` (`.pt`) |  ✅ | —                                         | 模型的权重文件        | 例如：`weights/best.pt`                                                              |
| `-o, --out-renamed`                            | `Path` (目录)    |  ✅ | —                                         | 重命名后图片的目标输出目录                 | **不能**与 `--input` 相同；若未指定 `--no-clean-out`，运行前会被清空                                |
| `--prompt`                                     | `str`          |  ✅ | —                                         | 发给 Ark 模型的 OCR 提示词            | 写清楚标签要识别的字段或格式                                                                    |
| `--duplicates`                                 | `True/False`   |  ✅ | —                                         | 如果你的样本中有重复，相同名称的标签可以生成重复  | `True`：自动生成 `BASE-1/-2/...`；`False`：直接用 OCR 文本作为文件名（有冲突则跳过）                       |
| `--class-name`                                 | `str`          |  — | `"WhiteTag"`                              | 模型中要裁剪的类别名（会把所有检测合并成最小外接框） | 与模型的 `names` 大小写不敏感                                                               |
| `--ark-key`                                    | `str`          |  — | 环境变量 `ARK_API_KEY`                | AI模型的API Key             | 优先级：命令行 `--ark-key` > 环境变量 `ARK_API_KEY` > 代码内默认值                 |
| `--ark-model`                                  | `str`          |  — | `"doubao-1-5-thinking-vision-pro-250428"` | 使用AI的模型版本               | 必须是你的模型端点已开通的模型                                                                |
| `--device`                                     | `str`          |  — | `"cpu"`                                   | 推理使用的设备(用于分割标签算法的运行)                          | 支持 `cpu`、`cuda`、`cuda:0`、`cuda:1`，或直接写数字表示 `cuda:<n>`                             |
| `--save-crops / --no-save-crops`               | flag           |  — | **保存** (True)                             | 是否保存标签裁剪图                     | 默认保存，若不想要裁剪图可加 `--no-save-crops`                                                  |
| `--crops-dir`                                  | `Path`         |  — | `cropped`                                 | 裁剪图的保存目录                      | 若为相对路径，则会建在 `--input` 目录下                                                         |
| `--clean-crops-after / --no-clean-crops-after` | flag           |  — | **删除** (True)                             | 结束后是否删除裁剪图目录                  | 默认删除，若想保留裁剪结果可加 `--no-clean-crops-after`                                          |
| `--clean-out / --no-clean-out`                 | flag           |  — | **清空** (True)                             | 运行前是否清空输出目录               | 默认清空，若想在原有输出目录上追加结果可用 `--no-clean-out`                                            
| `--recursive`                                  | flag           |  — | `False`                                   | 是否递归遍历 `--input` 的所有子目录       | 只会处理扩展名在 `IMG_EXTS` 列表中的图片                                                        |
| `--dry-run`                                    | flag           |  — | `False`                                   | 只做计划与日志输出，不真正移动/重命名文件         | 适合先查错或验证流程                                                                        |
| `--csv`                                        | `Path`         |  — | `<input>/rename_mapping.csv`              | 指定重命名映射 CSV 的输出路径             | CSV 字段包括：`src_dir, old_name, ocr_text, base_sanitized, index, final_name, status` |
| `--log-file`                                   | `Path`         |  — | 无（Linux一般会打印到stdout）                | 除了 stdout 再额外写一份日志到文件         |---                                                                         |

### CSV 中可能出现的状态码

* `OK`：已计划重命名/移动。
* `READ_FAIL`：图像文件读取失败。
* `NO_DET`：YOLO 未检测到目标类别。
* `NO_TEXT`：OCR 返回为空或出错。
* `NAME_CONFLICT`：在 `--duplicates False` 模式下，目标文件名已存在。
* 此外还可能有批量运行时的顶层错误信息。

---

# 3. 如何在命令行中运行？

## 默认运行方式（CPU、不递归；裁剪图保存在输入目录）

* 必须显式提供输出目录 `-o`
* 裁剪图默认写入到 `<INPUT>/cropped`
* 运行前会清空输出目录，运行后会删除裁剪目录（都是默认行为）

```bash
python detect_tags.py \
  -i /abs/path/to/raw_imgs \
  -w /abs/path/to/yolo_white_tag.pt \
  -o /abs/path/to/renamed_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --duplicates True
```

## GPU 运行示例（使用 `cuda:0`），递归扫描子目录，保留裁剪图，不清空旧输出

* 仍然必须提供 `-o`
* `--recursive` 用于扫描更深层级
* `--no-clean-crops-after` 保留 `<INPUT>/cropped`
* `--no-clean-out` 保留以往输出

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

## 演练模式（Dry run）：只模拟，不真正移动文件，同时导出 CSV 和日志

* 演练模式同样需要 `-o`，这样才能计算出目标路径
* `--duplicates False` 表示直接使用 OCR 识别到的文本作为文件名，如有冲突会在 CSV 中标记

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

## 自定义输出路径（重命名后的图片与裁剪图都放到绝对路径）

* 默认裁剪图会写到 `<INPUT>/cropped`，这里演示改成另一个目录

```bash
python detect_tags.py \
  -i /data/raw \
  -w /models/white_tag.pt \
  -o /data/renamed_out \
  --crops-dir /data/crops_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --duplicates True
```

## 显式指定清理行为（以下其实是默认值，仅作说明）

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

## 提供 Ark API Key 的方式

```bash
export ARK_API_KEY="YOUR_ARK_KEY"
python detect_tags.py \
  -i /data/imgs \
  -w /data/weights/white.pt \
  -o /data/renamed_out \
  --prompt "Output ONLY the sample ID on the white tag. No extra text." \
  --duplicates True
```

> 如果你的入口脚本名字不是 `detect_tags.py`，请替换成你自己的，比如 `main.py`。

---

# 4. 人工审核照片程序（旧版）
> 能够实时地预览、修改和保存你的图片，以免AI实现错误的重命名。
## 界面视图如下：
![image](https://github.com/Openmyeyesxz/Image-Renamd_By_AI-OCR/blob/main/Image/tag_check.png)


# 5. GUI界面程序
另外，我们提供了新的[集成AI-OCR和人工核验功能的GUI程序](https://github.com/Openmyeyesxz/Image-Renamd_By_AI-OCR_app)，可以通过AI-OCR进行识别，并且能够实时查看并修改图片文件名。目前只能在windows上进行运行。

---
## 引用要求
请注明本材料来源于 **中国农业大学农学院 小麦研究中心（WGGC, Center for Wheat Genetics and Genomics Center）**。


