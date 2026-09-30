# 意群分割评测数据集

本数据集由 [ACL6060](https://aclanthology.org/2023.iwslt-1.2) 和 [RealSI](https://arxiv.org/abs/2407.21646) 两个数据集二次处理得到，用于中英文意群分割任务的评估。

## ACL6060

### 文件组成

```sh
ACL6060/
|-- dev
|   |-- aligned_data
|   |-- full_wavs
|   |-- mfa_algin_data
|   |-- segments_timestamp
|   |-- text
|   |-- xml
|-- eval
|   |-- aligned_data
|   |-- full_wavs
|   |-- mfa_align_data
|   |-- segments_timestamp
|   |-- text
|   |-- xml
```

ACL6060 数据集中，只提供了 [SHAS](https://github.com/mt-upc/SHAS) 工具切分的语音片段和时间戳；对于人工标注的转写结果以及切分的语音片段，没有提供时间戳信息。为此，使用 [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/) 工具进行对齐，然后根据人工标注的分段转写结果，后处理得到分段的时间戳。

**文件说明**

- `aligned_data` MFA 工具对齐后的结果文件
- `mfa_algin_data` 用于 MFA 工具对齐的数据
- `segments_timestamp` 根据 MFA 对齐结果后处理得到的分段时间戳，时间戳的单位 ms
- `full_wavs` 未切分的原始语音
- `xml` 原始数据集中的 xml 文件
- `text` 根据 xml 文件后处理得到的分段转写结果和聚合后的转写结果


## RealSI

### 文件组成

```sh
RealSI/
|-- en2zh
|   |-- full_wavs
|   |-- gold_timestamp
|   |-- text_long
|   |-- text_segments
|   zh2en
|   |-- full_wavs
|   |-- gold_timestamp
|   |-- text_long
|   |-- text_segments
```

**文件说明**

- `full_wavs` 未切分的原始语音
- `gold_timestamp` 根据原始数据集中的 json 文件后处理得到的分段时间戳，时间戳的单位 ms
- `text_long` full_wavs 目录下语音对应的未分段的转写结果
- `text_segments` 根据原始数据集中的 json 文件后处理得到的分段转写结果

---

本数据集由 LiangHu 构建，有任何问题可以尝试联系 lianghu@hccl.ioa.ac.cn


## 整篇翻译合并与 COMET 评估

仓库根目录下的两个脚本用于将 segment 级推理结果合并成整篇音频翻译，并使用本地 COMET 模型计算每个音频的分数：

- `merge_segment_predictions.py`：合并同一音频的所有 segment 翻译，并生成四个评估输入文件。
- `score_document_comet.py`：读取四个整篇评估输入，逐个音频计算 COMET 分数，并统计均值。

### 1. 合并 segment 翻译

这个脚本直接读取推理结果中的 `wav_id`、`src`、`ref` 和 `mt`，不需要传入 `data-root`。可以直接在仓库根目录执行：

```bash
cd /data/baizhihao/tmp/semantic_group

python3 merge_segment_predictions.py \
  --input /path/to/predictions.json \
  --output-dir /path/to/merged
```

推理 JSON 可以是 JSON 数组、JSONL，或包含 `data`、`results`、`predictions` 等列表字段的 JSON 对象。每条记录默认类似：

```json
{
  "index": 0,
  "wav_id": "./2022.acl-long.268-65-413930-415290",
  "src": "Is ...?",
  "ref": "是 ...?",
  "mt": "{\"transcription\":\"Is ...?<SepPanGuPi>是 ...?\"}"
}
```

脚本会从 `wav_id` 中解析出原音频名、segment 编号和时间戳，然后按照 segment 顺序合并。同一个 `wav_id` 前缀的片段会合并成一篇音频。`mt` 如果是 JSON 字符串，脚本会提取 `transcription` 字段，并取 `<SepPanGuPi>` 后面的内容作为模型翻译。

如果四个数据集分别有四个推理文件，可以显式标明数据集：

```bash
python3 merge_segment_predictions.py \
  --input \
  acl_dev=/path/to/acl_dev.json \
  acl_eval=/path/to/acl_eval.json \
  realsi_en2zh=/path/to/realsi_en2zh.json \
  realsi_zh2en=/path/to/realsi_zh2en.json \
  --output-dir /path/to/merged
```

ACL 的 `dev` 和 `eval` 文件名格式相同，仅凭 `wav_id` 无法自动区分，因此需要使用 `--group` 或 `GROUP=PATH`。RealSI 的 `en2zh`、`zh2en` 可以根据 `wav_id` 前缀自动识别。

输出目录中会生成：

```text
acl_dev.json
acl_eval.json
realsi_en2zh.json
realsi_zh2en.json
```

每条整篇记录包含 COMET 所需的三个字段：`src`（源文）、`mt`（模型翻译）和 `ref`（参考译文）。如果推理文件中的字段名不同，可以指定：

```bash
python3 merge_segment_predictions.py \
  --input /path/to/predictions.json \
  --output-dir /path/to/merged \
  --name-key my_wav_id \
  --source-key my_source \
  --reference-key my_reference \
  --translation-key my_prediction
```

脚本会检查同一音频是否出现重复的 segment，但由于不再读取数据集目录，无法判断推理文件是否缺少某个理论上的 segment；如需完整性检查，应在推理文件生成阶段或根据原始时间戳另行检查。

### 2. 计算整篇音频的 COMET

`--model-path` 可以传 COMET checkpoint 文件，也可以传包含 `checkpoints/model.ckpt` 的模型目录：

```bash
python3 score_document_comet.py \
  --model-path /data/baizhihao/models/comet/Unbabel/wmt22-comet-da \
  --input \
  acl_dev=/path/to/merged/acl_dev.json \
  acl_eval=/path/to/merged/acl_eval.json \
  realsi_en2zh=/path/to/merged/realsi_en2zh.json \
  realsi_zh2en=/path/to/merged/realsi_zh2en.json \
  --output-dir /path/to/comet_results \
  --batch-size 8 \
  --gpus 1
```

没有 GPU 时使用：

```bash
--gpus 0
```

输出文件包括：

```text
acl_dev_comet.json
acl_eval_comet.json
realsi_en2zh_comet.json
realsi_zh2en_comet.json
summary.json
```

`summary.json` 中的主要字段：

- `groups.<group>.mean_comet`：该数据集所有音频的 COMET 均值。
- `four_group_mean`：四个数据集均值的平均，每个数据集权重相同。
- `all_audio_macro_mean`：把四个数据集的所有音频放在一起计算的均值。

第二个脚本需要当前 Python 环境已经安装 `unbabel-comet` 及其 PyTorch 依赖。
