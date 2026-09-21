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