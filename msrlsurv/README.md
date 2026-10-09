# 早期实现草稿：MS-RLSurv

本目录是早期单头 ABMIL、fine-only/空集先验、64/32 候选区域配置及增量奖励的草稿，不代表现行方法已经实现。唯一现行方案见[长期研究方案第 0 节](../docs/experiment_restart.md#0-当前长期研究方案)：HAFED 跨尺度融合、TSU、可变物理区域、分阶段训练与终末 BCE 加获取成本。以后代码适配留待服务器任务授权；本页命令只解释草稿接口，不是自动执行队列。历史脚本保留在原目录。

## 已开始实现

- `regions.py`：不读取UNI的空间划分、多切片区域分配、区域内空间覆盖采样。
- `data.py`：患者级manifest校验、拟合与内部选择集交集检查、NPZ特征入口。
- `predictor.py`：单头gated ABMIL、部分证据mask、空集先验、增量注意力更新。
- `environment.py`：缓存获取环境，初始无fine，合法动作/硬预算，ACQUIRE/STOP，训练标签与推理状态隔离。
- `train_predictor.py`：完整bag与随机子集联合训练的单个内部划分入口。它尚不是完整OOF流程。

## 真实输入

拟合和内部选择分别传一个JSON manifest。必须包含：

| 字段 | 要求 |
| --- | --- |
| `endpoint` | `five_year_death`；标签来源先核实，程序不自行重建临床结局 |
| `partition` | `train`；内部选择患者也来自Train，不能用外层OOF留出或Test来选checkpoint |
| `patients` | 每患者一条：唯一`patient_id`、整数0/1的`label`、`slides`列表 |
| `slides` | 每切片唯一`slide_id`及`features_path`；相对路径按manifest所在目录解析 |
| 特征文件 | NPZ内`img_features[N,1024]`、`coords[N,2]`；未确认公开Zarr与旧NPZ等价，转换工作尚未完成 |

程序拒绝患者/切片/特征文件在拟合与内部选择集重叠，不会覆盖已有输出目录。默认使用CPU；没有导入时训练或GPU初始化。

有真实数据和PyTorch环境后，训练入口为：

```text
python -m msrlsurv.train_predictor --fit-manifest FIT.json --select-manifest INNER_SELECT.json --output runs/predictor_seed42
```

这是使用说明，本轮未执行该命令。

## 当前验证与未完成部分

NumPy环境规则可用标准库unittest检查：`python -m unittest discover -s tests`。控制输入用于软件契约测试，不是患者实验、预后指标或论文结果。缺少PyTorch时网络张量测试会明确跳过。

目前尚未接入：真实WSI低分辨率读取与UNI在线提取、Gymnasium/MaskablePPO训练、集合Actor-Critic头、OOF编排、三模型集成/校准及正式评价。获取环境允许由调用方传入真实cheap描述；没有替缺失低分辨率图生成虚假特征。当前环境状态为单预测器的概率/熵/证据量/成本/预算；后续集成分歧是明确的扩展接口，尚未实现。

注意力源自Ilse等人的[ABMIL论文](https://proceedings.mlr.press/v80/ilse18a.html)和[原作者实现](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/tree/eb0434ba2795711a45d693d60120ae53532b1b93)，本工程适配UNI输入、部分证据和增量计算；不复用其MNIST训练入口。
