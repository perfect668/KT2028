# 开源病理生存模型候选核查（W146，2026-11-11）

本轮回答当前训练进度与其他开源强基线；只读研究，不启动新候选训练，不修改EAGLE、R3或既有数据协议。下面的优先级按本项目的问题、现有UNI资产和适配工作量判断，不是把跨论文指标拼成统一排行榜。

## 1. nnMIL：优先考察训练组织与轻量聚合

- 论文：*Nature Biomedical Engineering*，2026-08-25，DOI [10.1038/s41551-026-01767-8](https://www.nature.com/articles/s41551-026-01767-8)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/42642650/)。论文覆盖约4万张WSI、35项任务与4种病理基础模型，包括预后及外部队列。
- [官方仓库](https://github.com/Luoxd1996/nnMIL)，已实际读取README、TUTORIAL、`scripts/run_survival.sh`、`training/trainers/survival_trainer.py`、两份生存loss及`network_architecture/models/simple_mil.py`。
- 核心是图块和特征维度的随机采样、轻量门控注意力、较大批量优化、任务相关采样、推理时分块/窗口集成。公开实现SimpleMIL训练时随机选特征维度，不是靠额外庞大主干来表达创新。
- 原生生存链路已确认：独立survival trainer，Cox或NLLSurv；时间/事件字段、患者级划分、训练与推理入口均有。当前Cox为默认选项，默认batch32；原NLL分箱按Train分位点，不能宣称与本项目固定五年度/完整经历区间的删失语义自动等价。
- 对本项目的推断：可以直接检验采样、批量优化和轻量聚合能否改善泛化，是当前首选候选。可利用已有特征，仍须做Zarr读取/多切片患者聚合及固定split适配。不能直接运行会重写dataset_plan、自动重划分的完整wrapper，或复用其CRC DSS示例标签到当前OS。

## 2. PANTHER：无监督形态原型与紧凑表示

- 论文：[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Song_Morphological_Prototyping_for_Unsupervised_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html)。[官方仓库](https://github.com/mahmoodlab/PANTHER)。虽不是2026新论文，但仍是适合本项目的成熟强基线。
- 用K-means/GMM形态原型，将大量patch压缩为包含混合概率、均值/协方差的紧凑WSI表示，再接下游生存模型。主要表征构建不依赖生存标签，适合检验是否过多监督聚合导致泛化困难；下游模型仍可能过拟合，不能声称天然解决问题。
- 原生生存已核：`src/scripts/survival/brca_surv.sh`及`panther.sh`，UNI1024输入、16原型、batch64、Cox、固定prototype、公开训练入口，不是单纯分类网络换头。脚本`target_col=dss_survival_days`但目录名含overall_survival，实际适配必须审字段，不按文件夹名猜终点。
- 对本项目的推断：可复用现有UNI特征；先按每折Train构建prototype，再冻结到内验/外层，避免信息混用。当前Zarr资产需适配读取，不能为转格式重复下载原片或默认复制全库。官方也披露过中心/染色分布变化下原型分配塌缩的风险，需查看真实原型占用。

## 3. OTSurv：异质性感知最优传输与软选择

- 论文：[MICCAI 2025官方页面](https://papers.miccai.org/miccai-2025/0660-Paper1359.html)。[官方仓库](https://github.com/Y-Research-SBU/OTSurv)。
- 可学习生存tokens与patch之间建立unbalanced OT；全局约束处理形态长尾分布，局部不确定性约束控制传输质量，逐步纳入更多区域。它为区域抑噪/软选择提供机制参考，不需要先引入RL；已算好的特征上的软选择不等于真实按需像素读取或提特征成本节省。
- 原生生存已核：`src/scripts/train_otsurv.sh`调用`training.main_survival`，`in_dim=1024`、`loss_fn=cox`、batch16。脚本使用DSS字段，须与当前OS区分；不能把默认四分箱/alpha等无关Cox参数误当现年度NLL配置。
- 作者公开结果为六队列平均C-index约0.646；COADREAD0.667±0.111、LUAD0.638±0.077、STAD0.565±0.057、KIRC0.750±0.149。它不是每一癌种都比EAGLE/其他方法更高。特征协议、临床终点和分折不同，不能与本项目单折数值直接排名。
- 对本项目的推断：有明确聚合/软选区机制，适合后续创新与消融；优先级在nnMIL/PANTHER之后，先确认简单完整生存链路的泛化状况。

## 4. 后续空间序列候选

[2DMamba（CVPR 2025）](https://github.com/AtlasAnalyticsLab/2DMamba)及[MambaMIL](https://github.com/isyangshu/MambaMIL)也公开了代码，论文涉及WSI生存。2DMamba用二维状态空间扫描表达空间结构，官方工程依赖自定义CUDA kernel，当前5090/PyTorch2.8环境需要额外编译适配。MambaMIL README明确提供生存脚本。它们作为后续备选，本轮没有完成全部训练入口/损失/兼容性审计，不标成已经就绪。

## 建议的后续比较原则

下一候选优先nnMIL，再PANTHER，OTSurv作为异质性/软选择机制候选。若获准运行，应先保留作者原生生存链路，固定本项目OS与患者折，明确Cox风险与五年度概率任务的区别；若另加统一NLL适配，作为单独对照登记。所有prototype、分箱、采样计划和权重选择仅用Train/内部验证。不能按已见外层成绩换模块/调参数，不能因这些推荐自动启动新实验。

论文和代码阅读材料存于仓库外`KT2028-assets/open_models_review_w146`；训练快照与匿名指标存于`KT2028-assets/training_status_w146.json`。
