# 长期工作记录

建立日期：2026-10-09。此文件承接接下来约两个月的开题与实验协作，只记录实际工作和接续状态，不复制研究方案全文。

入口：[唯一现行方案](experiment_restart.md#0-当前长期研究方案) · [工作规则](../AGENTS.md#长期协作与记录规则) · [项目上下文](project_context.md) · [历史证据](historical_findings.md)。

## 当前状态

最后更新：2026-10-10；W011 align30轮完成并重载/备份，W012 partial首epoch及真实优化/冻结检查通过，后台继续30轮。TSU同配对表示误差暂高于直接coarse，负结果保留；单例Test几何异常仍为Q05。

W013已将用户要求的通俗解释和汇报顺序写入长期规则；本轮仅更新文档，实验状态仍以W012核查快照为准，未刷新服务器训练进度。

W014已按实际源码补充工作一网络架构图和维度/融合说明，原位保存于长期方案0.5；本轮仍未刷新训练状态或改算法。

| 项目 | 当前事实 |
| --- | --- |
| 研究方案 | P1；双尺度部分观察预后建模＋TSU/预算自适应选区与 STOP；完整定义只见长期方案第 0 节 |
| 工作阶段 | S3 full/align已完成；partial run=p1_partial_s41_20261010运行中，训练源码95b7fac、PID79335；既有255/84/84不变，Test不用 |
| 最近完成 | 项目环境/国内源/CUDA/预训练 ResNet18 检查；427 份 fine 全量解码；三张 Train 图像的成熟 Zarr 读取、物理尺度和坐标叠加检查 |
| 工作区基准 | `D:\codex\开题2028\KT2028`，`research/p1-restart`；P1 初始存档 909b0fc，全队列作业源码 2eed94f；当前 HEAD 另按 Git 核对 |
| 保存/备份状态 | 重要源码、P1 文档、S1/S2 汇总已提交并推送专用分支，服务器 Git checkout 通过 bundle 同步；大数据/标签清单与权重不进 GitHub。低倍率中转缓存另在本机资产目录 |
| 实现状态 | full/align已完成且可重载；partial实际优化、三模式Val与首次checkpoint已核实，冻结参数不变。策略未训练，SB3 padding/rollout未完成 |
| 实验状态 | full Val AUROC0.67963/BCE0.60787，后期过拟合。align 选第30轮，Val总MSE0.13817；TSU同配对MSE0.07936高于coarse0.05686，不称传播有效 |
| 当前执行范围 | 用户报告完成后按此前主方法要求接续：核查/备份 align，登记、启动及检查 partial；不改 P1、不扩数据、不启动 PPO/Test；持续阶段存档 |
| 当前接续位置 | 使用 W011 已核实 align 最佳权重训练零/随机/完整证据患者预测模块；冻结底层与TSU，按固定三模式平均Val BCE选权重。之后核查结果并准备PPO协议/接口 |

“未开展”与“因缺资产无法开展”分开：真实低倍率 coarse 已取得，full 模型实验已完成；原始高倍率 WSI 未整包保存，真实按需 UNI 提取计时尚未开展。

## 当前待解决事项

| 编号 | 缺口/需核准事项 | 依赖范围与处理方式 |
| --- | --- | --- |
| Q01 | 官方 CSV 与源码已有五年编码及 255/84/84 case 划分；原始随访/删失与独立临床患者映射仍缺 | 按官方 case 隔离核准，不重划分；CSV 的 slide_id 是各 split 内的占位字段，必须用 case_id 映射真实特征 |
| Q02 | 全队列 423/423 结构、哈希和官方划分检查通过，当前传输/处理错误 0；单例 Test 坐标异常转 Q05 | 队列完整性已解决；细尺度映射覆盖率保留，不能将哈希通过解释为全部图像视觉对齐通过 |
| Q03 | 区域几何工程候选已登记并用于full/align；PPO共同成本参考N_ref、预算/λ范围、候选上界及策略运行规模仍未锁定 | 现有区域/预测器训练不受该项阻塞；策略训练前登记，不恢复旧64/32/min-max或旧奖励 |
| Q04 | 2026-10-10 align 启动前：5090 32607 MiB、使用 0 MiB/利用率 0%；数据盘约 118 GiB 可用，原 25 核/90 GiB 配额 | GPU 当前空闲，沿用环境，不新增安装；数值为快照，不视为训练已启动 |
| Q05 | 一例 Test：1,918 fine 中 1,407 完全超出官方 OME 范围，fine 最大坐标与图像尺寸明显不匹配 | 原始 fine/当前区域产物保留，原因待核；不自动缩放、移位或剔除该病例，不影响 Train/Validation 入口准备，但正式 Test 前必须解决映射/处理协议 |
| Q06 | align最佳权重的同配对Val TSU MSE0.07936，高于直接coarse0.05686及来源fine0.06260 | 保留负结果，按登记路线完成partial，再做共同预测器下的TSU比较/消融；不是标签错误或程序崩溃证据，不自行改结构/重选checkpoint |

资产明细及历史错误已在长期方案档案/历史发现中保存，本表不另造重复资产清单。任务开展时新增问题沿用 Q 编号；已解决项记录证据与关闭条目。

## 记录方法

每完成一项实质工作，追加一个 W 条目，并更新“当前状态”和“当前待解决事项”。同一长任务的中间进展可更新该条目，明确“进行中/已完成/未完成”；不能把计划写成完成事项。不要求每天没有工作也生成记录。

任务记录至少包括：日期与状态、用户任务/范围、方案和源码基准、完成内容、证据/交付物、检查及其限制、决定/方案变化、未完成项、下一项输入输出与停止边界。路径优先使用仓库相对链接；敏感患者数据不写入公开日志。

实验开展后，每个真实 run ID 对应固定的方案版本、源码状态、配置、数据/划分版本、拟合资产、seed、命令/环境、checkpoint 与结果位置。日志只放入口和结论范围，原始运行记录留在实际产物中；不为未运行实验建立假结果行。

维护时使用下列字段即可，不为每轮另建文件：

```text
### Wxxx｜日期｜任务名（状态）
范围与基准：用户要求、执行边界、方案版本、commit/未提交修改。
完成与证据：真实交付物链接；来自源码、记录或本轮运行的判断。
检查与限制：检查结果和未验证部分。
决定与变化：是否改变方案；若改变，链接正文和修订项。
未完成/失败：缺口、影响、对应 Q 编号。
下一最小工作：输入、输出、范围、停止条件；建议不等于执行授权。
```

## 工作条目

### W001｜2026-10-09｜固定长期研究方案（已完成）

- 范围与基准：用户要求将讨论固定到一个长期文档；仅文档整理。方案 P1，源码基准 `main` / `7ed0dcdd023f0dd963dda95a60733729473d4766`，原有未提交修改保留。
- 完成与证据：[长期方案第 0 节](experiment_restart.md#0-当前长期研究方案) 写入两项研究工作、网络、信息权限、奖励/成本、分阶段训练、参数状态、对照及保守预期。AGENTS/项目上下文同步，早期代码 README 标为草稿。
- 决定与变化：取消旧预测器达标门槛；64/32 不再是固定方法定义；分阶段训练；终末 BCE 加正成本；预后模块需要本任务训练且不保证完美。详见长期方案 0.14，不另抄一套结构。
- 检查与限制：文档差异检查及相对链接检查通过；原第 1–9 节正文保留，与编辑前按文本归一后的 SHA256 一致。没有算法有效性验证或新实验。
- 未完成：Q01–Q04；文档尚未提交/推送。
- 下一项：建立长期工作日志与协作规则；用户随后明确提出此需求，进入 W002。

### W002｜2026-10-09｜建立长期工作记录与接续规则（已完成）

- 范围与基准：用户询问近两个月的工作是否需要长期记录与类似 skill 的规则；按文档协作范围建立。方案仍为 P1，HEAD 与 W001 相同。
- 完成与证据：新增本日志，提供顶部状态、待解决事项及任务记录字段；[AGENTS.md](../AGENTS.md#长期协作与记录规则) 作为唯一规则入口，新增启动、落盘、交接、变更和实验追溯要求；项目上下文及方案维护规则补充日志链接。
- 决定与变化：不另建重复的 SKILL.md；AGENTS 管规则、长期方案管方法、工作日志管进度，上下文只作导航。方法没有变化，不升级 P1。
- 检查与限制：文档差异、入口和本地相对链接检查；没有改算法源码或运行实验。长期记录只能保存已经落盘的信息，每次关键进展按规则更新，不能依赖对话中未保存的记忆。
- 未完成：Q01–Q04 和远端备份状态不变。
- 下一最小工作：建议按长期方案 0.12 核准数据与双尺度接口，输入为元数据、映射、split、备份和公开资产索引，输出为可追溯数据契约与区域/成本规格；范围不包含训练。当前文档工作完成即停止，等待后续具体任务。

### W003｜2026-10-09｜服务器与数据接手 S1（已完成初步盘点与访问验证）

- 范围与基准：用户提供新服务器并要求开始第一步；方法仍为 P1，实施工作安排加入长期方案 0.15，HEAD 与 W001 相同。连接凭据不存入文件。
- 已完成：服务器连接成功；容器 CPU/内存限额和 GPU/磁盘核实；上传根目录 `/root/autodl-tmp/SurGen-data`；16,846,959,224 字节 UNI ZIP 的 SHA256 与上传清单一致，MD5 与 bundle README 一致。
- 映射预检：SR386 427 个 Zarr、2,203,176 个 patch，features 为 float32 `[P,1024]`、coords 为 int64 `[P,2]`，元数据行数对应；官方五年 CSV 共 423 case（255/84/84），按整数 T 编号规范化后全部映射到真实特征，case 无跨 split 交集。`slide_id=slide_0...` 为各 split 内占位，不可用作真实切片标识；4 个有特征 case 无本任务标签。
- 标签证据：上传源码 `dataset_constants.py` 将 Died within 5y 编为 1、Survived 5y 编为 0；原始随访资料未提供，不能声称已经重算五年结局或核准临床患者身份。
- 真实产物：服务器 `/root/autodl-tmp/KT2028/tools/audit_s1.py` 完成审计，SR386 解压到 `/root/autodl-tmp/KT2028/data/sr386_uni_zarr`，占 8,508,946,570 字节；审计目录 `/root/autodl-tmp/KT2028/audit/s1_20261009` 保存 3 份 asset manifest、427 切片 inventory、4 个无标签 case 清单、完成回执、数据契约与图像访问记录。原上传包未改动，SR1482 未解压。数据盘结束时约 177 GiB 可用。
- 本地保留：[审计回执](../records/server_s1_20261009/completion_receipt.json)、[数据契约](../records/server_s1_20261009/data_contract.json)、[审计工具](../records/server_s1_20261009/audit_s1.py)；带 case/标签的清单未复制到公开仓库。服务器源与本地工具 SHA256 一致，原文件与派生物没有混写。
- 图像缺口与来源：用户确认只下载 16.8 GB 特征包，原图过大。[官方 BioImage Archive](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1285) API 实际返回 2026-08-03 新增 OME-Zarr 信息；文件列表有 427 个 SR386 ZIP，首次检查的云端展示存储只有 4 个前缀，故不据展示来源宣称全队列可读。已在一份真实 5.03 GB SR386 ZIP 上验证 HTTP Range，49,370 字节取得元信息；随后以 634,578 字节传输完成一张 RGB 缩略图解码，见[范围访问记录](../records/server_s1_20261009/remote_range_probe.json)、[像素读取回执](../records/server_s1_20261009/coarse_access_example/receipt.json)、[真实缩略图](../records/server_s1_20261009/coarse_access_example/thumbnail.png)、[探针源码](../records/server_s1_20261009/probe_remote_thumbnail.py)。正式读取应采用成熟 Zarr 解码接口，探针只作资产可行性检查。
- 检查与限制：原 ZIP SHA/MD5 预检通过；全部 SR386 元信息行数/维度和数据 chunk key 检查通过；选中条目解压 CRC 通过。图像实例 ZIP CRC/分片索引 CRC32C/PNG 哈希通过，本地已查看。fine 数组尚未解码，故未检查有限值/重复坐标；单图可读不等于全部病例和所选 coarse 尺度都通过。读取 Test CSV 仅作资产/划分核查，没有 Test 预测或指标。上传 SurGen 源码没有 .git，版本以文件摘要记录，不能编造 commit。
- 决定与变化：方法仍为 P1；新增执行排期及更新服务器阶段边界，不换预测器/任务。采用按需获取官方低倍率数据的接入方向，不要求用户完整下载原图；实际全队列适配待 S2。
- 未完成：Q01 的原始随访与独立患者身份来源、Q02 全病例图像映射/坐标解码、Q03 正式几何和成本配置。现有数据契约是资产级清单，不是已完成的双尺度训练接口。
- 接续：S2 最小数据读取与双尺度接口准备，输入为上述真实回执、Zarr、官方 CSV/图像索引；输出为数值/坐标检查、case 对应的低倍率视图、物理映射与候选区域配置。无需重划分或重提全部 UNI，不完整下载 CZI/OME-Zarr；训练与正式评价不包含在这一最小任务中。本轮完成即停止，未安装额外依赖，未运行 GPU 实验，SSH 已退出，无遗留后台任务。

### W004｜2026-10-09｜S2 环境与数据读取检查（已完成本项；正式区域接口待续）

- 授权：用户要求继续工作并配置服务器环境，注意国内网络与资源源配置；确需手动操作时再说明。
- 范围：保留系统 CUDA/PyTorch 和原始数据，建立项目隔离环境、国内 HTTPS 下载源与缓存；读取真实 fine，验证低倍率读取和物理映射。当前不训练或进行正式模型评价。
- 基准：P1，HEAD 与 W003 相同；服务器既有 torch 2.8.0+cu128、torchvision 0.23.0+cu128、numpy 2.3.2。开始时环境目录不存在。
- 已完成环境：项目 `.venv` 使用系统包继承原 miniconda PyTorch/CUDA，不替换原环境；阿里云 PyPI 实测 200/0.394 秒，清华 200/0.938 秒，中科大 403，选择阿里云默认、清华备用。依赖、缓存和已校验的 46.8 MB 官方 ResNet18 权重放数据盘，不关闭 TLS、不设置非官方 Hugging Face 端点。依赖检查无冲突；5090 CUDA 运算、真实缩略图的冻结 ResNet18 512 维前向通过，仅为软件检查。激活脚本、锁定清单、约束和检查回执已保存于 [S2 记录目录](../records/server_s2_20261009/README.md)。环境依赖已有 miniconda 基础环境，不能声称可独立重建 CUDA。
- 已完成数据：427 个 Zarr 的 2,203,176 行 fine 全部读取解码，非有限值/全零行/重复坐标/负坐标均为 0；全部 427 个切片有官方图像索引，423 个有监督 case 无匹配缺失。见 [数值审计汇总](../records/server_s2_20261009/fine_audit_summary.json) 与 [审计工具](../records/server_s2_20261009/audit_fine_values.py)。仅作资产检查，没有使用 Test 做预测或评价。
- 源码核准：上传 `WholeSlideImage.py:92,597,600,611` 和 `dataset_h5.py:151` 表明 coords 是扫描包围框相对的原图像素，原 CZI 读取需加扫描偏移，patch 按目标 MPP 换算并允许边缘填充；不能将旧坐标当作 1 MPP 坐标或给 OME 数组盲目添加偏移。
- 图像实测：按 Train fine patch 数最小/中位/最大选择三张实例，成熟 Zarr 3.4 ZipStore/codec 读取官方 Range 数据成功，分别传输 604,522 / 704,883 / 776,703 字节，耗时 157.852 / 182.299 / 164.620 秒。原 ZIP 分别约 5.71 / 48.25 / 89.09 GB，均未完整下载。三图 level-0 MPP 均为 0.1112310496 µm，最低层用于读取/坐标检查，不是正式区域输入。55 / 0 / 4 个 fine 中心超出原图边界，源码允许边缘填充；正式接口需标记有效范围，不能直接丢弃或跨图归属。实际绿色坐标叠加图已查看，整体与组织范围相符，不能据此声称逐 patch 解剖配准完成。见[三例回执](../records/server_s2_20261009/coarse/receipt.json)和[成熟读取检查工具](../records/server_s2_20261009/verify_coarse_reader.py)。外网请求延迟显著，正式全队列读取应缓存元信息/像素并限制并发，不能以下载字节比例声称算力节省。
- 接续位置：服务器 `/root/autodl-tmp/KT2028/environment/` 存环境配置/来源/检查记录；S1 输入与派生清单保持原位，后续结果使用独立 S2 目录。
- 保存/检查：公开小图、环境配置/锁定版本、工具和汇总回执已复制到本地 `records/server_s2_20261009`，按服务器 [SHA256 清单](../records/server_s2_20261009/artifact_sha256.json)核对。病例标签清单、大数组和权重未复制到仓库；原数据/系统环境保留。差异检查通过。SSH 已退出，无剩余后台作业；尚未提交/推送。
- 决定/限制：P1 不变；无预测器、PPO 训练或正式评价；未下载 UNI 权重，现成特征读取不需要它。环境完成不等于 P1 网络/环境已经实现。
- 下一最小工作：输入为 S1/S2 清单、官方图像索引与三例物理尺度，输出为登记过的候选区域几何、真实区域 RGB、fine 归属/有效范围及缓存接口；先在 Train 小样本检查正式裁剪和对齐，再扩展队列。不自动把最低层缩略图或全 fine 均值作为 coarse，不在该接口工作中训练模型。

### W005｜2026-10-09｜真实区域接口与 GitHub 分支持续存档（进行中）

- 用户授权：继续下一步；服务器接入仓库，建立专用分支并及时更新存档。三例是输入接口检查，不是缩小正式队列；有监督病例仍按官方 255/84/84 case 使用。
- 基准：从 main / `7ed0dcdd023f0dd963dda95a60733729473d4766` 创建 `research/p1-restart`，保留原有修改；先保存 P1 文档、S1/S2 工具/汇总和明确标记的早期草稿，再开发当前数据接口。
- 当前范围：正式区域 RGB 读取、物理网格/细 patch 映射/边缘有效性和缓存；先检查 Train 实例，满足检查后扩大队列。无训练和正式指标。
- 存档规则：服务器工作目录与数据分离；GitHub 上只保存重要代码、配置、文档和汇总；原始/派生大数组、标签清单、权重与凭据不提交。服务器不持久化本机 GitHub 令牌；推送可经本机现有 Git 认证完成。
- Git 接入：`research/p1-restart` 已建立并推送，首次归档 909b0fc；服务器直连 GitHub 443 超时，记录在 `/root/autodl-tmp/KT2028/audit/repo_clone.log`。改为本机认证推送、Git bundle 中转，服务器 checkout 为 `/root/autodl-tmp/KT2028/repo`，origin 仍指向正式仓库；只 fast-forward，保留数据、环境、原目录和修改。无需用户操作。后续服务器直接推送未配置，当前备份采用此中转流程。
- 实现：新增 `kt2028/` 当前数据接口，候选先由 RGB 阈值/非重叠物理网格生成，随后映射 fine 坐标，coarse 不读取 UNI 数值；冻结 ResNet18 输出 512 维，白色补边和 ImageNet 归一化，无额外 center crop。CSR 保留唯一 fine 归属与图像有效比例，零 fine 初态、真实获取和正 patch-count 成本可检查。输入程序不接收标签。
- 配置：`configs/regions_p1_trial.json` 登记 4 µm/px 目标、896 µm 视野/步长、224 输入、RGB 组织阈值及字节/内存限制；实际选取约 3.559 µm/px 原生层。它是首版接口几何，不是经预后效果验证的最优值；R/k 可变，不固定 64/32/16；N_ref 和预算仍待协议登记。完整候选不是原 WSI 全 patch。
- 三例结果：54/362/650 个 RGB 区域，合计 1066；15393 个原 fine patch 中 15384 个映射入候选，9 个 RGB 候选外 patch 留在原特征，不伪造归属；99 个部分边缘填充，0 个完全图像外 patch。三个病例的全部可获取区域实际读取 UNI 后检查通过：零 fine 初态、唯一归属、重复/空动作拒绝、累计 patch 数正确，见[真实获取检查](../records/server_s2_regions_20261009/region_contract_checks.json)、[三例汇总](../records/server_s2_regions_20261009/pilot_summary.json)；区域 crop 与网格图已查看。几何单元检查 4 项通过；这些都是软件/数据检查，不证明预后或策略有效。
- 传输改进：服务器直读原入口第一例约 362 秒/7.47 MB，已安全终止慢速后续读取并保留已完成结果/失败现场，见[中断记录](../records/server_s2_regions_20261009/slow_source_interruption.json)。官方 `/info` API 返回直接 HTTPS 源，尾部 64 KiB 与原入口匹配，服务器 1 MiB 仍约 51 秒，本机约 3.55 秒。采用本机纯标准库分段读取选定低倍率 ZIP 条目、CRC 检查后 SCP 中转，三例约 7.47/44.11/81.84 MB、16.33/69.05/96.66 秒，无完整原图下载。服务器使用原生 Zarr 解码本地选定数组，两个新案例分别约 7.24/11.95 秒完成区域数据。传输数字来自各自回执，不是完整端到端成本基准；S1/旧尝试/中转重复传输不能隐去。
- 后台接续（当前仍在运行）：服务器作业源码 commit `2eed94fdcc5dc28e4e35fc444a0cde1f8df4078d`，配置 SHA256 `257fe7699bcd29c622d44c76339eb11a780fbd6e10ba0cd558c7be14e68fdf61`。服务器 PID 29156，日志 `/root/autodl-tmp/KT2028/audit/s2_regions_20261009/cohort.log`，PID 回执同目录 `cohort_pid.json`，实时结果 `/root/autodl-tmp/KT2028/data/regions_p1_20261009/summary.json`；本机中转当前 PID 36452（原 13444 已按进程身份核对后停止），中转源码 33f5eab，日志 `D:\codex\开题2028\KT2028-assets\cohort_relay.log`、实时 `cohort_relay_receipt.json`、PID 回执同目录。现为 2 图像并行×3 分段，成功上传才设 ready；服务器只用 ready 缓存，禁止网络回退；字节/内存/磁盘限额保留。早期 12/423 [运行快照](../records/server_s2_regions_20261009/cohort_progress_snapshot.json)保留，后续[恢复快照](../records/server_s2_regions_20261009/recovery_progress_snapshot.json)已完成 31/423、服务器无错误；接手必须查实时文件，不能沿用旧计数。
- 连接：为持续传输建立本机专用 SSH 密钥，私钥保存在用户主目录 `.codex/ssh` 并限制当前用户权限，服务器仅追加公钥且保留已有 authorized_keys；仓库/日志不保存私钥或密码。中转依赖本机联网及该进程继续运行；没有创建定时任务。不得重复启动同一缓存目录的并发中转。
- 限制与下一项：423 是官方 CSV 的 case/切片清单，不能宣称 423 个独立临床患者；Q01 原始随访/独立患者关系仍待核准。当前不训练，不自动锁定预算/归一化成本或把 full-candidate 称作全 WSI。接下来等两项作业结束，核对 expected/completed/errors、所有来源/配置/数组哈希、覆盖率及 split 清单并存档；有失败先定位，不能删除失败、改倍率或改纳入病例掩盖问题。
- 传输恢复：初始 4×4 中转出现一次 SSH 就绪确认超时及一次 HTTPS TLS EOF；初始错误回执保存在本机资产目录 `cohort_relay_interrupted_20261009.json`，日志保留，不删除病例。降低到 2×3，加入有界 SSH/SCP/HTTP 重试、分段落盘复用及最终回执原子发布；已完成缓存不重下，未就绪的部分传输不会进入服务器处理。恢复中转当前无错误，TLS 失败切片已重新取得校验通过的低倍率条目，继续上传/处理。8 项几何/缓存/传输恢复软件检查通过，包括断网时缓存不回退、SSH 超时可重试、成功分段可复用；三例真实获取检查另记录，不等同算法有效。

### W006｜2026-10-09：传输期间并行源码核查与 P1 架构接口

- 授权/范围：用户要求低倍率传输继续时完成代码拉取和架构初步准备，并解释传输内容。本轮基准 `ae82fe9e1ddaaf765d2c2a132fe8a2d2dd29e534`；保持 P1、原划分及后台作业，不训练或正式评价。
- 传输说明：从官方已有 OME-Zarr 图像金字塔按 Range 下载约 3.559 µm/px 层的原始编码 ZIP 条目，CRC 校验后本机中转服务器；服务器原生解码 RGB，再按 896 µm 区域提取冻结 ResNet18。不是本机生成的小缩略图，不下载完整原图。原先三张图为小/中/大 Train 的尺度/接口检查，队列为全部 423 个官方已标注 case。开工时服务器完成 41/423，无当前错误，后台继续；数字是快照，结束和后续必须读取实时文件。
- 源码：SASHA 本机克隆到仓库外 `KT2028-assets/source/SASHA`，固定 commit `6385fe6bfcf4f1591b8f691ada2407c83002f62e`；选择源码压缩包留存服务器 audit/s2_network_20261009。查阅 HAFED、门控注意力、FGlobal、Train 损失与环境。原代码存在固定细 bag、估计视作访问、不同奖励等接口差异；没有直接运行原训练。首版按 P1 编写，不称原 HAFED 完整复现，差异/未完成项和精确文件 SHA256 见 `records/server_s2_network_20261009/implementation_check.json`。
- 交付：`kt2028/network.py` 可变长度 fine 聚合/coarse 投影/几何与观察标记/四分支患者预测/TSU/共享区域和 STOP 打分；`episode.py` 管理零 fine、显式正 N_ref、patch 硬预算、合法动作与终末损失；`check_network.py` 真实 Train CPU 前向检查；标签只可作为独立 Train 奖励参数，不进入推理状态，估计仍可获取、真 fine 永不被覆盖。
- 实际检查：服务器原有环境 CPU 上 13 项软件检查通过（原 8＋新 5），覆盖即时 STOP、终末 BCE 一次记账、整段收益等式、硬预算/重复/空获取、TSU 权限、coarse 梯度和置换不变；三例真实 Train 的 54/362/650 区域完成零 fine→单区域获取→TSU→Actor/STOP 前向。随机初始化，未输出患者风险或指标、未保存 checkpoint；真实前向回执存档于同目录。GPU 留给原有 frozen ResNet 作业。
- 失败保留：首次真实检查使用 N_ref=1，原始 patch 数/预算导致随机 Actor 的 STOP 概率下溢，断言失败；改为三例共同的检查参考 9603 后通过。只修检查尺度，没有把 9603 锁定为正式协议，也没有训练模型来通过检查。
- 限制：未实现原 HAFED 全损失/原论文复现、分阶段训练编排、Gymnasium 包装、SB3 固定 Discrete padding/rollout 适配或 PPO 学习。当前 Actor-Critic 只是可变 R 打分接口。未来训练前仍需完成 Q01/Q02 与 N_ref/预算/候选容量协议。
- 接续/停止：保存并同步 `research/p1-restart` 的本轮代码、证据及文档后停止本轮操作；原两项后台任务继续。后续先查实际进度/失败及全队列资产一致性，再完成训练数据加载/批处理与 PPO 适配；不自动训练。中转期间用户只需保持本机联网，暂停/重启前记录 PID/receipt 并复用缓存，不重复启动。

### W007｜2026-10-09 16:40–16:43：服务器状态与下一步可行性核查（完成）

- 用户任务：检查服务器状态，判断能否开展下一步；本轮只读资源、进程和现有数据，不安装、下载新增资源、修改算法或启动训练。核查源码基准为 `8e7bfe67ec2f2ef4ab5d142faea2662788160d13`，本机及服务器工作区开工时干净。
- 资源：SSH 正常；数据盘 200 GiB、剩余约 131 GiB；5090 显存 32607 MiB，快照使用 1097 MiB、利用率 0%。实际 cgroup 25 核/90 GiB，内存使用约 31.5 GiB；不把宿主机 754 GiB 当作本容器配额。既有 torch/CUDA/Zarr/NumPy/Gymnasium/sb3-contrib 导入正常，未改环境，详见[资源快照](../records/server_status_20261009/resource_status.json)。
- 作业：服务器 PID 29156 与本机中转 PID 36452 存活；16:40–16:42 为 325/423，16:43 中转升至 327，分段文件持续更新，当前错误为 0，尚无全队列完成标记。没有重启/中断后台任务。GPU 低利用率是快照，不据此推断计算故障。
- 资产核查：对快照中已完成的 325 份 receipt/NPZ 做完整 SHA256、共同配置、官方 cohort/split、fine 路径存在及一致、finite coarse/几何、CSR/数量/索引范围和唯一归属、必要 RGB 产物存在检查，325 份全部通过、错误 0。Train 255/255 完整，Validation 70/84，Test 0/84；区域数范围 54–672，已分配 fine 1,655,923/1,657,675，候选外 1,752。仅检查产物/结构，不是预后评价，没声称每张图视觉对齐均经人工确认。详见[资产检查快照](../records/server_status_20261009/asset_status.json)。
- 可追溯性：322 份来自全队列作业 commit 2eed94f，三份先导来自 a0c4a61/5da59f8，共同配置哈希一致、产物哈希及结构通过。保留实际来源差异，不把当前 Git HEAD 反写为全部产物来源。一轮临时只读核查程序留在服务器 audit 和仓库外本机资产目录，不另建算法版本脚本。
- 结论：可以立即继续 Train 数据加载、可变 R/k 批处理和训练入口准备，不必等待 Test 传完；这些是下一项最小准备工作，输出应为可复用加载/批处理接口和真实 Train 软件检查记录。正式训练尚未启动：训练编排、模型选择规则与具体运行配置未完成登记，PPO padding/rollout 适配也仍缺。Q01 临床随访/独立患者映射缺口继续保留，但不阻止接口准备。
- 本轮边界：完成状态核查并保存/同步文档与汇总后停止；本机继续联网以保持中转，暂无用户手动操作。后续先查实时进度再开展已登记的准备任务，不能把此快照当作全队列完成回执。

### W008｜2026-10-09 19:05 起：单例补传、全队列核查与预测器训练入口（完成）

- 用户授权：检查传输是否完成并开展下一步。基准 commit `aab9bd273e1b93556a0bae6367fac9f226b84a27`。本轮范围为补齐资产、全队列检查、Train 加载/病例批处理、full/align/partial 训练入口及软件检查；未启动正式优化、Validation/Test 预测或 PPO。
- 完成恢复：原队列实际为 422/423，1 例 Test 低倍率读取报 `Incomplete range`；原中转和区域任务均已退出，不是仍在传输。短 HTTP body 被 ValueError 抛出，未进入已有重试；修为 OSError，保留五次重试、Range/CRC/字节限额与分段复用。只补缺失图像，新增传输 6,777,045 字节，成功后服务器续跑，422 份完成资产校验复用。原错误 receipt/log 与恢复前 summary 保留；当前完成 423/423、pending=0、errors=[]，无后台准备任务继续。本机无需再为该队列保持下载。
- 数据结果：156,395 个 RGB 候选区域，2,179,406 fine patch 中 2,175,619 分配到候选，候选外 3,787；部分边缘补齐 3,357、完全图像外 1,407。全部 423 份 NPZ 哈希、共同配置、有限值、CSR/数量/唯一归属/索引和官方 255/84/84 划分检查通过。[全队列结构回执](../records/server_s2_training_interface_20261009/asset_status.json)保留不同作业 commit 来源；最终 summary 的当前 commit 不是全部产物的生成来源。
- Q05：全部 1,407 个完全越界 patch 来自一例 Test（原 fine 共 1,918），最大 fine 坐标显著大于官方 OME level0 尺寸；[异常事实](../records/server_s2_training_interface_20261009/geometry_anomaly.json)。仅确认坐标/尺寸不匹配，未认定源头原因，不自动缩放/移位/剔除。Train/Validation 无该完全越界异常；原 fine 和当前产物保留。结构通过不等于全部视觉对齐正确，正式 Test 前需解决。
- 实现：`training_data.py` 实际官方 Train manifest 加载，target/inputs 分离，仅按选区读取 UNI；按 case/seed/epoch 采样并保留 ragged 病例 batch，不固定 R/k、不跨病例汇聚，拒绝未适配的多切片 case。显式 Validation 仅可用于模型选择，Test 加载拒绝。`training_stages.py` 三个阶段的冻结/损失与部分 TSU 更新；`train_predictor.py` 分阶段优化、固定 Validation 选择、正确前驱权重/配置/资产指纹、私有 run/epoch/checkpoint/完成记录。默认执行真实 Train 检查，实际优化需显式 execute；本轮不调用 execute。
- 配置：登记 `configs/predictor_p1_candidate.json` 的起始参数及来源；partial Train 零/完整各 25%、中间 50%；30 epoch、病例 batch2、AdamW1e-4 等均为首个工程候选，不称最优。full 用 Validation BCE、align 用对齐＋TSU MSE、partial 用固定零/随机/完整 BCE 等权均值，平局首次；Test 不参与。P1 方法与奖励未改、PPO 预算/N_ref 未定。
- 实际验证：21 项软件测试通过；真实两例 Train 的 577/433 个区域、8,246/6,127 fine 分别在 CPU 和 RTX5090 完成 full/align/partial 的前向/反向及冻结检查，partial 另明确覆盖零和中间证据。损失/梯度有限、冻结参数无梯度；optimizer_steps=0，无 checkpoint、无预测性能指标。回执见[CPU](../records/server_s2_training_interface_20261009/preflight_cpu_complete.json)、[CUDA](../records/server_s2_training_interface_20261009/preflight_cuda_complete.json)、[实现来源/恢复说明](../records/server_s2_training_interface_20261009/implementation_receipt.json)。完整 epoch 优化、checkpoint 文件及 Validation 执行尚未实际验证，不能说正式训练已完成。
- 成本记录限制：旧 relay 每图失败时未返回累计 bytes，成功 receipt 的加和缺少该类失败传输；当前 summary network_bytes 也只描述服务器读取，不覆盖本机中转及旧尝试。不得以这些和数宣称完整实际网络或端到端计算成本。
- 下一最小工作：以 255 Train/84 Validation、区域资产及候选配置运行 full 阶段训练，记录独立 run ID、源码/配置/数据来源与模型选择；输出可重载的 best checkpoint 和真实日志，再开展 align/partial。PPO 适配与 Q05 调查可另行安排，不将当前未训练 TSU 用于正式策略实验。本轮保存并同步后停止，不自动启动优化或 Test。

### W009｜2026-10-09：第一阶段 full 预后训练（30 epoch 完成；2026-10-10 重载核实）

- 用户明确授权先训练一个看效果，并询问是否应先扩数据。执行范围：既有 255 Train/84 Validation、第一阶段 full、seed41、30 epoch、病例 batch2、AdamW1e-4；沿用已登记候选配置，Validation BCE 最小选 checkpoint（平局首次），同时记录 AUROC/Brier。Test、后续 align/partial/PPO、扩充数据均不在本轮范围。
- run ID：`p1_full_s41_20261009`；私有输出目录 `/root/autodl-tmp/KT2028/runs/p1_full_s41_20261009`，启动日志/PID 留独立 audit 目录。启动前核查现有区域资产与 Train/Val 不重叠；源码基准 f517efe，本次补充指标记录和 CPU 线程限制，无模型/损失/选择规则改变。
- 数据量判断：255 Train 对预后建模偏小，但冻结 UNI/ResNet18 后可以作为工程起点，不要求先扩充。patch、区域和部分状态不增加独立病例数。先观察既有模型的真实学习/Validation 行为，不自动引入不同临床来源、标签或研究终点。官网 SurGen 论文表4提供相同划分，整个 SurGen 的更多图像不能自动纳入当前五年任务；独立临床映射/原随访缺口仍保留。
- 实际启动：北京时间 19:35，源码 `534bc98e941768b8a60c5101574a6e4d06f52b62`，PID 56993；日志 `/root/autodl-tmp/KT2028/audit/p1_full_s41_20261009/training.log`，结果 `runs/p1_full_s41_20261009/epochs.jsonl`，完成必须看 `completion.json`。12 项相关软件检查通过；已有环境/数据，无新增依赖或下载。[启动记录](../records/p1_full_s41_20261009/process.json)。实际清单 label1（按官方编码为五年内死亡）Train100/255、Val30/84，参考[标签计数](../records/p1_full_s41_20261009/cohort_counts.json)。
- 首轮实际证据：epoch0 Train 在线 batch 平均 BCE 0.67252、Validation BCE 0.65546、AUROC 0.67716、Brier 0.23148；84 Val，非独立 Test，不把训练过程在线 BCE 叫作最终 Train 评价。best checkpoint 已生成并通过 weights_only＋strict state_dict 重载；[首 epoch 快照](../records/p1_full_s41_20261009/first_epoch_snapshot.json)记录当时 SHA，后续 best 更新时该 SHA 不再表示当前文件。训练继续，当前尚无 30 epoch 完成回执，不判定算法有效/最终性能。
- 用户中途询问扩数据：建议优先 [TCGA-COAD](https://portal.gdc.cancer.gov/projects/TCGA-COAD) / [TCGA-READ](https://portal.gdc.cancer.gov/projects/TCGA-READ)，先临床随访＋Diagnostic Slide 清单，筛五年结局可确定的病例再按需取 WSI；有 [GDC 官方获取说明](https://gdc.cancer.gov/access-data/gdc-data-portal)。[CPTAC-COAD](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70227852) 是后备，病理与临床来源明确，但实际五年随访完整病例数未查，不能承诺新增规模。五年前无死亡且失访不当作五年存活；GDC 随访字段需含 follow_ups，参见[2026 官方变更](https://docs.gdc.cancer.gov/Data_Portal/Release_Notes/Data_Portal_Release_Notes/)。仅建议，未下载或改主方案/当前队列；独立外部验证或扩 Train 用途应后续明确。
- 结束确认：30 行 epoch0–29 连续、完成回执 completed_epochs=30，PID56993 已退出。最佳 checkpoint 仍为 epoch1（第2轮），SHA256 `4d3f307f30efcb3f3fc0620518fd75282615d934be3754c63cd4dfefcc1bd6be` 与 completion 匹配；最终结果与局限见 W010，早期快照保留历史身份。

### W010｜2026-10-10：完整训练结果核查、重载与双份权重备份（完成）

- 任务：用户报告训练完成，核查此前授权的 full 实验产物、模型选择和实际效果。核查 checkout `de4fb33b2838913e624a7be561c6fdf1f2478c09`；训练源码仍为 534bc98，不修改算法、不启动新训练/后续阶段或 Test。
- 实际验证：完成回执、30 行连续日志、最小 Validation BCE epoch、权重 SHA 全部一致；weights_only/strict state_dict 重载，用原 255 Train/84 Validation 重新前向。Validation BCE/AUROC/Brier 与选中 epoch 日志差异 <1e-6。患者级预测仅留服务器私有 audit，不提交 GitHub。[重载核查](../records/p1_full_s41_20261009/completed_verification.json)、[完整 epoch 记录](../records/p1_full_s41_20261009/epochs.jsonl)、[完成回执](../records/p1_full_s41_20261009/completion.json)。
- 第2轮选中模型重载：Train AUROC0.77832、BCE0.59179、Brier0.20414；Validation AUROC0.67963、BCE0.60787、Brier0.21199。只按 Train 正例比例100/255给恒定风险的参考在 Val AUROC0.5、BCE0.65436、Brier0.23082；只是常数参考，不代替 fine ABMIL/其他成熟模型对照。不根据该差值声称统计显著或泛化已证实。
- 过拟合证据：第30轮 Train 在线 batch BCE0.0002066，Val BCE2.87003/Brier0.33903；最高 Val AUROC0.73210 出现在第17轮，但 BCE2.03448/Brier0.34120。不事后将该轮替代事先登记 BCE 选出的模型，也不把最高 AUROC 当作最终模型指标。后期排序指标与概率质量明显分离；初版有一定区分能力、容易过拟合，现有证据不能证明扩数据必需/充分，也不能证明可靠的部分观察或 RL 效果。
- 权重备份：服务器 `runs/p1_full_s41_20261009/best.pt` 与本机仓库外 `D:\codex\开题2028\KT2028-assets\checkpoints\p1_full_s41_20261009\best.pt` 均实际存在，SHA相同。仅约4.1MiB自训权重备份，不是新增预训练权重/数据下载；不入 GitHub。日志与小型汇总按专用分支持续存档。
- 限制/下一最小建议：单 seed，84 Val 已用于选模型，不能当独立 Test；Q01 原随访/临床身份、Q05 Test 几何保持开放。建议先登记一个受控过拟合处理/必要 fine ABMIL 对照，固定划分、成本范围与选择规则后比较，不直接认为需扩大数据；现行 P1 不自动修改，不把指标设为开展 RL 的通关门槛。本轮核查/备份/存档后停止。

### W011｜2026-10-10：进入主方法，粗细对齐与 TSU 训练（30轮完成、重载核查及双份备份完成）

- 用户要求：询问下一步并要求开始自己的方法。按最新指令推进 P1 阶段 2，不沿用 W010 的追加基线优先建议；不修改方法版本。开工 checkout `1c87db0bc4bacff703ad8372145d973b0e54e685`，本机与服务器干净且一致；GPU 空闲。
- 登记 run ID `p1_align_s41_20261010`，输入 `runs/p1_full_s41_20261009/best.pt`，前驱 SHA256 `4d3f307f30efcb3f3fc0620518fd75282615d934be3754c63cd4dfefcc1bd6be`。沿用 `configs/predictor_p1_candidate.json`、255 Train/84 Validation、seed41/30 epoch、batch2、AdamW1e-4；不使用 Test，不扩数据、不挑新 seed。
- 优化范围：冻结 fine 局部编码及患者预测模块，只训练 coarse 投影和 TSU；每病例余弦相似度≥0.9 的非自配对最多采样128对。损失为 coarse 对齐 MSE＋TSU MSE，选择固定 Validation 阶段损失最小 checkpoint（平局首次）。预测 AUROC 不作为本阶段选择指标。
- 实现准备：仅补充阶段诊断日志，分别报告对齐 MSE、TSU MSE、实际监督配对数及有配对病例数；不改目标、采样、参数或前驱。启动后核实实际优化、TSU 配对/参数改变、冻结参数与前驱逐 tensor 相同，不能用进程存在替代训练证据。
- 输出/范围：私有 `runs/p1_align_s41_20261010/{run.json,epochs.jsonl,best.pt,completion.json}` 和独立 audit 日志；小型运行/验证回执存档专用分支，权重不入 Git。完成启动确认后结束本轮，后台仅继续已启动 align。
- 实际启动：北京时间09:33，训练源码 `309a5250e84c2f6fa350281cef32ff7579eafafd`，PID77832；日志 `/root/autodl-tmp/KT2028/audit/p1_align_s41_20261010/training.log`。18项相关软件检查通过；09:35快照已完成epoch0/1，固定 Validation 总MSE从0.67623降至0.62618，第2轮对齐0.08021/TSU0.54597。Train255例均有TSU配对，第2轮共32640对；Val84例10752对。只是训练诊断，不等于预后获益。[启动记录](../records/p1_align_s41_20261010/process.json)、[来源配置](../records/p1_align_s41_20261010/run.json)、[优化/冻结快照](../records/p1_align_s41_20261010/first_epoch_snapshot.json)。
- 权重验证：前两轮最佳 checkpoint weights_only＋strict重载成功；18个冻结model tensor与full前驱逐元素相同，2个coarse和6个TSU tensor确已改变，参数有限。当前快照SHA仅表示当时第2轮文件，后续best可能更新。30轮完成、最终best与完成回执尚未核实，不能写成align完成；无需用户手动操作，也无需本机持续在线。
- 接续：align 结束并核实最佳权重后，输入其 checkpoint 训练 partial（零/随机/完整证据）；其后固定成本/预算并适配 PPO。强基线与消融按已有论文计划开展，不成为开始主方法的先决条件；不保证 TSU 提高预测或 PPO 超过随机。
- 完成核查：用户报告结束；本轮 checkout `37a0356a07281779900c3c8b1cb235cddeb4689b`，训练源码仍为309a525。30行epoch0–29、completion、最小Val阶段损失与最佳权重一致，选第30轮。SHA256 `1d0a9b33367ebd4350774b44908853497e55ae49f9236d733bb7a6db81ae56af`。strict重载、84 Val原配对规则重新前向通过，总MSE0.138168889与日志差<1e-6（coarse0.0588066/TSU0.0793623）；18个冻结tensor与full相同、2个coarse及6个TSU tensor改变。见[完成核查](../records/p1_align_s41_20261010/completed_verification.json)、[完整日志](../records/p1_align_s41_20261010/epochs.jsonl)、[完成回执](../records/p1_align_s41_20261010/completion.json)。
- 负结果诊断：在同一84 Val、10,752个有序采样配对上，TSU MSE0.07936，直接使用目标coarse为0.05686，直接复制来源fine为0.06260；当前TSU表示重建误差未优于两个简单参照。阈值合格配对占全部非自配对约17.88%，不是任意区域全连。此为训练后探索性诊断、Val已用于选择，不替换模型，不证明预后好坏或临床作用；后续保留TSU消融，不能仅因MSE下降称传播有效。
- 双份备份：服务器best.pt与本机仓库外 `KT2028-assets/checkpoints/p1_align_s41_20261010/best.pt` 均实际存在且SHA一致，约4.1MiB；权重不入Git。GPU核查空闲。Q01/Q05未解决，但不阻塞当前Train/Val partial训练。

### W012｜2026-10-10：部分证据预后训练（首epoch优化/冻结核查完成，30轮训练进行中）

- 范围：用户报告align完成，按此前要求开始主方法及已记录的接续顺序，核实前驱后继续partial；不修改P1、数据/标签、采样、结构、损失、seed或选择规则，不启动PPO/Test。
- run ID `p1_partial_s41_20261010`，前驱为W011最佳align权重，SHA `1d0a9b33367ebd4350774b44908853497e55ae49f9236d733bb7a6db81ae56af`；输入既有255 Train/84 Val、区域资产与 `configs/predictor_p1_candidate.json`，seed41、30 epoch、batch2、AdamW1e-4。
- 冻结fine局部编码、coarse投影及TSU，训练context、患者注意力汇聚和二分类头；Train采样零/完整各25%，其余随机中间证据量。Val固定zero/random/full三模式，以三者BCE等权平均最小选checkpoint（平局首次），分模式报告AUROC/BCE/Brier。random组本身包含零和完整抽样，不称纯中间证据队列。
- 输出：独立私有 `runs/p1_partial_s41_20261010` 和 audit日志；前驱/源码/清单/资产哈希与epoch指标按既有入口保存。核实首次实际优化、冻结参数不变及三模式Val执行后结束本轮，后台只继续partial；完成后才能报告最终部分证据效果。
- 风险/后续：TSU暂未优于coarse表示误差，仍按登记主方法完成预测训练，后续同协议去TSU比较，不假定必有收益、不把预测性能设为RL通关门槛。PPO前仍须明确成本参考/预算/λ并完成可变区域接口，不自动采用旧协议。
- 实际启动：北京时间10:18，源码 `95b7face17f1b72246847472c5cfd8d29a7d6b27`，PID79335，日志 `/root/autodl-tmp/KT2028/audit/p1_partial_s41_20261010/training.log`。18项相关软件检查通过；输入前驱SHA匹配，资产/清单前驱检查通过并生成run.json。首epoch/首次checkpoint核实仍待完成，进程存在不等于实验完成。[启动记录](../records/p1_partial_s41_20261010/process.json)。
- 首epoch实际核查（10:23）：Train在线BCE0.60302；Val zero BCE0.63305/AUROC0.67531/Brier0.22071，random BCE0.60479/AUROC0.68272/Brier0.20982，full BCE0.60677/AUROC0.67963/Brier0.21132，各84例；三模式平均BCE0.61487。仅首轮/model-selection Val，不是最终效果，不事后改选AUROC。10个冻结model tensor、6个TSU tensor与align前驱逐元素一致；10个允许训练tensor实际改变，69,381个可训练参数。weights_only＋strict重载通过，选择分数与三BCE均值一致。[真实优化/冻结快照](../records/p1_partial_s41_20261010/first_epoch_snapshot.json)、[来源配置](../records/p1_partial_s41_20261010/run.json)。快照checkpoint SHA会随后续best更新而过期。
- 本轮结束边界：已完成align核查/权重双份备份和partial启动检查，小型记录与长期文档持续提交推送并同步服务器；本机无需持续在线。仅partial后台继续，尚无30轮完成回执，不自动启动PPO/Test。下次先核查完成回执和最佳权重，再报告三模式最终行为并准备策略协议/接口。

### W013｜2026-10-10：固化通俗解释与汇报规则（完成）

- 用户明确要求将“先解释问题、输入、训练对象和输出，再报告简称与指标”写入长期工作规则。核查基准 `d6f9554e1e307d169f5a4f54f712806fd69c4512`，开工工作区干净。
- 已在 [AGENTS.md 的汇报规则](../AGENTS.md#汇报) 写入：每次先讲当前阶段的目的、输入、训练对象/不训练、输出及流程位置；模块简称先解释中文职责与必要例子；指标说明含义、方向、数据划分及证据范围；参数说明实际来源，不将工程候选称为最优或训练充分。
- 本轮只改规则和工作记录，P1方案、算法、配置及后台实验均未改；不启动新实验，不刷新训练状态。按既有授权提交推送专用分支并同步服务器规则，后续会话遵循此入口。

### W014｜2026-10-10：工作一方法与网络架构图（完成）

- 用户要求解释工作一的具体方法并用一张图展示网络。开工checkout `9dd8fe0d1a2a922ad7b51fda0830dd70b84ded14`、工作区干净；核对长期方案0.5/0.6及实际 `network.py`、`training_stages.py`。
- 将原简略文本图替换为长期方案0.5中的单张Mermaid架构图；明确全部低倍率区域/仅已观察精细patch输入，512/1024→256投影、区域内单分支与患者级四分支门控注意力、按状态取证据、7→256状态编码相加、1024→1线性输出。标注TSU为工作二辅助接口、训练期MSE对齐与推理路径；未把concat、Transformer、医学四类别或实时UNI提取写入实际实现。
- 补充阶段冻结与标签权限、状态比例按区域数计算，以及TSU尚无优势的证据边界；只澄清现有P1实现，不改方案版本/算法/配置，不检查或启动实验。图原位存入唯一长期方案，不另建重复报告。文档按既有授权存档并同步服务器。
