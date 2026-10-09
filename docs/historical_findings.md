# 历史实验与已知问题

更新日期：2026-10-06。本文件记录历史证据，不是继续实验的任务单。

2026-10-09 导航补充：现行方案只维护于[长期研究方案第 0 节](experiment_restart.md#0-当前长期研究方案)。本文“当前”“下一步”等按历史日期理解，不覆盖现行方案。

依据：GitHub [133740c 源码快照](https://github.com/perfect668/KT2028/tree/133740c6ade236d947b14d25f33430e1f37c3bdf)、仓库归档结果，以及用户关于旧服务器损坏的说明。本轮只核查了公开源码与记录，未重新运行数据实验，不能验证已经丢失的权重或患者级输出。

## 1. 已完成过的工作

| 工作 | 历史记录 | 能支持的判断 |
| --- | --- | --- |
| 数据/区域基础设施 | 全量 UNI、10 簇、K64、coarse/fine 构造代码 | 有可审查的处理方法，完整原始 WSI 协议仍待恢复 |
| 自建 E2 | EvidenceNet、三 seed 集成、partial 校准与 OOF 脚本 | 有预测器实现；不等于概率可靠性已充分证明 |
| Dense 闭环 smoke | 38 项通过，2025 Train transitions，8 名患者 | 报告显示核心语义与数值检查通过，不证明泛化 |
| 正式 Dense PPO | corrected 训练 200894，加 lambda3 5956 | 曾完成正式单 seed 实验，后来确认 warm-start 协议受影响 |
| Validation 对照 | 84 名患者，固定/自适应及 matched budget | 可查看历史指标，不能当作独立 Test 结论 |
| 后续效用审计 | 源码统一效用；未重训旧模型 | 发现了实质问题，旧结果仍受影响 |

Smoke 的 critic 固定批次过拟合 EV 约 0.97904、checkpoint 重载输出差异 0，是实现检查记录。正式训练 critic EV 仍为负，历史汇报约 -4.82 至 -0.33。过拟合通过与正式价值估计有效是不同命题。

## 2. corrected / balanced 历史结果

归档：[results/dense_formal_seed17_corrected](../results/dense_formal_seed17_corrected/)。

四档累计 transitions 为 51670、51717、51644、51819，总计 206850。按 `(max-min)/max` 复算约 0.34%，满足旧任务的 <5% 条件；历史文字所称“约 0.20%”不等于该公式的结果。

历史 lambda：0、0.0003935273813882034、0.0011797714321550856、0.003516254591272361。它们是当时生成的值，**后来确认生成路径有错误，不能直接作为重启参数**。

balanced checkpoint 的历史 SHA256：
`1fd38332787f3a76e481184dec6c16cbfb38778778703078577957cd67e535c7`。
该摘要只标识历史文件，不证明现在可取得权重。

主要 Validation 指标（四舍五入）：

| 方法 | AUROC | AUPRC | Brier | 平均区域数 | 平均代理成本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2_K0 | 0.7568 | 0.6555 | 0.2112 | 0 | 0 |
| E2_Full64 | 0.7506 | 0.6510 | 0.2008 | 64 | 0* |
| Random20_K4 mean | 0.7519 | 0.6518 | 0.2012 | 4 | 1.9162 |
| AttentionTopK_K4 | 0.7475 | 0.6485 | 0.2011 | 4 | 1.8426 |
| DensePPO_K4 | 0.7525 | 0.6536 | 0.2012 | 4 | 1.3763 |
| DensePPO_K8 | 0.7525 | 0.6534 | 0.2015 | 8 | 3.0475 |
| DensePPO adaptive | 0.7525 | 0.6534 | 0.2014 | 7.58 | 2.9157 |

*Full64 的归档成本 0 是该记录的记账问题，不能解释为处理全部区域免费。Random20 的 mean 汇总按评价脚本对重复预测取均值后计算指标，应与逐次随机运行的指标分布区分。

从这些数值可观察到：Dense K4 比 Random mean 的 AUROC 约高 0.0006、AUPRC 约高 0.0018；Brier 没有明确改善。没有独立、多 seed 或统计证据支持“显著超过随机”。E2_K0 的排序指标还略高于 Full64，但 Full64 的 Brier 较低；这也不能单凭一张表断言 fine 完全无价值。

adaptive 四档 lambda 产生相同的确定性动作和预算，平均 7.58 接近 8 步上限。没有证明策略学到成本参数控制下的预算适应。历史 one-step STOP regret 24 个状态，均值约 0.00324、中位数 0.00181、最大 0.00953，只是标签已知后的即时诊断，不能称为长期 regret。

这套结果不构成与 HeMiCoRe 等原始模仿基线的公平超越证明；需先确认原论文版本、数据、终点、划分和评价协议。

## 3. 后续审计发现：效用符号不一致

见 [审计摘要](../results/dense_formal_seed17_corrected/utility_consistency_audit/audit_summary.json) 与 [审计报告](../results/dense_formal_seed17_corrected/utility_consistency_audit/utility_consistency_audit_report.md)。

旧正式执行源码中的局部 `phi` 使用正 BCE，环境的 `evidence_potential` 使用负 BCE。结果是：
- 动作枚举、warm-start 目标与 lambda 分位数走错误效用；
- PPO 环境奖励走正确效用；
- 两条路径优化的方向不一致。

审计在 8 名 Train 患者、k=0/1/2/4 的 1992 个合法动作比较行上记录：
- 最大收益差异 0.8360994；
- 符号不一致率 99.1466%；
- 最佳动作改变率 100%；
- 旧规则所选动作在统一效用下平均增益 -0.0495886；
- 平均即时 regret 0.0400688。

JSON 的字段名是 `states=1992`，报告将其说明为合法动作比较行；不能误写成 1992 个独立患者状态。

另有非零历史的 warm-start 把累计成本置零，与实际已选成本不一致。审计修复了接口，统一调用 `evidence_potential`，复算缓存/环境 reward 差异为零。**审计没有重新 warm-start、训练、Validation 或 Test；旧 balanced 权重没有被更新。**

因此旧 Dense 结果应标为“协议受影响的历史结果”。修复源码不能恢复旧模型的正确性，也不自动否定每一个独立计算的 E2 指标。

## 4. OOF 的证据边界

[v7_run_oof.py](../msrlselect_v8/src/MSRLSelect/v7_run_oof.py) 将外层 fold 排除出梯度训练，但把该 fold 传给 `train_one` 作 checkpoint 选择。源码自己注明留出 fold 用于选择。

所以“训练不含该患者”是局部事实，“严格独立 OOF”不成立。外层标签间接参与了模型选择。Train 全体拟合的 PCA/形态中心及旧 teacher 的来源也需要在新协议中隔离并追溯。

此外，`v7_train_evidence.py` 主入口曾把旧 teacher 概率写为 teacher-aligned OOF 产物；另一个 OOF 脚本生成真实模型预测并可覆盖同名文件。重启不能仅根据文件名判定来源，必须记录生成命令、执行源码和拟合数据。

## 5. 成本与模型解释的历史问题

- coarse 已依赖全量 UNI；fine 缓存按 mask 开放模拟并未实测按需图像编码。
- log patch-count 归一化成本不等于实际计算代价。
- 四槽是聚类汇聚，不是已经证实的四个病理语义专家。
- Dense 策略先平均槽再编码，与 E2 拼接保留槽位置的路径不同。
- 完整证据上的预测器性能不能代表其部分证据概率和动作收益都可靠。
- 历史 Validation 曾用于预测器 checkpoint 选择和多轮开发，不能把现存 Validation 当成完全未触碰的最终独立集。
- corrected Dense 记录中 `test_run=false` 只约束这轮记录，不能推断所有历史脚本从未读取 Test。

## 6. 当前能得出的结论

已完成过较多工程实现、闭环检查和一轮正式 Dense 实验，但没有建立可靠的最终算法收益证据。整体论文仍缺少协议干净的预测器、真实成本论证、最终方法选择、必要对照/消融/重复和锁定后的独立评价。

重启不是继续追加 MoE。当前重点是数据与预测器基础；旧代码、负结果与审计可作为诊断资产，旧 checkpoint 和旧 lambda 不作为可信启动参数。后续选择以新协议和实际证据为准。

## 7. 2026-10-06 本轮源码与归档核查补充

读取 commit：`7ed0dcdd023f0dd963dda95a60733729473d4766`。以下是静态源码检查、Git 对照及归档 CSV 算术核查，不是数据实验复现；本轮未运行历史测试或评价入口。

### 已修复与历史来源

- `e1a6ec086d02ff2c29d26f0f4e216da7ce9495b2` 的 Git 中 `formal_dense.py` 字节 SHA256 复算为 `4d3f8909af406fa3691230b610cf92d243a02a099b605529ba72382610e8c918`，与审计报告所称服务器执行源摘要一致。Git 源码可直接确认正 BCE 的局部 `phi`、动作增益调用和 warm-start 历史累计成本为零。服务器文件当时与此相同的说法仍来自历史审计，并非本轮连接服务器验证。
- `3506ba8` 的源码差异确认：移除局部 `phi`、动作增益调用统一负 BCE `evidence_potential`、warm-start 训练及诊断观测改为所选成本之和。当前环境本来就正确累加 ACQUIRE 成本；不能将 warm-start 缺陷写成全部 rollout 都漏记成本。
- 归档的 `formal_dense_server_repaired.py` SHA256 为 `0034c23c…`，当前主路径 `formal_dense.py` 为 `91544ea7…`（均按 Git 内容计算），两者不是同一脚本，不能混用版本或执行命令。`source_manifest.json.files.formal_dense.py` 的 `7f8385…` 字段也不能当作当前主路径摘要；旧执行源应以明确 archive commit、复算的 Git 摘要及审计文字对应。
- 动作表实际为 8 个患者索引 × 4 个 k，即 32 个 patient/k 组、1992 个合法动作行；符号不一致率从 CSV 复算为 99.1466%。lambda 表为 7968 行，其存储的 cached/environment reward 差异最大值为 0。这里只验证了归档表内部记录，未用 E2 权重复算动作收益。
- corrected 日志中最高 lambda 因 CSV 浮点文本不同分成两串，合并 45863+5956=51819 后四档累计为 51670/51717/51644/51819，总计206850；差异率约0.3385%。训练日志 EV 范围为 -4.8190 至 -0.3328。没有发现效用修复后重新训练旧 balanced 模型的归档证据。

### OOF 与身份映射的额外限制

`v7_run_oof.py` 用 CSV 行号 `%5` 分外层 fold，`rl_core/assets.py` 同样按当前行索引 `%5` 路由。它不是按 `patient_id` 分组的划分器，CSV 换序也会改变患者所用模型。缺完整元数据/生成时名单，不能验证一人多片隔离及行号对应。

PCA 和球面四中心都在全部 Train 上拟合，OOF fold 直接复用。`v7_r1_revised_inference.py` 虽对报告用校准预测做按 fold cross-fitting，却把全 Train OOF 拟合的最终系数写入公共参数文件；Dense Train 实际读取该公共校准器，不能把报告的 cross-fit 隔离自动扩展到 RL 风险/奖励。外层 checkpoint 已泄漏的事实也不会被后续校准 cross-fitting 修复。

teacher 路径可追至 `v6_p1r.py` 的 Stage1C C3 五折三 seed 概率；其 StratifiedKFold 与 E2 行号 fold 不是同一分法。teacher 模型训练隔离是否满足每个 E2 外层 fold 的要求仍需追溯。E2 训练直接取 teacher CSV 的 `.values` 而未按患者 ID join，完整 slots 也依赖 CSV 行序；是否发生实际错配须用缺失资产验证。上述问题不能仅凭文件名或 shape 通过予以排除。

### 当前评价源码仍存在的问题（未修改算法）

`evaluate_dense.py` 在本轮 HEAD 与 `e1a6ec0` 中相同，但本轮未找到独立证明当时服务器评价执行源一致的摘要。以下可确认代码问题，历史数值影响应等待完整执行来源和逐患者轨迹核查：

1. `fixed` 的 coarse 分支使用 `.norm(-1)`（第41行）。首个参数是范数阶数 p，默认 dim=None，并非最后一维；得到整体标量后按 active mask 广播，未观察区域失去各自评分，argmax 退化为索引优先。它影响该源码的 `coarse-only` 对照，而非自动影响 Random/Attention/PPO 固定预算路径。依据：[PyTorch norm 参数定义](https://docs.pytorch.org/docs/2.14/generated/torch.norm.html)。
2. `adaptive` 第64行的 `ix[~stop] & (active[ix[~stop]].sum(1)>=64)` 将 long 患者索引与 bool 条件按位运算，结果用作整数索引，不是先用条件筛选 ix。8步内达不到64区域，此条件为 false 时可产生0索引并误将 `alive[0]` 置 false，造成不由 STOP 决定的提前结束。此问题位于 evaluator，不位于 `SequentialAcquireEnv` 的训练终止实现。
3. `adaptive` 第60行在更新获取之前保存轨迹，单步 `cost` 固定为0；后续 `total` 确有累加，故轨迹单步成本与最终汇总成本应区分。不能从轨迹0成本推出环境未收费。
4. Full64 第74行调用 `add(name,pp,k)` 时不传总成本，第71行的缺省值是0。源码直接解释了归档 Full64 成本0的记账缺陷；当前仍未修复。原始 UNI、启动 coarse、I/O、预测器及策略实际计算成本也未计入此代理成本。

这些新发现进一步限制 adaptive/对照的历史解释，但不允许反向从汇总表编造动作序列或断言某个患者当时必定受影响。仓库缺 `validation_predictions.csv`、`validation_action_sequences.jsonl`、`validation_stop_audit.csv` 的完整正式评价产物，因此四档“动作完全一致”当前只能引用历史记录；本轮仅从汇总表确认四档预算、代理成本及预测指标一致。
