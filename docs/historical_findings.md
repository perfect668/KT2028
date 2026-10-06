# 历史实验与已知问题

更新日期：2026-10-06。本文件记录历史证据，不是继续实验的任务单。

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
