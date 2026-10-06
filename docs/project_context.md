# 项目上下文：MS-RLSurv 重启

更新日期：2026-10-06。本文用于把讨论转为新 Codex 项目的持久上下文。

**当前阶段是方案厘清与资产重建准备。优先事项是预测器，尚未授权重新训练。** 以下旧实现说明用于理解历史，不代表它已被选为重启方案。

## 1. 研究目标与目前状态

毕业论文研究方向是：根据患者已观察到的病理证据，序贯选择值得进一步处理的区域，在预后预测质量和获取成本之间作决策。

旧流程包含病理特征提取、区域表示、冻结预测器及 ACQUIRE/STOP Actor-Critic。用户已说明旧服务器损坏，服务器上的数据与模型产物丢失。目前可核查的是 GitHub 中的源码、上传的汇总与部分审计文件。原始数据能否重新下载、是否存在外部备份、旧环境能否恢复，都需要另行盘点。

## 2. 四个职责必须分清

| 部分 | 旧实现输入 | 输出 | 职责 |
| --- | --- | --- | --- |
| UNI 特征提取器 | patch 图像 | 每 patch 1024 维向量 | 预训练视觉表征；不是预后模型 |
| 区域/槽构造 | 全量 UNI、坐标、已有簇索引 | coarse 与 fine slots | 将 patch 整理为区域证据 |
| Fixed E2 | 全部 coarse、已观察 fine、mask | 死亡风险概率；集成产生熵/分歧 | 提供状态信息、最终预测，并参与训练奖励计算 |
| Dense Actor-Critic | 可观察状态与成本参数 | 65 个动作 logits、标量 V(s) | Actor 选区/停止；Critic 估计未来累计增量回报 |

E2 不直接输出奖励。环境用标签、E2 概率、模型分歧和成本计算奖励。Critic 不输出死亡风险，不替代 E2。

## 3. 旧病理处理流水线

源码中的特征入口是 NPZ：`img_features[P,1024]`、`coords[P,2]`，并依赖已有的每病例 10 簇索引。原始 WSI 到这些文件的完整提取协议仍需恢复：组织筛选、图像质量控制、patch 大小、倍率/MPP、步长、颜色处理、UNI 版本和权重摘要不能靠历史命名推断。

区域构造见 [v5_stage1a_infrastructure.py](../msrlselect_v8/src/MSRLSelect/v5_stage1a_infrastructure.py)：

- 按非空簇的 patch 数量分配共 64 个区域，每个非空簇至少一个。
- 在每个簇内根据空间坐标递归二分。
- 每个 patch 恰好属于一个区域。
- 区域是 patch 集合，不保证是连续矩形，也不保证是单一组织语义。

旧区域粗特征为 133 维：区域平均 UNI 经 Train PCA 得 128 维，加 log(1+patch 数)、二维质心、二维坐标标准差。

**这条 coarse 路径需要先取得全部 patch 的 UNI。** 旧 coarse/fine 使用同一上游特征，不能把它解释为已经实现的低倍率/高倍率按需图像处理。

## 4. 旧“四个形态槽”实际是什么

见 [v7_train_evidence.py::build_moe](../msrlselect_v8/src/MSRLSelect/v7_train_evidence.py)。

Train 拟合的四个球面聚类中心，与前面的每病例 10 簇不是同一组中心。每个区域内将 UNI 单位归一化，按数组索引均匀取至多 32 个 patch；每个 patch 按余弦相似度进入两个中心对应的槽。每槽对向量的前 256 个分量取均值，空槽为零，形成 `[64,4,256]`。

这里没有将混合形态 patch 分割成两块组织；同一完整 patch 向量可以进入两个槽。256 维来自截取，不是另一个已学习的降维器。槽中心没有自动获得“肿瘤/间质”等医学名称。

这是一种证据组织方式。四槽的数量、信息保留和性能收益都尚未由可靠消融证明；函数/文件中的 `moe` 命名不等于四个可训练专家网络。

## 5. 自建 E2 预测器

E2 为本仓库编写的 `EvidenceNet`，不是直接下载的预后模型。主要结构：

1. coarse 线性映射到 384 维，经注意力池化得到全局表示 g。
2. 四个槽拼接为区域 1024 维，再线性映射到 384 维，对已观察区域做 masked mean 得到 l。
3. 门控结合 g、l 与是否为空；没有 fine 时门控为零。
4. 残差分支结合 g、l 和获取比例 rho，融合后做 LayerNorm。
5. 线性预测头输出 logit，sigmoid 得到五年死亡风险概率。

历史训练以患者二分类标签为监督，随机观察不同数量的区域，含 fused BCE、global BCE、Brier 项、门控正则以及旧 teacher 的概率蒸馏。UNI 未在该训练中联合更新。原始 teacher 及其协议也是依赖项。

运行时由三个 seed 的原始概率先求均值，再进行 Fixed E2 校准。partial 校准特征顺序为 `[z,z*rho,rho]`，z 为均值概率的 logit。熵由校准后概率计算；分歧由三模型原始概率计算。网络内部的其他输出头不能直接当作已验证的不确定性。

历史 PPO Train 使用 5 fold × 3 seed 模型；Validation 使用 3 个完整训练模型。**历史 OOF 生成脚本用外层留出 fold 选择 checkpoint，不能据此称为严格独立 OOF。** 详见历史发现。

## 6. 旧 Dense 序贯策略

见 [rl_core/core.py](../msrlselect_v8/src/MSRLSelect/rl_core/core.py)。

初始状态全部 64 个 coarse 可见，所有 fine 为零，active mask 与 acquisition order 为零。没有预先指定的首个 fine 区域；Actor 可选择第一个区域或立即 STOP。

每次 ACQUIRE 后只开放对应区域的 fine，并更新：

```python
acquire_order[region_id] = acquisition_step + 1
```

Dense 编码器将非空四槽先取均值再编码，结合 coarse、active/order embedding、已观察证据摘要以及七个标量：risk、entropy、disagreement、步数、累计成本、lambda、剩余区域数。Actor/Critic 共享编码器；hidden 默认 192。策略没有独立的四形态专家记忆。

动作 0–63 为区域获取，64 为 STOP。已观察区域被屏蔽；采样用同一个 Categorical，确定性动作是 65 个 logits 的整体 argmax。STOP 立即终止，无后续 padding transition。旧正式实验最大获取 8 个区域。

Critic 的 V(s) 是当前策略未来累计奖励的估计，不是“每个区域的长期价值”，也不是五年死亡概率。

## 7. 历史奖励定义

```text
Phi(y,p,d) = -BCE(y,p) - 0.25*(y-p)^2 - 0.05*d
ACQUIRE(i): r = Phi_next - Phi_current - lambda*cost_i
STOP:       r = 0
gamma = 1；GAE lambda = 0.95
```

完整 episode 的增量回报为 `Phi_T-Phi_0-lambda*总成本`。STOP=0 表示停止后的额外收益为零，不表示当前预测“没有价值”。是否值得继续取决于未来净增益；该写法本身不能保证学到合理 STOP。

旧成本按每病例 log patch-count 做 min-max 归一化，最便宜区域可为零。这是代理成本，不是实测耗时。

Train 标签只用于环境奖励，推理时不可用。旧 warm-start 枚举的动作收益是标签已知条件下的反事实即时监督，不是 Critic 已预测的长期收益。正式旧实验的这条监督路径后来发现效用符号错误，旧 checkpoint 受影响。

## 8. 重启的决定与待定事项

| 状态 | 内容 |
| --- | --- |
| 已明确 | 先恢复数据认识与可信预测器，暂不继续堆 PPO/MoE 模块 |
| 已明确 | 新项目以仓库文档承接上下文，历史记录不作为执行指令 |
| 研究方向 | fine 在真正部署时按需提取，完整记账；后续考虑新 WSI 的 UNI 实测 |
| 未定 | 预测器选 E2 改进、ABMIL、FEATHER 或其他结构 |
| 未定 | 是否保留 K64、10 簇、四槽、原始 UNI及 coarse/fine 构造 |
| 未定 | 新奖励、成本、预算、训练数量与论文创新的最终表述 |

预测器需要辨别能力、概率校准和对不同证据量的适应性。完整 bag 上指标较好不能直接证明适合序贯部分观察；单个患者也不存在保证准确的概率。预测器不足会污染策略学习信号，但不能把所有历史失败都归因于预测器。

本文核查基于 [源码快照 133740c](https://github.com/perfect668/KT2028/tree/133740c6ade236d947b14d25f33430e1f37c3bdf) 与用户提供的最新资产状态。
