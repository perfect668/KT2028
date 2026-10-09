# P1 当前数据接口

唯一现行方法见 `docs/experiment_restart.md` 第 0 节。本包与 `msrlsurv/` 早期草稿分开，包含真实数据预处理、部分获取、首版预后/TSU/策略打分与 episode 接口；尚未训练预后网络或 PPO。

`network.py` 接通可变长度 UNI 局部汇聚、512→256 coarse 投影、四分支患者汇聚、二分类 logit、三向量 TSU 及共享区域/STOP 打分。分支不是医学类别。仅接受已获取的 fine 字典；TSU 估计另存，永不冒充真观察。借鉴 SASHA 已锁定源码的分层门控注意力与 MLP 思路，是 P1 接口适配，尚非原 HAFED 全损失/训练协议的复现。[来源与软件检查记录](../records/server_s2_network_20261009/implementation_check.json)明确差异与未完成项。

`episode.py` 使用冻结前向、显式正 N_ref、整数 patch 硬预算和 λ；立即 STOP 合法，空/重复/预算不足获取拒绝。Train 可单独传奖励标签，推理无需标签；标签不进入 visible_state 或网络。获取不可再继续时也自动终止，终末 BCE 只记一次，γ=1 的目标为终末 BCE＋累计成本。`SharedActorCritic` 目前是可变 R 的 PyTorch 打分器，**不是已接通的 SB3 MaskablePPO**；固定 Discrete 的 padding、批量 rollout 和 Gymnasium 包装尚需实现。

`check_network` 只对已登记三例 Train 区域资产做 CPU 随机初始化前向检查，不输出预后指标、不保存模型权重。检查用共同 N_ref=9603 不代表已选择正式成本协议；真实数据流成功不等于算法有效。13 项几何/缓存/中转/网络环境软件检查通过，真实前向回执单列保存。

`prepare_regions` 用真实低倍率 RGB 的组织阈值和不重叠物理网格构建候选，随后读取 fine 坐标建立一对一归属。粗向量由冻结 ImageNet ResNet18 提取，计算过程不读取 UNI 数值。区域内所有可用 fine 均保留，数量可变；无 fine 的 RGB 区域仍提供 coarse，但不能获取。边缘 patch 按有效图像交集中心归属，保留有效比例，不造新患者/patch。

`configs/regions_p1_trial.json` 是 S2 接口配置：4 µm/px、224 像素对应 896 µm 视野；选择不粗于目标的原生层。这个几何候选用于落实接口，尚非最终实验超参数，不将每区约 16 个视野写成固定 patch 数。组织阈值是 RGB 工程筛选，不是医学组织分类器。白色补边后仅做 ImageNet 归一化，不再 center crop 改变物理视野。

运行方式（路径由实际资产传入）：

```sh
python -m kt2028.prepare_regions --samples /path/to/samples.json \
  --config configs/regions_p1_trial.json --output /path/to/region_assets \
  --range-cache /path/to/http_cache --device cuda --batch 32
```

输入 samples 是 `slide_id/source(path,size)/fine_path` 数组，不需要结局标签。输出 NPZ 包括 coarse、几何、fine CSR 索引和有效比例，fine 原始数值仍从原始 Zarr 按需读取。私有病例级 receipt、图片和数组放数据目录，不提交 GitHub。完成输出可校验后续跑；不覆盖失败/不同配置的产物。Range 缓存和读取设字节/内存上限，遇到源大小变化拒绝使用；官方数据源之外不使用未知镜像。

`RegionCase.visible_state()` 初始无任何 fine；`acquire(i)` 才读取相应 UNI 行并记实际新增 patch 数，拒绝重复和空获取。标签不属于该对象。导入的完整 fine 坐标索引用于缓存模拟，前处理成本不能省略，cached_patch_counts 不代表真正在线提取已完成。N_ref/预算与 λ 待实验协议登记。

GitHub 同步：服务器 origin 指向正式仓库，使用 `research/p1-restart`。国内 GitHub 直连失败时通过本机 Git 认证与 bundle 中转：本机推送已审查的提交，再将包含相同提交的 bundle 传到服务器；服务器使用 bundle 的 ref 进行 fast-forward。服务器的重要改动可提交后导出 bundle，由本机读取并推送。避免把认证令牌放到代码/服务器配置中；禁止 force push/覆盖未提交修改。阶段结束核对本机、服务器和 GitHub 的 commit，落盘记录差异。

慢速外网接入：官方 `/info` API 提供直接 HTTPS 文件入口，`prefetch_rgb_cache` 只获取选定低倍率层的 ZIP 条目并检查 CRC。`relay_cohort` 在本机分段下载后传至服务器，成功传输才写 `.relay_ready`；`prepare_regions --wait-for-relay` 只处理 ready 的病例并禁止网络回退。服务器将所选数组的原始编码分片暂存到本地 Zarr，再交由原生 codec 解码，避免每个内层 chunk 重读整块 ZIP。完整原图不下载；低倍率缓存仍占空间并产生传输/前处理成本。中转任务依赖本机联网，进度和 PID 在工作日志/资产目录，不是定时任务。

更换低倍率层或数据来源时重新核准缓存就绪标记和最终中转回执，不将旧标记视为新层已下载。缺片/超出字节或内存上限时保留失败，不默默降倍率、改变队列或改组织阈值。
