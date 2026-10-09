# P1 当前数据接口

唯一现行方法见 `docs/experiment_restart.md` 第 0 节。本包与 `msrlsurv/` 早期草稿分开，当前只有真实数据预处理和部分获取接口，未实现/训练 P1 预后网络或 PPO。

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
