# W139：DSCA与当前R3的关键差别（2026-11-09）

本轮核读论文及官方公开main源码，仅解释与登记，不修改现有运行、网络、标签或划分。DSCA不是当前R3的已完成复现；以下差异不能当作已证实的失败原因。

## 已核事实

- [论文](https://arxiv.org/html/2206.05782)主攻多尺度计算成本与语义融合：5×父区域对应16个20×子块，粗引导cross-attention将16个细token压为1个，再做双流Transformer与区域级融合/GAP。square pooling压缩token数，不等于减少参数；论文报告模型6.31M，比当前R3的2.340615M和128版0.761351M更大。
- [官方WSIHierNet](https://github.com/liupei101/DSCA/blob/main/model/HierNet.py)在post模式分别编码两路区域token，然后区域级相加/拼接，最后统一GAP与离散hazard头。当前R3由粗Transformer后token查询变长有效细子块，含观察状态跨门控、两路分别池化和患者融合；不是同一融合次序。局部粗引导聚合已存在于R3，不能宣称再添加一次就是新解决方案。
- [公开配置](https://github.com/liupei101/DSCA/blob/main/config/config_hier.yaml)：1024-384-384-4、1层Transformer、dropout0.6、reg_l1=1e-5、weight_decay=5e-4、lr8e-5、累积16病例、alpha0.1、monitor loss；公开opt为lookahead_adam，论文正文写Adam，不能混为原样一致。
- [训练器](https://github.com/liupei101/DSCA/blob/main/model/model_handler.py)执行L1、验证Plateau10轮学习率减半、30轮早停；逐轮验证使用cur_alpha=0，默认按loss选best。当前W135/W136是dropout0.25/AdamW/WD1e-3/固定学习率/早停15、partial/full/coarse_aux/fine_aux四项加权。更大WD不等价于整套正则更强，也不能仅比较dropout数值保证有效。
- [时间编码](https://github.com/liupei101/DSCA/blob/main/dataset/WSIPatchData.py)按未删失患者时间四分位设4档，不是固定五年度；这使事件分配更均衡，但不增加事件，且改变输出日历语义。其公开代码先从标签表设边界再收集折患者；受控适配应只由训练折估计边界，不盲拷全表流程。
- [SurvMLE](https://github.com/liupei101/DSCA/blob/main/loss/loss_interface.py)：训练L=event项+(1-alpha)*censored项，alpha0.1使删失项乘0.9；没有直接C-index排序损失。更重要，删失loss使用S_padded[t+1]，将所在离散档纳入存活项；当前只计完整经历年度，首年内删失为零项。不能声称当前目标与DSCA alpha0逐人完全等价，也不能把直接加入当前档称为无假设的信息回收。
- 论文三队列分别NLST447、BRCA978、LGG486患者，五折且训练部分20%内部验证；报告C-index0.66803±0.05817、0.61224±0.05350、0.70181±0.04321。BRCA死亡比例13.5%，说明少事件是困难因素而非模型失效的充分解释。癌种、特征提取器、随访/离散化、划分均与当前不同，不能作同条件排名。主评价是排序与风险分层，不证明五年概率已校准或无过拟合。

## 对当前诊断的影响

参数量/事件稀疏仍是风险，不能据此解释全部性能差距。优先补官方DSCA主体的受控适配对照（同现有UNI、患者折与终点）以检验结构/训练差别，再分别消融区域融合、完整输入单目标、正则/调度；时间编码与删失处理应另登记，不能和结构同时替换后归因。现运行队列继续，未因本轮调研启动新训练或改监督。
