# TGOD-SD：UR5e 单示范模仿学习

本项目依据论文《基于引导多样性的护理机器人模仿学习》（DOI：`10.13973/j.cnki.robot.240269`）实现 TGOD-SD，并使用已有 MuJoCo UR5e 场景和单条专家示范。任务是把白杯从红垫搬到蓝垫。当前以复现论文公开的算法为主，未公开的细节作为实现假设单独说明。

## 论文对应关系与实现边界

1. 每回合从固定均匀先验采样技能 `z`，整条轨迹保持该技能。SAC 的 actor、双 Q 以状态和 `z` 为输入，保留条件动作熵目标。
2. MINE 损失固定为两个 DV 下界的负和，SAC 伪奖励固定为两个现有 MINE 逐样本项的等权和。环境奖励恒为 0；不再叠加示范接近度或进度奖励，不对伪奖励做运行均值归一化或裁剪。
3. 训练执行预先设定的 episode 预算。成功、抓取、抬杯、放置、末态距离仅用于日志，不参与奖励、检查点选择或早停。
4. 训练结束后生成候选，按全部候选中的最小 Sinkhorn divergence 选轨迹。成功标记不用于预先过滤候选。

**这仍不能称为原论文的完全等价实现。** 论文没有给出单条示范下 `I(Z;D)` 的联合采样与编码定义。如果把完整示范视作恒定随机变量，该互信息退化为零；这不说明论文作者实际采用了这种定义。现有代码保留 `I(Z;f(S,D_t,t))` 的关系特征代理，包含当前状态、时间对齐的专家状态、差值、接近度与相位。接近度仍是 MINE 的一个输入，但已不再形成独立奖励。此代理与论文第二项是否等价，尚需作者代码或补充说明验证；本轮不另行设计替代算法。

以下既有实现假设也保留并披露：

- 离散 one-hot 技能、批内打乱技能构造 MINE 负样本、网络结构、自动温度、优化器与梯度裁剪参数均为项目设置，未核实为论文原始设置。
- 动作为三维末端位移；环境沿用运动学控制及自动抓取/释放。该单臂仿真任务与论文双臂护理实验存在差异。
- SD 使用经专家统计量缩放的关节角、TCP、杯位置和时间特征、均匀 OT 质量及平方欧氏代价；提前成功轨迹用末态补齐到 episode 时间范围。这些预处理并非论文公开的完整规范；时间特征也不等于严格保序 OT。
- 论文约“2000轮次”不能严格换算成 2000 episodes。本项目将 2000 episodes、每回合最多 500 步作为明确的训练预算假设。

## 本轮修改说明

| 文件 | 修改及原因 |
| --- | --- |
| `tgod_sd/agent.py` | 两个 MINE 损失/奖励项固定等权；删除 support/progress 奖励及奖励归一化、裁剪执行路径。保留 SAC、双 Q、温度、目标网络、MINE 负采样与梯度裁剪实现。旧奖励统计仅保留检查点读取兼容。 |
| `tgod_sd/expert.py` | 仅补充关系特征是论文未公开细节的代理假设的注释，计算方式不变。 |
| `tgod_sd/trainer.py` | 删除按任务指标比较续训检查点、最佳模型、成功专属快照、周期性任务评估、早停和预算外 replay 预填充。只按指定回合数训练和固定间隔保存；保留阶段日志、完整 replay/RNG、配置与来源记录。 |
| `train.py` | 删除 `--resume-candidates`；保留指定检查点 `--resume`、总预算 `--episodes` 和追加预算 `--additional-episodes`。 |
| `tgod_sd/trajectory.py` | 删除成功优先筛选分支，始终对全部候选取最小 SD；保留原 SD 数值求解与特征处理。 |
| `evaluate.py`、`tgod_sd/evaluation.py` | 删除成功优先 CLI 和任务排名函数；独立评估保留配置恢复与统计，输出明确的全候选最小 SD 规则。 |
| `tgod_sd/config.py` | 新增训练专用复现约束，拒绝额外奖励、非等权 MI、奖励归一化/裁剪及已撤销的任务驱动训练选项。历史配置仍能加载用于只读评估。 |
| `configs/ur5e_pick_place.yaml` | 默认配置对齐两项等权奖励与全候选最小 SD，其余既有数值尽量保留。 |
| `configs/paper_seed45.yaml` | 新的独立复现配置；保留 seed45 实验的环境尺度 `0.015`、8 技能、500 步上限及原 SAC/MINE 学习率 `3e-4/5e-5`，撤销上一轮的 SAC 降学习率方案。 |
| `configs/retrain_seed42.yaml`、`configs/finetune_seed45.yaml` | 数值保留作历史实验记录，标注不可用于新的复现训练。 |
| 测试文件 | 核对实际 TD 目标、固定预算、续训状态和候选选择规则，替换先前针对早停/选优的测试。 |

上一轮加入的 `checkpoint_config.py`、replay 数据校验、实际学习率恢复、配置/资源哈希和独立评估日志继续保留，用于核对执行情况。旧的 `outputs/analysis_20260907` 报告记录的是当时的工程调参建议；当前训练方案以本说明为准。

## 资产与验证

YAML 的路径相对项目根目录解析：

```text
universal_robots_ur5e/scene.xml
data/similar_expert/expert_demo.npy
data/similar_expert/expert_qpos.npy
data/similar_expert/expert_cup.npy
data/similar_expert/expert_initial_state.npz
```

专家数组形状分别为 `(T,12)`、`(T,6)`、`(T,3)`，加载时检查长度、有限值和初始状态。需要 Python 3.10 或更高版本：

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python -m unittest discover -s tests -v
```

## 新的训练与续训

已有 `motion_only_seed45` 检查点使用了旧奖励目标，不能接到新目标后称为同一复现实验。因此从头运行，并使用独立目录：

```bash
python -u train.py --config configs/paper_seed45.yaml --episodes 2000 --no-match
```

输出默认写到 `outputs/paper_seed45/`。2000 是本次预先声明的总预算，日志可用于检查有限值、损失与行为，不按成功率切换最终模型。若采用 800 episodes 的预算，在开始前将命令的 `2000` 改为 `800`，并记录预算；不能据此认定训练已经或必然收敛。

- `metrics.jsonl`：每回合阶段诊断、技能、累计步数、replay 大小及更新批次的 MI/损失均值。更新统计来自历史 replay，不代表该回合专属回报。
- `checkpoints/latest.pt`：最近保存的完整回合边界状态，包括策略、Q、MINE、优化器、有效 replay 及随机数状态。
- `checkpoints/episode_*.pt`：按固定间隔保存的轻量模型，可做事后评估，不用于连续续训。
- `config.resolved.yaml`、`run_manifest.json`：实际配置、学习率、预算、协议标记以及代码/资产来源。

新协议训练产生的完整 `latest.pt` 可以继续同一实验。下面的 `--episodes` 是包含已完成回合的**总上限**：

```bash
python -u train.py --config configs/paper_seed45.yaml --resume outputs/paper_seed45/checkpoints/latest.pt --episodes 2000 --no-match
```

`--additional-episodes N` 追加 N 回合，与 `--episodes` 互斥。续训要求 seed、环境、网络、TGOD/SAC 参数和资产一致，并恢复完整 replay 与采样状态。旧协议、缺少 replay、损坏 replay 或训练参数变化会明确报错，不进行额外预填充或静默改变训练目标。

中途 Ctrl+C 不把半个回合的状态覆盖到 `latest.pt`。默认最多需要重跑最近固定保存点以后的回合；若输出中已有更晚回合日志，应通过 `--output-dir` 指定新的续训目录，保留原记录。不同设备/软件环境仍可能产生数值差异，状态保存不保证跨平台逐位一致。

## 独立评估与回放

使用固定预算结束时的 `latest.pt`，另设评估 seed 生成 160 条候选：

```bash
python -u evaluate.py --checkpoint outputs/paper_seed45/checkpoints/latest.pt --seed 46 --candidate-count 160 --output-dir outputs/paper_seed45/eval_seed46
```

评估默认读取检查点内的环境、网络、训练 seed 和匹配设置；CLI `--seed` 在其后覆盖。可用 `--config` 提供资源路径和匹配参数，但环境、网络、技能数冲突会报错，避免动作尺度误用。

选择规则始终为**全部候选的最小 SD**。即使历史检查点保存 `prefer_successful=true`，当前评估也会覆盖为 false，并在 `evaluation_manifest.json` 和 `candidate_scores.json` 记录旧选项已忽略；这属于对旧策略采用当前选择协议的事后评估。所选轨迹可能失败，因此同时报告成功率和 SD 求解收敛标记，不能把最小 SD 当作任务成功或把未收敛的近似值当作可靠排名。

新配置保留旧实验的 `epsilon=0.1`、`max_iterations=2000`、`tolerance=1e-5`，没有引入新的 SD 公式。后续求解精度调整应单独记录；本次短测试不证明大规模候选全部收敛。

评估输出包括 `candidates/`、`candidate_scores.json`、`selected_trajectory.npz`、实际配置、来源及分技能诊断。回放命令：

```bash
python replay.py outputs/paper_seed45/eval_seed46/selected_trajectory.npz
```

加入 `--render --realtime` 可打开 MuJoCo 窗口。回放沿用已保存的关节角和杯轨迹；本项目仍是仿真研究实现。
