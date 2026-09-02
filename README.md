# TGOD-SD：UR5e 单示范模仿学习

本项目根据论文《基于引导多样性的护理机器人模仿学习》（DOI: `10.13973/j.cnki.robot.240269`）实现一个可运行的 TGOD-SD 工程，并复用相邻 `SAC_ur5e` 项目中的 MuJoCo UR5e 场景和单条专家示范。任务是让 UR5e 将白色杯子从红色垫子拿起并放到蓝色垫子。

## 算法对应关系

论文中的 TGOD-SD 是两阶段方法：

1. TGOD：技能变量 `z` 条件化 SAC 策略；SAC 负责最大化条件动作熵。论文把内部伪奖励写成 `MINE(Z;S)+MINE(Z;D)`，环境不提供人工任务奖励。
2. SD：训练结束后生成多条完整候选轨迹，计算候选与专家示范之间的熵正则最优传输（Sinkhorn）距离，输出距离最小的轨迹。

论文没有公开网络层数、技能数、MINE 正负样本构造、SAC 超参数、Sinkhorn 正则系数以及单条固定示范下 `I(Z;D)` 的可计算定义。本实现将这些缺失项集中放在 `configs/ur5e_pick_place.yaml`，并采用以下明确的工程补全：

- 使用离散 one-hot 技能变量，每个回合采样一次；
- 使用 Donsker–Varadhan MINE 下界和批内打乱技能构造乘积分布样本；
- 单条固定示范在数学上会使 `I(Z;D)=0`。代码因此把第二项明确实现为 `I(Z;f(S,D_t,t))` 的关系互信息代理，并加入通用的 `log(proximity)` 示范支持先验，使该项确实偏好靠近按时间对齐的专家状态；它是示范派生的内在信号，不是红垫/蓝垫等任务奖励，权重可设为 0 做字面公式消融；
- SAC 的 actor、双 Q 网络均以 `[observation, z]` 为输入；
- Sinkhorn 特征默认包含关节角、TCP 位置、杯子位置和归一化时间，并按专家统计量归一化；时间特征只能缓解反向经过相同空间点的轨迹误判，并非严格保序 OT；
- 成功候选会在 SD 评分前用末态补齐到 500 帧，以保留专家轨迹的真实时间轴；
- 默认按论文描述对全部候选取最小 SD；`prefer_successful=true` 是可选的任务安全扩展，且成功必须经历抓取、抬杯和释放；
- reward normalization/clip、自动温度、超时是否 bootstrap、均匀 OT 权重、平方欧氏 ground cost、随机候选采样等也都是配置中公开的工程选择；回放池中的历史转移按当前 MINE 重算伪奖励，而不是冻结采集时分数。

这些选择是可运行复现所需的补全，不冒充论文原始参数。

## 资产

默认直接读取相邻项目中的原始文件，不复制或修改它们：

```text
../SAC_ur5e/universal_robots_ur5e/scene.xml
../SAC_ur5e/data/similar_expert/expert_demo.npy
../SAC_ur5e/data/similar_expert/expert_qpos.npy
../SAC_ur5e/data/similar_expert/expert_cup.npy
../SAC_ur5e/data/similar_expert/expert_initial_state.npz
```

专家数据必须分别具有 `(T,12)`、`(T,6)`、`(T,3)` 的形状。加载器会检查长度、有限值和初始状态。

## 安装与验证

需要 Python 3.10 或更高版本。

```powershell
python -m pip install -r requirements.txt
python smoke_test.py
python -m unittest discover -s tests -v
```

`smoke_test.py` 会验证场景、专家数据、环境单步、MINE/SAC 一次参数更新以及 Sinkhorn 距离，不启动长时间训练。

## 训练

论文只展示约 2000 个含义并不完全明确的“轮次”后伪奖励收敛。默认配置把它映射为 2000 个 episode（每个最多 500 步），这是复现假设，不是可由论文核实的原始超参数：

```powershell
python train.py --config configs/ur5e_pick_place.yaml
```

先做短流程检查：

```powershell
python train.py --episodes 2 --candidate-count 2 --device cpu
```

训练结果写入 `outputs/ur5e_pick_place/`：

- `metrics.jsonl`：每回合任务成功、伪奖励和各网络损失；
- `checkpoints/latest.pt`：策略、双 Q、目标网络、MINE、优化器和配置；
- `candidates/candidate_*.npz`：训练后采样的完整候选轨迹；
- `candidate_scores.json`：每条候选的 Sinkhorn 距离和成功标记；
- `selected_trajectory.npz`：最终选中的轨迹。

训练可中断续跑：

```powershell
python train.py --resume outputs/ur5e_pick_place/checkpoints/latest.pt
```

检查点不包含体积很大的 replay buffer；恢复后会先重新填充回放池，再继续梯度更新。

## 单独生成候选并匹配

```powershell
python evaluate.py --checkpoint outputs/ur5e_pick_place/checkpoints/latest.pt
```

默认就是严格按论文描述对所有候选直接选最小 SD。若要先过滤成功候选，可把 `matching.prefer_successful` 改为 `true`，或在独立评估时加 `--prefer-successful`。

## 回放

无窗口检查最终位姿：

```powershell
python replay.py outputs/ur5e_pick_place/selected_trajectory.npz
```

打开 MuJoCo 窗口并按保存的 q-pos/杯子轨迹回放：

```powershell
python replay.py outputs/ur5e_pick_place/selected_trajectory.npz --render --realtime
```

## 代码结构

```text
tgod_sd/
  env.py             UR5e 白杯搬运环境，环境奖励恒为 0
  expert.py          单条专家示范加载、对齐与关系特征
  networks.py        条件 actor、双 Q 和 MINE
  replay_buffer.py   TGOD 回放池
  agent.py           MINE 伪奖励与 SAC 更新
  sinkhorn.py        数值稳定的熵正则最优传输距离
  trajectory.py      候选生成、SD 匹配和保存
  trainer.py         训练流程、日志和检查点
```

当前实现是仿真研究代码。白杯的抓取由真实接触或 1.5 cm 近接触容差触发，并在抓取期间进行运动学附着，因为提供的 Menagerie 场景没有真实夹爪自由度。实体机器人执行前还必须增加限位、碰撞监控和急停，不应直接发送这些 q-pos。
