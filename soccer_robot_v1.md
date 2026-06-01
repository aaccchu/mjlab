# Soccer Robot v1 —— 实现总结(walk & kick)

> 对应 git commit **`2f5af962` "v1-walk and kick"**(及其前置 `be89bbd9`、`50e4c399`)。
> 基线为上游 commit **`5a433e83`**(Remove stale commented-out foot collision geoms)。
> 本文档描述的是 **v1 相对最初基线做了什么、怎么改、加减了什么、哪些模块客观可验证有效**。
> 所有数据来自 `git diff 5a433e83..2f5af962`,非凭记忆。

---

## 1. 一句话总览

在 mjlab 速度跟踪框架上,**新建了足球场虚拟平台 + 足球实体 + 端到端带球到目标点的 RL 任务**
(`Mjlab-Soccer-Unitree-G1`),让 Unitree G1 人形机器人从零学会"走路 + 走向球 + 把球带到
随机目标点"。核心设计是用一个 `DribbleCommand` 把带球目标编码成 base-frame twist,从而
**零改动复用全部现有步态奖励**,保持单一策略端到端训练。

---

## 2. 改动规模(客观 git 数据)

`git diff --stat 5a433e83..2f5af962`:**17 个文件,+1420 行 / -4 行**。

按三次提交拆分:

| commit | 主题 | 改动 | 关键内容 |
|--------|------|------|----------|
| `50e4c399` demo通 | 跑通 demo | 2 文件 +6/-2 | README + demo.py 微调 |
| `be89bbd9` zuqiuchangjustbuilt | **建球场** | 10 文件 +907 | 球场几何、终止条件、任务注册、场地文档/XML/示意图 |
| `2f5af962` v1-walk and kick | **加带球** | 7 文件 +512/-7 | DribbleCommand、dribble 奖励/观测、足球实体、changelog |

> 注:这是纯新增工作(几乎只加不删,-4 行仅为 README/demo 的少量替换),
> 未改动 mjlab 核心框架代码 —— 全部通过现有扩展点(spec_fn 回调、CommandTerm、
> RewardTerm、ObservationTerm、TerminationTerm)接入。

---

## 3. 加了什么(逐模块)

### 3.1 场地平台(commit be89bbd9)
- **`src/mjlab/terrains/soccer_field.py`(+344 → v1 共 390 行)**
  - `build_soccer_field(spec, cfg)`:按中心坐标系直接注入足球场几何 —— 22×14m 外边、
    线宽 0.125m、完整球场线(外框/中线/中圈/两侧禁区/球门区),中圈与角球弧用多段短 box
    拼接近似(MuJoCo 无圆环 primitive),线无碰撞。
  - 走"路径 B":基于现有 `unitree_g1_flat_env_cfg`(plane 地面)+ `SceneCfg.spec_fn`
    回调注入,**不走 terrain generator 框架**(规避网格偏移坐标坑)。
  - 球门保留碰撞体(立柱/横梁用 cylinder);外围挡板不做(纯软边界)。
- **`src/mjlab/tasks/velocity/mdp/terminations.py`(+21)**
  - `out_of_field_bounds`:软边界 —— 机器人越出场地范围即截断 episode 并重置,
    靠奖励学会留在场内。
- **`src/mjlab/tasks/velocity/config/g1/__init__.py`(+9)**:注册任务 `Mjlab-Soccer-Unitree-G1`。
- **文档/资产**:`docs/soccer_field/`(dimensions.md 尺寸表、soccer_field.xml、
  示意图 schematic/persp/topdown PNG)、根目录 `soccer_field_plan.md` 计划文档。

### 3.2 足球 + 带球(commit 2f5af962)
- **`src/mjlab/terrains/soccer_field.py`(+46)**
  - `SoccerBallCfg` + `get_soccer_ball_spec(cfg)`:FIFA 5 号 / RoboCup MSL 球 ——
    半径 0.11m、质量 0.43kg、橙色、friction (0.5,0.02,0.01)、solref (0.02,1)、
    **solimp 5 元组 (0.9,0.95,0.001,0.5,2.0)**(必须 5 值,否则 MuJoCo 报错)。
    作为带 freejoint 的独立实体加入 `scene.entities["ball"]`,不画进场地 worldbody。
- **`src/mjlab/tasks/velocity/mdp/dribble_command.py`(+257,全新文件)**
  - `DribbleCommand(CommandTerm)`:核心创新。同时
    (a)拥有球/目标状态并在 reset 重置球位,
    (b)`command` 属性返回派生的 base-frame twist `[vx, vy, wz]`。
  - 派生 twist 只是**目标编码**(approach 阶段瞄球后方、push 阶段穿球推向目标),
    **不作为控制施加** —— 策略仍从零学每个关节力矩,保持端到端。
  - reset 坑:`_resample_command` 在 `sim.forward()` 之前跑,`root_link_pos_w`(读 xpos)
    陈旧,改为直接读 freejoint qpos 放置球;`_update_command` 每步在 forward 之后跑,
    xpos 派生量新鲜可用。
- **`src/mjlab/tasks/velocity/mdp/rewards.py`(+59)**:四个带球奖励 ——
  `dribble_approach`(靠近球)、`dribble_ball_to_target`(球到目标,主导)、
  `dribble_ball_velocity_to_target`(球速投影到球→目标方向)、`dribble_success_bonus`(成功奖励)。
- **`src/mjlab/tasks/velocity/mdp/observations.py`(+48)**:三个带球观测 ——
  `robot_to_ball`、`ball_to_target`、`ball_velocity_b`(均转到 base frame,各 (B,3))。
- **`src/mjlab/tasks/velocity/config/g1/env_cfgs.py`(+99)**:`unitree_g1_soccer_env_cfg` ——
  注入球实体、把 commands 换成单一 `dribble`、所有步态奖励 command_name 从 "twist"
  重指向 "dribble"、追加 dribble 奖励/观测、保留 out_of_field_bounds 终止。
- **`src/mjlab/tasks/velocity/mdp/__init__.py`(+1)**:导出 dribble_command。
- **`docs/source/changelog.rst`(+9)**:Added 条目记录球 + DribbleCommand。

---

## 4. 减了什么 / 没做什么

- **几乎没删**:diff 仅 -4 行(README/demo 的少量文本替换),无功能删除。
- **刻意不做(v1 范围裁剪)**:
  - 不加物理挡板(纯软边界,靠 reward+termination 约束)。
  - 不走 terrain generator(避免坐标偏移坑)。
  - 不做射门 —— v1 目标是"带球到随机目标点"(盘球),非射门;球门只保留碰撞体几何。
  - 不改 mjlab 核心(entity/data.py 等),全靠公开扩展点。

---

## 5. 哪些模块客观可验证有效

下列均为**本次实际跑过命令**核对,非推断:

| 验证项 | 方法 | 结果 |
|--------|------|------|
| 任务真实注册 | `uv run list-envs` | `Mjlab-Soccer-Unitree-G1` 出现(列表第 7 项)✅ |
| 球场/球几何函数存在 | grep soccer_field.py | `build_soccer_field` / `SoccerBallCfg` / `get_soccer_ball_spec` 均在 ✅ |
| 带球奖励存在 | grep rewards.py | 4 个 `dribble_*` 函数均在(L474/489/502/517)✅ |
| 带球观测存在 | grep observations.py | `robot_to_ball`/`ball_to_target`/`ball_velocity_b` 均在(L61/73/85)✅ |
| 软边界终止存在 | grep terminations.py | `out_of_field_bounds`(L71)✅ |
| 框架健康 | 之前 make check + 48 测试 | 通过;obs 维度 actor 108 / critic 120 ✅ |
| 运行无 NaN | 100 步 smoke test | 无 NaN;球放置 z=0.11、离机器人 0.6–1.5m 不穿模 ✅ |
| **训练有效** | 3000 迭代完整训练 | fell_over 450→个位数(走路+平衡学会);episode_success 0→峰值 0.13(带球出信号)✅ |
| **可视化有效** | model_2999.pt 单 env 渲染抽帧 | G1 在绿场稳定站立/行走,朝橙球走过去并带球推进,步态自然不摔 ✅ |

**结论:走路 + 平衡彻底验证有效(fell_over 趋零);带球能力已客观出现并提升,
但 v1 训练未完全收敛(success 仍偏低、随机目标波动)——属"功能跑通、行为初现"阶段。**

---

## 6. 关键文件路径

- 球场+球几何:[src/mjlab/terrains/soccer_field.py](src/mjlab/terrains/soccer_field.py)
- 带球命令(核心):[src/mjlab/tasks/velocity/mdp/dribble_command.py](src/mjlab/tasks/velocity/mdp/dribble_command.py)
- 带球奖励:[src/mjlab/tasks/velocity/mdp/rewards.py](src/mjlab/tasks/velocity/mdp/rewards.py)
- 带球观测:[src/mjlab/tasks/velocity/mdp/observations.py](src/mjlab/tasks/velocity/mdp/observations.py)
- 软边界终止:[src/mjlab/tasks/velocity/mdp/terminations.py](src/mjlab/tasks/velocity/mdp/terminations.py)
- 环境配置:[src/mjlab/tasks/velocity/config/g1/env_cfgs.py](src/mjlab/tasks/velocity/config/g1/env_cfgs.py)
- 任务注册:[src/mjlab/tasks/velocity/config/g1/__init__.py](src/mjlab/tasks/velocity/config/g1/__init__.py)
- 计划文档:[soccer_field_plan.md](soccer_field_plan.md)
- 场地尺寸/资产:[docs/soccer_field/](docs/soccer_field/)
- 阶段成果记录:[stage_eval.md](stage_eval.md)
- v1 训练 run:`logs/rsl_rl/g1_velocity/2026-05-31_21-19-24/`(model_2999.pt)
- v1 可视化:[soccer_eval/2026-05-31_dribble_final/](soccer_eval/2026-05-31_dribble_final/)

---

## 7. 复现命令

```sh
# 训练(后台,RTX 3090 24GB,4096 env 占用 ~6.5GB)
MUJOCO_GL=egl uv run train Mjlab-Soccer-Unitree-G1 \
  --env.scene.num-envs 4096 --agent.max-iterations 3000

# 渲染可视化(必须 --num-envs 1,否则离屏渲染拉远成俯视看不清)
MUJOCO_GL=egl uv run play Mjlab-Soccer-Unitree-G1 \
  --agent trained --checkpoint-file logs/.../model_2999.pt \
  --num-envs 1 --video True --video-length 400 \
  --video-height 480 --video-width 640

# 提交前必跑(行宽 88,uv run)
make check
```
