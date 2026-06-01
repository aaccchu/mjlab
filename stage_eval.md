# 阶段性成果记录 (stage_eval)

本文件记录足球场 / G1 带球任务每次 commit 时的阶段性成果。
每条目固定包含五部分:**干了什么 / 怎么执行 / 验证评价 / 可视化效果评估 / 重要文件路径**。

可视化产物统一存放在仓库内 `soccer_eval/<日期>_<阶段名>/`(`logs/`、`videos/`
被 .gitignore 忽略,因此评估用的视频/截图需手动拷贝到 `soccer_eval/` 才会进版本库)。

---

## 阶段 3:G1 端到端带球到目标点(3000 迭代训练已完成,2026-05-31,对应未来 commit)

### 干了什么
- 在足球场任务上新增足球(FIFA 5 号 / RoboCup MSL:半径 0.11m、质量 0.43kg、
  橙色、friction (0.5,0.02,0.01)、solref (0.02,1)、solimp 5 元组
  (0.9,0.95,0.001,0.5,2.0)),作为带 freejoint 的独立实体加入
  `scene.entities["ball"]`。
- 新建 `DribbleCommand`:同时(a)拥有球/目标状态并在 reset 重置球位,
  (b)`command` 属性返回派生的 base-frame twist [vx,vy,wz],让现有所有步态奖励
  (track_*/variable_posture/feet_*)零改动复用。派生 twist 只是目标编码,不作为
  控制施加 —— 仍是端到端单一策略从零学走路+走向球+带球到目标。
- 奖励:dribble_approach + dribble_to_target(主导)+ dribble_ball_velocity +
  dribble_success,叠加在继承的步态奖励之上。
- reset 时机坑:`_resample_command` 在 `sim.forward()` 之前跑,`root_link_pos_w`
  (读 xpos)是陈旧的,改为直接读 freejoint qpos 放置球。

### 怎么执行
```sh
# 训练(后台)。RTX 3090 24GB,4096 env 仅占用 ~6.5GB,显存富余,可上调 env 数提速。
MUJOCO_GL=egl uv run train Mjlab-Soccer-Unitree-G1 \
  --env.scene.num-envs 4096 --agent.max-iterations 3000

# 渲染可视化(单 env 才是贴身侧视;多 env 会拉远成俯视看不清)
MUJOCO_GL=egl uv run play Mjlab-Soccer-Unitree-G1 \
  --agent trained --checkpoint-file <logs/.../model_XXXX.pt> \
  --num-envs 1 --video True --video-length 400 \
  --video-height 480 --video-width 640

# 提交前必跑
make check   # ruff format + lint + ty + pyright,行宽 88,uv run
```

### 验证评价
- 框架健康检查:任务注册成功;config 接线正确(entities=robot+ball、只有 dribble
  命令、无 twist 残留、步态奖励全部重指向、obs 维度 actor 108 / critic 120);
  运行时 100 步无 NaN;球放置正确(z=0.11,离机器人 0.6–1.5m 不穿模);48 测试通过;
  make check 通过。
- 训练健康(2026-05-31 run `2026-05-31_21-19-24`,**3000/3000 迭代已跑完**):
  - 进度:中途 1408 iter 抽查 Mean reward 54–56,2401 iter 时 episode_success 升到
    **0.13**(峰值);末段(model_2999.pt 评估快照)Mean reward 49.6、
    episode_success 0.065、fell_over 0.125、robot_to_ball_error 2.06
  - 指标随随机配置波动(成功 episode 重置到新随机远目标会拉高 ball_to_target_error、
    降低瞬时 success);整体趋势:走路+平衡彻底学会(fell_over 从 450→个位数),
    带球能力出现并提升,但仍处早期,未完全收敛
  - 迭代速度:~1.39s/iter,总训练约 70 分钟
  - **下一步建议:** 显存富余(详见文末 GPU 备注),下一轮可用 8192 env 提速 +
    更多迭代(或调 dribble 奖励权重)以推动带球收敛

### 可视化效果评估
- **最终 model_2999.pt 单 env 渲染(640×480,400 帧):** 绿色球场上 G1 人形稳定站立/
  行走,橙球在身前,机器人朝球走过去并带球推进,行为与 DribbleCommand 的
  approach→push 设计一致。抽帧确认机器人+球都清晰可见,步态自然不摔。
- 中途 model_1300.pt 快照(~43% 训练)行为类似,带球处于更早期。
- **泛化/鲁棒性测试(改变初始条件,2026-05-31):** 临时把球的生成距离改远
  (spawn_dist_range 0.6–1.5m → 2.5–4.0m)、目标改远(target_dist_range
  2.0–6.0m → 6.0–9.0m),用同一个 model_2999.pt 渲染。抽帧确认:即使球离机器人
  更远、目标更远(比训练分布更难),G1 仍能稳定站立/行走并朝远处的球走过去,
  没有摔倒或穿模——说明策略对初始条件有一定泛化,不是只记住了训练时的固定距离。
  测试用的 env_cfgs.py 改动已**还原**,仓库配置保持训练默认值。
- **坑记录:** `--num-envs >1` 时离屏渲染器会拉远镜头框住相邻环境 → 俯视看不清
  机器人/球。渲染评估视频必须 `--num-envs 1`。改初始条件只需临时改
  `DribbleCommandCfg` 的 spawn_dist_range / target_dist_range,测完还原即可。

### 重要文件路径
- 球+球场几何:[src/mjlab/terrains/soccer_field.py](src/mjlab/terrains/soccer_field.py)
  (`SoccerBallCfg` / `get_soccer_ball_spec` / `build_soccer_field`)
- 带球命令:[src/mjlab/tasks/velocity/mdp/dribble_command.py](src/mjlab/tasks/velocity/mdp/dribble_command.py)
- 奖励:[src/mjlab/tasks/velocity/mdp/rewards.py](src/mjlab/tasks/velocity/mdp/rewards.py)
  (`dribble_approach` / `dribble_ball_to_target` / `dribble_ball_velocity_to_target` / `dribble_success_bonus`)
- 观测:[src/mjlab/tasks/velocity/mdp/observations.py](src/mjlab/tasks/velocity/mdp/observations.py)
  (`robot_to_ball` / `ball_to_target` / `ball_velocity_b`)
- 环境配置:[src/mjlab/tasks/velocity/config/g1/env_cfgs.py](src/mjlab/tasks/velocity/config/g1/env_cfgs.py)
  (`unitree_g1_soccer_env_cfg`)
- RL 超参:[src/mjlab/tasks/velocity/config/g1/rl_cfg.py](src/mjlab/tasks/velocity/config/g1/rl_cfg.py)
  (`num_steps_per_env=24`)
- 计划文档:[soccer_field_plan.md](soccer_field_plan.md)
- 可视化产物:全部汇总在
  [soccer_eval/2026-05-31_dribble_soccer-robot_v1_final/](soccer_eval/2026-05-31_dribble_soccer-robot_v1_final/)
  下,按验证着重点分子文件夹(详见该目录 README.md):
  - **最终标准评估(model_2999.pt,3000 迭代,默认初始条件):**
    [02_final_default_model2999/](soccer_eval/2026-05-31_dribble_soccer-robot_v1_final/02_final_default_model2999/)
  - 中途快照(model_1300.pt,~43%,默认初始条件):
    [01_midtrain_default_model1300/](soccer_eval/2026-05-31_dribble_soccer-robot_v1_final/01_midtrain_default_model1300/)
  - 改初始条件鲁棒性测试(球/目标更远,model_2999.pt):
    [03_robustness_farball_model2999/](soccer_eval/2026-05-31_dribble_soccer-robot_v1_final/03_robustness_farball_model2999/)

---

## 阶段 2:足球场平台 + G1 在场内移动 (commit be89bbd9 / 2f5af962)

### 干了什么
- 基于 `unitree_g1_flat_env_cfg`(plane 地面)+ `SceneCfg.spec_fn` 回调,按中心坐标
  注入足球场几何(22×14m 外边、线宽 0.125m、完整球场线:外框/中线/中圈/禁区/
  球门区,中圈与角球弧用多段短 box 拼接近似)。
- 软边界:越界即截断 episode 并重置(`out_of_field_bounds` 终止条件),靠奖励
  学会留在场内。无物理挡板。球门保留碰撞体(立柱/横梁用 cylinder)。
- 注册任务 `Mjlab-Soccer-Unitree-G1`。

### 怎么执行
```sh
uv run train Mjlab-Soccer-Unitree-G1 --env.scene.num-envs 4096
make check
```

### 验证评价
- 任务注册成功、make check 通过、48 测试通过。
- 训练时机器人能在场地范围内移动,越界正确触发重置。

### 可视化效果评估
- egl 渲染:橙球在绿场上,踢一脚滚约 3.46m(阶段 3 加球后验证)。
- (阶段 2 当时主要验证场地线与软边界,详见 git 历史。)

### 重要文件路径
- 场地:[src/mjlab/terrains/soccer_field.py](src/mjlab/terrains/soccer_field.py)
- 终止条件:[src/mjlab/tasks/velocity/mdp/terminations.py](src/mjlab/tasks/velocity/mdp/terminations.py)
  (`out_of_field_bounds`)
- 计划文档:[soccer_field_plan.md](soccer_field_plan.md)
