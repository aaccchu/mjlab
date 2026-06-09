# Soccer Robot v4 — 自定位方法论与重构方案(EKF + 对称消歧 + 解析主动视觉)

> 2026-06-09 创建。本文件是 v4 自定位(线A)的**方法论参考 + 重构设计**,独立于实验流水账
> `soccer_robot_v4_experiment.md`。来源:RoboCup SPL/Humanoid League 几十年经验 + 近年 arXiv 调研,
> 结合 EXP6-11 的实测结论。**结论先行:当前"点云拼接+单次Kabsch"架构选错了,应转 EKF/UKF 递归滤波。**

---

## 一、问题定位:为什么 pos_err 卡在 1.5-2m

### 当前架构(EXP6-11)
单目 head_cam(挂 neck,neck_yaw ±90°)→ 检测23个已知3D坐标地标的像素 → depth抬升到世界坐标 →
与地图做 Kabsch(SE2刚体配准)→ 位姿。FusedPoseBelief 把 N 帧的点用里程计搬到当前帧 → 拼成大点云 →
**单次 Kabsch**(observations.py:572-596)。

### 三个根因(调研 + 实测交叉确认)
1. **点云拼接 ≠ 信息累积**(架构级,最重要)。机器人朝同方向走时,N 帧看到同 ~4 个点,拼接后约束
   不增加。EXP7 实测"唯一关键点 3.9→4.4"、EXP11 修了 append bug 后 pos_err 仍 ~2.1m,均印证:
   **这个架构有天花板,不是调参/修bug能突破的**。
2. **从未处理球场镜像对称**。对称场地下 Kabsch 会随机跳到镜像解,单这一项就贡献大误差。
3. **单帧欠定**。前向窄视角 4/23 点,Kabsch 数学上欠约束;主动扫视(EXP8-10)想解决它,
   但用 RL 间接学扫视教不动(EXP9 决定性负结论)。

---

## 二、RoboCup 标准答案(调研核心)

**"永远不要靠单帧/多帧几何配准做定位。用 EKF/UKF 递归滤波,跨帧累积稀疏观测。"**

### 方法1:EKF/UKF 递归滤波(最高优先,改动最确定)
- 维护 SE2 位姿分布 [x, y, yaw] + 协方差。**预测步**:用里程计(RL walk 的命令速度/IMU)推进位姿+膨胀协方差。
  **更新步**:每帧把检测到的**每个地标**作为独立观测,用观测模型(地标投影/世界坐标)更新分布——
  **哪怕只看到 1 个点也能更新**,信息天然跨帧累积。
- 这是**带已知地图的定位(非SLAM)**:观测模型干净(地标3D坐标已知),无需回环/边缘化。
- 算力 O(1)/步、常数内存,RoboCup 在 Nao 弱 CPU 实时跑。**比 factor graph/滑窗更适合本场景**(后者
  是无地图VIO/SLAM的过度工程)。
- 参考:B-Human `SelfLocator`/`FieldFeatures`(github.com/bhuman/BHumanCodeRelease,每年release+技术报告PDF);
  PythonRobotics UKF 定位实现(github.com/AtsushiSakai/PythonRobotics,Localization/unscented_kalman_filter)。

### 方法2:对称消歧(与方法1正交,可同时上)
- **CLAP**(Clustering to Localize Across n Possibilities,arXiv:2509.08495):对每观测生成 n 个对称变换
  假设,在位姿空间聚类——真位姿假设跨多帧/多点聚成一致簇,虚假对称解分散。几何方法、无需训练。
- 嫁接方式:对 Kabsch/EKF 输出的位姿,同时维护其镜像对称解为多假设,用跨帧一致性(或 EKF 多假设/
  粒子簇)淘汰错误解。
- 背景:Clustered Particle Filtering(AAAI);RoboCup Symmetry Operator(Springer 10.1007/978-3-540-25940-4_24)。

### 方法3:解析主动视觉(绕开 RL 教不动扫视的困境)
- **熵减 next-best-view**(Seekircher/Laue/Röfer,Springer 10.1007/978-3-642-20217-9_1):维护定位 belief 熵,
  选最大化**期望熵减**的头部角度。
- **关键:不必用 RL**。我有地标3D坐标+相机模型,可**解析预测**"neck 转到角度 θ 能看到哪些地标 →
  EKF 协方差/熵下降多少",贪心选最优 θ。比 EXP8-10 用 RL 间接学扫视直接得多。
- RL 版备选:Real-time Active Vision Using Deep RL(Bestmann/Bit-Bots,arXiv:2011.13851)。

### 方法4:多技能冲突(线B 收尾用,优先级在定位之后)
- **Teacher distillation**(DeepMind "Learning Agile Soccer Skills",arXiv:2304.13653):先训单技能专家
  (找球定位 / 踢球)再蒸馏成统一策略,避免互相覆盖。解 codex C8 射门退化 + 我 EXP10 neck甩头破稳的同源张力。
- 备选:phase-gating、residual policy(冻结基策略只学残差)。

---

## 三、重构方案(线A 下一步,我的成熟设计)

### 阶段 R1:EKF 替换点云拼接(核心,先做)
- 新建 `field_ekf.py`:SE2 EKF。状态 [x,y,yaw]+3x3协方差。
- **预测步**:用 robot.data 的 base 线/角速度(或命令速度)+ dt 推进;协方差加过程噪声 Q。
- **更新步**:对每个 vis>0 的地标,观测=该地标在 base 系的 depth-lifted xy(已有,_collect_frame 在算);
  观测模型 h(pose)=地标世界坐标变换到 base 系的预测 xy;雅可比解析可写;按观测噪声 R 更新。
- FusedPoseBelief 改为:每 __call__ 跑一次 EKF predict+update,输出 belief=[x_n,y_n,sin,cos,
  trace(协方差)→不确定度, vis_now]。**有状态滤波器替代点云缓冲**,per-env 维护,reset 时该 env 重置为
  宽先验(高协方差)。
- **判据**:对比 EXP11 基线(点云拼接 ~2.1m),EKF 版纯视觉 pos_err 是否显著下降、能否趋近 0.79m。

### 阶段 R2:对称消歧(R1 有效后叠加)
- EKF 维护 2 个假设(本位姿 + 镜像位姿),按观测似然/跨帧一致性给权重,输出主假设。
- 或粒子化:小粒子群覆盖对称解,重采样收敛。

### 阶段 R3:解析主动视觉(R1+R2 后)
- 不用 RL。每步用 EKF 协方差解析算各 neck_yaw 候选角的期望熵减,输出最优 neck 目标角作为
  **低层控制目标**(不是 RL action),叠加到现有步态。避免 EXP8-10 的 RL 扫视困境 + 稳定性张力。

### 工程纪律
- 每阶段保留旧版可回退;先 smoke(几何正确性:已知位姿→EKF 应收敛到真值)再训练。
- eval 借鉴 codex 指标集(scripts/summarize_v4_plain_log.py:goal_rate/fell_over/not_scored/selfloc/ball_path)。
- 线A(定位)与线B(踢球链)解耦推进;goal_rate 是线B 的命门,不指望线A 提升它。

---

## 四、关键链接(真实可访问)
- B-Human 代码:github.com/bhuman/BHumanCodeRelease
- PythonRobotics UKF:github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/unscented_kalman_filter/unscented_kalman_filter.py
- CLAP 对称消歧:arxiv.org/html/2509.08495 ;泛化版 arxiv.org/abs/2509.13605
- 迭代地标匹配定位:arxiv.org/html/2503.11020v2
- 熵减主动视觉:link.springer.com/chapter/10.1007/978-3-642-20217-9_1
- RL 主动视觉:arxiv.org/abs/2011.13851
- DeepMind 足球技能蒸馏:arxiv.org/abs/2304.13653
- 多技能干扰理论:huggingface.co/papers/2606.02398
- legged RL 综述:arxiv.org/html/2406.01152v2
