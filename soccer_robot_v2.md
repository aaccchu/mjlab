# Soccer Robot v2 — 踢球/带球物理建模迭代计划

> 状态:**规划文档,尚未执行**。当前 v1 训练(`2026-05-31_21-19-24`,3000 迭代)跑完后再启动 v2。
> 目标:让仿真"像真的比赛",并最终让机器人能**逐步控球、推进、射门,
> 把球踢进球门**。物理核心是**踢球冲量 + 球旋转 + 滚动摩擦**三者耦合,
> 以及**足部接触点/力道**和**机器人质量**对球运动的真实影响。

---

## 0.5 仿真时间分辨率与接触稳定性(必须先确认,放最前)

踢球是**短时高力接触**,MuJoCo 是离散时间求解,timestep 不对会让接触力严重失真,
后面所有标定都不可信。**先确认现状,再标定。**

**现状(已读 velocity_env_cfg.py:446):**
- `timestep = 0.005`(物理 200 Hz),`decimation = 4`(控制 50 Hz),
  `iterations=10`,`ls_iterations=20`,G1 软场 `ccd_iterations=500`、`nconmax=70`。
- 即一次控制步内物理积分 4 次。球半径 0.11m、球速 6 m/s 时,单物理步位移
  6×0.005=0.03m ≈ 0.27R,**接触不会被一步跨过**,基本够用。

**v2 要做的验证(不是拍脑袋改 dt):**
- **dt 收敛性测试**:同一踢球初速,分别用 `timestep ∈ {0.005, 0.0025, 0.002}`
  跑被动滚动,球速/滚动距离变化应 **< 10%**。若 0.005 与更细的差 >10%,
  说明 200 Hz 不够、要降 timestep(代价:训练变慢)。
- **接触求解收敛**:高力接触时若球抖动/穿透,提高 `iterations`/`ls_iterations`
  或 `ccd_iterations`,而不是盲调 solref。
- **注意 MuJoCo 没有 "substeps" 概念**(那是 Isaac/PhysX 的)。MuJoCo 用
  `timestep` + solver `iterations` 控制精度。参考文档的 "substeps≥5" 不适用。

**验收**:dt 减半后球速变化 <10% → 当前 200 Hz 可信,标定结果可用。

---

## 0.6 设计哲学(贯穿全程的红线,防止偷偷加"作弊物理")

1. **不用 fake physics**:不 `set ball velocity`、不 teleport、不 `apply_impulse`
   当作踢球、不 `enforce_rolling()` 强改角速度。
2. **所有行为来自接触动力学**:踢球的冲量、接触点、球速、旋转,全部由 MuJoCo
   求解器从"脚 capsule 撞球"算出来,我们只标定物理参数和设计奖励。
3. **标定优先于建模复杂度**:能用参数标定解决的,不引入额外约束/力场。
   只有当真实接触**确实无法**产生需要的行为时,才考虑显式机制(并在文档里写明理由)。

> 这条红线很重要:未来任何人(包括我们自己)想"加个力让球听话"时,
> 必须先证明真实接触做不到,否则就是退化成假物理。**§3.6 带球力场是唯一
> 可能触碰这条线的地方,见该节的取舍说明。**

---

## 0.7 关键路径 / MVP(先做这四步就能量化对比 v1)

> 下面列的完整迭代(v2.0~v2.8)是按技术深度排列的,但**不需要全做完才能证明进步**。
> 为了避免"2-3 周还没跑出一个可对比结果",先定义一条关键路径:

### v2-core(关键路径,1-2 天内拿到 D-vs-B 对比)

| 步骤 | 内容 | 改动量 | 依赖 |
| --- | --- | --- | --- |
| **v2.0** | 球 `condim=3→6` + `roll=0.01`(已实测 9m,一行改动) | 1 行 | — |
| **v2.2** | 足↔球 ContactSensor(found/force/pos) | ~30 行 cfg | v2.0 |
| **v2.3** | 接触/带球奖励 + 重训(8192 env,≥5000 iter) | ~60 行 | v2.2 |
| **§8 eval** | 固定 seed 评估脚本 + D-vs-B 对比 | 新脚本 | v2.3 |

完成这四步即可回答"**v2 是否比 v1 强**"(§8.6 验收门槛 1-5)。

### v2-extended(core 通过后再做,按价值排序)

| 优先级 | 内容 | 价值 |
| --- | --- | --- |
| P1 | v2.8 进球任务闭环(缺口 G) | 最终目标 |
| P1 | v2.1b 壳层惯量 + v2.1c 回弹标定 | 仿真保真度 |
| P2 | §3.5 踢球动力学标定曲线 | 证明"会踢"不是"蹭" |
| P2 | v2.5 MSL 合规指标 | 规则约束 |
| P3 | v2.6 domain randomization | sim-to-real |
| P3 | v2.7 观测 ablation | sim-to-real |
| P3 | §3.6 dribbler 路线 B | 只有 A 失败才做 |

> **原则:每完成一步都能独立验证,不用等全做完。core 不通过就不进 extended。**

---

## 0. 先纠正参考文档里的 5 个 MuJoCo 错误(很重要,避免走弯路)

你贴的物理方案大方向对(冲量+转动+滚动摩擦),但里面有几条是
**通用引擎/Isaac Gym 的写法,直接套到 MuJoCo 会错或会让仿真倒退**。
在动手前必须澄清:

1. **MuJoCo 的 geom 没有 `restitution`(弹性)属性。**
   文档里 `restitution="0.6"` 在 MuJoCo 里是无效字段,会被忽略或报错。
   MuJoCo 的"弹性/回弹"由 `solref`(尤其第一个时间常数)隐式决定,要让球
   撞墙/落地有回弹,得调 `solref` 而不是加 restitution。**我们现在的球没有
   restitution,这是对的**;真要回弹可用 `solref=(0.01~0.02, <1)`(欠阻尼)。

2. **不要用"直接设置球速度 / teleport / apply impulse"来踢球——那是 v1 之前的
   假物理,会让我们倒退。** 文档推荐 `ball.qvel[:3] = velocity` 和
   `enforce_rolling()` 强行改角速度。但 mjlab **现在已经是真实接触动力学**:
   - 球是 `condim=3` 的可碰撞球体(freejoint),
   - G1 每只脚有 **7 个 `foot_capsule` 碰撞体**(见 g1.xml,`left/right_footN_collision`),
   - 脚踢到球时,MuJoCo 求解器**自己算冲量、接触点、由此产生的线速度+角速度**。

   所以"踢球点位置→旋转""冲量→球速"这些**不需要我们手写公式去 set**,而是
   **自然涌现**。我们要做的是把**物理参数标定对**(质量/摩擦/接触刚度),
   让涌现出来的行为符合真实比赛。手动 set 速度反而会破坏这套真实接触。

3. **G1 不是圆柱体,是已经建好的全关节人形(~35kg)。** 文档给的
   `<body name="robot"> <geom type="cylinder" mass="30">` 是把机器人当刚性圆柱,
   完全不适用——我们用的是 asset_zoo 里的 Unitree G1 全身 mesh+关节+actuator。
   "机器人重量"这一项**已经满足**(G1 总质量已在 XML 里定义),不需要新建圆柱。
   v2 里"机器人质量影响球"是**自动成立**的(动量守恒由求解器保证)。

4. **滚动距离公式 `d = v²/(2·μ_r·g)` 只是直觉估算,MuJoCo 的滚动摩擦不是这么算的。**
   文档自己也算出"v=6 → 滚 61m"然后说"摩擦得调大"。在 MuJoCo 里,滚动阻力来自
   geom friction 的**第 3 个分量(roll friction)**,单位/语义和教科书 μ_r 不同,
   **必须用实测标定**(见 §4 验证),不能直接套公式定参数。

5. **更关键:roll friction 只有在高维接触里才会按预期生效。已实测证实。**
   MuJoCo 的 `condim=3` 只包含法向 + 两个切向摩擦维度;`condim=4` 才包含
   torsional/spin friction,`condim=6` 才包含 rolling friction。当前代码已确认:
   - `SoccerBallCfg` 的 `ball_geom` 是 `condim=3`;
   - G1 足部碰撞体是 `condim=3`;
   - 门柱/场地碰撞也多为 `condim=3`。

   **实测验证(2026-05-31,最小 MuJoCo 脚本,球 6 m/s 纯滚动出发):**

   | condim | roll friction 扫描 | 滚动距离 |
   | --- | --- | --- |
   | **3** | 0.001 → 0.1(100 倍) | **全部 52.45m,20s 不停**(roll 项完全失效) |
   | 6 | 0.001 | 40.9m(几乎不停) |
   | 6 | **0.01** | **9.04m**(正中 RoboCup 8–15m) |
   | 6 | 0.05 | 4.86m |
   | 6 | 0.1 | 4.31m |

   结论:`condim=3` 下扫 `friction[2]` 100 倍滚动距离纹丝不动 —— roll friction
   是**死参数**。这也追溯解释了 v1:球能停只是滑动相(slide friction)短暂耗能,
   一旦进入纯滚动就基本不停(52m)。

   **混合 condim 取 max,已实测(2026-05-31):** ground=3 + ball=6 → 4.86m,
   与 6/6 完全相同;ground=6 + ball=3 同样。**因此只需把球 geom 升到 `condim=6`
   即可**,地面和脚保持 `condim=3` 不变(对走路平衡零影响,也省去 pair override
   和全局升维的求解成本)。`condim=6 + roll≈0.01` 已落在目标区间,v2.1 搜索范围很窄。

**一句话:v2 不是"换成冲量模型",而是"把已有的真实接触物理标定到比赛级真实度"。**

---

## 1. 现状盘点(v1 已经有什么)

| 项目 | 现状 | 文件 |
| --- | --- | --- |
| 球质量 | 0.43 kg ✓ | `SoccerBallCfg.mass` |
| 球半径 | 0.11 m ✓ | `SoccerBallCfg.radius` |
| 转动惯量 | **当前是自动实心球惯量**:I=2/5·m·R²≈0.00208;真实充气足球更接近壳层,I≈2/3·m·R²≈0.00347,需验证 | (求解器 / 显式 inertial) |
| 球摩擦 | `(0.5, 0.02, 0.01)` = (slide, spin, roll),但当前 `condim=3` 下 roll 项不应直接假设生效 | `SoccerBallCfg.friction` |
| 球接触维度 | `ball_geom condim=3`,需为 rolling friction 做 `condim=6` 或 pair override 验证 | `soccer_field.py` / pair |
| 接触刚度 | `solref=(0.02,1)`,`solimp=(0.9,0.95,0.001,0.5,2.0)` | `SoccerBallCfg` |
| 足部碰撞 | 每脚 7 个 capsule(真实接触面) | `g1.xml` `*_footN_collision` |
| 机器人质量 | G1 全身 mesh,质量已定义 | `g1.xml` |
| 踢球机制 | **真实接触**(脚 capsule 撞球,求解器算冲量) | (涌现,非手写) |
| 球↔目标奖励 | approach/to_target/ball_velocity/success,当前主要是随机 target point,还不是完整进球任务 | `rewards.py` |

**结论:架构大方向对,但不能再假设"第三个 friction 分量已经在起作用",
也不能默认"实心球惯量"足够真实。v2 的工作是参数标定 + 接触维度验证 +
接触可观测 + 旋转/滚动/回弹真实度,不是重写。**

---

## 2. v2 要解决的关键缺口

### 缺口 A:球速衰减/停止/可追赶性不真实(球滑太远 或 粘地)
- 现 roll friction = 0.01 看起来偏小,但更根本的问题是**当前球 `condim=3`,
  rolling friction 可能没有按预期进入接触约束**。
- 目标不是固定规定"6 m/s 必须滚 8~15 m",而是保证球被踢动后会**自然减速并停下**,
  且在轻踢/传球等常见速度下,机器人有机会追上并重新控球。8~15 m 只能作为
  初期人工参考区间,最终以实测标定、场地尺寸和可追赶性指标为准。
- 手段:已实测确认只需把球 geom 升到 `condim=6`(地面/脚保持 3 即可,取 max 规则);
  `condim=6 + roll=0.01` 已落在 9m,搜索范围很窄,微调即可。不需要 pair override。

### 缺口 B:球的旋转/滑动→滚动过渡没验证过
- 真实球:踢出后先**滑动**,逐渐过渡到**纯滚动**(v=ωR)。
- mjlab 求解器**会自然产生**这个过渡(只要 spin/roll friction 合理),
  **不要用文档的 `enforce_rolling()` 强行覆盖角速度**(那会破坏真实接触)。
- 手段:把球的角速度纳入**可观测/可记录**,渲染时确认球真的在转、有滚动过渡。
  spin friction(第 2 分量,现 0.02)控制自旋衰减,偏击产生的弧线/侧旋由它影响。

### 缺口 B2:足球转动惯量可能不真实(实心球 vs 充气壳层)
- 当前 MuJoCo 会根据 `sphere + mass` 自动给出**实心球**惯量
  `I=2/5·m·R²≈0.00208 kg·m²`。
- 真实足球不是实心橡胶球,质量主要在外壳和内胆,惯量更接近**薄壳球**
  `I=2/3·m·R²≈0.00347 kg·m²`,比实心球大约 67%。
- **实测影响(2026-05-31,同一偏心冲量 J=1 N·s 作用在球顶):**

  | 惯量模型 | 角速度 ω | 球面旋转线速度 ωR |
  | --- | --- | --- |
  | 实心(2/5) | 52.9 rad/s | 5.81 m/s |
  | 壳层(2/3) | 31.7 rad/s | 3.49 m/s |

  **差 40%**——不是可忽略的差异。壳层惯量下球旋转更慢、滑→滚过渡更长、
  搓球/弧线手感更接近真实足球。
- 手段:做显式球惯量对照测试。不要先入为主认为自动惯量"一定更真实"。
  如果壳层惯量让旋转/滚动行为更接近实测,则在球 body 上显式设置 inertial。

### 缺口 B3:回弹/接触顺应性没有标定
- MuJoCo 没有 `restitution` 字段,但真实足球的弹跳、撞门柱反弹、脚触球停留时间
  仍然由 `solref/solimp` 等接触参数隐式决定。
- 当前只计划测滚动距离,不足以判断球是否"像真球"。一个球可以滚动距离正确,
  但落地像石头、撞门柱像海绵,仍然不真实。
- 手段:增加 drop test、post/wall rebound test、foot impact dwell-time test。
  回弹参数只在这些测试不合理时小步调整,并同步做 dt/solver 收敛检查。

### 缺口 C:足→球接触"位置/力道"不可观测、无奖励引导
- 你强调"足部踢到球的位置、力道、物理分析都要有"。现在脚能踢到球(真实接触),
  但**没有一个传感器/观测量告诉策略"我踢中了没、踢在球的哪个部位、力多大"**。
- 手段:加一个**足↔球的 ContactSensor**(类比已有的 `feet_ground_contact`),
  暴露接触 found/force/pos,用于:
  - **观测候选**:可选让策略知道脚和球的接触状态,但必须做可迁移性 ablation;
  - **奖励**:鼓励"用脚正面接触球"而非膝盖/胫骨乱碰;
  - **记录/分析**:把接触点相对球心位置、接触力写进 metrics,做物理分析。

### 缺口 D:MSL 规则约束还没有进入仿真指标
- RoboCup MSL 使用 regular size FIFA soccer ball,并对机器人尺寸/重量、ball handling、
  kicker 等有比赛规则约束。v2 不能只追求"踢得动",还要避免学到不合规动作。
- 手段:把规则相关量做成评估 metric,至少包括:
  - ball handling/夹球:球是否被机器人几何包住、卡住、持续贴身超过合理范围;
  - 射门速度上限监控:MSL 规则中最大射门速度量级约 80 km/h(≈22 m/s),
    各年规则有出入,不作硬阈值,仅作异常检测上界;训练中不应出现远超此量级的球速;
  - 若未来加 dribbler 几何,必须验证对应真实硬件机构且不形成非法持球。

### 缺口 E:场地/球参数过于理想,缺少标定后的鲁棒随机化
- 真实场地不是完美平面,球也不会永远 0.430 kg、0.110 m、固定压力/摩擦。
- 但随机化不能抢在标定前做,否则无法判断问题来自物理参数还是随机扰动。
- 手段:先固定参数完成 §4 物理保真测试,再加入小范围 domain randomization:
  球质量/半径、球-地摩擦、场地局部摩擦、轻微地面不平、接触参数小扰动。

### 缺口 F:观测过于理想,接触传感器是否可迁移未区分
- 接触传感器很适合做 reward/metric,但如果真机没有等价足-球接触信号,
  直接作为 policy observation 可能造成 sim-to-real 依赖。
- 手段:把 foot-ball contact 分成三类用途:
  - **metric/logging**:永远允许,用于物理分析;
  - **reward shaping**:允许,但权重要小,避免刷接触;
  - **policy observation**:必须可选,并做 ablation。默认优先不给策略强依赖,
    或至少保留一个"无接触观测"训练配置。

### 缺口 G:最终任务还没有明确变成"逐步踢进球门"
- 当前 `DribbleCommand` 的核心是把球带到随机 target point,这对学习控球和方向推进
  很有用,但**不等价于比赛进球**。如果 v2 只优化随机点成功率,可能得到一个会把球
  推到场上任意点的策略,但不能保证会组织到球门方向、最后完成射门。
- 最终目标必须拆成闭环:
  1. 找球/接近球;
  2. 用脚触球并恢复控球;
  3. 多次轻推/带球把球逐步推进到进攻方向;
  4. 到射门区后把球朝球门口踢出;
  5. 球穿过球门线且在门柱之间,判定进球。
- 手段:保留随机 target 作为课程学习的早期阶段,但 v2 后期必须切到
  **goal-scoring command/evaluation**:target 不再是任意点,而是球门口/球门线后的
  合法进球区域;success 不再是"离 target <0.5m",而是"球穿过球门线且在门框内"。

---

## 3. 详细实施步骤(多个可独立验证的小迭代)

> 每个小迭代单独 `make check` + 短训练 smoke test,确认不退化再进下一步。
> 顺序经过设计:先确认接触维度/惯量/回弹这些被动物理,再加可观测(传感器),
> 最后调奖励、规则指标和随机化。

### 迭代 v2.0 — 接触维度生效性验证(缺口 A 的前置条件)
- **改什么**:暂不改训练配置,新增离线物理测试。对同一初速度滚动测试比较:
  - 当前 `ball_geom condim=3`;
  - 球-地接触 `condim=6` 或 pair override;
  - 不同 `friction[2]` 候选值。
- **怎么测**:给球 `{2,4,6,8}` m/s 初速度,记录滚动距离、停止时间、
  `|v|-ωR` 滑动量衰减。若 `condim=3` 下扫 `friction[2]` 几乎无变化,
  说明原方案不能直接标定 roll friction。
- **验收**:找到一个能让 rolling friction 明确影响滚动距离的接触配置。
- **文件**:`src/mjlab/terrains/soccer_field.py`;可选新增 pair 配置;
  新增 `tests/test_soccer_ball_roll.py`。

### 迭代 v2.1 — 球滚动摩擦标定(缺口 A)
- **改什么**:在 v2.0 确认生效的接触配置上,扫 roll friction 候选值,
  从 `[0.02, 0.04, 0.06]` 起;必要时微调 `solref/solimp` 让落地不抖也不粘。
- **怎么测**:写一个**离线脚本/测试**(不训练),给球 `{1,2,4,6,8}` m/s
  初速度,让它在场地平面上自由滚,记录停球时间、停球距离、速度衰减曲线。
  8~15 m 只作为 6 m/s 的初期参考区间,不是硬编码目标。
  - 注意:这里给初速度是**做物理标定测试**(纯被动滚动),不是训练时的踢球机制,
    两者不要混。
- **验证**:球最终自然停下;轻踢/传球速度下不会长期滑出场外;速度衰减曲线平滑;
  渲染确认球在"滚"不是"滑"。
- **文件**:`src/mjlab/terrains/soccer_field.py`(改 cfg);
  新增 `tests/test_soccer_ball_roll.py`(标定测试)。

### 迭代 v2.1a — 追球可达性测试(缺口 A 的任务可学习性)
- **改什么**:新增评估脚本/metric,不改物理。把机器人放在球后方固定距离,
  给球 `{1,2,4,6}` m/s 初速度,让策略或脚本化追球控制去重新接近球。
- **怎么测**:记录 `catch_up_rate`、`time_to_recover_possession`、
  `ball_stop_time`、`ball_stop_distance`、追上时球-机器人距离。
- **验收**:轻踢/传球速度下,球停下前或停下后应能被机器人重新接近到 `<0.5m`。
  如果球长期不停、频繁滚出场外,或中低速球也系统性追不上,说明摩擦/奖励/追球策略
  至少有一项还不合理。
- **文件**:`src/mjlab/scripts/evaluate_soccer.py` 或
  `tests/test_soccer_ball_catchability.py`。

### 迭代 v2.1b — 足球转动惯量对照(缺口 B2)
- **改什么**:比较默认实心球惯量与显式壳层惯量:
  - 实心球 `I=2/5·m·R²≈0.00208`;
  - 壳层近似 `I=2/3·m·R²≈0.00347`。
- **怎么测**:偏心接触/斜踢脚本记录球线速度、角速度、滑→滚过渡时间、
  滚动距离。不要只看单个 episode,至少扫多个接触点和初速度。
- **验收**:选择更接近目标真实行为的一组惯量;如果差异显著,把惯量显式写入球模型,
  并在 `stage_eval.md` 记录原因。
- **文件**:`src/mjlab/terrains/soccer_field.py`;
  新增 `tests/test_soccer_ball_inertia.py` 或合并到 roll 标定脚本。

### 迭代 v2.1c — 回弹/接触顺应性标定(缺口 B3)
- **改什么**:一般不先改参数,先测。若落地/撞柱明显不真实,再小步调
  `solref/solimp`,并同步跑 dt/solver 收敛测试。
- **怎么测**:
  - drop test:球从固定高度落地,记录第一次反弹高度、接触时间、峰值力;
  - post/wall rebound:球以 `{2,4,6,8}` m/s 撞门柱/墙,记录反弹速度比例;
  - foot impact dwell-time:脚触球持续时间和峰值力是否爆炸。
- **验收**:无明显穿透/抖动/异常弹射;dt 减半后关键指标变化 <10%。
- **文件**:`tests/test_soccer_ball_rebound.py`;必要时改 `SoccerBallCfg.solref/solimp`。

### 迭代 v2.2 — 足↔球接触传感器(缺口 C/F 的可观测部分)
- **改什么**:在 `unitree_g1_soccer_env_cfg` 里加一个 `ContactSensorCfg`,
  primary = 14 个 `*_footN_collision`,secondary = `ball_geom`,
  开 force / pos(参考已有 `feet_ground_contact` 的写法 env_cfgs.py:67)。
- **暴露**:接触 found(是否踢中)、force(力道)、pos(世界系接触点)。
  接触点相对球心 `r = contact_pos - ball_pos`,可推断击球部位(上/下/侧)。
- **可迁移性要求**:默认先用于 metric/reward;是否加入 policy observation 必须做
  ablation。若真机没有等价接触观测,保留一个"无 foot-ball contact 观测"配置。
- **验证**:渲染一脚踢球,确认接触瞬间 sensor.found=1、force>0、pos 在球面附近。
- **文件**:`env_cfgs.py`(加 sensor);可选 `observations.py` 加
  `foot_ball_contact` 观测函数。

### 迭代 v2.3 — 接触/带球奖励整形(缺口 C 的奖励部分)
- **改什么**:基于 v2.2 的接触量,新增/调整奖励:
  - `kick_contact`:脚正面接触球给正奖励(鼓励用脚而非身体);
  - 可选 `kick_power`:把接触力/球获得的速度投影到"球→目标"方向,鼓励有效推进
    (注意:这是 shaping,真实冲量仍由求解器算,我们只奖励结果);
  - 复用现有 `dribble_ball_velocity_to_target` 作为"踢出后球朝目标滚"的奖励。
- **验证**:短训练(12~50 iter)看 `kick_contact` 有信号、无 NaN、不摔。
- **文件**:`rewards.py`(加函数)、`mdp/__init__.py`(导出)、`env_cfgs.py`(挂奖励)。

### 迭代 v2.4 — 旋转/滚动真实度确认(缺口 B)
- **改什么**:一般不改物理,主要是**记录与确认**。把球角速度、|v|-ωR(滑动量)
  写进 metrics;必要时微调 spin friction(第 2 分量)让偏击有合理弧线。
- **明确不做**:不加 `enforce_rolling()` 强制改角速度(会破坏真实接触)。
- **验证**:见 §4 测试 2(斜踢有旋转)、测试 3(惯量对照)和测试 5(机器人撞球球飞、人基本不退)。
- **文件**:`dribble_command.py` 或 `rewards.py` 的 metrics 部分。

### 迭代 v2.5 — MSL 规则合规指标(缺口 D)
- **改什么**:新增评估 metrics,不一定进入训练 reward。
- **指标**:
  - `max_ball_speed`:监控是否出现远超 ~22 m/s(量级参考,非硬阈值)的异常射门速度;
  - `ball_handling_time`:球长期被机器人几何包住/夹住/贴身的时长;
  - `dribbler_legality`(若启用 dribbler 几何):验证对应真实机构且不非法持球。
- **验证**:v2 评估报告里给出上述指标分布,异常 episode 存视频。
- **文件**:`dribble_command.py` / `evaluate_soccer.py` / `stage_eval.md`。

### 迭代 v2.6 — 标定后 domain randomization(缺口 E)
- **改什么**:只在 v2.0~v2.4 固定参数标定通过后启用小范围随机化。
- **范围建议**:
  - 球质量/半径在 FIFA/IFAB 合法范围附近小幅随机;
  - ball-ground slide/spin/roll friction 小范围随机;
  - 场地局部摩擦/轻微高度扰动;
  - `solref/solimp` 小范围随机,但必须保持物理测试不过界。
- **验证**:随机化开启后,§4 物理保真测试仍在容忍区间内,训练不 NaN。
- **文件**:`env_cfgs.py` events/domain randomization 配置。

### 迭代 v2.7 — 观测噪声/延迟与传感器 ablation(缺口 F)
- **改什么**:增加评估/训练配置,比较:
  - 完美球状态;
  - 球位置/速度带噪声和延迟;
  - 无 foot-ball contact policy observation;
  - 有 foot-ball contact policy observation。
- **验收**:最终推荐配置不能强依赖真机没有的观测;若依赖,必须在文档写清楚
  对应真实硬件传感器来源。
- **文件**:`observations.py` / env cfg variants / `evaluate_soccer.py`。

### 迭代 v2.8 — 进球任务闭环(缺口 G,最终目标)
- **改什么**:在保留随机 target 课程的基础上,新增 goal-scoring 模式:
  - `target_pos` 可采样到对方球门口/球门线后方,形成进攻方向;
  - success 从"球接近 target 点"升级为"球穿过球门线且 y 在门柱内";
  - 终止/metric 明确区分 `goal_scored`、`shot_on_goal`、`out_of_bounds`。
- **课程建议**:
  1. **Target Dribble**:继续用随机 target point 学会找球、触球、推进;
  2. **Goal-Directed Dribble**:target 固定在球门方向,学会沿场地纵向推进;
  3. **Shooting Zone**:从禁区/射门区附近开始,学会把球踢进门;
  4. **Full Field Goal**:随机开局,逐步控球推进并完成射门。
- **奖励建议**:
  - `goal_progress`:球朝对方球门线方向推进;
  - `shot_on_goal`:球速方向穿过球门口投影;
  - `goal_scored`:球穿过球门线且在门框内的稀疏成功奖励;
  - 保留 `kick_contact`/`effective_kick` 作为小权重 shaping,主导仍是进球结果。
- **验证**:单 env 视频必须能看到"多次触球/追球恢复/推进/射门/进球"完整链路。
  评估时不能只看 random target 成功率,必须报告 Goal Rate。
- **文件**:`dribble_command.py`(goal-scoring command / goal-line success),
  `terminations.py`(goal scored / out of bounds),
  `rewards.py`(goal progress / shot on goal / goal scored),
  `evaluate_soccer.py`(goal evaluation protocol)。

---

## 3.5 踢球动力学标定曲线(最关键缺口:脚→球初速度的映射)

> 这是当前计划**最致命的缺口**。§4 测试只是"给球 6 m/s"看滚多远,但训练里
> **球速是接触的结果,不是我们给的**。我们从没建立过"脚怎么踢 → 球获得多少速度"
> 的映射,也就无法判断策略到底是"会踢球"还是"碰巧把球蹭走"。

### 为什么致命
- 现实比赛:轻推/传球/射门 → 球速完全不同,靠的是**脚速 + 接触方式**。
- 现在:我们既不能控制、也不能评估这个映射 → 看不出策略有没有"踢球能力"。

### 实施方法(依赖 v2.2 的接触传感器)
有了 foot↔ball ContactSensor 后,在每次接触事件记录:
```text
{
  contact_force,            # 接触力(传感器)
  contact_duration,         # 接触持续步数
  contact_point_offset,     # 接触点相对球心 r = contact_pos - ball_pos
  foot_velocity,            # 接触瞬间脚 site 的世界系速度
  ball_velocity_after       # 接触结束后球的线速度(+角速度)
}
```
聚合方式:**不是训练,是采集脚本**——用已训练策略(或脚本化踢球)跑大量接触事件,
把上面的量 dump 成 csv,离线拟合关系。

### 输出曲线(直接证明"会踢球")
- **球初速 v_ball vs 接触冲量 J**(J 由 force×duration 近似,或动量差 m·Δv)。
- **球初速 v_ball vs 脚速 v_foot**(应大致单调:脚踢得快→球飞得快)。
- **接触点 offset vs 球角速度 ω**(偏上→后旋、偏下→前旋,验证旋转涌现)。

### 验收标准(球速分档要覆盖真实区间)
- 轻踢 → 1~2 m/s,传球 → 2~4 m/s,射门 → 5~8 m/s 都能涌现出来。
- 若策略只会产生 <2 m/s 的慢推 → 说明"只会蹭球不会踢",需在 v2.3 用
  踢球力度奖励引导,或检查脚速是否上得去。

### 文件
- 采集脚本 `scripts/collect_kick_data.py`(读 contact sensor + 球速,dump csv);
  画图脚本 `scripts/plot_kick_curve.py`。指标可同时进 `MetricsManager` 做训练监控。

---

## 3.6 带球(dribbler)物理模型(现实 MSL 最关键能力,但有红线取舍)

> 现实 MSL 里**最关键的不是踢,是带球控制**——低速移动球不滚走、加速球不飞出。
> 但这一节直接触碰 §0.6 红线(不加 fake 力场),所以**取舍要写清楚**。

### 红线下的两条路线(先试 A,A 不行再考虑 B)

**路线 A(首选,纯接触):靠脚的持续接触 + 摩擦带球。**
- 不加任何额外约束力。靠 v2.3 的 `kick_contact` 奖励引导策略学会
  "用脚内侧/脚背持续轻触球、小步推进",让真实接触+摩擦自然完成带球。
- 这完全符合 §0.6:行为来自接触动力学。**应优先训练验证 A 能不能work。**

**路线 B(兜底,显式带球力场,触碰红线):仅当 A 证明做不到时才用。**
- 现实机器人很多带 dribbler(滚轮/凹槽),是**真实硬件机构**,不是作弊——
  所以建模它不算 fake physics,但必须**对应真实机构**而非凭空加力。
- MuJoCo 实现思路:**不要凭空 apply force**;若要建 dribbler,应在脚/小腿前方
  加一个**真实几何**(小凹面/挡板 geom)用接触+摩擦兜住球,让"兜住"也来自接触。
- 参考文档说的 "dribbler_force ≈ 5~15 N、接触高度在球心以下"可作为**真实机构的
  目标握持力量级参考**,但实现必须落到几何+接触,不是 `xfrc_applied` 直接加力。
- **规则约束**:任何 dribbler 几何都必须通过 §3/v2.5 的 ball handling 合规指标。
  如果形成"夹球/藏球/长时间持球",即使任务成功率提高也不能算有效进步。

### 验证指标
- 低速移动(<1 m/s)时球保持在脚前 <0.3m,不滚走。
- 加速时球不飞出(球-脚距离不突然 >0.5m)。
- 复用 §8.2 的 Possession(控球率)量化。

### 取舍声明(写进文档防止以后偷懒)
- **默认走路线 A**。只有 A 在合理训练后控球率仍上不去,才启用 B,
  且 B 必须是"真实 dribbler 机构的几何+接触建模",不是 `apply_force` 黑魔法。
- 若用 B,必须在 stage_eval.md 里注明"引入了显式带球机构,对应真实硬件 XXX",
  保持 sim-to-real 的可解释性。

### 文件
- 路线 A:`rewards.py`(`kick_contact`/带球 shaping)。
- 路线 B(若启用):`g1.xml` 或 soccer env 的 spec_fn 加 dribbler 几何。

---

## 4. 验证标准(每轮训练后必跑,沿用 stage_eval.md 五段式)

参考文档的测试口径是好的起点,但 v2 需要扩展成完整物理/规则验收:

- **测试 0 — rolling friction 生效性**:`condim=3` 与 `condim=6/pair override`
  对照;扫 `friction[2]` 时滚动距离必须有可解释变化。若无变化,不能进入 v2.1 标定。
- **测试 1 — 球速衰减/停球曲线**:被动给球 `{1,2,4,6,8}` m/s 初速度,
  记录停球时间、停球距离和速度衰减。6 m/s 的 8~15 m 只作为初期参考区间,
  不是硬验收;硬验收是球会自然停下且曲线稳定、可解释。
- **测试 1b — 追球可达性**:轻踢/传球速度下,机器人应能在球停下前或停下后
  重新接近到 `<0.5m`;记录 `catch_up_rate` 和 `time_to_recover_possession`。
- **测试 2 — 斜踢有旋转**:偏心接触后球有角速度、轻微弧线;
  渲染单 env 确认,metrics 记录角速度非零。
- **测试 3 — 惯量对照**:实心球惯量 vs 壳层惯量下,记录偏心踢球角速度、
  滑→滚过渡时间和滚动距离;选择更符合真实球行为的一组,并记录依据。
- **测试 4 — 回弹/顺应性**:drop test 和撞门柱/墙测试无明显异常弹射、
  粘地、穿透或数值抖动;dt 减半后关键指标变化 <10%。
- **测试 5 — 机器人撞球**:球被踢飞,**机器人质心位移很小**(G1 ~35kg 是
  主动平衡体,不是刚性块——踢球时会靠关节/落脚反应维持平衡,不能简单套
  刚体动量守恒;但质量比 m_ball/M_robot ≈ 1/80,球获得的速度远大于机器人扰动)。
- **测试 6 — MSL 合规**:球速不出现非比赛级异常尖峰(量级参考:MSL 规则中
  最大射门速度约 80 km/h ≈ 22 m/s,各年规则有出入,不作硬阈值,仅作异常检测上界);
  dribbler/脚部接触不形成长时间夹球或非法持球。
- **测试 7 — 进球闭环**:从不同初始球位/机器人位姿开始,策略能通过多次触球
  将球推进到射门区并踢过球门线。记录 `goal_scored`、`shot_on_goal`、
  `time_to_score`、`touches_before_goal`。
- **训练健康指标**(沿用 v1):Mean reward 上升、`fell_over`→0、
  `episode_success`/`goal_scored` 上升、`robot_to_ball_error` 收敛、无 NaN。
- **可视化**:必须 `--num-envs 1` 渲染(多 env 会拉远成俯视看不清,
  见 stage_eval.md 阶段 3 坑记录),存到 `soccer_eval/<日期>_v2/`。

---

## 5. 风险与取舍

- **风险 1:roll friction 调大后带球变难。** 真实球滚得远→机器人要学会控制力度。
  这是**期望的难度提升**,但可能需要更长训练(>3000 iter)。
  缓解:课程学习——先小摩擦学会接触,再逐步增大。
- **风险 1b:在 `condim=3` 下扫 roll friction 是无效标定。**
  缓解:把 v2.0 生效性验证作为硬门槛;没有通过就不记录"roll friction 已标定"。
- **风险 2:接触奖励引导过强→策略学会"贴着球蹭"刷奖励。**
  缓解:`kick_contact` 权重要小,主导奖励仍是 `dribble_to_target`。
- **风险 3:solref 调成欠阻尼(要回弹)可能引入数值抖动/接触爆炸。**
  缓解:回弹是"撞墙/撞门柱"才需要,带球阶段可暂不开;若开,小步调 + smoke test。
- **风险 4:三体接触(脚+球+地面同时接触)导致不稳定。** 球贴地被脚踢时,
  脚、球、地面三者同时接触,处理不好会"夹球"或异常弹射。
  缓解:先只升级 ball-ground 的接触维度,不要全局升维;适当调 solimp 让碰撞不过硬;
  §0.5 的 dt 收敛测试要**专门覆盖"贴地踢球"这一最易出问题的场景**;
  若仍弹射,提高 solver iterations 而非降摩擦。
- **风险 5:显式壳层惯量改善旋转,但破坏数值稳定或带球难度。**
  缓解:惯量改动必须和滚动距离、斜踢旋转、贴地踢球三项一起验收。
- **风险 6:把 foot-ball contact 放进 observation 后形成不可迁移依赖。**
  缓解:contact 默认先用于 metrics/reward;作为 observation 必须有 ablation 和真机传感器依据。
- **风险 7:domain randomization 太早启用导致无法定位物理问题。**
  缓解:先固定参数完成物理标定,再小范围随机化;评估时能一键关闭随机化。
- **风险 8:随机 target 成功率提高,但不会真正进球。**
  缓解:把 random target 只作为课程阶段;最终验收必须看 Goal Rate、
  Shot-on-Goal Rate 和完整视频链路,不能用 target success 代替进球。
- **取舍:不追求空气动力学(马格努斯力/真实弧线球)。** MuJoCo 不模拟流体,
  弧线只能靠地面摩擦产生的侧旋近似。冠军队级别的空气弧线超出当前范围,先不做。

---

## 6. 重要文件路径(v2 会动到的)

- 球 + 球场几何:[src/mjlab/terrains/soccer_field.py](src/mjlab/terrains/soccer_field.py)
  (`SoccerBallCfg.friction` / `condim` / 显式惯量 / `solref` 标定在这)
- pair / domain randomization 工具:
  [src/mjlab/envs/mdp/dr/pair.py](src/mjlab/envs/mdp/dr/pair.py),
  [src/mjlab/envs/mdp/dr/geom.py](src/mjlab/envs/mdp/dr/geom.py)
- G1 资产(足部碰撞 capsule、机器人质量):
  [src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml](src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml)
- 环境配置(加足↔球 ContactSensor、挂奖励):
  [src/mjlab/tasks/velocity/config/g1/env_cfgs.py](src/mjlab/tasks/velocity/config/g1/env_cfgs.py)
- 奖励:[src/mjlab/tasks/velocity/mdp/rewards.py](src/mjlab/tasks/velocity/mdp/rewards.py)
- 观测:[src/mjlab/tasks/velocity/mdp/observations.py](src/mjlab/tasks/velocity/mdp/observations.py)
- 带球命令 / metrics:[src/mjlab/tasks/velocity/mdp/dribble_command.py](src/mjlab/tasks/velocity/mdp/dribble_command.py)
- 物理标定测试(新增):
  `tests/test_soccer_ball_roll.py`,
  `tests/test_soccer_ball_inertia.py`,
  `tests/test_soccer_ball_rebound.py`
- 评估脚本(新增):
  `src/mjlab/scripts/evaluate_soccer.py`,
  `scripts/collect_kick_data.py`,
  `scripts/plot_kick_curve.py`
- 阶段成果记录:[stage_eval.md](stage_eval.md)
- v1 计划:[soccer_field_plan.md](soccer_field_plan.md)

---

## 7. 一句话总结

v2 的本质**不是**换成"冲量+手动设速度"模型(那会倒退成假物理),而是:
**把 mjlab 已有的真实足↔球接触动力学,通过(a)接触维度确认后的滚动摩擦标定、
(b)足球惯量/回弹标定、(c)足球接触传感器、(d)接触/带球奖励整形、
(e)MSL 合规与鲁棒评估、(f)进球任务闭环,标定到 RoboCup 比赛级真实度**,并用
"rolling friction 生效性/球速衰减与停球/追球可达性/斜踢旋转/惯量对照/
回弹/撞球动量/规则合规/进球率"一组测试验收。
机器人质量由 G1 资产保证;球转动惯量必须经过实心球 vs 壳层球对照后再定。

---

## 8. v2 vs v1 的量化进步如何验证(可靠指标,不靠主观)

> 你说得对:可视化只能给主观判断。下面是一套**可复现、带统计显著性**的量化对比方案。

### 8.0 先讲清楚一个陷阱:不能直接比 reward / episode_success 的训练曲线

**v2 改了物理(滚动摩擦变大、加了接触传感器/奖励),所以:**
- v2 的奖励函数和 v1 不一样 → **Mean reward 跨版本不可比**(分母都变了)。
- v2 的物理更难(球滚得远、要控力度)→ 即使策略更强,`episode_success` 训练
  曲线也可能更低。**直接比训练日志会得出错误结论。**

**正确做法:把"评估"和"训练"分开**。用一个**固定的、与版本无关的评估协议**,
在**同样的测试条件**下分别跑 v1 和 v2 的策略,比的是**评估指标**而非训练指标。
评估时关掉探索噪声(用确定性 mean action)、固定随机种子、固定球/目标生成。

### 8.1 评估协议(Evaluation Protocol,v1/v2 完全一致)

建一个独立评估脚本(当前仓库**没有** evaluate 入口,只有 train/play/list-envs,
需新建 `src/mjlab/scripts/evaluate_soccer.py` 或一次性脚本):

```
- num_envs = 512(或更多,统计需要量)
- 固定 seed 列表(如 [0..9]),每个 seed 跑满 episode
- 确定性策略:用 policy 的均值动作(关掉高斯采样)
- 固定评估分布:球/目标在场地内均匀采样,但用固定 RNG → v1/v2 看到完全相同的局面
- 关掉训练专用的 domain randomization(评估要可复现)
- 每个 episode 记录下面 §8.2 的指标,聚合成 mean ± 95% CI
```

**关键:同一个评估脚本,只换 `--checkpoint-file`(v1 的 vs v2 的),其余全相同。**

### 8.2 与版本无关的"任务能力"量化指标(核心)

这些指标衡量**完成任务的能力**,不依赖具体奖励函数,因此可跨版本比较:

| 指标 | 定义 | 越好 | 实现 |
| --- | --- | --- | --- |
| **目标点成功率 Target Success Rate** | episode 内球进入 target 0.5m 的比例 | ↑ | 课程阶段指标,不能替代进球 |
| **进球率 Goal Rate** | episode 内球穿过球门线且在门柱之间的比例 | ↑ | 最终主指标 |
| **射正率 Shot-on-Goal Rate** | 有效射门轨迹穿过球门口投影的比例 | ↑ | 新增 metric |
| **首达时间 Time-to-Goal** | 从开始到首次到点的步数(仅成功 episode) | ↓ | 新增 metric,记录首次 at_goal 的 step |
| **进球时间 Time-to-Score** | 从开始到进球的步数/秒数(仅进球 episode) | ↓ | 新增 metric |
| **进球前触球次数 Touches-before-Goal** | 进球前足-球有效接触次数 | 看分布 | 新增 metric |
| **末端球-目标距离** | episode 结束时 ball_to_target_error | ↓ | 已有 `ball_to_target_error`,`reduce="last"` |
| **带球控球率 Possession** | 球在机器人 0.5m 内的步数占比 | ↑ | 由 `robot_to_ball_error<0.5` 统计 |
| **追球恢复率 Catch-up Rate** | 球被踢出后重新接近到 0.5m 内的 episode 比例 | ↑ | 新增 metric |
| **恢复控球时间 Recover Possession Time** | 从球失控到重新接近球的步数/秒数 | ↓ | 新增 metric |
| **丢球率 Loss-of-control** | 球失控(离机器人 >1.5m 且非朝目标)的次数 | ↓ | 新增 metric |
| **摔倒率 Fall Rate** | episode 因 fell_over 终止的比例 | ↓ | 已有终止统计 |
| **越界率 Out-of-bounds** | 因 out_of_field_bounds 终止比例 | ↓ | 已有终止统计 |
| **路径效率 Path Efficiency** | 球实际滚动路程 / 球-目标直线距离 | →1 | 新增 metric(累计球位移) |
| **球速分布 Ball Speed Dist.** | 平均球速 / 球速方差 / 高速踢球(>5m/s)比例 | 看分布 | 新增 metric |
| **有效踢球率 Effective-Kick** | 球速方向与"球→目标"方向夹角 <30° 的接触比例 | ↑ | 新增 metric(配合 §3.5 接触数据) |

> 最终任务里 **Goal Rate 是主指标**;Target Success Rate 只是课程阶段指标。
> Time-to-Score / Possession / Path Efficiency / Catch-up Rate 是辅助质量指标。
> 全部在**相同评估协议**下测,带 95% 置信区间。
>
> **球速分布**和**有效踢球率**是区分"真会踢"和"乱碰"的职业级指标:
> - 球速分布能看出 v1 是不是"全是慢推"、v2 有没有出现真实"射门"(>5m/s);
> - 有效踢球率衡量**方向性**——球被踢中后是不是真朝目标走,而非乱飞。
> 这两个指标和 §3.5 的踢球标定数据共用同一套接触采集。

### 8.3 解决"物理变了没法比"的关键:交叉评估(Cross-Evaluation)

光比"v1 策略在 v1 物理" vs "v2 策略在 v2 物理"**说明不了是策略变强还是物理变简单**。
所以做 **2×2 交叉评估**(策略 × 物理环境):

| | v1 物理环境 | v2 物理环境 |
| --- | --- | --- |
| **v1 策略** | A(基线) | B(v1 策略迁移到新物理) |
| **v2 策略** | C(v2 策略回到旧物理) | D(v2 目标) |

- **D vs A**:端到端进步(新策略+新物理 对 老策略+老物理)——你最终关心的。
- **B vs A**:量化"物理变难了多少"(同一个 v1 策略,新物理掉了多少分)。
- **D vs B**:在**同一个 v2 物理下**,v2 策略比 v1 策略强多少 → **这才是干净的"策略进步"**。
- **C vs A**:v2 策略在旧物理上是否退化(应不差于 A,否则是过拟合新物理)。

> 实现上:v1/v2 的物理差异就是 `SoccerBallCfg.friction` 等几个参数 + 是否挂接触奖励。
> 评估时用 CLI 覆盖物理参数即可在"v1 物理"和"v2 物理"间切换,不用改代码。
> 注意:观测维度若 v2 变了(加了接触观测),v1 策略无法直接跑 v2 物理 → 要么
> 评估时给 v1 补零观测,要么 v2 把新观测设计成"可选/向后兼容",这点在 v2.2 设计时就要定。

### 8.4 物理保真度指标(验证"仿真本身"是否更真实,与策略无关)

§4 的物理测试在这里量化成数字,**不需要策略**,纯物理标定:

- **Rolling friction 生效性**:`condim=3` 与 `condim=6/pair override` 对照,
  证明 `friction[2]` 的变化确实会影响滚动距离。
- **球速衰减/停球曲线**:给球 {1,2,4,6,8} m/s 初速度,记录停球时间、
  停球距离和速度曲线。v2 应表现为自然减速并停下;8~15 m 仅可作为 6 m/s 的
  初期参考,不能写成硬目标。
- **追球可达性曲线**:给球 {1,2,4,6} m/s 初速度,记录机器人重新接近球的成功率
  和恢复控球时间,证明"球会停,机器人追得上"。
- **滑→滚过渡时间**:记录 |v - ωR| 衰减到≈0 的时间。真实球应在踢出后短时间内进入纯滚动。
- **惯量敏感性**:比较默认实心球惯量和壳层惯量下的角速度、滑→滚过渡和滚动距离。
- **回弹/顺应性**:drop test 和撞门柱/墙测试记录反弹高度/速度比、接触时间、峰值力。
- **撞球动量比**:机器人撞球后 `Δv_robot / Δv_ball`,应 ≪ 1(重机器人几乎不退)。
- **MSL 合规**:最大球速、ball handling time、dribbler legality 不出现异常。
- **进球闭环**:Goal Rate、Shot-on-Goal Rate、Time-to-Score、Touches-before-Goal
  证明策略不是只会把球推到随机点,而是能逐步推进并射入球门。

### 8.5 统计严谨性(避免"看一两个 episode 就下结论")

- 每个指标在 **≥512 env × ≥10 seed** 上聚合,报 **mean ± 95% 置信区间**。
- 版本对比用**配对比较**(相同 seed/局面下 v1 vs v2),做配对 t 检验或
  bootstrap,报 **p 值或效应量**,而不是只看均值差。
- 所有评估结果(csv/json + 曲线图)存到 `soccer_eval/<日期>_v1_vs_v2/`,
  并把对比表追加进 [stage_eval.md](stage_eval.md)。

### 8.6 验收门槛(v2 才算"相对 v1 进步")

v2 必须同时满足(在 8.1 协议下):
1. **D vs B**:同一 v2 物理下,v2 策略 Goal Rate 显著高于 v1 策略(p<0.05);
2. **物理保真**:§8.4 的 rolling friction 生效性、球速衰减/停球、滑→滚、惯量/回弹测试通过;
3. **不退化**:v2 的 Fall Rate / Out-of-bounds 不显著高于 v1;
4. **质量提升**:Time-to-Score、Time-to-Goal 或 Path Efficiency 至少一项改善;
5. **会踢球(非乱碰)**:v2 的有效踢球率显著高于 v1,且球速分布出现真实射门
   (>5m/s 比例 v2 > v1),证明策略学到的是"踢球能力"而非"慢推蹭走"。
6. **可恢复控球**:轻踢/传球速度下 Catch-up Rate 和 Recover Possession Time 达到可接受水平;
7. **能完成最终目标**:视频和指标都显示"接近球→多次触球推进→射门→进球"完整链路;
8. **规则合规**:没有非比赛级异常高速、长时间夹球/持球或不对应真实硬件的 dribbler 行为。

> 只有"可视化好看"不算数;以上门槛带统计显著性/物理标定记录才算 v2 真正进步。

### 8.7 落地最小工作量

- 新增评估脚本(确定性 rollout + 固定 seed + 指标聚合 + csv 输出)。
- 在 `dribble_command` / metrics 里补 Time-to-Goal、Possession、Path Efficiency、
  Catch-up Rate、Recover Possession Time、Loss-of-control、Max Ball Speed、
  Ball Handling Time、Goal Rate、Shot-on-Goal Rate、Time-to-Score 几个 metric。
- 新增物理标定脚本/测试:rolling friction 生效性、球速衰减/停球曲线、
  追球可达性、惯量对照、drop/rebound。
- 一个画图脚本:球速衰减/停球曲线 + 追球可达性曲线 + 惯量/回弹曲线 +
  进球率/射正率曲线 + 2×2 交叉评估柱状图(带 CI)。
- 文件:新增 `src/mjlab/scripts/evaluate_soccer.py`、`scripts/plot_v1_v2.py`(或放 soccer_eval/);
  改 `dribble_command.py` 加 metric。
