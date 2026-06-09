# Soccer Robot v4 实验记录

> 自主执行记录。每个实验:目标 → 理由 → 做法 → 结果(实跑数字)→ 评估 → 下一步调整。
> 方案见 `soccer_robot_v4.md`。v3 代表模型 04_e2e。开始:2026-06-07。

## 记录纪律
- **本文件是 v4 实验结果的正式总账**。每次实验结束后,无论成功/失败/中止,都要补:
  目标、理由、关键改动、结果、评估、下一步。
- **方案变更** 不写在这里单独悬空,必须同步回写 `soccer_robot_v4.md`。
- **较成功实验** 必须同时产出正式工件:
  - `mjlab/checkpoints/` 下的 checkpoint + params
  - `mjlab/soccer_eval/` 下的视频、曲线、诊断图、报告
- `*-codex.md`、memory note、临时 supervisor report 只作为辅助,**不能替代** 本文件和
  `soccer_robot_v4.md`。

---

## Exp 1 — 分离相机,恢复踢球(地基)

**目标**:真场线 env 里 depth 相机保持 64×48(踢球 CNN 完整迁移,reinit 0),RGB 相机独立。
从 model_2800 bootstrap,dribble_success 恢复到 ~0.5(对标 04_e2e),并确认 depth CNN 不再 reinit。

**理由**:05 不踢球的根因是 depth/RGB 共用 head_cam,提 RGB 分辨率连带毁了 depth-ball CNN 的
spatial_softmax。分离相机让 depth 分辨率不变 → 踢球能力完整继承。这是 v4 一切的前提。

**做法**:新增 `head_cam_rgb` 第二个 CameraSensorCfg(独立分辨率),depth obs 用原 `head_cam`
(64×48),RGB obs(StackedCameraRGB)改用新传感器。

**做法补充**:新 env `mos92_soccer_selfloc_dualcam_env_cfg`——depth 用原 head_cam(64×48,
仅 depth),新增 head_cam_rgb(96×72,仅 rgb)给 StackedCameraRGB。脚本 spike_v4_dualcam.py。

**smoke 验证(关键判据已达成)**:load 日志 `cnns.camera(depth).spatial_softmax` **reinit 0**
(05 时被 reinit 2 个),只 `camera_rgb` reinit 3 个(本就是 fresh 分支)。depth-ball/踢球 CNN
完整迁移。obs:camera (1,48,64) + camera_rgb (18,72,96),双传感器分离确认。

**全量训练**:3800 iter,384 envs。判读门槛:dribble_success >0.4(对标 04 的 0.51)、
goal_rate >0.2。跑完自动判读并接 Exp 2(RGB 提分辨率压自定位误差)。

**状态**:✅ 完成(3800 iter)。**判定:第一层 bug 修复验证成功,但未恢复到 v3 代表行为。**

**结果(current/best/best@iter/趋势,按监督口径)**:
| 指标 | current@iter | best@iter | 后半段趋势 |
|---|---|---|---|
| dribble_success | 0.00@3793 | **0.25@3604** | 0~0.25 跳,**best<0.30** |
| goal_rate | 0.00 | 0.50@1947 | 不稳 |
| selfloc_pos_err_m | 6.73@3799 | **1.88@1836** | GT 全抽后从 1.9m **塌陷到 6.7m** |
| fell_over | 0 | 0 | 全程零摔倒 ✓ |

**评估**:
- ✅ **第一层根因确认修复**:depth CNN reinit 0(05 时被毁),selfloc 一度到 **1.88m**(远好于 05 的 5.8m)。
- ❌ **触发 EXP1A 退出规则**:best dribble_success 0.25 < 0.30 → "dualcam selfloc 链不足以恢复 v3 代表行为"。
- ⚠️ **中后期塌陷**:selfloc 在 GT 全抽(iter>3500)后从 1.9m 塌到 6.7m,踢球也回落到 0。

**监督者反馈(已采纳)**:此 run 是 `selfloc_realspec` 链,**不是 04_e2e 同口径**——几何/奖励/课程都不同,
不能当 04 的直接替身。根本问题是 bootstrap 起点(model_2800 是 selfloc 模型,非会完整踢球的 04)和
env 链(selfloc vs e2e)都不对。

**下一步**:分叉到 **EXP1B = realspec_e2e_dualcam**(最高优先级):从 `mos92_soccer_e2e_env_cfg` 出发
+ 0.125m 真场线 + 分离双相机 + 从 **04_e2e/model_1499 bootstrap**。门槛:dribble_success≥0.35、
goal_rate≥0.20、fell_over=0、selfloc 优于 05。

**工件**:checkpoint 在 `logs/.../2026-06-07_19-49-28_spike_v4_dualcam/`(best selfloc @ model_1800 区)。
因未达踢球门槛,暂不提升为正式 checkpoints/ 交付(留作 dualcam selfloc 链验证记录)。

---

## Exp 1B — realspec_e2e_dualcam(监督者最高优先级,04 同口径)

**目标**:在 **04_e2e 同口径**(进攻半场几何+进球奖励层+e2e fade)下,叠加真场线 0.125m + 分离双相机,
从 **04_e2e/model_1499 bootstrap**。验证:踢球能力被继承(非重学),真场线对 selfloc 的真实影响。

**理由(采纳监督者反馈)**:Exp1 用的是 selfloc_realspec 链 + model_2800 起点,**不是 04 同口径**——
几何/奖励/课程/起点都不同,不能当 04 替身。监督者指出根本问题是起点(model_2800 是 selfloc 模型,
非会完整踢球的 04)和 env 链(selfloc vs e2e)都不对。本实验切到正确口径。

**做法**:新 env `mos92_soccer_e2e_dualcam_env_cfg`(基于 mos92_soccer_e2e_env_cfg)+ 0.125m 真场线 +
depth 64×48 仅深度 + head_cam_rgb 96×72 仅 RGB + RGB 6帧×stride6。脚本 spike_v4_e2e_dualcam.py,
bootstrap 从 04_e2e/model_1499。

**smoke 验证**:**depth CNN reinit 0**(踢球完整继承),只 camera_rgb reinit 3(fresh 分支)。exit 0。

**早期信号(iter 42)**:dribble_success **0.59**(vs Exp1 后期才 0.25)——**踢球能力确实从 04 继承**。
selfloc 12.6 为冷启动。

**门槛**:dribble_success≥0.35、goal_rate≥0.20、fell_over=0、selfloc 优于 05(5.8m)/Exp1(1.88m best)。

**状态**:训练中(3800 iter,384 envs)。跑完自动判读 current/best/best@iter/趋势并接下一步。

**结果(current/best/best@iter/趋势)**:
| 指标 | current@iter | best@iter | 后半段 |
|---|---|---|---|
| goal_rate | 0.00 | **0.50@679** | 0~0.50 波动 |
| dribble_success(reward和) | 1.20 | 2.35@2470 | 保住 |
| selfloc_pos_err_m | 10.94@3799 | 10.42@3502 | **全程卡 11m,从未收敛** |
| fell_over | 0 | 0 | 零摔倒 ✓ |

**评估**:✅ **踢球+进球+稳定从 04 完整继承**(goal_rate peak 0.5、fell_over 0)——监督者口径方向正确。
❌ **selfloc 彻底失败(10.4m)**,从 iter 0 卡 11m 从未收敛,比 Exp1 的 1.88m 差一个量级。

**根因(已确认)**:e2e fade 课程 `start=0/end=1` → GT 位姿 obs 从第 0 步就≈0(纯视觉)。但 EXP1B 的
**RGB 分支是全新 reinit 的**(新 96×72 相机),**fresh CNN 没有 GT 暖身就被要求纯视觉定位 → 学不出来**
(v3 exp8 教训:fresh RGB 分支需 GT 暖身)。原 e2e 能用零暖身是因为它从已会纯视觉的 model_2800 起跑,
RGB 非 fresh;EXP1B 从 04 起跑但换了新相机,RGB 又变 fresh,却套用了零暖身课程——不匹配。

**下一步 EXP1B-v2(对症修复)**:给 EXP1B 的 GT fade 课程加暖身 `start=800/end=2500`(对齐 selfloc 链),
让 fresh RGB CNN 先在 GT 辅助下学好定位再撤拐杖。踢球已从 04 继承不受影响。门槛同 EXP1B + selfloc 收敛到 <3m。

**状态**:EXP1B 完成,判定"踢球继承成功但 selfloc 因零暖身失败",转 EXP1B-v2。

---

## Exp 1B-v2 — e2e_dualcam + GT 暖身修复(对症 EXP1B 的 selfloc 失败)

**目标**:在 EXP1B(04 同口径+分离相机)基础上,给 fresh RGB 分支加 GT fade 暖身 [800,2500] iter,
让 selfloc 收敛到 <3m,同时保住从 04 继承的踢球+进球。

**理由**:EXP1B selfloc 全程卡 11m 从未收敛,根因是 e2e fade [0,1] 对 fresh RGB 分支无暖身
(v3 exp8 教训)。EXP1B-v2 改 fade [800,2500] 给暖身。**已知风险(OOD)**:04 是用 e2e fade[0,1]
训练的,GT 位姿 obs 训练时本就≈0,现在暖身重新开到 1 对 04 是 OOD 输入——若暖身因此救不动 selfloc,
则证明需要 EXP1C 的 model_2800 起点(它训练时见过 GT)。

**做法**:脚本 spike_v4_e2e_dualcam.py 全量分支设 selfloc_gt_mask start=800×24/end=2500×24。
smoke 确认 depth reinit 0、exit 0。

**门槛**:selfloc 收敛 <3m + dribble/goal 保住(goal_rate≥0.2、fell_over=0)。

**状态**:训练中。跑完自动判读;若 selfloc 仍不收敛→坐实 OOD→转 EXP1C(model_2800 起点对照)。

**结果(current/best/best@iter/趋势)**:
| 指标 | best@iter | iter>2500(撤拐杖后) |
|---|---|---|
| goal_rate | 0.625@1074 ✓ | — |
| dribble_success(reward和) | 2.06@2158 | 保住 |
| selfloc_pos_err_m | 9.73@187 | **10.4~11.6,仍卡 11m** ✗ |
| fell_over | 0 ✓ | — |

**评估**:❌ **GT 暖身没救动 selfloc**——加 [800,2500] 暖身后,撤拐杖段仍卡 11m,与无暖身 EXP1B 相同。
**坐实预判的 OOD 假设**:04 用 e2e fade[0,1] 训练,GT 位姿 obs 训练时本就≈0,现重新开 GT 对 04 是 OOD,
fresh RGB 分支无法借暖身学定位。踢球+进球+稳定仍从 04 完整继承(goal_rate 0.625、零摔倒)。

**结论**:**从 04 bootstrap 走不通 selfloc**——04 策略不"知道"怎么用 GT 位姿 obs。需 model_2800 起点
(训练时见过 GT 暖身、会纯视觉定位)。

**下一步 EXP1C(监督者状态机指定的 bootstrap 对照)**:同一 e2e_dualcam env + 暖身课程,改从
**model_2800** bootstrap。目的:拆分"起点不对"vs"env/感知链不对"。model_2800 会纯视觉定位(强项),
e2e 奖励层重新塑造踢球。门槛:selfloc<3m + dribble/goal 保住。

**状态**:EXP1B-v2 完成,判定"04 起点 selfloc 不可救(OOD)",转 EXP1C。

---

## Exp 1C — bootstrap 对照:model_2800 起点(监督者状态机指定)

**目标**:同一 e2e_dualcam env + GT 暖身 [800,2500],改从 **model_2800** bootstrap(而非 04)。
拆分"起点不对"vs"感知链不对":model_2800 训练时见过 GT 暖身(暖身对它 in-distribution),
会纯视觉定位;看能否同时拿到 selfloc<3m 和踢球(e2e 奖励层重塑)。

**理由**:EXP1B/v2 证明从 04 起点 selfloc 不可救(04 训练时 GT obs≈0,暖身重开 GT 是 OOD)。
model_2800 是在 GT 暖身调度下训练的,策略分布与"GT obs 存在"兼容——这是 EXP1C 的真正变量。

**做法**:脚本 spike_v4_e2e_dualcam_m2800.py(仅改 BASE_CKPT→model_2800)。smoke:depth reinit 0、exit 0。
注:model_2800 的 RGB CNN 此处也 fresh(相机 64→96 维度变),变量是 trunk+head 的 GT 兼容性。

**门槛**:selfloc<3m + dribble/goal 保住(goal_rate≥0.2、fell_over=0)。

**⚠️ GPU 争用记录**:启动时检测到外部进程 `scripts/train_custom.py`(PID 295424,非本会话启动)
占用 GPU,两进程争用致 EXP1C ETA 虚高(~5h44m vs 常态 1h40m)。**按安全准则不擅自杀外部进程**,
EXP1C 慢速运行。若持续过慢将在记录中说明窗口约束。

**状态**:训练中。

**结果(分阶段 selfloc,关键)**:
| 阶段 | selfloc min |
|---|---|
| GT全在(<800) | **7.07m** ← GT 直接在 obs 里却仍 7m |
| 暖身中(800-2500) | 5.29m |
| 撤拐杖(>2500) | 9.20m(末 10.6m) |
goal_rate best 0.50@1561、fell_over 0(踢球+稳定仍继承)。

**决定性诊断(排除 bootstrap 假设,定位真根因)**:
三个起点(04 / 04+暖身 / model_2800+暖身)selfloc 撤拐杖后**全卡 ~10m** → **问题不在 bootstrap 起点**。
更关键:对比**同为 model_2800 起点、同 selfloc head(都 reinit 0 迁移)、GT 都在 obs**:
- v3 realspec(**全场几何**)GT全在期 selfloc = **2.80m**
- v4 EXP1C(**进攻半场几何**)GT全在期 selfloc = **7.07m**
唯一大差异是**出生几何**,差 2.5 倍。**坐实根因:不是 selfloc 通路坏/起点错/没学会视觉,而是
进攻半场几何下定位本质就难**(位姿多样性低、地标可见性差)——这正是 v3 识别过的"几何张力"
(踢球要近球门聚焦 ↔ 定位要全场多样性)在 v4 重现。监督者让走 04 同口径是对的(暴露真问题),
但 04 同口径的几何本身就是 selfloc 瓶颈。

**下一步 EXP2(对症:几何课程)**:出生分布从全场渐变到进攻半场,让 selfloc 先在多样几何学好定位,
再聚焦进球。这是解几何张力的正路,而非继续换 bootstrap/暖身。门槛:selfloc<3m + goal_rate≥0.2。

**状态**:EXP1C 完成,根因从"bootstrap/暖身"澄清为"进攻半场几何对 selfloc 是毒药",转 EXP2 几何课程。

---

## Exp 2 — 几何课程(对症几何张力,EXP1C 诊断的正路)

**目标**:出生几何从全场(x∈[-9,9],yaw±π)随训练 morph 到进攻半场(x∈[-2,8.5],yaw±1.2),
与 GT 暖身 [800,2500] 同步。让 selfloc 先在易几何(全场,位姿多样)学好,再随聚焦进球收紧几何。
门槛:selfloc<3m + goal_rate≥0.2 + fell_over=0。

**理由**:EXP1C 决定性诊断——同 model_2800+GT 在,全场 2.80m vs 进攻半场 7.07m,根因是几何张力
(①定位要全场多样 ↔ ③进球要近球门聚焦)。几何课程在时间上化解张力,而非固定二选一。

**做法**:新 curriculum `spawn_geometry_curriculum`(全场↔进攻半场逐键线性插值 pose_range)+
新 env `mos92_soccer_e2e_dualcam_geomcurric_env_cfg`,从 model_2800 bootstrap。smoke:课程激活、
depth reinit 0、exit 0。GPU 外部进程已退,全速运行。

**状态**:训练中。

**结果(分阶段 selfloc + 总指标)**:
| 阶段 | selfloc min |
|---|---|
| 全场期(<800) | **6.47m** ← 全场几何也没到 v3 的 2.80m! |
| morph(800-2500) | 5.65m |
| 进攻半场(>2500) | 9.01m(末 10.6m) |
goal_rate best 0.50@1513、fell_over 后期 0.22(略升)。

**评估(负结果 + 复盘,触发监督红线第4条:停下复盘)**:
❌ 几何课程没用,且**全场期 6.47m 连 v3 realspec 全场期的 2.80m 都没复现**。这**推翻"几何是唯一根因"**:
若几何是唯一因素,EXP2 全场期应接近 2.80m。

**逐变量复盘 v3 realspec(2.80m) vs v4 EXP2(6.47m),同为 model_2800+全场+GT在**:
- ❌ 假设"fresh RGB CNN 是根因":**排除**——两者 camera_rgb CNN 都 reinit(都 fresh,6帧 stride 通道12→18)。
- ⏳ **剩余最大差异:任务结构**。v3 realspec=纯 selfloc 任务(无进球层);v4 EXP2=e2e(goal_progress/
  goal_scored/time_penalty + dribble 命令)。**新候选根因:e2e 进球奖励层稀释了 selfloc 梯度**——
  v3 全力学定位,v4 同时踢球进球,selfloc 只是众多目标之一。

**下一步(避免盲目再训,先便宜诊断)**:用 EXP3 区分——但先做诊断,不连续"换一个变量再训"。
候选:对比 selfloc_accuracy 权重(v3=0.8 已知;查 v4 e2e 是否被进球层压低)。

**便宜诊断结论(坐实根因,无需再训)**:reward 权重对比——
- v3 realspec(纯 selfloc 任务):selfloc_accuracy=0.8,几乎是唯一认知目标。
- v4 EXP2(e2e):selfloc_accuracy=0.8,但要和 dribble_to_target(3.0)+dribble_success(5.0)+
  goal_progress(2.0)+goal_scored(5.0)+dribble_approach(1.5) 共约 **16.5 总权重的踢球/进球奖励竞争**。
- **根因坐实:e2e 多目标稀释了 selfloc 梯度**(与几何无关)。v3 exp8 同款教训:selfloc 0.3→0.8 是为
  对抗被其它信号 scatter;v4 目标更多,0.8 仍不够。

**下一步 EXP3**:提 selfloc_accuracy 权重(0.8→2.5),让定位在 e2e 多目标里有足够梯度压力;保留几何
课程(全场期略好、无害)。门槛:selfloc<3m(尤其全场期回到 ~2.8m)+ goal_rate≥0.2 + fell_over=0。
若提权重后 selfloc 回到 ~2.8m → 坐实稀释根因;若仍卡 → 任务本质冲突,转 EXP4(分离 selfloc 预训练)。

---

## Exp 3 — 提 selfloc 权重(对症 EXP2 诊断的梯度稀释)

**目标**:selfloc_accuracy 0.8→2.5,保留几何课程,从 model_2800 bootstrap。让定位在 e2e 多目标
里有足够梯度。门槛:selfloc<3m(尤其全场期回到 ~2.8m)+ goal_rate≥0.2 + fell_over=0。

**理由**:EXP2 便宜诊断坐实——e2e 里 selfloc(0.8)被 ~16.5 总权重的踢球/进球奖励稀释,全场期
6.47m vs v3 纯 selfloc 2.80m。提权重对抗稀释(v3 exp8 同款逻辑)。

**做法**:脚本 spike_v4_e2e_selflocw.py 设 selfloc_accuracy.weight=2.5。smoke:权重 2.5 生效、
depth reinit 0、exit 0。

**判别意义**:若 selfloc 回 ~2.8m→坐实稀释根因;若仍卡→任务本质冲突,转 EXP4(分离 selfloc 预训练/PnP)。

**状态**:训练中。

**结果(分阶段,关键突破+暴露下一层)**:
| 阶段 | selfloc min |
|---|---|
| 全场期(<800) | **2.13m** ← 提权重后回到 v3 水平!✓ |
| morph(800-2500) | **1.81m**(best) |
| 进攻半场(>2500) | 5.12m → 末 8.37m ← 几何收紧又塌 |
goal_rate best **0.60@1569**、fell_over 后期 0、dribble_success best 1.83。

**评估(坐实稀释根因 + 隔离出最后一层)**:
✅ **稀释根因确认**:提 selfloc 权重 0.8→2.5 后全场期从 6.47m **回到 2.13m**(best 1.81m),复现 v3 水平。
证明 EXP2 学不好确实是 e2e 多目标稀释 selfloc 梯度,与几何/bootstrap/分辨率无关。**真正进展。**
⚠️ **最后一层(干净隔离)**:selfloc 在全场/morph 期 1.8-2.1m,但几何收紧到纯进攻半场(>2500)
塌回 5-8m。与 v3 对照(model_2800 进攻半场 4.65m)一致——**这是进攻半场几何的真实定位上限,非 bug**:
近球门、朝球门、位姿单一,可见地标本就少。

**完整诊断闭环**:v4 selfloc 问题 = ①梯度稀释(已解,提权重)+ ②进攻半场几何物理上限(~5m,真实约束)。

**下一步 EXP4 选项**:(a) 几何课程**终点不收到纯进攻半场**,保留一定全场比例(如 alpha 上限 0.6),
让定位维持多样性同时仍能进球——直接对症"几何收紧致塌";(b) 距离自适应精度——远处(地标少)放松
selfloc 要求,近球门用其它信号。先做 (a),改动小且直接。

---

## Exp 4 — 几何课程封顶 alpha_max=0.6(对症几何收紧致塌)

**目标**:几何 morph 终点封在 alpha=0.6(保留 40% 全场多样性),叠加 EXP3 的 selfloc 权重 2.5。
让 selfloc 维持低误差(全场期已证 2.13m)同时仍有足够近球门 episode 进球。
门槛:selfloc 全程<3m(尤其后期不塌回 5m+)+ goal_rate≥0.2 + fell_over=0。

**理由**:EXP3 坐实——梯度稀释已解(提权重后全场 2.13m),但纯进攻半场几何把 selfloc 物理限到 5-8m。
封顶保留多样性直接对症"几何收紧致塌",是完整诊断闭环后的对症修复。

**做法**:curriculum 加 alpha_max 参数;脚本 spike_v4_e2e_geomcap.py 设 alpha_max=0.6 + selfloc 2.5。
smoke 通过。

**状态**:训练中。

**结果**:selfloc best=2.04@1363,末 8.66m;封顶 0.6 **未阻止塌陷**(末值比 EXP3 还略差)。
goal_rate best 0.583、fell_over 后期 0。

**评估(封顶假设被推翻,触发跨实验复盘)**:封顶保留 40% 多样性没用——推翻"几何终点太集中"假设。

**决定性跨实验对齐(6 个 v4 e2e run 的 selfloc best@iter)**:
| run | best@iter | 末值 |
|---|---|---|
| selflocw(权重2.5) | 1.81@**1132** | 8.37 |
| geomcap(+封顶0.6) | 2.04@**1363** | 8.66 |
| m2800 | 5.29@**1032** | 10.59 |
| geom | 5.65@**1291** | 10.64 |

**最终根因(决定性)**:所有低 selfloc 的 best **全部出现在 iter 1032-1363**,即 GT 暖身窗口 [800,2500]
**前半段(GT 仍大量存在)**;撤光 GT 后(>2500)**全部塌到 8-10m**。这是 6 个实验的共同模式,与几何/
权重/封顶无关。**真根因:fresh RGB CNN 在 e2e 任务下无法在 GT 撤光后独立维持纯视觉定位**——GT 在时
靠 GT obs 到 1.8m(假象),撤 GT 后 RGB CNN 没真正学会从 0.125m 真场线视觉定位就塌。与 v3 realspec
末期反弹 6.5m 同现象。

**结论**:fade 调度/几何/权重调参已连续证明无效(监督红线:连续同向失败换机制)。需转**机制级方法**:
EXP5 = 几何方法(关键点检测+PnP),或先验证"RGB CNN 到底能否从真场线学到视觉定位"(隔离 RGB 分支
单独预训练,不被 e2e 多目标干扰)。这是监督者 EXP4 状态机预留的"高分辨率回归仍卡→上几何方法"触发条件。

**状态**:EXP4 完成,最终根因锁定为"fresh RGB CNN 撤 GT 后纯视觉定位维持不住",转机制级方法。

---

## Exp 5a — 判别实验:纯 selfloc + 高分辨率 RGB(隔离分辨率变量)

**目标**:纯 selfloc 任务(无 e2e 多目标干扰)+ RGB 提到 160×120(2.78x 像素)+ 分离相机,
看撤 GT 后 selfloc 能否维持 <3m。判别"分辨率是否是撤 GT 后维持的关键"。

**理由**:跨实验对齐发现所有 v4 selfloc best 都在 GT 暖身期、撤 GT 后塌。EXP1(纯 selfloc dualcam 96×72)
也塌(1.88→6.73m),说明 96×72 信号太弱、RGB CNN 学不到能撑过撤 GT 的视觉定位。高分辨率视角图证
0.125m 线高分辨率下可辨。本实验隔离分辨率:纯任务+高分辨率,撤 GT 后能否维持。

**做法**:脚本 spike_v4_selfloc_hires.py,RGB 相机 160×120,NUM_ENVS 384→192(显存)。smoke:
camera_rgb (18,120,160)、depth reinit 0、exit 0、无 OOM。

**判别意义**:撤 GT 后维持<3m→分辨率是关键,后续 e2e 也提分辨率;仍塌→信号/架构问题,转 EXP5c(PnP)。

**状态**:训练中。

**用户决策(2026-06-08,分辨率路线)**:160×120 仍属低分辨率。计划:等 EXP5a 跑完看撤 GT 后表现,
然后用**中分辨率 384×288 验证**(比 160 高 6x 像素、比 960 省很多)。960×960 直接套现有架构跑不起来
(133x 显存,仅能 2-3 envs),需中分辨率先验证"分辨率单调有用"再决定是否冲更高。
- **下一步 EXP5b = 中分辨率 384×288**:6帧栈降到 3 帧、envs 降到 ~96(显存)。纯 selfloc 任务,
  看撤 GT 后 selfloc 是否随分辨率单调改善。

**监督者 09:42 留言 + 判别分析(已采纳)**:
- 监督纠偏:EXP5a 当前 2.5m 不算成功(mask_factor≈0.88,GT 未撤净)。真判据=GT fade≈0 后(iter>2500)
  稳定<3m。**与我落盘的"撤GT后维持"判据一致。** 监督红线:GT fade 后仍塌→不再做 fade/geometry/reward
  小调参,直接转关键点+PnP(必要时 MCL)。
- **协调用户(中分辨率验证)vs 监督者(无效转PnP)的判别逻辑**:
  - 用 EXP5a 撤 GT 后**相对 96×72 基线(EXP1:撤GT后 min 2.05→末 6.73)是否好转**作分叉判据:
  - 若**明显好转**(撤GT后塌得更轻/维持更久)→ 分辨率路线有效,按用户意见上中分辨率 EXP5b(384×288),暂不转 PnP。
  - 若**与 96×72 同样塌、无改善**→ 分辨率路线证伪,按监督者直接转 PnP(EXP5c),跳过中分辨率。
  - 这样既尊重用户分辨率验证意愿,又守监督红线(无效即换机制)。

**结果(撤 GT 后,真判据)**:
| | 撤GT后 min | 撤GT后 末值 | mask_factor末 |
|---|---|---|---|
| EXP1 96×72 基线 | 2.05m | 6.73m | 0 |
| EXP5a 160×120 | 2.36m | 6.68m | 0 |

**评估(判别分析,采纳监督者真判据)**:GT 完全撤光后(mask=0),160×120 与 96×72 **几乎完全相同**
(都从 ~2m 塌到 ~6.7m)。**160×120 相对 96×72 无好转。**

**判别决策(协调用户中分辨率意愿 + 监督者无效转PnP红线)**:160×120 仅比 96×72 高 2.78x 像素,
跳跃太小,不足以判定"分辨率路线整体无效"。384×288(6x 像素、高分辨率视角图证 0.125m 线更清晰)
是更公平的**决定性判别点**。故按用户指示跑 EXP5b(384×288)作分辨率假设的最终测试:
- 若 384×288 撤 GT 后**明显好转**→ 分辨率有效,继续走分辨率(必要时配 patch/单帧高清架构冲更高)。
- 若 384×288 撤 GT 后**仍塌到 ~6.7m**→ 分辨率路线**证伪**,直接转 EXP5c(关键点+PnP,监督者预留)。

**状态**:EXP5a 完成(160×120 无好转),转 EXP5b 中分辨率决定性测试。

## Exp 5b — 中分辨率 384×288 决定性判别(分辨率路线最终测试)

**目标/理由/做法**:见上。384×288(6x于160、16x于96)是分辨率假设的公平判别点。48 envs(96 envs OOM,
384×288 CNN 激活太大)。

**结果(撤 GT 后,决定性)**:
| 分辨率 | 撤GT后 min | 末值 |
|---|---|---|
| 96×72(EXP1) | 2.05m | 6.73m |
| 160×120(EXP5a) | 2.36m | 6.68m |
| **384×288(EXP5b)** | **3.15m** | **6.51m** |
mask_factor 末=0(GT 全撤),best 2.89@2173(暖身期)。

**评估(分辨率路线证伪,满足用户验证+监督红线)**:三个分辨率跨 16x 像素,撤 GT 后**全塌到 ~6.5m,
无改善**;384 的 min 反因 48 envs 样本少而略差。**提分辨率没让 RGB CNN 学会撑过撤 GT 的视觉定位。**
**根因确认:瓶颈不是分辨率,而是"回归式 CNN 从 RGB 直接学位姿"范式本身**——学不到脱离 GT 拐杖的稳定映射。
（既满足用户"中分辨率验证"要求,也满足监督者"分辨率无效→转 PnP"红线。）

**下一步 EXP5c(监督者状态机 EXP4_GEOMETRIC_LOC)**:关键点检测+PnP。CNN 检测场地关键点(角点/罚球点/
中圈交点),用已知地图 3D 坐标解 PnP 得位姿。几何约束硬、精度上限高、sim 关键点 GT 可自动生成监督。
必要时加 MCL 解对称歧义。

**状态**:EXP5b 完成,分辨率路线证伪,转 EXP5c 几何方法(关键点+PnP)。

## Exp 5c — 关键点检测 + Depth + Kabsch(用户选定几何架构)

**地基验证(已实测确认)**:
- depth 是**米制**(head_cam mean 4.39m,max 1032=远裁剪面)→ 可抬 3D。注意:camera_depth obs 归一化到[0,1]
  不能用于抬3D,须用 `sensor.data.depth` 原始米制值 + 屏蔽远裁剪面(>cutoff)像素。
- 相机外参 `sim.data.cam_xpos`(num_envs,num_cams,3)+`cam_xmat` 可取 → 可投影已知3D关键点成2D监督标签。
- 相机内参 fovy=60°、64×48(depth)→ K 矩阵可推。

**数据通路(绕开渐隐拐杖的关键)**:
已知场地关键点3D坐标 →(相机外参投影)→ 图像2D监督标签 → 训练检测头(逐帧稠密监督,**无需GT渐隐**)。
推理:检测头出2D像素 →(depth抬)→ 相机系3D → 与地图3D做Kabsch刚体配准 → 机器人位姿。

**为何能解前面所有实验的塌陷**:之前回归式CNN靠GT位姿obs"抄答案",撤GT就塌。关键点检测的监督是
"像素坐标"(每帧由几何投影自动生成、永不撤除),CNN学的是稳定的视觉特征→像素映射,位姿由几何确定性
求解(Kabsch非学习),从根上不依赖会被撤掉的拐杖。

**关键点选取(地图3D已知)**:场地四角、罚球区角、中圈与中线交点、球门柱根部等几何显著点。

**状态**:地基已验证,开始实现(关键点3D表 → 投影标签 → 检测头 → Kabsch 求解 → reward 接入)。

**几何模块(field_keypoints.py)已实现 + 闭环自检通过**:
- `field_keypoints_3d`:23 个地图已知关键点(四角/罚球区角/球门区角/中线端点/中圈交点/罚球点/球门柱根)。
- `project_keypoints`:3D→图像2D(生成检测头逐帧监督标签 + 可见性掩码)。
- `lift_pixels_to_world`:2D像素+米制depth→世界3D。
- `kabsch_se2`:加权刚体配准 src→dst 解 (yaw, t),闭式非学习。
- 闭环自检(合成位姿):visible 6/23、lift_err **0.0000m**、Kabsch 恒等正确 → 三函数坐标约定自洽。

**下一步**:真实数据验证——用 sim 真实 cam_xpos/cam_xmat + 真实米制 depth 跑全链,Kabsch 复原位姿对比
`robot_field_pose` GT。过关后再建检测头(CNN+spatial_softmax 出像素,投影标签监督)。

**几何链真实sim验证(非循环,决定性里程碑)**:用 sim 真实相机外参 + 米制 depth,经
相机系3D→**基座系**(运动学已知,不用世界位姿)→Kabsch,复原位姿对比 robot_field_pose GT:
| env | 可见点 | 复原(x,y,yaw) | GT |
|---|---|---|---|
| 0 | 9 | 1.07,0.37,-0.04 | 1.07,0.37,-0.04 |
| 1 | 7 | 0.83,-4.41,1.06 | 0.83,-4.42,1.06 |
| 2 | 7 | 1.53,-3.06,0.75 | 1.53,-3.06,0.75 |
| 3 | 6 | 2.99,2.85,-0.76 | 2.99,2.86,-0.76 |
**误差 ~1cm,4 视角全中**。证明整条几何链坐标约定正确,且非循环(基座系推理,真机可复现)。

**关键结论**:关键点检测准确时,几何法精确复原位姿(<3cm ≪ 3m 目标)。**唯一待学=CNN关键点检测器**,
监督为每帧投影标签(永不撤除、无渐隐拐杖)→ 从根上绕开前6实验的撤GT塌陷。

**下一步**:建关键点检测头(CNN+spatial_softmax 出 K 个像素坐标),用 project_keypoints 标签 +
可见性掩码做监督损失;推理时检测像素→depth抬→基座系→Kabsch→位姿,接入 selfloc obs/reward。

**EXP5c 启动(关键点检测+Kabsch,范式级)**:smoke 通过(action 46、head 加宽 24→66、kp_pixel_err
随机起点 1.5、修复 weight=0 监控被跳过→改 1e-8、kp_selfloc_pos_err_m 正常输出)。全量启动:iter40
kp_pixel_err 2.02、kp_selfloc_pos_err_m 5.64m(未训检测头随机预测)。ETA ~2h。
- **判据**:训练后 kp_pixel_err↓→关键点检测变准→几何链(已证检测准时~1cm)→kp_selfloc_pos_err_m
  应稳定<3m **且撤GT概念后不塌**(本就无GT obs拐杖,监督是每帧投影标签)。
- 若达标→v4首个真场线纯视觉定位方案,补 checkpoints/ + soccer_eval/。

**EXP5c 结果判读(reward gaming,非范式失败)**:
| 指标 | best | 末值 |
|---|---|---|
| kp_pixel_err | 0.014@3744 | 0.042(撤GT段min0.014/max0.093,**不塌**) |
| kp_visible | 4.02@4 | **0.059(崩溃)** |
| kp_selfloc_pos_err_m | 5.3@24 | 11.2 |
| goal_rate | 0.111 | 0 |

**根因(代码可证)**:keypoint_detection reward=exp(-masked_err/std²),分母 clamp(min=1.0)。可见点→0 时
masked_err→0→reward→1 白拿。policy 学会**转头不看场地**(看不见=免费满分),位姿无点可解→11m。
**与前6实验塌陷本质不同**:检测范式有效(kp_pixel_err 全程低、撤GT不塌,离线几何已证~1cm),
失败仅在**奖励漏洞**,非感知能力。属奖励设计bug,不是范式问题——故做1处对症修复,非小调参循环。

**EXP5d 修复**:reward 改 `(visible/K)×exp(-err/std²)`,可见点→0 时奖励→0(堵死免费午餐),
施加"既看见又看准"双向梯度。若修复后 kp_visible 维持且 pos_err<3m→达标;若 gaming 转移→重审任务结构。

## EXP5d 趋势判读(visible-gated reward)+ 纯视觉边界审查 + codex 18:31 建议采纳

**判读工具**:新增 `scripts/extract_kp_metrics.py`,训练结束即客观抽取 current/best/best@iter/后段趋势
(kp_visible / kp_pixel_err / kp_selfloc_pos_err_m / goal_rate / fell_over + keypoint_detection reward)。

**EXP5d 后段判读(iter~3295,接近完成)**:
| 指标 | cur | best@iter | 趋势 | 判定 |
|---|---|---|---|---|
| kp_visible | 4.41 | 4.91@3259 | 稳定3.9-4.9 | ✓ 可见性修复成功(EXP5c崩到0.06) |
| kp_pixel_err | 2.05 | 0.46@380 | ↑变差 | ✗ 检测精度退化卡~2 |
| kp_selfloc_pos_err_m | 6.91 | 6.09@155 | 卡6-7m | ✗ 远未达<3m |
| goal_rate | 0 | 0 | → | ✗ 不踢球 |
| keypoint_detection(reward) | **0.0000** | 全程 | → | ✗ 奖励饱和无梯度 |

**根因(数学可证)**:std=0.15,reward=(vis/K)·exp(-err/0.15²)。pixel_err≈2 → exp(-2/0.0225)=exp(-89)≈0。
**检测精度部分梯度完全消失**,policy 只学到 visible/K(可见性维持但精度学不动)。即 codex 警示的
`keypoint_detection:0.0000` 风险——**RL reward 不适合学 23×2 维稠密回归**(codex 也独立提出此点)。

**纯视觉边界审查(codex 18:31 红线)**:
- ✓ 合规:keypoint env 的 selfloc_gt_mask fade 区间 _E2E_FADE_START=0→END=1,GT 位姿 obs(ball_to_target
  槽位持 robot_field_pose)从 iter1 即乘 0(mask_factor=0.0000 已证)。策略训练**不吃 GT 位姿**。
- ✓ 可还原非特权:几何链 base-frame 验证只依赖 cam→base 相对变换(真机标定/运动学可得);world cam pose
  仅用于训练标签投影 + monitor,codex 明确允许作监督信号、非部署输入。
- 待办:正式评估须给出完全不含 world pose 的推理路径(用标定 cam→base)。

**采纳 codex 建议 → EXP5e 方向**:把关键点检测从纯 RL reward **拆为监督学习辅助 loss**(稠密视觉任务本质),
或先增大 std 让梯度回来。下一步定方案。

**EXP5d 最终(iter3799,确认 iter3295 判断)**:kp_visible 终4.69/best4.98@3723(✓修复)、
kp_pixel_err 2.27、kp_selfloc_pos_err_m 6.67/best6.09@155、goal_rate 0、keypoint_detection reward 全程0。
确认:可见性漏洞已修,但 PPO 学不动46维稠密检测。下一步 EXP5e 二选一,先查 rsl_rl 注入监督loss 可行性。

## EXP5e — std 0.15→1.0(廉价证伪:PPO 能否在健康梯度下学稠密检测)

**目标**:仅改 keypoint_detection 的 std=0.15→1.0,让奖励核不饱和(smoke 确认 reward 0.0000→0.0030),
看 PPO 在有梯度时能否把 kp_pixel_err 降下来、pos_err<3m。
**理由**:EXP5d reward 全程 0 是核饱和(err~2→exp(-89)≈0),非 gaming。std=1.0 把梯度铺到 err∈[0,1.5]。
**判据(廉价证伪)**:若 pixel_err 明显下降且 pos_err 改善→PPO 可学,继续调;若仍卡~2→**坐实 PPO 不适合
46维稠密回归**,转 EXP5f(监督 aux-loss,需按 rsl_rl 的 rnd/symmetry 模式接入,较侵入)。
**rsl_rl 调查结论**:辅助 loss 有 rnd/symmetry 先例,但需独立模块+optimizer 接入 PPO 构造器,改动较大,
故先做 std 廉价证伪再决定是否投入。
**早期(iter43)**:keypoint_detection 0.0025(梯度回来✓)、fell_over 8.3 偏高(待观察)。ETA ~1.8h。

**EXP5e 结论(用户决定停于 iter511,转监督loss)**:
- std=1.0 后 kp_pixel_err 降到 0.50(✓梯度恢复,证实 EXP5d 是核饱和)。
- 但 **fell_over 暴涨到 32-37**(EXP5d~1)、kp_visible 掉到 1.6、pos_err 仍 6.5m、goal_rate 0。
- **坐实**:PPO 用标量奖励学 46 维稠密检测,会与 upright/dribble **抢梯度**(检测 reward 占比一大就压垮本体)。
  这正是 codex + 我共同判断"关键点检测应拆为监督 loss"的根据——监督 loss 不进 PPO 奖励混合,不挤压本体。
- (进程清理注:pgrep -f "spike...py" 会误匹配等待器循环命令里的同名字符串,确认进程死活应看 `ps`+GPU占用。)

## EXP5f — 关键点检测拆为监督 aux-loss(codex 主张,范式内最终形态)
**目标**:关键点检测改用 project_keypoints 投影标签做 dense supervised loss(可见性掩码),与 PPO 解耦,
不进奖励混合;Kabsch 几何照常解位姿。保住 upright/dribble。
**理由**:EXP5c/d/e 三轮证明 RL reward 学稠密检测必失败(gaming→饱和→抢本体梯度)。监督学习是稠密视觉
任务的正确工具。
**待定实现**:按 rsl_rl 的 rnd/symmetry 辅助 loss 模式接入(独立 head + 投影标签 loss + 自己的 backward)。
下一步:评估最小侵入改法。

## EXP5f A' 启动(监督 aux-loss,本地 KeypointAuxPPO,不碰 pip 包)
**实现**:① obs 函数 keypoint_uv_label 每帧出 (K*3,) 投影标签(交错 uv+vis,不可见点uv置0);
② 训练专用 obs 组 keypoint_label 进 rollout storage、不入 actor/critic obs_groups(策略不吃标签,codex 机制已验证);
③ 本地 KeypointAuxPPO(继承PPO)在 update() 前跑独立监督 pass:masked smooth_l1(预测selfloc slice[20:66] vs 标签),
PPO优化器,梯度流入共享CNN+检测头;④ class_name 点路径指向本地类。
**调试**:smoke 先遇 shape bug(actor forward 漏 stochastic_output=True→读到陈旧分布参数)→修;
再遇 kp_aux=6M(标签布局 [ux..,uy..] 与 reshape(K,2) 交错错位→掩码错乘)→改交错布局+不可见置0→loss 0.20 正常。
**早期(iter28)**:kp_aux 0.14、pix_err 0.47、pos_err 6.41m。ETA ~2.6h。
**判据**:kp_aux_pix_err 持续降→kp_selfloc_pos_err_m<3m **且 fell_over/goal_rate 不退化**(监督loss不进奖励,
不应再抢本体梯度——这是 A' 相对 EXP5e 的核心优势)。

## EXP5f 判读 + 范式级根因(2026-06-08,决定全局)
**EXP5f 被污染**:残留 EXP5e 的 keypoint_detection reward(weight2.5/std1.0)与新 aux loss 叠加,
fell_over iter322 已 52(>EXP5e 37),非干净测试,已停。
**但暴露范式级根因(网络结构已证)**:actor 是**单一共享 MLP trunk** 259→512→256→128→66,
mlp.6 同层吐 joint_pos[0:20]+selfloc[20:66]。→ **任何感知学习信号(reward 或 supervised loss)都经共享
trunk 反传污染步态**。这解释 fell_over 随感知梯度强度单调恶化(5d≈1→5e 37→5f 52)。
**"reward vs loss"是症状,共享网络主干才是病根**。6 实验一直在和此耦合搏斗而未命名。

## 范式转向:模块分离 + oracle 优先(采纳 codex 第13/14节)
**架构必须翻转**:当前策略把关键点/位姿当 46-d action 预测(感知在 actor trunk 内);
需改为 detector→几何/belief 估计器产出**位姿 belief**,belief 作为 **obs 喂给 soccer policy**(感知移出 trunk)。
**EXP6 = GT-landmark oracle(codex #1,最高性价比首步)**:喂投影关键点+可见性(**不喂 robot pose**),
Kabsch 出单帧 belief→belief 摘要进 policy obs。无 learned detector。
- 一石二鸟:① 验证后端(Kabsch→belief→主动扫视→踢球)在"感知完美"时是否可行(codex 诊断顺序1);
  ② oracle 供感知→**无感知梯度穿过步态 trunk**→直接消除 fell_over 污染源。
- 判据:oracle 下 pos_err<1m + goal_rate≥0.2 + fell_over≈0。若仍不行→病在融合/动作/协调,不在 CNN。
**后续阶梯**(codex):noisy-oracle 给误差预算→learned detector+frozen belief→+active perception→e2e 微调。
**待用户定**:入口选 oracle-first(推荐),还是先建模块骨架,还是先离线验 belief。

## EXP6 oracle 中期判读(iter ~1100/3000,2026-06-08)
**结构性假设证实(本轮核心成果)**:感知移出 actor trunk(belief 作 obs、非 action)后,
**fell_over 从 EXP5f 的 52 → 0.2~0.7**。步态污染根源(感知梯度穿共享 trunk)彻底消除。
6 轮实验追的结构性病根已解。
**但坐实真瓶颈(= codex #2/#3 预言)**:
- selfloc_pos_err_m ≈ 4.5m(远差于静态校验的可见时 1.1m);belief_vis_frac 0.17(走动+前向窄视角
  只见 ~4/23 关键点)→ Kabsch 欠定 → belief 误差大。
- goal_rate 仍 0:belief 不准→无法定位瞄门。
**结论**:单帧几何 + 窄视角不够。**必须上 ① 时序融合(EKF/MCL,跨帧累积关键点降低 vis 依赖)
+ ② 主动转身扩大视野**。这不是架构失败(fell_over 已证架构对),是 codex 分层路线的下一阶段入口。
**下一步 EXP7**:在 belief obs 上叠加时序滤波(滑窗多帧关键点累积→Kabsch/EKF),先离线验 multi-frame
pos_err < single-frame,再接训练。仍 oracle 检测(隔离检测器变量)。

## EXP7 离线多帧融合 gate(2026-06-08,决定性洞察)
对 EXP6 model_2999 rollout 120帧,对比单帧 vs 多帧(odometry 完美融合,窗口10)Kabsch:
| | pos_err mean | median | 可见点 | **唯一关键点(满分23)** |
|---|---|---|---|---|
| 单帧 | 2.57m | 1.24m | 3.9 | **3.9** |
| 多帧融合 | 2.23m | 1.06m | 43.5 | **4.4** |
改善仅 13.4%。

**决定性诊断**:10帧堆了43.5个观测,但**唯一关键点仅 3.9→4.4**——几乎全是同4个点的重复观测。
Kabsch 几何约束由唯一点决定,重复观测只做降噪(13%≈降噪量级),不增新几何方向。

**整个 v4 的决定性结论**:
- **被动时序融合无效**:机器人朝同方向走,视野永远是那~4个点,唯一覆盖上不去。
- **真正解锁点 = codex 主动视觉**:必须主动转身/扫视让相机指向不同关键点,唯一覆盖 4→15+,
  Kabsch 才良态,pos_err 才可能破 1m。
- 单帧 median 已 1.24m(好可见角度下接近1m),证明几何链本身对——缺的是**视野覆盖**,非精度。

**下一步 EXP8 = 主动视觉(跳过被动时序,直击根因)**:在 belief obs(含 vis_frac/residual 不确定性)
上,让策略学会"不确定时转身扫视"。codex 顺序:head scan→+body yaw→step-turn,奖励=唯一覆盖↑/
不确定性↓,惩罚 fell_over/丢球/jerk,安全课程先行。oracle 检测仍保留(隔离检测器变量)。

## EXP8 可行性上界诊断(2026-06-08,纯几何+sim 实测,确立 EXP8 价值)
追查 EXP7 矛盾"为何单帧只见~4点"。三步诊断:
1. **真实 project_keypoints CPU 扫 yaw**(含真实相机 mount quat):静止单朝向画面内 9-12 点(最佳朝向),
   全扫 yaw unique **22-23/23**。几何上界充足。
2. **sim 实测固定朝向**:画面内仅 **4.98 点**(策略追球,朝向几乎不变,只切到前向一片)。
3. **深度有效性排除**:画面内点深度 100% 有效(range 1.58-15.21m,median 6.58m,无一被 30m 掩码或
   3m cutoff 截断——cutoff 只作用 RGB-keypoint env 的 obs,不影响 belief 几何链)。

**完整证据链(EXP8 主动视觉价值确立)**:
① 几何允许全扫覆盖 23(上界足)② 深度全程有效(转到即可用)③ 当前策略不转身→卡 ~5 点→
Kabsch 欠定→pos_err 4.2m。→ **转身把 unique 5→20+ 是解锁 <1m 的真实且唯一路径**。
**EXP8 = 主动视觉确定值得做**(非被动时序)。瓶颈是行为(不转身),非几何/深度/精度。

**EXP8 设计**:belief obs 已含 vis_frac/residual 不确定性。加主动转身能力 + 不确定性驱动奖励
(unique 覆盖↑ / residual↓),安全课程先行(codex 红线:全身运动勿一次性放进踢球奖励)。
仍 oracle 检测(隔离检测器变量)。先想清动作空间:复用现有关节(neck/base yaw)还是加显式扫视动作。

## EXP8 动作空间核查 + neck 扫视上界(2026-06-08,确立分阶段设计)
**关键事实**:head cam 挂 robot/head,body 链 base→neck(neck_yaw±90°)→head(neck_pitch±0.5)→cam,
相机随 neck 转;neck_yaw/pitch 已在 actuator 动作空间内(无需加新动作)。
**真实投影测 neck-only 扫视 unique 覆盖**(body 朝+x 典型追球朝向):
| 位置 | neck中位(fixed) | neck±90°扫视 | 全身360° |
|---|---|---|---|
| (5,0) | 5 | 9 | 23 |
| (7,3) | 3 | 9 | 23 |
| (3,-5) | 4 | 11 | 23 |
| (-3,-4) | 7 | 14 | 23 |
neck 中位 3-8 ≈ sim 实测~5(模型验证)。

**EXP8 分阶段设计(契合 codex 红线 head→body→step 渐进)**:
- **阶段1 neck-only 主动扫视(最安全,优先)**:neck±90° 把 unique 5→9-14,**不动脚/不碰步态/不打断踢球**。
  可能已足够 Kabsch 破 1m。零步态风险。
- **阶段2**(若 neck 不够):加 body yaw 慢转,需安全课程(勿一次性放进踢球奖励)。
**分水岭**:先验证 neck-only 9-14 unique 够不够破 1m;够则 EXP8 全程不需危险全身动作。
**机制**:neck 已可控,EXP8 = 不确定性(vis_frac/residual)驱动的扫视激励 + 当前 pose reward 对 neck 的
回中拉力需放松(否则 neck 被拉回中位不扫视)。oracle 检测仍保留。

## EXP8 FusedPoseBelief obs 验证(2026-06-08,融合机制确认对)
实现有状态时序融合 belief obs（FusedPoseBelief,仿 StackedCameraRGB 的 CircularBuffer+reset 范式）:
N=8帧/stride=4(跨~0.64s),odometry 把各帧关键点累积到当前 base 帧→Kabsch,输出7维
[x,y,sin,cos,vis_now,uniq_frac,resid]。用 EXP6 model_2999 rollout 150帧验证:
| | pos_err mean | median | 覆盖 |
|---|---|---|---|
| 单帧 belief | 3.07m | 1.44m | vis_frac 0.177(~4点) |
| 融合 belief | 2.17m | **1.03m** | uniq_frac 0.230(~5.3点) |
改善 29.3%(median 1.44→1.03,已接近1m)。比离线脚本13%更好(stride=4 时间基线更长)。

**结论**:融合机制几何正确、有状态缓冲/reset 正常。被动行走下已 median 1.03m。但 uniq_frac 仅0.23——
因 EXP6 策略不主动扫视。**EXP8 训练解锁点**:策略学会 neck 扫视→uniq_frac 应冲 0.6+(13-14点)→
pos_err 大幅破1m。因果链闭合:融合对,缺主动扫视行为。
**EXP8 = FusedPoseBelief obs(就绪)+ neck 扫视激励 + 放松 neck 回中拉力**。

## EXP8 全量训练启动(2026-06-08 夜,主动扫视+时序融合)
实现并验证全部接线后启动:`scripts/spike_v4_e2e_fused_scan.py`,2000 iter,384 envs,
从 EXP6 model_2999 bootstrap。env=`mos92_soccer_e2e_dualcam_fused_scan_env_cfg`。
组件(均经 smoke 验证无误):
- FusedPoseBelief obs(8帧/stride4 odometry 融合,7维,actor obs 87→88)
- active_scan_coverage reward(weight 0.5,驱动 uniq_frac→0.6,只能靠转 neck 达成)
- neck_yaw pose std 0.15→1.5(放松回中,允许扫视)
- fused_belief_error 监控(weight 1e-8,记录融合 belief 真实 pos_err + fused_uniq_frac)
smoke 8 iter 已见 pos_err 4.2→1.81m。**判读三判据**:① scan_uniq_frac 是否升>0.4(学会扫视)
② selfloc_pos_err_m 是否破<2m 趋向1m ③ fell_over~0 且 goal_rate 不崩(扫视不吃稳定/踢球,codex红线)。
等待器 bndxy1o3a 挂着。日志 /tmp/v4_exp8_full.log。
**关键实现细节**:有状态 obs term 注册传实例 func=mdp.FusedPoseBelief(None,...)(env 惰性注入),
仿 StackedCameraRGB;reward 经 om._group_obs_term_cfgs[grp][idx].func 拿 term 实例读缓存,不重算几何。

## EXP8 结果 + 根因(2026-06-08 夜)
2000 iter 完成。三判据:
- ③ fell_over ~0.1 ✅(扫视没破坏步态)
- ② selfloc_pos_err_m(融合)4.2→**1.8m** 🟡(大幅改善但卡 1.8m,未破 1m)
- ① **scan_uniq_frac ~0.24 平,没升 ❌**(策略没学会主动扫视)
- goal_rate 偶发 0.05-0.11 🟡
pos_err 降到 1.8 几乎全来自**被动融合**(smoke 第8 iter 已 1.81m),主动扫视行为未涌现。

**根因(确凿)**:`gaze_center`(weight **1.0**,让 neck 对准球保持球在画面中心)+ `gaze_search`(0.5,
球不可见时转向球)与 `active_scan`(0.5,扫场地找地标)**争夺同一个 neck 关节**。单 neck 盯球与扫视
物理互斥,盯球奖励(1.0+0.5)碾压扫视(0.5)→ 策略选盯球,neck 不扫,uniq 卡 0.24。
证据:Reward/gaze_search 0.276 > active_scan 0.178。

**这是奖励冲突,非架构问题**。解法=codex 预见的**不确定性门控时间分工**:belief 差→扫视找地标
(压制 gaze);belief 好/将踢球→盯球。**EXP9 设计**:① gaze_center/search 按 belief 确定性门控
(uniq_frac 高才奖励盯球)② active_scan 仅 belief 不确定时强激励 ③ 二者时间互补而非同时竞争。

## EXP9 启动(2026-06-09,不确定性门控扫视)
针对 EXP8 根因(盯球奖励碾压扫视)的修复:`scripts/spike_v4_e2e_gated_scan.py`,2000 iter,
384 envs,从 EXP8 model_1999 bootstrap(同7维 belief,全载无 reinit)。
env=`mos92_soccer_e2e_dualcam_gated_scan_env_cfg`。改动:
- gaze_center/search → gated 版(乘 clamp(uniq_frac/0.4) 门:belief 差时盯球不给分)
- active_scan weight 0.5→1.0(强化扫视梯度)
smoke 验证门控生效:gaze_center reward 0.0037(被门压到~0)< active_scan 0.030,扫视首次主导。
**判据**:① scan_uniq_frac 是否升>0.4(门控解锁扫视)② pos_err 是否随之破<1.5m 趋1m
③ fell_over~0 且 goal_rate 不崩(门开后盯球恢复,能踢球)。
等待器 bbghb4r4k。日志 /tmp/v4_exp9_full.log。
若 EXP9 仍不扫视→说明 active_scan 奖励本身梯度不足/uniq_frac 对 neck 动作不敏感,需重审扫视激励
设计(可能要直接奖励 neck_yaw 的运动幅度/方差,而非间接奖励覆盖结果)。

## EXP9 结果:门控成功但扫视仍未涌现(2026-06-09,决定性负结论)
2000 iter 完成。三判据:
- ③ fell_over ~0.05-0.1 ✅(门控没破坏步态)
- ② pos_err ~1.6m(偶达1.24)🟡(与 EXP8 持平,无突破)
- ① **scan_uniq_frac 全程卡 0.25-0.28，纹丝不动 ❌**
- goal_rate 偶发 0.06-0.25 不稳定 🟡
**奖励分解证明门控机制本身成功**:gaze_center 被压到 0.03(EXP8 时主导)、active_scan 升到 0.43 主导。

**决定性负结论**:即使扫视成为唯一高额奖励来源,策略依然学不会转 neck。
→ 排除"奖励冲突"假说(EXP8 诊断)。真问题:**active_scan 间接奖励"覆盖结果"(uniq_frac),
梯度传不到"转 neck_yaw"这个动作**。策略拿到 0.43 scan 奖励是被动行走的偶然覆盖,
未建立"主动转头→覆盖↑→奖励↑"的因果关联。与"RL 教不动稠密视觉检测"同类问题(间接奖励教不动低层行为)。

## EXP10 设计:直接奖励 neck 运动(绕过间接覆盖奖励)
不再奖励覆盖结果,改**直接奖励 neck_yaw 的运动幅度/方差**(逼策略先动起来,建立动作-效果关联),
belief 不确定时尤其激励。可能配合:① 奖励 neck_yaw 角速度绝对值(不确定时)② 或对 neck_yaw 加
探索噪声/课程,先强制扫视再让覆盖奖励接管。仍 oracle 检测。从 EXP9 model_1999 bootstrap。

## EXP10 启动(2026-06-09,直接奖励 neck 运动)
针对 EXP9 负结论(间接覆盖奖励教不动扫视动作):`scripts/spike_v4_e2e_neck_motion.py`,2000 iter,
384 envs,从 EXP9 model_1999 bootstrap。env=`mos92_soccer_e2e_dualcam_neck_motion_env_cfg`。
核心新增:`neck_scan_motion` reward(weight 0.8)=(1-确定度门)*tanh(|neck_yaw角速度|/2.0)——
**直接奖励 neck 转动动作本身**(belief 不确定时),给 PPO 短 credit path 的稠密梯度;覆盖/gaze 奖励
负责塑造"往哪看"。保留 EXP9 门控 gaze + active_scan。
**判据**:① neck_scan_motion 是否真驱动 neck 动起来 → scan_uniq_frac 是否首次升破 0.4
② pos_err 是否随之破 1m ③ fell_over~0 且 goal_rate 不崩。
**关键早期信号**(~iter 100-300 即可见):scan_uniq_frac 若开始爬升=动作涌现,路径打通;
若仍卡 0.25=连直接运动奖励都教不动,需重审(可能 neck 动作被其他约束锁死/动作空间问题)。
日志 /tmp/v4_exp10_full.log。

## EXP10 中期观察(iter ~157,2026-06-09,直接运动奖励有效但引发稳定性张力)
**关键突破信号**:直接奖励 neck 角速度**确实教会了 neck 动**——neck_scan_motion reward 0.005→0.23 持续爬升,
scan_uniq_frac 多次冲到 **0.42**(EXP8/9 两轮死卡 0.25,首次实质突破)。证明"间接覆盖奖励教不动、
直接动作奖励教得动"的判断正确。
**代价(codex 红线预警的稳定性张力)**:初期 neck 狂甩→fell_over 飙到 **5.17**(灾难级),uniq 因频繁摔崩到 0.001。
**但 PPO 正自行化解**:fell_over 单调恢复 5.17→3.6→2.0→1.0→0.5→0.27,已基本稳住;uniq 恢复到 ~0.27 并多次破 0.4。
→ "边扫边稳"正在被 RL 学会。趋势向好,继续跑完 2000 iter 判读。
**注意**:之前一次 task-notification 是误报(等待器被误打包进落盘命令导致 orphan,EXP10 实际未停);
已重挂独立等待器 b7yhxvm9r。教训:run_in_background 的等待器不要和其他命令打包。

## 代码审查发现(2026-06-09,EXP10 训练期间审查 FusedPoseBelief)
**BUG #1(重要,影响 EXP8/9/10 的融合质量)**:FusedPoseBelief 的 append 节流用**全局标量**
_last_step/_steps_since_append,但 reset() 在**每次部分 reset**(terminated 子集)时把它们清零。
384 envs 下几乎每步都有 env 终止→计数器几乎每步被清零→steps_since_append 永远到不了 stride-1→
**缓冲在初帧后基本停止 append 新帧,时序窗口变陈旧**。纯 Python 模拟证实:200 步 stride=4,
若每步都有 env reset,实际 append 仅 **1 次**(健康应 ~50 次)。
→ 后果:EXP8/9/10 的"8帧融合"实际退化成近似单帧(被动融合增益被悄悄抹掉)。这解释了为何
EXP8 融合改善有限(1.8m)、EXP10 uniq_frac 卡 0.30——融合窗口根本没正常滚动。
**注**:EXP8 离线验证脚本(median 1.03m)用独立 fused.reset(slice(None)) 全量 reset,不触发此 bug,
所以验证时融合是健康的——这正是验证(1.03m)与训练(1.8m)差距的一个嫌疑根因。
**StackedCameraRGB 同模式**:realspec 用 stride=6 也中招;默认 stride=1 的实例免疫(每步必 append)。
**修复方向**:append 节流应 per-env(用 buffer 的 _num_pushes 或 per-env step 计数),
reset 只清被 reset 的 env 行,不动全局节奏。或最简:stride 逻辑改为基于 env.common_step_counter
的全局整除(step % stride == 0)而非易被 reset 清零的累加器。

**BUG #2(非 bug,但需记录的设计不对称)**:critic.ball_to_target 仍是单帧 ball_to_target 向量,
actor.ball_to_target 是 FusedPoseBelief(7维)。reward 查 term 用 isinstance 过滤,安全;但 critic
看到的 belief 与 actor 不同(critic 用的是球向量而非融合位姿)——这是继承自 oracle env 的原有结构,
非本次引入,但值得确认 critic 价值估计是否因此有偏。

## 从 codex 实验借鉴的关键教训(2026-06-09,跨 agent 学习)
查阅 mjlab_codex_v4 的 EXP7C8-C21 实验(codex 走无训练阶段机+手写搜索+GT override 诊断路线)。
codex 还独立监督了我的 EXP8 训练,第三方判读与我一致:not_scored_frac≈1,"不能算纯视觉踢球成功"。
**三条直接改变我下一步判断的教训**:
1. **"会扫视/看见球" ≠ "会踢球"**(最重要)。codex C8/C18/C21 反复证明:可见性/持球率/搜索能力都能提升,
   但 goal_rate 始终≈0,球几乎不动(path/step≈0.0001,stuck 19s)。**真瓶颈不是感知,是"把球从可见态
   稳定输送到近脚窗口[x_b≈0.08-0.15m,|y_b|<0.1m]并推出"的踢球链**。→ 我一直盯 pos_err/uniq_frac,
   但即使自定位完美,踢球链是断的,goal_rate 不会自己上来。
2. **GT 作弊都救不动踢球链**(C21a):上帝视角在近脚窗口注入推球命令,goal 仅 0.018(上界 0.68)。
   说明冻结的 04_e2e actor 无法把视觉链状态转成踢球动作,需专门训练/蒸馏 belief-to-window + contact primitive。
3. **新增搜索/扫视行为会挤掉已学的踢球/稳定**(C8 射门链退化 ≈ 我 EXP10 的 neck 狂甩破坏步态,同源张力)。
**codex 现成工具**:scripts/summarize_v4_plain_log.py(统计 goal_rate/fell_over/not_scored/selfloc/ball_path),
scripts/probe_v4_contact_windows.py(近脚触球窗口诊断)。下次 eval 应借鉴其指标集,不只看 pos_err。
**结论修正**:v4 真正的拦路虎是**踢球链(belief→近脚窗口→推球)**,自定位只是前置。我应在修完融合 bug、
确认自定位能到 1m 后,把重心转向 codex 已诊断清楚的踢球链问题,而非继续在扫视上深挖。

## EXP11 启动(2026-06-09,融合bug修复后重跑,隔离"健康融合"变量)
目的=回答线A问题:**修好融合 bug 后,健康时序融合能把纯视觉 pos_err 推到多少?能否接近 0.79m?**
脚本 `scripts/spike_v4_e2e_fused_fixed.py`,与 EXP8 完全相同配置(fused_scan env + EXP6 model_2999 bootstrap),
**唯一变量=融合 append bug 已修**(commit 63a851ab)。这样 pos_err 改善可干净归因于健康融合。
对照:EXP8(bug 未修)末100 pos_err 1.70m、uniq 0.25。
**判据**:① pos_err 是否显著低于 1.70m、能否趋近离线验证的 1.03m 甚至 0.79m
② uniq_frac 是否因缓冲正常滚动而升 ③ fell_over 保持低。
若 pos_err 明显改善→证明此前"被动融合无效"的结论部分是 bug 假象,自定位线有救;
若仍 1.7m→说明被动融合确实弱,自定位需靠主动扫视(回 EXP10 线但要解决稳定性)。
日志 /tmp/v4_exp11_full.log。

## GitHub/文献调研:RoboCup 自定位方法(2026-06-09,可能重塑线A)
查阅 RoboCup SPL/Humanoid League 几十年经验 + 近年 arXiv。**核心洞察(改变方法判断)**:

### 洞察1:不该用单帧/多帧 Kabsch,应用 EKF/UKF 递归滤波(最高优先)
RoboCup 全行业标准答案:**永远不靠单帧几何配准定位**,而用 EKF/UKF 维护 SE2 位姿分布,
每帧把检测到的每个地标作为独立观测更新滤波器(哪怕只看到1个点也能更新),里程计做预测步。
**我的现状**:FusedPoseBelief 是"N帧点用里程计搬到当前帧→拼大点云→单次Kabsch"(observations.py:572-591),
**这不是递归滤波**。朝同方向走时 N 帧看到同几个点,拼接约束不增加→这正是 EXP7"被动融合无效"的根因。
→ EKF/UKF 把"单帧看4点欠定"变成"信息跨帧累积",且算力 O(1)、RoboCup 在 Nao 弱 CPU 实时跑。
**这是带已知地图的定位(非SLAM),观测模型=地标投影,EKF 非常干净**。参考 B-Human SelfLocator、
PythonRobotics UKF 实现。**性价比最高、改动最确定。**

### 洞察2:对称二义性需单独处理(与洞察1正交)
对称球场 Kabsch 会随机跳到镜像解→单这一项就造成大误差。CLAP(arXiv:2509.08495)思路:
对每观测点生成对称假设→位姿空间聚类→真解聚成簇、虚假解分散。几何方法无需训练,可直接嫁接 Kabsch 输出。
**我从未处理过镜像对称**——这可能是 pos_err 卡 1.5m 的一个隐藏贡献者。

### 洞察3:主动视觉用熵减/信息增益驱动(我 EXP8-10 走的路,但方法可改进)
Seekircher/Laue/Röfer 的熵减 next-best-view:维护 belief 熵,选最大化期望熵减的头角。
**关键**:不必用 RL!我有地标3D坐标+相机模型,可解析预测"neck转到θ能看到哪些地标→协方差降多少",
贪心选最优θ。比我 EXP8-10 用 RL 间接学扫视(教不动)更直接。arXiv:2011.13851 是 RL 版备选。

### 洞察4:多技能冲突用 teacher distillation(线B 收尾用)
DeepMind "Learning Agile Soccer Skills"(arXiv:2304.13653):先训单技能专家(找球定位/踢球)再蒸馏成
统一策略,避免互相覆盖。这正是 codex C8 射门退化 + 我 EXP10 neck甩头破稳的解法。优先级排定位之后。

### 路线重排(线A 方法升级)
原线A=修融合bug重跑(EXP11进行中,仍是点云拼接Kabsch)。**新认知**:即使bug修好,点云拼接的
天花板可能就在那;真正的提升要靠 EKF/UKF 递归滤波 + 对称消歧。
→ EXP11 结果出来后,无论好坏,线A 下一步应是**把 FusedPoseBelief 从"点云拼接Kabsch"改造成"EKF/UKF递归滤波"**,
并加对称消歧。主动扫视改用解析熵减(绕开 RL 教不动扫视的问题)。

## 代码审查发现 #3(2026-06-09,EXP11 期间)— 扫视/门控 reward 一步滞后
**机制**:step() 内 reward_manager.compute(line440) 在 observation_manager.compute(line464) **之前**运行。
我的 active_scan/门控/monitor reward 读 term._last_uniq,而该缓存只在 FusedPoseBelief.__call__
(obs compute 内)写入。→ 第 N 步 reward 读到的是第 N-1 步 obs 写的 uniq_frac。
**严重度评估(诚实)**:
- 稳态时:50Hz 下一步=20ms,belief 变化极小,门控/扫视奖励慢半拍→影响轻微,非系统性偏置,只是噪声。
- **reset 后第一步更糟**:reset() 不清 _last_uniq(只清 buffer 行),第一步 reward 用的是**上一 episode**
  的确定度→门控/扫视奖励在每个 episode 开头错位一帧。
- active_scan 用 _last_uniq 当 reward 值,一步滞后让"奖励"与"产生它的动作"错配半拍,弱化 credit assignment。
**结论**:这是真 bug 但严重度中低——不是 EXP8/9/10 扫视学不动的主因(那是间接奖励+点云拼接架构问题),
但会给扫视学习加噪声。**修法**:reward 改为现场计算 uniq_frac(不读缓存),或 reset 时把 _last_uniq 清零。
鉴于线A 要转向 EKF 重构,此 bug 留待重构时一并处理(EKF 版 reward 接口会重写),现在不单独热修。

## EXP12 离线 EKF 验证:递归滤波大幅胜出(2026-06-09,决定性正面结果)
线A R1 重构前的廉价 gate:同一 EXP11 策略 rollout(model_1500),同样的 per-landmark depth 观测,
三方法公平对比 pos_err(EKF 从宽先验起步,非给 GT):
| 方法 | mean | median |
|---|---|---|
| 单帧 Kabsch | 3.65m | 1.90m |
| 8帧点云拼接 Kabsch(现架构)| 2.68m | 1.31m |
| **递归 EKF** | **1.21m** | **1.10m** |
**EKF vs 点云拼接 +55%,vs 单帧 +67%**。
**结论(决定性)**:
1. 调研论点在真实数据成立——递归滤波远胜几何配准。"被动融合无效"是点云拼接架构错,非融合本身无用。
2. EKF 离线就把 mean 2.68→1.21m、median→1.10m,**已接近 1m 目标**(且用的是未为 EKF 优化的旧策略)。
3. 点云拼接 mean(2.68)>>median(1.31)的长尾大误差,EKF 消除了(mean≈median)——长尾很可能是
   镜像对称跳变,印证 R2 对称消歧价值。
**纪律**:两步验证——① 合成自检(每帧仅见3/8点,EKF 30步收敛到0.0000m,证数学正确)
② 真实 rollout 对比(本结果)。EKF 公式见 scripts/exp12_offline_ekf.py:_ekf_step。
**下一步**:R1 把 FusedPoseBelief 重构为在线 EKF(用此验证过的 _ekf_step),重训看 pos_err 能否破 1m。
踩坑:验证脚本须用训练时同一 env(fused_scan,obs 88维)才能加载策略;_collect_frame 前须 _ensure_init。

## EXP13 启动(2026-06-09,线A R1:在线 EKF 重构)
EXP12 离线验证(EKF 1.21m vs 点云拼接 2.68m,+55%)后的正式重构:
- 新类 `EkfPoseBelief`(observations.py,继承 FusedPoseBelief 复用 _collect_frame)= 有状态 per-env SE2-EKF。
  predict 用 GT 里程计 delta,update 对每个可见地标做序贯 EKF 校正。输出仍 7 维同布局
  [x_n,y_n,sin,cos,vis_now,uniq_frac,uncertainty],第7维 resid→EKF 协方差迹(真不确定度)。
- 新 env `mos92_soccer_e2e_dualcam_ekf_env_cfg`(派生 fused_scan,保留 active_scan+松 neck,换 EKF belief)。
- 脚本 `scripts/spike_v4_e2e_ekf.py`,从 EXP11 model_1999(88维同结构)全载 bootstrap,继承步态+踢球。
- monitor fused_belief_error 因 isinstance(EkfPoseBelief 是 FusedPoseBelief 子类)自动匹配,pos_err 日志正确。
**smoke 结果(关键)**:selfloc_pos_err_m 训练初就 **~1.08-1.26m**(吻合离线 1.21m),远好于点云拼接 ~2.1m,
bootstrap 全载无 mismatch,在线 EKF 正常运行。
**判据**:① pos_err 能否随训练稳定破 1m、趋近 0.79m ② fell_over 保持低(EKF 是 obs 不污染步态,应稳)
③ goal_rate(不期望此步提升,线B 才是命门)。日志 /tmp/v4_exp13_full.log。
对照:EXP11(点云拼接)末100 pos_err 2.125m。

## EXP13 最终结果(2026-06-09,线A R1 EKF 成功 + 踢球链意外正向)
末100 多指标判读(scripts/readout_v4_metrics.py,三维度):
**自定位 ✅ 达标**:pos_err **0.98m**(median 0.98,max 仅1.06,长尾极小)。
  EXP11点云拼接2.13m → EKF 0.98m,**破1m目标**(理想0.79m仅差一点)。线A核心目标达成。
**稳定性 ✅**:fell_over 0.057(median 0)——EKF纯obs零污染步态,最稳一轮。
**踢球链 🟡 意外正向(未达标)**:goal_rate 0.03→**0.080**、episode_success 0.19→**0.40**、
  ball_to_tgt_err 4.74→**2.97m**、ball_speed 0.39。全线改善。
**解读**:原判断"定位准救不了踢球链"被部分修正——定位准**间接帮了**踢球(机器人更清楚自己/球门
  相对位置,把球带向门更准,ball_to_target 4.7→3.0m)。但goal_rate仍0.08,**临门一脚仍断**,
  印证核心判断:踢球动作技能(B2)是独立缺失件,定位只能把球带到离门3m,从3m到进门的踢击仍缺技能。
**线A 收官**:0.98m 达标。是否再加R2对称消歧榨到0.79m,优先级低于线B(goal_rate命门)。
**下一步**:转线B,先做B2(近脚窗口spawn球专训踢球动作技能)。EXP13 model 作为后续bootstrap基线。

## EXP14 启动(2026-06-09,线B B2:近脚踢球技能)
针对踢球链断裂(EXP13 goal_rate 0.08,根因=动作技能缺失非感知/奖励)。三处改动:
1. near_foot_spawn_fraction=0.5:一半episode球spawn在脚前踢击窗口(0.08-0.20m)密集练踢,
   一半保持完整任务防遗忘。解决spawn_dist≥0.6m致踢击样本稀疏。
2. dribble_kick_impulse(w1.5):接触步奖励球速朝门投影(踢击质量)。
3. ekf_kick env+spike从EXP13 model_1999 bootstrap(继承定位0.98m+步态+接近)。
smoke通过:全载/奖励激活/无错。
**判据**:① goal_rate能否从0.08显著上升(目标→0.2)、episode_success↑ ② 护栏:pos_err不退化(<1.1)、
fell_over<0.1(踢球训练不能破坏线A定位与稳定) ③ ball_speed↑/ball_stuck↓。
对照EXP13:goal 0.08,success 0.40,pos_err 0.98,fell_over 0.057。日志/tmp/v4_exp14_full.log。

## 线A 里程碑留痕归档(2026-06-09)
线A EKF 自定位成功后的记录留痕工作(权重归档+评估确证):
**权重归档** `checkpoints/v4_soccer/lineA_ekf_exp13/`:
- model_1999.pt(最终权重,md5 28bdd74600d6f5a511ad11a94cc502ab)+ model_0.pt(起点,可复现)
  + tensorboard events + git 快照 + README.md(完整背景/指标/复现方法)。
- 防 logs 目录被后续实验覆盖,关键节点权重永久留存。
**评估产物** `soccer_eval/2026-06-09_v4/lineA_ekf/`(脚本 scripts/eval_v4_lineA.py):
- 视频 mp4(9env 500步)+ 4 预览 PNG + metrics.json + README(含方法学局限)。
- **核心确证**:独立 headless 评估(128env,warmup150)pos_err median **0.99m**,
  精确复现训练稳态 0.98m → 线A 归档数字可信。
- 诚实记录局限:goal_rate/ball_to_target 等 episode 级累积量因短窗口(400步)被低估
  (评估 goal 0.004 vs 训练 0.08),以训练日志为准;fell_over n=4 采样不足无意义。
  逐步量(pos_err/ball_speed)适合短窗口确证,episode 级量以训练日志为准。

## EXP15 启动(2026-06-09,线B B2 改进:bug修复+调研驱动突破)
EXP14 结果:goal_rate 0.08→0.11(+37%)、ball_to_tgt 2.97→2.45m,方向有效但卡0.11;
out_of_bounds 0.21→0.28(印证spawn穿透bug)。codex经验+网络调研双重诊断后改动:
**bug修复(确定)**:
1. near_foot_dist下界0.08→0.25m(球半径0.11从root量会spawn进脚→穿透弹飞)。
2. rear_spawn_fraction显式置0(原从fused_scan继承0.5与near_foot冲突)。
**调研驱动突破**:
3. kick_impulse加speed_threshold=0.6:只奖朝门球速>0.6的真踢,掐掉'温吞推球'局部最优
   (EXP14 ball_speed卡0.39正是此症状)。
4. spike重置std_param→1.0:bootstrap继承坍缩动作std(0.44-0.98)抑制'大力踢'探索,重开。
**codex反证**:其冻结策略+外挂BC/teacher路线离线AUC0.99但闭环零进球(步态不兼容+部署分布shift),
端到端RL天然绕开两杀手,方向正确,不必上BC。
**判据**:① goal_rate能否突破0.11量级(目标→0.2)、ball_speed能否升(脱离0.39说明学会踢) ②
护栏pos_err<1.1、fell_over<0.1、out_of_bounds回落(<0.25说明spawn修好) ③多seed验证防假阳性。
对照EXP14: goal 0.11,success 0.42,ball_speed 0.40,oob 0.28,pos_err 0.97。日志/tmp/v4_exp15_full.log。

## 指标体系审查 + EXP15 中期诊断(2026-06-09)
用户质疑"goal_rate 0.2够吗",触发对全指标体系的代码级审查,发现多个问题:

### 指标定义的真相(读dribble_command.py确认)
- **goal_rate**=球曾穿球门线(x>10.8,|y|<1,门宽2m)。真足球进球。定义干净。
  **但goal_target_fraction=0.5**: 只有一半episode目标是球门,另一半是球周围1-3m随机点(纯带球)。
  => 非球门episode的goal_rate≈0,**全局goal_rate理论上限被压到~0.5**。
  => "goal_rate 0.2"实际=球门子集里~40%进球。0.2是合理中期目标,但偏低;
     会踢球的策略子集应达60-70%(全局~0.3)。
- **物理可达性约束**: episode仅20s(1000步@50Hz),球门11m外,robot随机spawn平均离门6-8m,
  很多球门episode物理上来不及接近+带球11m进门 => 进一步压低goal_rate上限。
- **episode_success不纯**: 混合"到随机点"和"进球门"两种目标,无法单独反映踢球能力。
- **ball_speed均值掩盖关键区别**: 含静止时刻平均,0.4既可能"一直慢推"也可能"偶尔快踢+大量静止",
  区分不出"慢推vs真踢"——而这正是EXP15要解决的核心,却没指标能直接验证!

### 指标改进建议
1. 报告"target=goal子集"的进球率而非全局goal_rate(真实踢球能力)。
2. 加"接触时球速/最大球速"指标,区分慢推vs真踢。
3. episode_success分离"到点"和"进门"两类统计。

### EXP15中期诊断(508 iter): 我的两个改动都过头了
- ball_speed 0.39→0.34**不升反降**: threshold=0.6太高,策略踢不出0.6球速,
  kick_impulse奖励几乎为0(0.01),门槛没引导"更用力"反而掐死信号(稀疏奖励冷启动陷阱)。
- fell_over 0.05→0.126**翻倍且剧烈震荡**(0~0.33反复): std重置1.0过猛,动作太乱,508iter未自愈。
- goal_rate 0.109无突破。
=> 结论: threshold应设低或用渐进课程; std重置1.0太激进应temper(如0.85)或配entropy退火。
