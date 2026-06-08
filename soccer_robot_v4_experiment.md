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
