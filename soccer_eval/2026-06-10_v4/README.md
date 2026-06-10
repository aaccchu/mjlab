# v4 soccer — 关键节点验证评估产物索引(2026-06-10)

本目录留痕 2026-06-10 当天的关键节点评估。每个子目录有独立 README + 视频 + 静帧 + metrics.json。
前一天的节点(lineA EKF / lineB kick / EXP16 / EXP19)见 `../2026-06-09_v4/`。

| 子目录 | 节点 | 源权重 | 核心确证 |
|---|---|---|---|
| `kick_aim_exp20b/` | **当前最佳部署模型**:aim-the-kick(EXP20b) | `checkpoints/v4_soccer/kick_aim_exp20b/model_1999.pt` | 真实射门率 **0.38-0.41**(+20% vs EXP16-19 平台,v4 单次最大增益);ball_to_tgt 1.88 / robot_to_ball 0.60 / pos_err 0.85 / OOB 0.20 / success 0.58 五项历史最好;8 联画视频含真实进球(scored:true)+ 低头看球 0.7m 帧(突破 1.25m 盲区线) |

## 配套文档(repo 根目录)
- `soccer_robot_v4_experiment.md` —「EXP20 调研与瓶颈诊断」+ EXP20/20b 最终判读。
- `v4评估指标详解.md` — 指标定义与判读规则。
- 面板逐指标详解:`../2026-06-09_v4/kick_exp16/kick_exp16_pov_belief.md`(8 联画定义相同)。

## 方法学
与 2026-06-09_v4 一致:训练 stdout 稳态窗口判读(四轮同法,趋势可信)+ 单 env0 定性演示视频
(play=False、GT 课程拉到纯 EKF 终态、关 corruption/push、goal_target_fraction=1.0)。
仍 oracle 球检测阶段。
