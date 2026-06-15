# EXP23 — strike(近门发力,v4 踢球链当前最佳模型)

`model_1999.pt` 来自 `logs/rsl_rl/mos92_velocity/2026-06-11_11-09-06_spike_v4_e2e_ekf_kick_strike/`
(EXP23 bootstrap 自 EXP20b;核心改动 = 近门 `kick_impulse` 增益 ×3)。
`params/{env,agent}.yaml` 为复现配置;`git/mjlab.diff` 为归档时 git 版本信息。

## 为什么是当前最佳
**forensics 口径射门率 0.410(158/385)** — v4 单次评估最高,且对 EXP22b(0.286)显著
(z=3.61),对 EXP21(0.357)优势 z=1.54(样本量下不显著,但全部辅助指标同向更好):

| 指标 | EXP20b | EXP21 | **EXP23** |
|---|---|---|---|
| forensics 射门率 | 0.38-0.41 | 0.357 | **0.410** |
| NEVER_ARRIVED | — | 122/387 | **101/385** |
| ball_to_target_error | 1.88 | — | **1.689** |
| robot_to_ball_error | 0.60 | — | **0.575** |
| episode_success | 0.58 | — | **0.614** |
| fell_over | 0.063 | — | **0.027** |
| ball_in_finish_box | — | — | 0.370 |
| ball_speed_peak | 2.70 ⚠️ | — | 2.673(持平,未破 2.5) |
| kicks/ep(中位) | — | 8 | **11** |
| 近门球速(mean) | — | 0.42-0.52 | **0.58** |

## 谱系与改动
EXP20b(aim 修复线最佳)→ **EXP23**(近门 kick_impulse 增益 ×3,针对"近门球速 0.42-0.52
徘徊、kicks/ep 卡 9"的技能局部最优:轻推稳赚 goal_progress、真踢冒摔倒/出界险)。
env=`mos92_soccer_e2e_dualcam_ekf_kick_strike`(脚本 `spike_v4_e2e_ekf_kick_strike.py`)。
全程无 std 重开(codex"低速贴球吸引子"教训:噪声微调可洗掉踢球链)。

**机制验证**:近门增益生效 —— kicks/ep 9→11、近门球速 0.42-0.52→0.58、fell_over 0.063→0.027
(发力没换来更多摔倒,反而更稳)。NEVER_ARRIVED 122→101 是射门率提升的主要来源。

## 评估产物
`soccer_eval/2026-06-11_v4_showdown/exp23_forensics_big.json`(385 episodes 四分类 + 2000 次
kick 的分区质量;ckpt 字段已核对为 EXP23 真权重,非误测)。三方对决:EXP21/EXP22b/EXP23。

## 局限
- forensics 口径(oracle 球检测,非真纯视觉);
- 对 EXP21 优势统计上不显著,需更大样本或下一轮增益坐实;
- ball_speed_peak 2.67 仍在 codex 警戒带(2.82→2.65 缓降),破 2.5 应停;
- 近门球速 0.58 距 0.5-0.6+ 终极目标仍有空间,期望-差结构未根除(轻推仍稳赚)。
