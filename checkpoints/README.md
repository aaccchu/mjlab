# Soccer Robot Checkpoints

## v1_soccer/model_2999.pt
- 训练 run: `logs/rsl_rl/g1_velocity/2026-05-31_21-19-24/`
- 迭代: 3000
- 物理: condim=3（球无 rolling friction）
- 评估: success_rate 6.5%, ball_to_target 3.79m
- 网络: MLP 108→512→256→128→29 (actor), 120→512→256→128→1 (critic)

## v2_soccer/model_4999.pt
- 训练 run: `logs/rsl_rl/g1_velocity/2026-06-01_11-09-25/`
- 迭代: 5000
- 物理: condim=6（球有 rolling friction）+ kick_contact 奖励
- 评估: success_rate 99.8%, ball_to_target 0.20m, time_to_goal 122步
- 网络: 同 v1（obs 维度相同，contact sensor 不入 actor obs）
- wandb: run gvuf7h8r

## 兼容性
v1/v2 的 actor obs 维度相同（108），可以交叉加载（D-vs-B 测试已验证）。
区别仅在物理环境（condim）和奖励函数（v2 多了 kick_contact）。
