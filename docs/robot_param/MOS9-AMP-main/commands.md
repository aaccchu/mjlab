### Git Push
```bash
git remote -v
git remote set-url origin git@github.com:THMOS2025/MOS9-AMP.git
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa" git pull origin main --allow-unrelated-histories
git config pull.rebase false

GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa" git push -u origin main

git remote set-url origin git@8.141.22.226:Wegg/amp.git
```


### Forward Kinematic
```bash

# all file 
python scripts/mos9_fk_npz.py \
    --input_dir data/motions/GMR_kick \
    --output_dir data/motions/fk_kick \
    --output_fps 50 \
    --headless

# single file
python scripts/mos9_fk_npz.py \
    --input_file data/motions/GMR_kick/sideflip.npz \
    --output_dir data/motions/sideflip \
    --output_fps 50 \
    --headless


# config env (debug)
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset LD_PRELOAD

python scripts/mos9_fk_npz.py --input_file data/motions/mos9_GMR_rotate_motion/0313-39.npz --output_fps 50 --output_dir data/motions/mos9_fk_rotate_motion --headless
python scripts/mos9_fk_npz.py --input_file data/motions/mos9_GMR_rotate_motion/0313-40.npz --output_fps 50 --output_dir data/motions/mos9_fk_rotate_motion --headless
python scripts/mos9_fk_npz.py --input_file data/motions/mos9_GMR_rotate_motion/0319-06.npz --output_fps 50 --output_dir data/motions/mos9_fk_rotate_motion --headless
```


### Clip Motion
```bash
python data/motions/clip_motion.py --input_dir data/motions/mos9_fk_motion --output_dir data/motions/mos9_fk_motion_clipped_simple 
python data/motions/clip_motion.py --input_dir data/motions/mos9_fk_rotate_motion --output_dir data/motions/mos9_fk_rotate_motion_clipped

```




### replay motion
```bash
python scripts/replay_mos9_fk_motion.py --input_dir data/motions/mos9_fk_motion --output_fps 50 --enable_cameras
python scripts/replay_mos9_fk_motion.py --input_dir data/motions/mos9_fk_motion_clipped_simple --output_fps 50 --enable_cameras --plot_speed_curve
python scripts/replay_mos9_fk_motion.py --input_file data/motions/mos9_fk_motion/B15_-__Walk_turn_around_stageii.npz --output_fps 50 --enable_cameras
python scripts/replay_mos9_fk_motion.py --input_dir data/motions/mos9_fk_motion --output_fps 50 --video --enable_cameras

python scripts/replay_mos9_fk_motion.py --input_dir data/motions/mos9_fk_rotate_motion_clipped --output_fps 50 --enable_cameras --plot_speed_curve
python scripts/replay_mos9_fk_motion.py --input_dir data/motions/mos9_fk_rotate_motion --output_fps 50 --enable_cameras --plot_speed_curve

python scripts/replay_mos9_fk_motion.py --input_file data/motions/mos9_fk_motion_clipped/B10_-__Walk_turn_left_45_stageii.npz --output_fps 50 --enable_cameras



python scripts/replay_mos9_fk_motion.py --input_file data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg1.npz --output_fps 50 --enable_cameras



python scripts/replay_mos9_fk_motion.py --input_file data/motions/fk_kick/walk_kick.npz --output_fps 50 --enable_cameras
python scripts/replay_mos9_fk_motion.py --input_file data/motions/sideflip/sideflip.npz --output_fps 50 --enable_cameras


```


### Train & Play
```bash

# train
python scripts/amp_rsl_rl/train.py \
    --task AMP_MOS9_V7_ALIAS \
    --logger wandb \
    --log_project_name MOS9_AMP \
    --reward_track_lin_vel_xy=1.2 \
    --event_reset_robot_joints.params.position_range=0.95,1.05 \
    --agent_amp_lr_coef=0.1 \
    --agent_algorithm.learning_rate=3e-4 \
    --agent_policy.init_noise_std=0.5 \
    --agent_max_iterations=6000 \
    --headless


python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V7_ALIAS --headless





# play
python scripts/amp_rsl_rl/play.py \
    --task AMP_MOS9_V9 \
    --target logs/rsl_rl/mos9_loco/walk_vx_0415_137/model_500.pt \
    --commands.base_velocity.debug_vis=true \
    --num_envs 32 \
    --video \
    --length 1500

python scripts/amp_rsl_rl/play.py \
    --task AMP_MOS9_V11 \
    --target logs/rsl_rl/mos9_loco/2026-05-31_13-34-30_amp_v11_clean_bucket/model_3000.pt \
    --commands.base_velocity.debug_vis=true \
    --num_envs 32 \
    --video \
    --length 1500




# sim2sim
python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/walk_v7_0409_210/exported/policy_6000.onnx --cmd_hold_seconds 3.0 --duration=50.0

python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/walk_v7_0410_127/exported/policy_6000.onnx --cmd_hold_seconds 3.0 --duration=50.0

python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/walk_v2_0319_207/exported/policy_2000.onnx --sim_dt 0.005 --control_decimation 4
    


python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/2026-05-16_16-20-43_amp_v6_alias_turn_as_lr/exported/policy_1000.onnx --cmd_hold_seconds 10.0 --duration=2000.0

python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/2026-05-16_16-20-43_amp_v6_alias_turn_as_lr/exported/policy_1000.onnx --cmd_hold_seconds 10.0 --duration=20.0 --terrain

python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/2026-05-18_22-48-00_amp_v6_alias_turn_as_lr/exported/policy_5000.onnx --cmd_hold_seconds 4.0 --duration=50.0 --xml data/assets/MOS/MOS92_urdf_0517_v3/xml/MOS92_urdf_0517_v3_simplified.xml

python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/2026-05-20_16-42-36_amp_v6_alias_turn_as_lr/exported/policy_4000.onnx --cmd_hold_seconds 5.0 --duration=50.0 --xml data/assets/MOS/MOS92_urdf_0517_v3/xml/MOS92_urdf_0517_v3_simplified.xml

python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/2026-05-31_13-34-30_amp_v11_clean_bucket/exported/policy_3000.onnx --cmd_hold_seconds 3.0 --duration=50.0 --xml data/assets/MOS/MOS92_urdf_0517_v3/xml/MOS92_urdf_0517_v3_simplified.xml




python scripts/mos9_amp_sim2sim_mujoco.py --model logs/rsl_rl/mos9_loco/2026-05-25_20-52-28_amp_v11_clean_bucket/exported/policy_6000.onnx --cmd_hold_seconds 1.0 --duration=50.0 --xml data/assets/MOS/MOS92_urdf_0517_v3/xml/MOS92_urdf_0517_v3_simplified_boxfoot.xml



# replay policy cmd hanging test
python scripts/mos9_hanging_joint_test_mujoco.py --input_npz logs/rsl_rl/mos9_loco/walk_v7_0410_127/mujoco_plot/policy_6000/00_plot_data.npz 


python scripts/mos9_hanging_joint_test_mujoco.py --input_npz logs/rsl_rl/mos9_loco/walk_v7_0410_127/mujoco_plot/policy_6000/00_plot_data.npz --xml data/assets/MOS/MOS92_urdf_0517_v3/xml/MOS92_urdf_0517_v3_simplified.xml



# real replay mujoco
python scripts/mos9_hanging_action_replay_mujoco.py \
    --input_npz logs/rsl_rl/mos9_loco/walk_v7_0410_127/mujoco_plot/policy_6000/policy_plot_data.npz \
    --base_height 1.0

# sim2real bridge
python scripts/mos9_sim2real_bridge.py \
    --npz logs/hanging_test/MOS92_urdf_0308_simplified/walk_motion_mujoco_v7_0409/hanging_plot_data.npz \
    --out_dir logs/hanging_test/real_robot \
     --no_torque_clip

python scripts/mos9_sim2real_bridge.py \
    --npz logs/hanging_test/MOS92_urdf_0308_simplified/walk_new_model_0517/hanging_plot_data.npz \
    --out_dir logs/hanging_test/real_robot_policy0410 \
    --no_torque_clip \
    --duration 5 


python scripts/mos9_sim2real_bridge_with_imu.py \
    --npz logs/hanging_test/MOS92_urdf_0308_simplified/walk_motion_mujoco_v7_0410/hanging_plot_data.npz \
    --out_dir logs/hanging_test/real_robot_policy0410 \
    --no_torque_clip



# all motor zero
python scripts/mos9_all_zero_slow.py --duration 5.0 --hold_duration 20.0

# set motor zero
python scripts/mos9_set_zero_by_joint.py --joint right_knee --joint left_knee
python scripts/mos9_set_zero_by_joint.py --joint right_shoulder_roll

# hanging on policy test
python scripts/mos9_hanging_policy_test_mujoco.py \
  --model logs/rsl_rl/mos9_loco/walk_v7_0409_210/exported/policy_6000.onnx \
  --duration 8 \
  --cmd_hold_seconds 8.0 \
  --out_dir logs/hanging_policy

python scripts/mos9_hanging_policy_test_mujoco.py \
  --model logs/rsl_rl/mos9_loco/walk_v7_0410_127/exported/policy_6000.onnx \
  --duration 8 \
  --cmd_hold_seconds 8.0 \
  --out_dir logs/hanging_policy0410

python scripts/mos9_hanging_policy_test_mujoco.py \
  --model logs/rsl_rl/mos9_loco/walk_v7_0410_127/exported/policy_6000.onnx \
  --duration 8 \
  --cmd_hold_seconds 8.0 \
  --out_dir logs/hanging_policy0410 \
  --sim_dt 0.002 \
  --control_decimation 10 \
  --xml data/assets/MOS/MOS92_urdf_0517_v3/xml/MOS92_urdf_0517_v3_simplified.xml



# sim2real
python scripts/mos9_sim2real_deploy.py \
  --model logs/rsl_rl/mos9_loco/walk_v7_0409_210/exported/policy_6000.onnx \
  --out_dir logs/sim2real_deploy \
  --pre_move_duration 3.0 \
  --duration 5 \
  --cmd_hold_seconds 5.0

python scripts/mos9_sim2real_deploy_copy.py \
  --model logs/rsl_rl/mos9_loco/walk_v7_0410_127/exported/policy_6000.onnx \
  --out_dir logs/sim2real_deploy_policy0410 \
  --pre_move_duration 10.0 \
  --duration 12 \
  --cmd_hold_seconds 10.0

python scripts/mos9_sim2real_deploy.py \
  --model logs/rsl_rl/mos9_loco/walk_v7_0518/policy_6000.onnx \
  --out_dir logs/sim2real_deploy_policy0410 \
  --pre_move_duration 10.0 \
  --duration 5 \
  --cmd_hold_seconds 5.0



python scripts/mos9_sim2real_deploy_copy.py \
  --model logs/rsl_rl/mos9_loco/walk_v11_0518/policy_6000.onnx \
  --out_dir logs/sim2real_deploy_policy0410 \
  --pre_move_duration 10.0 \
  --duration 15 \
  --cmd_hold_seconds 10.0


python scripts/mos9_sim2real_deploy_copy.py \
  --model logs/rsl_rl/mos9_loco/walk_v1_0520/policy_6000.onnx \
  --out_dir logs/sim2real_deploy_policy0410 \
  --pre_move_duration 10.0 \
  --duration 12 \
  --cmd_hold_seconds 10.0


python scripts/mos9_mimic_deploy.py \
  --model logs/mimic/turn_left/2026-05-21_20-38-01_v1.onnx \
  --obs_txt data/motions/mimic_motion/turn_left_fast.txt \
  --pre_move_duration 10.0 \
  --duration 5


# test delay
python scripts/estimate_motor_delay.py \
  --npz logs/sim2real_deploy_policy0410/20260514_214108/deploy_data.npz \
  --max_lag_s 0.2

# system id
python3 scripts/identify_pd_from_npz.py \
  --npz logs/hanging_test/MOS92_urdf_0517_v3_simplified/20260517_211755/hanging_plot_data.npz \
  --out logs/hanging_test/identified_pd_results.npz
```



### IMU Test
```bash
python scripts/imu_sim.py --arb_axis 1,0,1 --arb_deg 60 --arb_duration 2.5
```


### Recommand Params
```bash
# walk_v3_no_curri_0324_309
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_Velocity_LessMotion_NoCurriculum --headless --agent_amp_lr_coef=0.02 --reward_alive=0.001 --reward_action_rate=-0.001 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_track_lin_vel_xy.params.std=0.35 --reward_track_ang_vel_z.params.std=0.35

# walk_v3_stdplus_0324_307
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_Velocity_LessMotion_LooseConstraints --headless --agent_amp_lr_coef=0.02 --reward_alive=0.001 --reward_action_rate=-0.001 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_track_lin_vel_xy.params.std=0.35 --reward_track_ang_vel_z.params.std=0.35

# V6
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V6 --headless --agent_max_iterations=10000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.1 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0

# v6 ALIAS walk_v6_0408_234
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V6_ALIAS --headless --agent_max_iterations=10000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.1 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0

# v7 ALIAS walk_v7_0409_210
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V7_ALIAS --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.1 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0


# v7 New kpkd
scripts/amp_rsl_rl/train.py --task AMP_MOS9_V7_ALIAS --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.03 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0
```

### resume
```bash
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V7_ALIAS_RANDOM --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.03 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0 --resume true --load_run walk_v7_0410_127 --checkpoint 6000 --logger wandb --log_project_name MOS9_AMP


python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V7_ALIAS_RANDOM --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.06 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0 --resume true --load_run walk_v7_0410_127 --checkpoint 6000 --logger wandb --log_project_name MOS9_AMP


python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V7_ALIAS_RANDOM --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.3 --reward_action_rate=-0.07 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0 --resume true --load_run walk_v7_0410_127 --checkpoint 6000 --logger wandb --log_project_name MOS9_AMP


python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V11 --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.6 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.15 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0 --logger wandb --log_project_name MOS9_AMP

python scripts/amp_rsl_rl/train.py --task MOS9EnvCfgV11 --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=2.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.1 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0 --logger wandb --log_project_name MOS9_AMP  --resume true --load_run walk_forward --checkpoint 6000




python scripts/amp_rsl_rl/train.py --task MOS9EnvCfgV11 --headless --agent_max_iterations=6000 --agent_amp_lr_coef=0.1 --reward_track_lin_vel_y=2.0 --reward_track_lin_vel_x=3.0 --reward_track_ang_vel_z=2.0 --agent_amp_task_reward_lerp=0.7 --agent_amp_reward_coef=0.2 --reward_action_rate=-0.15 --reward_alive=0.0001 --reward_yaw_rate_when_xy_cmd=-1.0 --logger wandb --log_project_name MOS9_AMP  --resume true --load_run 2026-05-20_21-04-09_amp_v6_alias_turn_as_lr --checkpoint 6000

```
