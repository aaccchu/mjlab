"""v4 EXP5f: local PPO subclass adding a supervised keypoint-detection aux loss.

Rationale: three RL-reward attempts (EXP5c/d/e) proved PPO's scalar reward can't
learn the 46-dim dense keypoint regression — it games (invisible=free), saturates
(exp kernel -> 0 gradient), or steals gradient from upright/dribble (fell_over 37).
Keypoint detection is a dense supervised vision task. This subclass keeps the full
PPO objective for motor control and ADDS a masked smooth-L1 loss between the
actor's selfloc output (the 46-dim predicted keypoint pixels) and the per-frame
projected labels carried in a TRAINING-ONLY obs group ``keypoint_label`` (in the
rollout storage, NOT in actor/critic obs_groups — the policy never sees labels).

Lives in-project and is selected via ``algorithm.class_name`` so the vendored
rsl_rl package is never modified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO


class KeypointAuxPPO(PPO):
  """PPO + supervised keypoint-detection aux loss on the selfloc action slice."""

  def __init__(
    self,
    *args,
    keypoint_cfg: dict | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    kpc = keypoint_cfg or {}
    # selfloc action slice within the full action vector (joint_pos[0:20] then
    # selfloc[20:66]); each keypoint is (u,v) so K = slice_len/2.
    self._kp_slice = slice(
      int(kpc.get("action_start", 20)), int(kpc.get("action_end", 66))
    )
    self._kp_label_group = kpc.get("label_group", "keypoint_label")
    self._kp_coef = float(kpc.get("coef", 1.0))
    self._kp_num = int(kpc.get("num_keypoints", 23))
    self._kp_mean_loss = 0.0
    self._kp_mean_pix_err = 0.0

  def _train_keypoint_aux(self) -> None:
    """Supervised pass: regress the actor's selfloc slice to projected labels.

    Runs over the SAME rollout storage PPO will use, BEFORE super().update()
    (which clears storage). Loss = masked smooth_l1 between predicted keypoint
    pixels and the training-only ``keypoint_label`` obs (uv_x|uv_y|vis, K each).
    Gradient flows into the shared RGB CNN + selfloc head — a clean supervised
    signal, NOT a reward competing with motor control. Uses the PPO optimizer.
    """
    gen = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    n, loss_acc, pix_acc = 0, 0.0, 0.0
    K = self._kp_num
    for batch in gen:
      obs = batch.observations
      if self._kp_label_group not in obs.keys():
        return  # label group absent — nothing to train (e.g. play mode)
      label = obs[self._kp_label_group]  # (B, K*3): uv_x|uv_y|vis
      uv_gt = label[:, : 2 * K].reshape(-1, K, 2)
      vis = label[:, 2 * K : 3 * K]  # (B, K)
      # Forward actor to refresh its distribution; take the deterministic mean.
      # Must pass stochastic_output=True (matches PPO.update) so
      # output_distribution_params is recomputed for THIS batch — otherwise it
      # stays stale at the rollout num_envs and shapes mismatch.
      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      mean = self.actor.output_distribution_params[0]  # (B, num_actions)
      pred_uv = mean[:, self._kp_slice].reshape(-1, K, 2)
      per = F.smooth_l1_loss(pred_uv, uv_gt, reduction="none").sum(-1)  # (B,K)
      w = vis
      denom = w.sum().clamp(min=1.0)
      loss = (per * w).sum() / denom
      self.optimizer.zero_grad()
      (self._kp_coef * loss).backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()
      with torch.no_grad():
        pix = (pred_uv - uv_gt).norm(dim=-1)
        pix_acc += ((pix * w).sum() / denom).item()
      loss_acc += loss.item()
      n += 1
    if n:
      self._kp_mean_loss = loss_acc / n
      self._kp_mean_pix_err = pix_acc / n

  def update(self) -> dict[str, float]:
    self._train_keypoint_aux()
    loss_dict = super().update()
    loss_dict["kp_aux"] = self._kp_mean_loss
    loss_dict["kp_aux_pix_err"] = self._kp_mean_pix_err
    return loss_dict

