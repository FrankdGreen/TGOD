from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .expert import ExpertTrajectory
from .schema import OBS_DIM


def _solve_spd_3x3(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Small Cholesky solve without dispatching to a BLAS/OpenMP runtime."""
    a00, a01, a02 = (float(value) for value in matrix[0])
    _, a11, a12 = (float(value) for value in matrix[1])
    _, _, a22 = (float(value) for value in matrix[2])
    b0, b1, b2 = (float(value) for value in right_hand_side)
    l00 = math.sqrt(max(a00, 1e-15))
    l10 = a01 / l00
    l20 = a02 / l00
    l11 = math.sqrt(max(a11 - l10 * l10, 1e-15))
    l21 = (a12 - l20 * l10) / l11
    l22 = math.sqrt(max(a22 - l20 * l20 - l21 * l21, 1e-15))
    y0 = b0 / l00
    y1 = (b1 - l10 * y0) / l11
    y2 = (b2 - l20 * y0 - l21 * y1) / l22
    x2 = y2 / l22
    x1 = (y1 - l21 * x2) / l11
    x0 = (y0 - l10 * x1 - l20 * x2) / l00
    return np.asarray([x0, x1, x2], dtype=np.float64)


class UR5ePickPlaceEnv(gym.Env):
    """UR5e white-cup pick-and-place task used by TGOD-SD.

    The returned environment reward is always zero. TGOD supplies the internal
    MINE pseudo-reward inside the learner, so task success is only an evaluation
    signal. The supplied scene has no gripper DOF; grasping is a contact/proximity
    latch followed by the same kinematic cup attachment used by SAC_ur5e.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path,
        expert: ExpertTrajectory,
        config: dict[str, Any],
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene does not exist: {self.xml_path}")
        self.model = self._load_model_unicode_safe(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.expert = expert
        self.config = config
        self.render_mode = render_mode
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.max_episode_steps = int(config["max_episode_steps"])
        self.frame_skip = int(config["frame_skip"])
        self.kinematic_control = bool(config.get("kinematic_control", True))
        self.max_ee_step = float(config["max_ee_step"])
        self.ik_iterations = int(config["ik_iterations"])
        self.ik_damping = float(config["ik_damping"])
        self.grasp_radius = float(config["grasp_radius"])
        self.grasp_command_threshold = float(config["grasp_command_threshold"])
        self.release_command_threshold = float(config["release_command_threshold"])
        self.cup_tcp_offset = np.asarray(config["cup_tcp_offset"], dtype=np.float64)
        self.blue_mat_center = np.asarray(config["blue_mat_center"], dtype=np.float64)
        self.success_radius = float(config["success_radius"])
        self.success_z_max = float(config["success_z_max"])
        self.minimum_lift_height = float(config["minimum_lift_height"])
        self.workspace_low = np.asarray(config["workspace_low"], dtype=np.float64)
        self.workspace_high = np.asarray(config["workspace_high"], dtype=np.float64)
        self.initial_joint_noise = float(config["initial_joint_noise"])
        self.initial_cup_xy_noise = float(config["initial_cup_xy_noise"])

        if self.workspace_low.shape != (3,) or self.workspace_high.shape != (3,):
            raise ValueError("workspace_low and workspace_high must each contain three values")
        if np.any(self.workspace_low >= self.workspace_high):
            raise ValueError("workspace_low must be below workspace_high")

        self._ee_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self._cup_joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        self._cup_qpos_adr = int(self.model.jnt_qposadr[self._cup_joint_id])
        self._cup_dof_adr = int(self.model.jnt_dofadr[self._cup_joint_id])
        self._cup_geom_ids = {
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "cup_body"),
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "cup_rim"),
        }
        self._wrist_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        self.action_space: spaces.Box = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self._viewer: Any = None
        self._step_count = 0
        self._grasped = False
        self._placed = False
        self._ever_grasped = False
        self._cup_lifted = False

    @staticmethod
    def _load_model_unicode_safe(xml_path: Path) -> mujoco.MjModel:
        """Load via an ASCII basename so MuJoCo on Windows accepts Chinese paths."""
        previous_directory = Path.cwd()
        scene_directory = xml_path.parent.resolve()
        try:
            os.chdir(scene_directory)
            return mujoco.MjModel.from_xml_path(xml_path.name)
        finally:
            os.chdir(previous_directory)

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, object_type, name)
        if identifier < 0:
            raise RuntimeError(f"Required MuJoCo object is missing: {name}")
        return int(identifier)

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_cup_pos(self) -> np.ndarray:
        return self.data.qpos[self._cup_qpos_adr : self._cup_qpos_adr + 3].copy()

    def _cup_anchor(self) -> np.ndarray:
        return self._get_cup_pos() + self.cup_tcp_offset

    def _actual_cup_wrist_contact(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 in self._cup_geom_ids:
                other_geom = int(contact.geom2)
            elif contact.geom2 in self._cup_geom_ids:
                other_geom = int(contact.geom1)
            else:
                continue
            if int(self.model.geom_bodyid[other_geom]) == self._wrist_body_id:
                return True
        return False

    def _cup_collision_with_robot(self) -> bool:
        allowed = {"wrist_3_link", "red_mat", "blue_mat"}
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 in self._cup_geom_ids:
                other_geom = int(contact.geom2)
            elif contact.geom2 in self._cup_geom_ids:
                other_geom = int(contact.geom1)
            else:
                continue
            body_id = int(self.model.geom_bodyid[other_geom])
            if body_id == 0:
                continue
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name not in allowed:
                return True
        return False

    def _solve_position_ik(self, target_position: np.ndarray) -> np.ndarray:
        self._ik_data.qpos[:] = self.data.qpos
        self._ik_data.qvel[:] = 0.0
        qpos = self._ik_data.qpos[:6].copy()
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(self.ik_iterations):
            mujoco.mj_forward(self.model, self._ik_data)
            error = target_position - self._ik_data.site_xpos[self._ee_site_id]
            if np.linalg.norm(error) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self._ik_data, jacobian, None, self._ee_site_id)
            arm_jacobian = jacobian[:, :6]
            # Avoid NumPy's BLAS-backed ``@`` here. Conda NumPy and pip Torch
            # can bundle different Intel OpenMP versions on Windows; these are
            # tiny 3x6 products, so explicit reductions are both cheap and avoid
            # a duplicate-runtime abort after the policy has run.
            gram = np.sum(
                arm_jacobian[:, None, :] * arm_jacobian[None, :, :], axis=2
            )
            task_update = _solve_spd_3x3(
                gram + self.ik_damping**2 * np.eye(3), error
            )
            update = np.sum(arm_jacobian * task_update[:, None], axis=0)
            qpos += np.clip(update, -0.08, 0.08)
            qpos = np.clip(qpos, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
            self._ik_data.qpos[:6] = qpos
        return qpos.astype(np.float32)

    def _attach_cup(self) -> None:
        self.data.qpos[self._cup_qpos_adr : self._cup_qpos_adr + 3] = self._get_ee_pos() - self.cup_tcp_offset
        self.data.qpos[self._cup_qpos_adr + 3 : self._cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._cup_dof_adr : self._cup_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _observation(self) -> np.ndarray:
        ee_position = self._get_ee_pos()
        cup_position = self._get_cup_pos()
        cup_anchor = cup_position + self.cup_tcp_offset
        cup_goal = np.asarray([*self.blue_mat_center, self.expert.cup_initial_position[2]], dtype=np.float64)
        observation = np.concatenate(
            [
                self.data.qpos[:6],
                self.data.qvel[:6],
                ee_position,
                cup_position,
                cup_anchor - ee_position,
                cup_goal - cup_position,
                np.asarray([float(self._grasped), float(self._placed)]),
                np.asarray([self._step_count / self.max_episode_steps]),
            ]
        )
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"Internal observation shape error: {observation.shape}")
        return observation.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.expert.initial_qpos.astype(np.float64).copy()
        if self.initial_joint_noise > 0:
            qpos += self.np_random.normal(0.0, self.initial_joint_noise, size=6)
        self.data.qpos[:6] = np.clip(qpos, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
        cup_position = self.expert.cup_initial_position.astype(np.float64).copy()
        if self.initial_cup_xy_noise > 0:
            cup_position[:2] += self.np_random.uniform(
                -self.initial_cup_xy_noise, self.initial_cup_xy_noise, size=2
            )
        self.data.qpos[self._cup_qpos_adr : self._cup_qpos_adr + 3] = cup_position
        self.data.qpos[self._cup_qpos_adr + 3 : self._cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        if self.model.nu >= 6:
            self.data.ctrl[:6] = self.data.qpos[:6]
        self._step_count = 0
        self._grasped = False
        self._placed = False
        self._ever_grasped = False
        self._cup_lifted = False
        mujoco.mj_forward(self.model, self.data)
        info = self._info(contact=False, collision=False, joint_target=self.data.qpos[:6])
        return self._observation(), info

    def _info(self, *, contact: bool, collision: bool, joint_target: np.ndarray) -> dict[str, Any]:
        cup_position = self._get_cup_pos()
        goal = np.asarray([*self.blue_mat_center, self.expert.cup_initial_position[2]], dtype=np.float64)
        return {
            "success": bool(self._placed),
            "grasped": bool(self._grasped),
            "placed": bool(self._placed),
            "ever_grasped": bool(self._ever_grasped),
            "cup_lifted": bool(self._cup_lifted),
            "contact": bool(contact),
            "collision": bool(collision),
            "step": int(self._step_count),
            "phase": "placed" if self._placed else ("carry" if self._grasped else "reach"),
            "ee_pos": self._get_ee_pos().astype(float).tolist(),
            "cup_pos": cup_position.astype(float).tolist(),
            "cup_goal": goal.astype(float).tolist(),
            "cup_goal_distance": float(np.linalg.norm(cup_position - goal)),
            "joint_target": np.asarray(joint_target, dtype=float).tolist(),
            "qpos": self.data.qpos[:6].astype(float).tolist(),
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"Action must have shape (4,), got {action.shape}.")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_position = np.clip(
            self._get_ee_pos() + action[:3] * self.max_ee_step,
            self.workspace_low,
            self.workspace_high,
        )
        joint_target = self._solve_position_ik(target_position)
        if self.kinematic_control:
            self.data.qpos[:6] = joint_target
        control_low = self.model.actuator_ctrlrange[:6, 0]
        control_high = self.model.actuator_ctrlrange[:6, 1]
        self.data.ctrl[:6] = np.clip(joint_target, control_low, control_high)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        contact = bool(
            self._actual_cup_wrist_contact()
            or np.linalg.norm(self._get_ee_pos() - self._cup_anchor()) <= self.grasp_radius
        )
        if self._grasped and action[3] < self.release_command_threshold:
            self._grasped = False
        elif not self._grasped and action[3] > self.grasp_command_threshold and contact:
            self._grasped = True
            self._ever_grasped = True
        if self._grasped:
            self._attach_cup()
            if self._get_cup_pos()[2] >= self.minimum_lift_height:
                self._cup_lifted = True

        self._step_count += 1
        cup_position = self._get_cup_pos()
        cup_goal = np.asarray(
            [*self.blue_mat_center, self.expert.cup_initial_position[2]], dtype=np.float64
        )
        goal_distance = float(np.linalg.norm(cup_position - cup_goal))
        if (
            not self._grasped
            and self._ever_grasped
            and self._cup_lifted
            and goal_distance < self.success_radius
            and cup_position[2] <= self.success_z_max
        ):
            self._placed = True
        collision = self._cup_collision_with_robot()
        terminated = bool(self._placed)
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        info = self._info(contact=contact, collision=collision, joint_target=joint_target)
        return self._observation(), 0.0, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        import mujoco.viewer

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        assert self._viewer is not None
        self._viewer.sync()
        return None

    def set_replay_state(self, qpos: np.ndarray, cup_position: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        cup_position = np.asarray(cup_position, dtype=np.float64)
        if qpos.shape != (6,) or cup_position.shape != (3,):
            raise ValueError(f"Replay state requires qpos (6,) and cup_position (3,), got {qpos.shape}, {cup_position.shape}.")
        self.data.qpos[:6] = qpos
        self.data.qpos[self._cup_qpos_adr : self._cup_qpos_adr + 3] = cup_position
        self.data.qpos[self._cup_qpos_adr + 3 : self._cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:6] = np.clip(
            qpos,
            self.model.actuator_ctrlrange[:6, 0],
            self.model.actuator_ctrlrange[:6, 1],
        )
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
