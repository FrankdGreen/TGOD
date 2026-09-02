import os
from typing import Optional

import mujoco
import numpy as np
try:
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    from gymnasium import spaces  # pyright: ignore[reportMissingImports]
except ImportError:
    import gym
    from gym import spaces


DEFAULT_MENAGERIE_PATH = os.environ.get("MUJOCO_MENAGERIE_PATH", os.path.expanduser("~/mujoco_menagerie"))
DEFAULT_UR5E_XML = os.path.join(DEFAULT_MENAGERIE_PATH, "universal_robots_ur5e", "ur5e.xml")


class UR5eSingleArmEnv(gym.Env):
    """Single UR5e MuJoCo trajectory-planning task without expert demonstrations.

    The goal is to move the end-effector to a target pose with a dense reward.
    This is a minimal but practical SAC training baseline for UR5e trajectory planning.
    """

    def __init__(self, xml_path: Optional[str] = None, render_mode: Optional[str] = None):
        xml_path = xml_path or DEFAULT_UR5E_XML
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"UR5e MuJoCo XML not found at {xml_path}. "
                "Set MUJOCO_MENAGERIE_PATH or pass xml_path explicitly."
            )

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.n_controls = 3
        self._ee_site_id = self._find_ee_site()
        self._max_ee_step = 0.03
        self._cup_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint"
        )
        self._cup_qpos_adr = (
            int(self.model.jnt_qposadr[self._cup_joint_id])
            if self._cup_joint_id >= 0
            else None
        )
        self._cup_target = np.array([0.45, 0.35, 0.115], dtype=np.float32)
        self._cup_tcp_offset = np.array([0.0, 0.0, 0.105], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        obs_dim = 6 + 6 + 3 + 3 + 3 + 3 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._target = np.array([0.45, 0.35, 0.22], dtype=np.float32)
        self._goal_radius = 0.05
        self._time_step = 0
        self._last_dist = None
        self._grasped = False
        self._placed = False

        self.reset()

    def _find_ee_site(self) -> int:
        for index in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, index) or ""
            if any(token in name.lower() for token in ("ee", "tool", "gripper", "attachment")):
                return index
        raise RuntimeError("Could not find an end-effector site in the MuJoCo model.")

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_cup_pos(self) -> np.ndarray:
        if self._cup_qpos_adr is None:
            return np.zeros(3, dtype=np.float64)
        return self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr + 3].copy()

    def cup_contact_with_ee(self) -> bool:
        cup_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_body"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_rim"),
        }
        ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )
        if ee_body_id < 0:
            return False

        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 in cup_geom_ids:
                other_geom = contact.geom2
            elif contact.geom2 in cup_geom_ids:
                other_geom = contact.geom1
            else:
                continue
            if self.model.geom_bodyid[other_geom] == ee_body_id:
                return True
        return False

    def cup_collision_with_robot(self) -> bool:
        cup_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_body"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_rim"),
        }
        allowed_bodies = {"wrist_3_link", "red_mat", "blue_mat", "world"}
        for contact in self.data.contact[:self.data.ncon]:
            if contact.geom1 not in cup_geom_ids and contact.geom2 not in cup_geom_ids:
                continue
            other_geom = contact.geom2 if contact.geom1 in cup_geom_ids else contact.geom1
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[other_geom]),
            ) or ""
            if body_name not in allowed_bodies:
                return True
        return False

    def inverse_kinematics(self, target_pos: np.ndarray, iterations: int = 30) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if target_pos.shape != (3,):
            raise ValueError(f"target_pos must have shape (3,), got {target_pos.shape}.")

        qpos = self.data.qpos[:6].copy()
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            error = target_pos - self.data.site_xpos[self._ee_site_id]
            if np.linalg.norm(error) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, self._ee_site_id)
            jacobian_6 = jacobian[:, :6]
            update = jacobian_6.T @ np.linalg.solve(
                jacobian_6 @ jacobian_6.T + 0.05**2 * np.eye(3), error
            )
            qpos += np.clip(update, -0.08, 0.08)
            joint_range = self.model.jnt_range[:6]
            qpos = np.clip(qpos, joint_range[:, 0], joint_range[:, 1])
            self.data.qpos[:6] = qpos
        self.data.qpos[:6] = qpos
        mujoco.mj_forward(self.model, self.data)
        return qpos.astype(np.float32)

    def _set_initial_pose(self):
        q0 = np.array([
            0.09319, -1.65091, 2.08823, -1.15862, -1.69686, 0.0,
        ], dtype=np.float64)
        self.data.qpos[:6] = q0
        self.data.qpos[6:] = self.model.qpos0[6:]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _obs(self):
        q = self.data.qpos[:6].copy()
        qdot = self.data.qvel[:6].copy()
        ee = self._get_ee_pos()
        target = self._target.copy()

        obs = np.concatenate([
            q.astype(np.float32),
            qdot.astype(np.float32),
            ee.astype(np.float32),
            target.astype(np.float32),
            (target - ee).astype(np.float32),
            (self._cup_target - self._get_cup_pos()).astype(np.float32),
            np.array([float(self._grasped), float(self._placed)], dtype=np.float32),
            np.array([float(self._time_step)], dtype=np.float32) / 1000.0,
        ])
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._set_initial_pose()
        self._time_step = 0
        self._last_dist = None
        self._grasped = False
        self._placed = False
        self._target = np.array([0.45, 0.35, 0.22], dtype=np.float32)
        obs = self._obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        target_pos = self._get_ee_pos() + action[:3] * self._max_ee_step
        joint_target = self.inverse_kinematics(target_pos)
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        ctrl = np.clip(joint_target, ctrl_low, ctrl_high)

        if self.model.nu > 0:
            self.data.ctrl[: self.model.nu] = ctrl[: self.model.nu]

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        contact = self.cup_contact_with_ee()
        if not self._grasped and action[3] > 0.0 and contact:
            self._grasped = True
        elif self._grasped and action[3] < -0.2:
            self._grasped = False

        if self._grasped and self._cup_qpos_adr is not None:
            self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr + 3] = (
                self._get_ee_pos() - self._cup_tcp_offset
            )
            self.data.qpos[self._cup_qpos_adr + 3:self._cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[self._cup_qpos_adr:self._cup_qpos_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self._time_step += 1
        ee = self._get_ee_pos()
        cup_pos = self._get_cup_pos()
        ee_to_cup = float(np.linalg.norm(ee - cup_pos))
        cup_to_target = float(np.linalg.norm(cup_pos - self._cup_target))
        reward = -0.5 * ee_to_cup - 0.8 * cup_to_target
        reward += 2.0 if self._grasped else 0.0
        reward += 3.0 if contact and action[3] > 0.0 else 0.0
        if not self._grasped and cup_to_target < 0.06 and cup_pos[2] <= 0.14:
            self._placed = True
        if self._placed:
            reward += 10.0

        qvel = np.abs(self.data.qvel[:6])
        reward -= 0.02 * float(np.sum(qvel))
        collision = self.cup_collision_with_robot()
        if collision:
            reward -= 50.0

        if self._placed:
            reward += 100.0
            done = True
        else:
            done = False

        if self._time_step >= 300:
            done = True

        info = {
            "distance": cup_to_target,
            "ee_pos": ee.tolist(),
            "target_pos": target_pos.tolist(),
            "joint_target": joint_target.tolist(),
            "target": self._target.tolist(),
            "cup_pos": cup_pos.tolist(),
            "grasped": self._grasped,
            "placed": self._placed,
            "contact": contact,
            "collision": collision,
            "phase": "placed" if self._placed else ("carry" if self._grasped else "reach"),
        }
        return self._obs(), float(reward), bool(done), bool(done), info

    def render(self):
        if self.render_mode in (None, "rgb_array"):
            return None

        if self.model is None:
            return None

        try:
            import mujoco.viewer
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        except Exception:
            pass
        return None

    def close(self):
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
