#!/usr/bin/env python3
"""
ROS2 NumPad keyboard -> Cartesian velocity -> DLS Jacobian pseudo-inverse -> joint velocities publisher.

Controls (NumPad):
  Num8 / Num2 : +/-X
  Num4 / Num6 : +/-Y
  Num9 / Num3 : +/-Z
  Num7 / Num1 : +/-Roll (wx)
  Num/ / Num* : +/-Pitch (wy)
  Num+ / Num- : +/-Yaw (wz)
  SHIFT (Hold): Precision / Fine-motion mode (scales speed by shift_scale, default 0.2x)
  ESC         : stop (clears motion immediately)
  P           : save one waypoint (Cartesian pose + joint positions) to CSV
  X           : exit (clean shutdown; configurable)

Run:
  python3 ros2_jacobian_numpad_record_path.py --ros-args -p tip_link:="end-effector"
"""

from __future__ import annotations

import threading
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

import pygame


class JacobianNumpadTeleop(Node):
    def __init__(self):
        super().__init__("jacobian_numpad_teleop")

        # -------------------------
        # Params (edit / override)
        # -------------------------
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("robot_description_topic", "/robot_description")
        self.declare_parameter("cmd_topic", "/velocity_controller/commands")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("tip_link", "end-effector")
        self.declare_parameter("rate_hz", 200.0)
        self.declare_parameter("v", 0.075)               # cartesian linear m/s (and angular rad/s)
        self.declare_parameter("shift_scale", 0.2)       # speed multiplier when holding Shift (0.2 = 20% speed)
        self.declare_parameter("smoothing_alpha", 0.12)  # exponential smoothing factor (0.05 = very smooth, 1.0 = instant)
        self.declare_parameter("damping", 0.05)          # lambda in DLS

        # Pygame options
        self.declare_parameter("keyboard_hz", 200.0)
        self.declare_parameter("pygame_window_w", 480)
        self.declare_parameter("pygame_window_h", 240)
        self.declare_parameter("pygame_grab_input", True)
        self.declare_parameter("exit_key", "x")
        self.declare_parameter("exit_on_window_close", True)

        # -------------------------
        # Read params
        # -------------------------
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        self.robot_description_topic = self.get_parameter("robot_description_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.base_link = self.get_parameter("base_link").value
        self.tip_link = self.get_parameter("tip_link").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.v = float(self.get_parameter("v").value)
        self.shift_scale = float(self.get_parameter("shift_scale").value)
        self.smoothing_alpha = float(np.clip(self.get_parameter("smoothing_alpha").value, 0.01, 1.0))
        self.damping = float(self.get_parameter("damping").value)

        self.keyboard_hz = float(self.get_parameter("keyboard_hz").value)
        self.pygame_w = int(self.get_parameter("pygame_window_w").value)
        self.pygame_h = int(self.get_parameter("pygame_window_h").value)
        self.pygame_grab_input = bool(self.get_parameter("pygame_grab_input").value)
        self.exit_key_char = str(self.get_parameter("exit_key").value).strip().lower()[:1]
        if not self.exit_key_char:
            self.exit_key_char = "x"
        self.exit_on_window_close = bool(self.get_parameter("exit_on_window_close").value)

        # -------------------------
        # NumPad Key Mapping (6 DOF)
        # -------------------------
        v = self.v
        self.key_mapping = {
            "num8": np.array([ v, 0, 0, 0, 0, 0], dtype=float),      # +X
            "num2": np.array([-v, 0, 0, 0, 0, 0], dtype=float),      # -X
            "num4": np.array([0,  v, 0, 0, 0, 0], dtype=float),      # +Y
            "num6": np.array([0, -v, 0, 0, 0, 0], dtype=float),      # -Y
            "num9": np.array([0, 0,  v, 0, 0, 0], dtype=float),      # +Z
            "num3": np.array([0, 0, -v, 0, 0, 0], dtype=float),      # -Z
            "num7": np.array([0, 0, 0,  v, 0, 0], dtype=float),      # +Roll
            "num1": np.array([0, 0, 0, -v, 0, 0], dtype=float),      # -Roll
            "num_div": np.array([0, 0, 0, 0,  v, 0], dtype=float),   # +Pitch (K_KP_DIVIDE)
            "num_mul": np.array([0, 0, 0, 0, -v, 0], dtype=float),   # -Pitch (K_KP_MULTIPLY)
            "num_plus": np.array([0, 0, 0, 0, 0,  v], dtype=float),  # +Yaw (K_KP_PLUS)
            "num_minus": np.array([0, 0, 0, 0, 0, -v], dtype=float), # -Yaw (K_KP_MINUS)
        }

        # Pygame key -> token mapping
        self._pg_key_to_token = {
            pygame.K_KP_8: "num8",
            pygame.K_KP_2: "num2",
            pygame.K_KP_4: "num4",
            pygame.K_KP_6: "num6",
            pygame.K_KP_9: "num9",
            pygame.K_KP_3: "num3",
            pygame.K_KP_7: "num7",
            pygame.K_KP_1: "num1",
            pygame.K_KP_DIVIDE: "num_div",
            pygame.K_KP_MULTIPLY: "num_mul",
            pygame.K_KP_PLUS: "num_plus",
            pygame.K_KP_MINUS: "num_minus",
            # Fallback normal number keys if NumLock is off
            pygame.K_8: "num8",
            pygame.K_2: "num2",
            pygame.K_4: "num4",
            pygame.K_6: "num6",
            pygame.K_9: "num9",
            pygame.K_3: "num3",
            pygame.K_7: "num7",
            pygame.K_1: "num1",
        }

        self._exit_pg_key = self._pygame_key_from_char(self.exit_key_char)
        if self._exit_pg_key is None:
            self.exit_key_char = "x"
            self._exit_pg_key = pygame.K_x

        # -------------------------
        # Keyboard & Smoothing state (thread-safe)
        # -------------------------
        self._lock = threading.Lock()
        self.pressed_keys: set[str] = set()
        self.shift_held: bool = False
        self._waypoint_requests = 0
        self._p_was_down = False
        self._waypoint_count = 0
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        self.target_twist = np.zeros(6, dtype=float)
        self.smoothed_twist = np.zeros(6, dtype=float)

        # -------------------------
        # ROS pub/sub
        # -------------------------
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.current_joint_state: JointState | None = None
        self._js_name_to_index = None

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 50)

        robot_desc_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._robot_desc_sub = self.create_subscription(
            String, self.robot_description_topic, self._robot_description_cb, robot_desc_qos
        )

        # -------------------------
        # KDL state
        # -------------------------
        self._kdl_ready = False
        self.chain = None
        self.n_joints = 0
        self.chain_joint_names: list[str] = []
        self.jac_solver = None
        self.fk_solver = None

        # -------------------------
        # Exit flag
        # -------------------------
        self._exit_requested = False

        # -------------------------
        # Pygame keyboard thread
        # -------------------------
        self._kb_running = True
        self._kb_thread = threading.Thread(target=self._pygame_loop, daemon=True)
        self._kb_thread.start()

        # -------------------------
        # Control timer
        # -------------------------
        self._timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)

        self.get_logger().info(
            f"Started NumPad Teleop. Waiting for {self.robot_description_topic} (URDF) and {self.joint_states_topic}..."
        )
        self.get_logger().info(
            f"Publishing joint velocities to {self.cmd_topic} at {self.rate_hz:.1f} Hz. "
            f"base_link='{self.base_link}', tip_link='{self.tip_link}'."
        )
        self.get_logger().info(
            "Controls (NumPad):\n"
            "  Num8/2: +/-X | Num4/6: +/-Y | Num9/3: +/-Z\n"
            "  Num7/1: +/-Roll | Num/ / Num*: +/-Pitch | Num+/Num-: +/-Yaw\n"
            f"  SHIFT (Hold): Precision smooth mode ({self.shift_scale}x speed)\n"
            "  P: Record Waypoint | ESC: Stop | X: Exit. Click Pygame window to focus."
        )

    @staticmethod
    def _pygame_key_from_char(ch: str) -> int | None:
        if not ch:
            return None
        c = ch.lower()[0]
        if "a" <= c <= "z":
            return getattr(pygame, f"K_{c}", None)
        return None

    def _recompute_twist(self):
        tw = np.zeros(6, dtype=float)
        for k in self.pressed_keys:
            vec = self.key_mapping.get(k, None)
            if vec is not None:
                tw += vec
        if self.shift_held:
            tw *= self.shift_scale
        self.target_twist = tw

    def _clear_keys(self, hard_stop: bool = False):
        with self._lock:
            self.pressed_keys.clear()
            self.shift_held = False
            self.target_twist = np.zeros(6, dtype=float)
            if hard_stop:
                self.smoothed_twist = np.zeros(6, dtype=float)

    def request_exit(self, reason: str = ""):
        if self._exit_requested:
            return
        self._exit_requested = True
        if reason:
            self.get_logger().warn(f"Exit requested: {reason}")
        else:
            self.get_logger().warn("Exit requested.")
        self._kb_running = False
        self._clear_keys(hard_stop=True)
        try:
            self._publish_cmd(np.zeros(self.n_joints if self.n_joints > 0 else 7, dtype=float))
        except Exception:
            pass

    def _pygame_loop(self):
        try:
            pygame.init()
            screen = pygame.display.set_mode((self.pygame_w, self.pygame_h))
            pygame.display.set_caption("NumPad Teleop - Buzz Wire (CLICK HERE)")
            clock = pygame.time.Clock()

            try:
                pygame.event.set_grab(self.pygame_grab_input)
            except Exception:
                pass

            font = pygame.font.Font(None, 20)
            info = [
                "CLICK THIS WINDOW (focus required).",
                "Num8/2: +/-X  |  Num4/6: +/-Y  |  Num9/3: +/-Z",
                "Num7/1: Roll  |  Num//*: Pitch |  Num+/-: Yaw",
                f"HOLD SHIFT: Precision Smooth Mode ({self.shift_scale:.2f}x speed)",
                "P: Record Waypoint to CSV  |  ESC: Emergency Stop",
                f"{self.exit_key_char.upper()}: Exit clean shutdown.",
            ]

            while rclpy.ok() and self._kb_running and (not self._exit_requested):
                pygame.event.pump()

                for event in pygame.event.get([pygame.QUIT]):
                    if event.type == pygame.QUIT:
                        if self.exit_on_window_close:
                            self.request_exit("Pygame window closed.")
                        else:
                            self.get_logger().warn("Pygame window closed -> stopping keyboard thread.")
                            self._kb_running = False
                            self._clear_keys(hard_stop=True)
                        break

                if (not self._kb_running) or self._exit_requested:
                    break

                if not pygame.key.get_focused():
                    self._clear_keys(hard_stop=True)
                    self._p_was_down = False
                else:
                    keys = pygame.key.get_pressed()
                    p_down = bool(keys[pygame.K_p] or keys[pygame.K_KP_PERIOD])
                    if p_down and not self._p_was_down:
                        with self._lock:
                            self._waypoint_requests += 1
                    self._p_was_down = p_down

                    if self._exit_pg_key is not None and keys[self._exit_pg_key]:
                        self.request_exit(f"Exit key '{self.exit_key_char}' pressed.")
                    elif keys[pygame.K_ESCAPE]:
                        # ESC triggers an immediate zero-velocity stop
                        self._clear_keys(hard_stop=True)
                    else:
                        held = set()
                        for pgk, token in self._pg_key_to_token.items():
                            if keys[pgk]:
                                held.add(token)

                        # Extra check for operators if NumLock affects them
                        if keys[pygame.K_KP_DIVIDE]:
                            held.add("num_div")
                        if keys[pygame.K_KP_MULTIPLY]:
                            held.add("num_mul")
                        if keys[pygame.K_KP_PLUS]:
                            held.add("num_plus")
                        if keys[pygame.K_KP_MINUS]:
                            held.add("num_minus")

                        # Check Left or Right Shift for precision mode
                        is_shift = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

                        with self._lock:
                            self.pressed_keys = held
                            self.shift_held = is_shift
                            self._recompute_twist()

                screen.fill((25, 25, 25))
                y = 10
                for line in info:
                    screen.blit(font.render(line, True, (230, 230, 230)), (10, y))
                    y += 22

                with self._lock:
                    keys_str = ", ".join(sorted(self.pressed_keys)) if self.pressed_keys else "(none)"
                    mode_str = f"PRECISION ({self.shift_scale}x)" if self.shift_held else "NORMAL (1.0x)"
                    mode_color = (120, 200, 255) if self.shift_held else (180, 220, 180)

                screen.blit(font.render(f"Mode: {mode_str}", True, mode_color), (10, y + 4))
                screen.blit(font.render(f"Active NumPad: {keys_str}", True, (180, 220, 180)), (10, y + 26))

                pygame.display.flip()
                clock.tick(max(60.0, self.keyboard_hz))

        except Exception as e:
            self.get_logger().error(f"Pygame loop crashed: {e}")
            self._clear_keys(hard_stop=True)
        finally:
            try:
                pygame.quit()
            except Exception:
                pass

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg
        self._js_name_to_index = {n: i for i, n in enumerate(msg.name)}

    def _robot_description_cb(self, msg: String):
        if self._kdl_ready:
            return

        urdf_xml = msg.data.strip()
        if not urdf_xml or "<robot" not in urdf_xml:
            self.get_logger().error("robot_description received but doesn't look like URDF XML.")
            return

        try:
            robot = URDF.from_xml_string(urdf_xml)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                self.get_logger().error("Failed to construct KDL tree from URDF.")
                return

            chain = tree.getChain(self.base_link, self.tip_link)
            n = int(chain.getNrOfJoints())
            if n == 0:
                self.get_logger().error(
                    f"KDL chain has 0 joints for {self.base_link} -> {self.tip_link}."
                )
                return

            names: list[str] = []
            for i in range(chain.getNrOfSegments()):
                seg = chain.getSegment(i)
                j = seg.getJoint()
                jt = j.getTypeName()
                if jt not in ("None", "Fixed"):
                    names.append(j.getName())
                    if len(names) == n:
                        break

            if len(names) != n:
                self.get_logger().error(f"Joint-name extraction mismatch: getNrOfJoints()={n}, extracted {len(names)}")
                return

            self.chain = chain
            self.n_joints = n
            self.chain_joint_names = names
            self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
            self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

            self._kdl_ready = True
            self.get_logger().info(f"KDL ready: n_joints={self.n_joints}, chain={self.chain_joint_names}")
            self.destroy_subscription(self._robot_desc_sub)

        except Exception as e:
            self.get_logger().error(f"Exception creating KDL from URDF: {e}")

    def _compute_jacobian(self, q_chain: np.ndarray) -> np.ndarray:
        jnt_array = kdl.JntArray(self.n_joints)
        for i in range(self.n_joints):
            jnt_array[i] = float(q_chain[i])

        jac = kdl.Jacobian(self.n_joints)
        self.jac_solver.JntToJac(jnt_array, jac)

        J = np.zeros((6, self.n_joints), dtype=float)
        for i in range(6):
            for j in range(self.n_joints):
                J[i, j] = jac[i, j]
        return J

    def _dls_pinv_times(self, J: np.ndarray, v6: np.ndarray) -> np.ndarray:
        lam = float(self.damping)
        A = J @ J.T + lam * np.eye(6)
        qdot = (J.T @ np.linalg.inv(A)) @ v6
        return qdot

    def _record_waypoint(self, q_chain: np.ndarray):
        if self.fk_solver is None:
            self.get_logger().warn("P ignored: KDL forward kinematics not ready.")
            return
        jnt_array = kdl.JntArray(self.n_joints)
        for i, v in enumerate(q_chain):
            jnt_array[i] = float(v)
        pose = kdl.Frame()
        status = self.fk_solver.JntToCart(jnt_array, pose)
        if status < 0:
            self.get_logger().error(f"Forward kinematics failed: status={status}")
            return
        qx, qy, qz, qw = pose.M.GetQuaternion()
        if self._csv_writer is None:
            prefix = datetime.now().strftime("buzz_wire_waypoints_%Y%m%d_%H%M%S")
            counter = 0
            while True:
                name = f"{prefix}{('_' + str(counter)) if counter else ''}.csv"
                try:
                    self._csv_file = Path(name).open("x", newline="", encoding="utf-8")
                    self._csv_path = Path(name).resolve()
                    break
                except FileExistsError:
                    counter += 1
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "waypoint", "utc_timestamp", "base_link", "tip_link",
                "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
                "joint_names_json", "joint_positions_rad_json"
            ])
            self.get_logger().info(f"Recording waypoints to: {self._csv_path}")
        self._waypoint_count += 1
        self._csv_writer.writerow([
            self._waypoint_count, datetime.now(timezone.utc).isoformat(),
            self.base_link, self.tip_link,
            float(pose.p[0]), float(pose.p[1]), float(pose.p[2]),
            qx, qy, qz, qw,
            json.dumps(self.chain_joint_names), json.dumps(q_chain.tolist()),
        ])
        self._csv_file.flush()
        self.get_logger().info(
            f"Saved waypoint {self._waypoint_count}: "
            f"XYZ=({pose.p[0]:.4f}, {pose.p[1]:.4f}, {pose.p[2]:.4f}) m"
        )

    def _on_timer(self):
        if self._exit_requested:
            try:
                if self.n_joints > 0:
                    self._publish_cmd(np.zeros(self.n_joints, dtype=float))
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass
            return

        if (not self._kdl_ready) or (self.current_joint_state is None) or (self._js_name_to_index is None):
            return

        with self._lock:
            target = self.target_twist.copy()
            is_shift = self.shift_held
            # Use extra-smooth ramp when Shift is held; brisk ramp otherwise
            alpha = (self.smoothing_alpha * 0.5) if is_shift else self.smoothing_alpha
            self.smoothed_twist = (1.0 - alpha) * self.smoothed_twist + alpha * target
            if np.allclose(target, 0.0) and np.linalg.norm(self.smoothed_twist) < 1e-4:
                self.smoothed_twist = np.zeros(6, dtype=float)
            cart = self.smoothed_twist.copy()

            pending_waypoints = self._waypoint_requests
            self._waypoint_requests = 0

        if pending_waypoints:
            names_ok = all(j in self._js_name_to_index for j in self.chain_joint_names)
            positions_ok = (
                all(
                    self._js_name_to_index[j] < len(self.current_joint_state.position)
                    for j in self.chain_joint_names
                )
                if names_ok
                else False
            )
            if not (names_ok and positions_ok):
                self.get_logger().warn("P ignored: /joint_states lacks joint positions.")
            else:
                q_snapshot = np.array(
                    [
                        self.current_joint_state.position[self._js_name_to_index[j]]
                        for j in self.chain_joint_names
                    ],
                    dtype=float,
                )
                if np.all(np.isfinite(q_snapshot)):
                    for _ in range(pending_waypoints):
                        try:
                            self._record_waypoint(q_snapshot)
                        except (OSError, ValueError, RuntimeError) as exc:
                            self.get_logger().error(f"Failed to write waypoint: {exc}")

        if np.allclose(cart, 0.0):
            self._publish_cmd(np.zeros(self.n_joints, dtype=float))
            return

        q_chain = np.zeros(self.n_joints, dtype=float)
        for i, jn in enumerate(self.chain_joint_names):
            idx = self._js_name_to_index.get(jn, None)
            if idx is not None:
                q_chain[i] = float(self.current_joint_state.position[idx])

        try:
            J = self._compute_jacobian(q_chain)
            qdot = self._dls_pinv_times(J, cart)
        except np.linalg.LinAlgError:
            self.get_logger().error("Jacobian inversion failed (singular).")
            self._publish_cmd(np.zeros(self.n_joints, dtype=float))
            return

        qdot[np.abs(qdot) < 1e-4] = 0.0
        self._publish_cmd(qdot)

    def _publish_cmd(self, qdot: np.ndarray):
        msg = Float64MultiArray()
        msg.data = qdot.tolist()
        self.cmd_pub.publish(msg)

    def destroy_node(self):
        try:
            if self.n_joints > 0:
                self._publish_cmd(np.zeros(self.n_joints, dtype=float))
        except Exception:
            pass

        self._kb_running = False
        self._clear_keys(hard_stop=True)

        if self._csv_file is not None:
            try:
                self._csv_file.close()
                self.get_logger().info(f"CSV saved: {self._csv_path} ({self._waypoint_count} waypoints)")
            except OSError as exc:
                self.get_logger().error(f"CSV close failed: {exc}")
        super().destroy_node()


def main():
    rclpy.init()
    node = JacobianNumpadTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()