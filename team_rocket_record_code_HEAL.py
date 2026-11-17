#!/usr/bin/env python3
"""
ROS2 keyboard->Cartesian velocity -> DLS Jacobian pseudo-inverse -> joint velocities publisher.

UPDATE (per your request):
- Replaced pynput with pygame for robust "hold key" behavior (keydown/keyup-like).
- Pygame runs in a dedicated thread, uses pygame.key.get_pressed() for smooth motion.
- Clears motion on focus loss, ESC clears keys, configurable EXIT key (default 'x').
- Optionally grabs input so terminal doesn't receive keys.

Controls:
q/a: +/-X, w/s: +/-Y, e/d: +/-Z
r/f: +/-Roll(wx), t/g: +/-Pitch(wy), y/h: +/-Yaw(wz)
ESC: stop (clears motion)
P  : save one waypoint (Cartesian pose + joint positions) to CSV
X  : exit (clean shutdown; configurable)

Run:
  python3 ros2_jacobian_keyboard_record_path.py --ros-args -p tip_link:="end-effector"

Hoop tracing (after confirming which base axes match your board):
  Rising side -> upper bend -> descending side -> lower bend -> next rising side.
  For a bend turning to the right in the board plane, use short holds:
  UP, UP+RIGHT, RIGHT, DOWN+RIGHT, DOWN (upper arch).
  A lower U bend: DOWN, DOWN+RIGHT, RIGHT, UP+RIGHT, UP.
  Use P at start, mid-bend(s), and end. These are approximate manual motions,
  NOT automatic timing or collision-safe commands.
  Map UP/DOWN/RIGHT to verified X/Y/Z controls for your actual robot base frame.
  Normal v=0.075 unchanged. Be careful near physical obstacles.
"""

from __future__ import annotations

import time
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


class JacobianKeyboardTeleop(Node):
    def __init__(self):
        super().__init__("jacobian_keyboard_teleop")

        # -------------------------
        # Params (edit / override)
        # -------------------------
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("robot_description_topic", "/robot_description")
        self.declare_parameter("cmd_topic", "/velocity_controller/commands")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("tip_link", "end-effector")
        self.declare_parameter("rate_hz", 200.0)
        self.declare_parameter("v", 0.075)          # cartesian linear m/s (and angular rad/s)
        self.declare_parameter("damping", 0.05)     # lambda in your DLS

        # Pygame options
        self.declare_parameter("keyboard_hz", 200.0)
        self.declare_parameter("pygame_window_w", 420)
        self.declare_parameter("pygame_window_h", 195)
        self.declare_parameter("pygame_grab_input", True)
        self.declare_parameter("exit_key", "x")              # press/hold to exit cleanly
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
        # Key mapping (same semantics)
        # -------------------------
        v = self.v
        self.key_mapping = {
            "q": np.array([ v, 0, 0, 0, 0, 0], dtype=float),
            "a": np.array([-v, 0, 0, 0, 0, 0], dtype=float),
            "w": np.array([0,  v, 0, 0, 0, 0], dtype=float),
            "s": np.array([0, -v, 0, 0, 0, 0], dtype=float),
            "e": np.array([0, 0,  v, 0, 0, 0], dtype=float),
            "d": np.array([0, 0, -v, 0, 0, 0], dtype=float),
            "r": np.array([0, 0, 0,  v, 0, 0], dtype=float),
            "f": np.array([0, 0, 0, -v, 0, 0], dtype=float),
            "t": np.array([0, 0, 0, 0,  v, 0], dtype=float),
            "g": np.array([0, 0, 0, 0, -v, 0], dtype=float),
            "y": np.array([0, 0, 0, 0, 0,  v], dtype=float),
            "h": np.array([0, 0, 0, 0, 0, -v], dtype=float),
        }

        # pygame key -> char mapping (robust, no unicode dependency)
        self._pg_key_to_char = {
            pygame.K_q: "q",
            pygame.K_a: "a",
            pygame.K_w: "w",
            pygame.K_s: "s",
            pygame.K_e: "e",
            pygame.K_d: "d",
            pygame.K_r: "r",
            pygame.K_f: "f",
            pygame.K_t: "t",
            pygame.K_g: "g",
            pygame.K_y: "y",
            pygame.K_h: "h",
        }

        self._exit_pg_key = self._pygame_key_from_char(self.exit_key_char)
        if self._exit_pg_key is None:
            self.exit_key_char = "x"
            self._exit_pg_key = pygame.K_x

        # -------------------------
        # Keyboard state (thread-safe)
        # -------------------------
        self._lock = threading.Lock()
        self.pressed_keys: set[str] = set()
        self._waypoint_requests = 0
        self._p_was_down = False
        self._waypoint_count = 0
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        self.active_twist = np.zeros(6, dtype=float)

        # -------------------------
        # ROS pub/sub
        # -------------------------
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.current_joint_state: JointState | None = None
        self._js_name_to_index = None

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 50)

        # /robot_description is latched TRANSIENT_LOCAL
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
        # KDL state (built once)
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
        self._last_warn_missing = 0.0

        self.get_logger().info(
            f"Started. Waiting for {self.robot_description_topic} (URDF) and {self.joint_states_topic}..."
        )
        self.get_logger().info(
            f"Publishing joint velocities to {self.cmd_topic} at {self.rate_hz:.1f} Hz. "
            f"base_link='{self.base_link}', tip_link='{self.tip_link}'."
        )
        self.get_logger().info(
            "Controls: q/a w/s e/d (XYZ), r/f t/g y/h (RPY), P record, ESC stop, "
            f"{self.exit_key_char.upper()} exit. Click pygame window to focus."
        )

    # -------------------------
    # Pygame utilities
    # -------------------------
    @staticmethod
    def _pygame_key_from_char(ch: str) -> int | None:
        if not ch:
            return None
        c = ch.lower()[0]
        if "a" <= c <= "z":
            return getattr(pygame, f"K_{c}", None)
        if "0" <= c <= "9":
            return getattr(pygame, f"K_{c}", None)
        return None

    def _recompute_twist(self):
        tw = np.zeros(6, dtype=float)
        for k in self.pressed_keys:
            vec = self.key_mapping.get(k, None)
            if vec is not None:
                tw += vec
        self.active_twist = tw

    def _clear_keys(self):
        with self._lock:
            self.pressed_keys.clear()
            self._recompute_twist()

    def request_exit(self, reason: str = ""):
        if self._exit_requested:
            return
        self._exit_requested = True
        if reason:
            self.get_logger().warn(f"Exit requested: {reason}")
        else:
            self.get_logger().warn("Exit requested.")
        self._kb_running = False
        self._clear_keys()
        # stop motion immediately
        try:
            self._publish_cmd(np.zeros(self.n_joints if self.n_joints > 0 else 7, dtype=float))
        except Exception:
            pass

    def _pygame_loop(self):
        """
        Robust, smooth "hold" teleop:
          - pygame.key.get_pressed()
          - clears on focus loss
          - ESC clears keys
          - exit key triggers clean shutdown
        """
        try:
            pygame.init()
            screen = pygame.display.set_mode((self.pygame_w, self.pygame_h))
            pygame.display.set_caption("Teleop Keys (CLICK HERE)")
            clock = pygame.time.Clock()

            # Grab input so keystrokes don't go to terminal
            try:
                pygame.event.set_grab(self.pygame_grab_input)
            except Exception:
                pass

            font = pygame.font.Font(None, 22)
            info = [
                "CLICK THIS WINDOW (focus required).",
                "Hold keys -> continuous motion.",
                "ESC: stop (clear motion).",
                "P: record waypoint to CSV.",
                f"{self.exit_key_char.upper()}: exit (clean shutdown).",
                "Close window exits (if enabled).",
            ]

            while rclpy.ok() and self._kb_running and (not self._exit_requested):
                pygame.event.pump()

                # Handle quit event
                for event in pygame.event.get([pygame.QUIT]):
                    if event.type == pygame.QUIT:
                        if self.exit_on_window_close:
                            self.request_exit("Pygame window closed.")
                        else:
                            self.get_logger().warn("Pygame window closed -> stopping keyboard thread.")
                            self._kb_running = False
                            self._clear_keys()
                        break

                if (not self._kb_running) or self._exit_requested:
                    break

                if not pygame.key.get_focused():
                    self._clear_keys()
                    self._p_was_down = False
                else:
                    keys = pygame.key.get_pressed()
                    # Rising edge: one CSV waypoint per P press, no auto-repeat.
                    p_down = bool(keys[pygame.K_p])
                    if p_down and not self._p_was_down:
                        with self._lock:
                            self._waypoint_requests += 1
                    self._p_was_down = p_down

                    # Exit key
                    if self._exit_pg_key is not None and keys[self._exit_pg_key]:
                        self.request_exit(f"Exit key '{self.exit_key_char}' pressed.")
                    # ESC clears motion
                    elif keys[pygame.K_ESCAPE]:
                        self._clear_keys()
                    else:
                        held = set()
                        for pgk, ch in self._pg_key_to_char.items():
                            if keys[pgk]:
                                held.add(ch)

                        with self._lock:
                            self.pressed_keys = held
                            self._recompute_twist()

                # UI
                screen.fill((25, 25, 25))
                y = 10
                for line in info:
                    screen.blit(font.render(line, True, (230, 230, 230)), (10, y))
                    y += 24

                with self._lock:
                    keys_str = "".join(sorted(self.pressed_keys)) if self.pressed_keys else "(none)"
                screen.blit(font.render(f"Pressed: {keys_str}", True, (180, 220, 180)), (10, y + 4))

                pygame.display.flip()
                clock.tick(max(60.0, self.keyboard_hz))

        except Exception as e:
            self.get_logger().error(f"Pygame loop crashed: {e}")
            self._clear_keys()
        finally:
            try:
                pygame.quit()
            except Exception:
                pass

    # -------------------------
    # ROS callbacks
    # -------------------------
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
                    f"KDL chain has 0 joints for {self.base_link} -> {self.tip_link}. "
                    "Check base_link/tip_link names."
                )
                return

            # Extract ONLY movable joints (stop once we have exactly n)
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
                self.get_logger().error(
                    f"Joint-name extraction mismatch: getNrOfJoints()={n}, extracted {len(names)}: {names}"
                )
                return

            self.chain = chain
            self.n_joints = n
            self.chain_joint_names = names
            self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
            self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

            self._kdl_ready = True
            self.get_logger().info(
                f"KDL ready: n_joints={self.n_joints}, chain joint order={self.chain_joint_names}"
            )

            # only need it once
            self.destroy_subscription(self._robot_desc_sub)

        except Exception as e:
            self.get_logger().error(f"Exception creating KDL from URDF: {e}")

    # -------------------------
    # Core math (same as yours)
    # -------------------------
    def _compute_jacobian(self, q_chain: np.ndarray) -> np.ndarray:
        """Return J (6 x n) in chain joint order."""
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
        """
        Your exact DLS formula:
            J# = J^T (J J^T + λ I)^-1
            qdot = J# * v
        Note: kept as + lambda * I (not lambda^2) exactly like your ROS1/ROS2 draft.
        """
        lam = float(self.damping)
        A = J @ J.T + lam * np.eye(6)
        qdot = (J.T @ np.linalg.inv(A)) @ v6
        return qdot

    # -------------------------
    # CSV waypoint capture (runs on ROS timer thread, not pygame thread)
    # -------------------------
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
            # New file only; never overwrite any existing directory file.
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
            f"P saved waypoint {self._waypoint_count}: "
            f"XYZ=({pose.p[0]:.4f}, {pose.p[1]:.4f}, {pose.p[2]:.4f}) m"
        )

    # -------------------------
    # Timer loop
    # -------------------------
    def _on_timer(self):
        # Exit requested by pygame thread
        if self._exit_requested:
            # publish zeros once more and shutdown
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

        # Must have KDL + joint states
        if (not self._kdl_ready) or (self.current_joint_state is None) or (self._js_name_to_index is None):
            return

        with self._lock:
            cart = self.active_twist.copy()

        # Collect pending P presses even when no movement key is held.
        with self._lock:
            pending_waypoints = self._waypoint_requests
            self._waypoint_requests = 0
        if pending_waypoints:
            # Reject incomplete or invalid states; never invent missing joint values.
            names_ok = all(j in self._js_name_to_index for j in self.chain_joint_names)
            positions_ok = all(self._js_name_to_index[j] < len(self.current_joint_state.position)
                               for j in self.chain_joint_names) if names_ok else False
            if not (names_ok and positions_ok):
                self.get_logger().warn("P ignored: /joint_states lacks one or more chain joint positions.")
            else:
                q_snapshot = np.array([
                    self.current_joint_state.position[self._js_name_to_index[j]]
                    for j in self.chain_joint_names], dtype=float)
                if np.all(np.isfinite(q_snapshot)):
                    for _ in range(pending_waypoints):
                        try:
                            self._record_waypoint(q_snapshot)
                        except (OSError, ValueError, RuntimeError) as exc:
                            self.get_logger().error(f"Failed to write waypoint: {exc}")
                else:
                    self.get_logger().warn("P ignored: invalid joint positions.")

        # If no keys: publish zeros
        if np.allclose(cart, 0.0):
            self._publish_cmd(np.zeros(self.n_joints, dtype=float))
            return

        # Build q in chain order using names
        q_chain = np.zeros(self.n_joints, dtype=float)
        missing = False
        for i, jn in enumerate(self.chain_joint_names):
            idx = self._js_name_to_index.get(jn, None)
            if idx is None:
                missing = True
                q_chain[i] = 0.0
            else:
                q_chain[i] = float(self.current_joint_state.position[idx])

        if missing:
            now = time.time()
            if now - self._last_warn_missing > 2.0:
                self._last_warn_missing = now
                self.get_logger().warn(
                    "Some chain joint names were not found in JointState. "
                    "Check naming consistency between URDF and /joint_states."
                )

        # Jacobian + DLS
        try:
            J = self._compute_jacobian(q_chain)
            qdot = self._dls_pinv_times(J, cart)
        except np.linalg.LinAlgError:
            self.get_logger().error("Jacobian inversion failed (singular). Try different input.")
            self._publish_cmd(np.zeros(self.n_joints, dtype=float))
            return

        # Small velocity deadband
        qdot[np.abs(qdot) < 1e-3] = 0.0

        self._publish_cmd(qdot)

    def _publish_cmd(self, qdot: np.ndarray):
        msg = Float64MultiArray()
        msg.data = qdot.tolist()
        self.cmd_pub.publish(msg)

    # -------------------------
    # Shutdown
    # -------------------------
    def destroy_node(self):
        # stop robot
        try:
            if self.n_joints > 0:
                self._publish_cmd(np.zeros(self.n_joints, dtype=float))
        except Exception:
            pass

        # stop keyboard loop
        self._kb_running = False
        self._clear_keys()

        if self._csv_file is not None:
            try:
                self._csv_file.close()
                self.get_logger().info(f"CSV saved: {self._csv_path} ({self._waypoint_count} waypoints)")
            except OSError as exc:
                self.get_logger().error(f"CSV close failed: {exc}")
        super().destroy_node()


def main():
    rclpy.init()
    node = JacobianKeyboardTeleop()
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
