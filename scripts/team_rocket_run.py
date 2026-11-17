#!/usr/bin/env python3
"""
team_rocket_run.py - replay a taught buzz-wire path exactly (ROS 2, feed-forward DLS IK)

Input: RAW teleop recording (buzzwire_waypoints_<date>_<time>.csv).
The Cartesian path is the FK of the recorded poses; the arm tracks it with feed-forward
damped-least-squares IK. Teaching never touched the wire, so replay stays at least as
clear as the teaching (plus SMOOTH_MM).

CHANGES vs prior version (overconstraint reduction)
  - Path metrics (arc length, retrace removal, smoothing deviation) now use only the loop
    centre position. Full 6-DoF pose is still tracked in control; planning no longer forces
    rigid-body (3-point) identity for shortcuts/smoothing. This removes unnecessary
    orientation overconstraint during path simplification while preserving taught poses.
  - Smoothing levels simplified; fewer iterative constraints.
  - Dry-run and live abort thresholds modestly relaxed for practical tracking.

WHAT IT DOES
  1. Load joints + Cartesian pose from recording(s).
  2. Remove waiting / retraced segments (poses within RETRACE_MM of a later pose, limited
     by path-length horizon). Shortcut lands on a taught pose.
  3. Mild stair-step rounding, never more than SMOOTH_MM from a taught loop pose.
     --smooth-mm 0 keeps the recording exactly.
  4. Time the path: SPEED mm/s max at loop, curvature and joint-velocity limited.
  5. 100 Hz control: feed-forward DLS IK
         desired pose/twist from taught path point
         xdot = xdot_f + K * pose_error
         qdot = J^T (J J^T + λ² I)^-1 xdot
     Path progress slows if the arm lags (does not run away).
  6. Pre-flight dry-run of the IK (pose error, sigma_min, configuration consistency).

SAFETY  Blocks if another node publishes on the command topic. Requires Enter to start.
        Zero-velocity stop on Ctrl+C, lost joint states, excessive joint/pose error, or
        prolonged lag. Start pose must be within MAX_START_DEG of taught start.

RUN
  python3 team_rocket_run.py buzzwire_waypoints_....csv --dry-run
  python3 team_rocket_run.py buzzwire_waypoints_....csv
  --speed-mm-s 10  --smooth-mm 0.5  --retrace-mm 1.0  --plan-only
Produces team_rocket_path.png and run_log_<date>_<time>.csv.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CMD_TOPIC = "/velocity_controller/commands"
CMD_JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
JOINT_STATES_TOPIC = "/joint_states"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
H = 0.0001            # fine path spacing [m]
LEVELS_MM = [0.0, 0.5, 1.0, 2.0]      # reduced set of rounding strengths (less iterative over-opt)


# ============================================================================== recording
def load_recordings(files):
    def key(f):
        m = re.findall(r"(\d{8}_\d{6})", os.path.basename(f))
        return m[-1] if m else os.path.basename(f)
    Q, P, Qt, names = [], [], [], None
    for f in sorted(files, key=key):
        with open(f) as fh:
            rd = csv.DictReader(fh)
            cols = sorted([c for c in (rd.fieldnames or []) if re.fullmatch(r"joint\d+", c)], key=lambda c: int(c[5:]))
            if len(cols) < 6:
                sys.exit(f"{f}: no joint1..joint6 columns. Give the RAW teleop recording "
                         "(buzzwire_waypoints_<date>_<time>.csv), not a fitted waypoint file.")
            names = names or cols
            for r in rd:
                try:
                    Q.append([float(r[c]) for c in names])
                    P.append([float(r["x"]), float(r["y"]), float(r["z"])])
                    Qt.append([float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])])
                except (ValueError, KeyError):
                    continue
    if len(Q) < 5:
        sys.exit("not enough points in the recording")
    return np.array(Q), np.array(P), Rotation.from_quat(np.array(Qt)), names


def tool_points(P, R, loop, side):
    """Flange, loop centre, side point (kept for optional FK checks). Planning metrics use loop only."""
    return np.hstack([P, P + R.apply(loop), P + R.apply(loop + side)])


def loop_only(Y):
    """Extract loop-centre coordinates (columns 3:6) for reduced-constraint path metrics."""
    return Y[:, 3:6]


def arc(Y):
    """Path length based on loop centre only (avoids orientation overconstraint in planning)."""
    L = loop_only(Y)
    d = np.diff(L, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])


def cut_retraces(Y, eps, horizon):
    """Skip from a pose to the LAST later pose whose loop centre is within eps (waiting / back-forth).
    Uses 3-D loop position only → less orientation overconstraint on shortcuts."""
    L = loop_only(Y)
    N, cum, tree = len(L), arc(Y), cKDTree(L)
    keep, i = [], 0
    while i < N:
        keep.append(i)
        cand = [c for c in tree.query_ball_point(L[i], eps) if c > i and cum[c] - cum[i] <= horizon]
        i = (max(cand) if cand else i) + 1
    if keep[-1] != N - 1:
        keep[-1] = N - 1
    return np.array(keep)


# ============================================================================== plan
def make_plan(Q, P, R, a, quiet=False):
    say = (lambda *x: None) if quiet else print
    loop = np.array([float(v) for v in a.loop_mm.split(",")]) / 1000.0
    side = np.array([a.side_mm / 1000.0, 0.0, 0.0])
    Y = tool_points(P, R, loop, side)
    raw_len = arc(Y)[-1]
    gaps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if gaps.max() > 0.005:
        k = int(np.argmax(gaps))
        say(f"WARNING: the recording jumps {gaps.max() * 1000:.1f} mm between points {k} and {k + 1} "
            "(two sessions joined?). The replay goes straight across - check that is free.")
    keep = cut_retraces(Y, a.retrace_mm / 1000.0, a.retrace_horizon_mm / 1000.0)
    Qk, Yk = Q[keep], Y[keep]
    s = arc(Yk)
    ok = np.concatenate([[True], np.diff(s) > 1e-9])
    Qk, Yk, s = Qk[ok], Yk[ok], s[ok]
    say(f"taught: {len(Q)} points, {raw_len * 100:.0f} cm of motion. Waiting / retraced motion removed: "
        f"{len(Q) - len(Qk)} points, {(raw_len - s[-1]) * 100:.0f} cm -> {s[-1] * 100:.0f} cm left")

    # fine polyline through the taught poses
    sf = np.append(np.arange(0.0, s[-1], H), s[-1])
    Qf = np.column_stack([np.interp(sf, s, Qk[:, j]) for j in range(Q.shape[1])])
    Yf = np.column_stack([np.interp(sf, s, Yk[:, j]) for j in range(9)])
    # Tree on loop centre only → planning deviation measured in position, not full rigid pose
    tree = cKDTree(loop_only(Yf))
    N = len(sf)

    tol = a.smooth_mm / 1000.0
    Qs, Ys, dev = Qf, Yf, np.zeros(N)
    if tol > 0:
        Ql = [Qf] + [gaussian_filter1d(Qf, m / 1000.0 / H, axis=0, mode="nearest") for m in LEVELS_MM[1:]]
        Yl = [Yf] + [gaussian_filter1d(Yf, m / 1000.0 / H, axis=0, mode="nearest") for m in LEVELS_MM[1:]]
        lvl = np.zeros(N)
        still = np.ones(N, bool)
        for k in range(1, len(LEVELS_MM)):
            still &= tree.query(loop_only(Yl[k]))[0] <= 0.8 * tol
            lvl[still] = k
        w = int(0.003 / H)
        lvl = gaussian_filter1d(minimum_filter1d(lvl, 2 * w + 1, mode="nearest"), 0.001 / H, mode="nearest")
        lvl *= np.clip(np.minimum(sf, sf[-1] - sf) / 0.005, 0.0, 1.0)      # exact at start and end
        Ql, Yl = np.array(Ql), np.array(Yl)
        idx = np.arange(N)

        def blend(lv):
            lo = np.clip(np.floor(lv).astype(int), 0, len(LEVELS_MM) - 2)
            f = (lv - lo)[:, None]
            return (1 - f) * Ql[lo, idx] + f * Ql[lo + 1, idx], (1 - f) * Yl[lo, idx] + f * Yl[lo + 1, idx]

        for _ in range(15):   # fewer iterations – less over-optimisation
            Qs, Ys = blend(lvl)
            dev = tree.query(loop_only(Ys))[0]
            bad = dev > tol
            if not bad.any():
                break
            grow = np.convolve(bad.astype(float), np.ones(2 * w + 1), mode="same") > 0
            lvl[grow] *= 0.7
            lvl[grow & (lvl < 0.05)] = 0.0
        else:
            lvl[:] = 0.0
            Qs, Ys, dev = Qf, Yf, np.zeros(N)
        say(f"stair-steps rounded: never more than {dev.max() * 1000:.2f} mm from a taught loop pose "
            f"(limit {a.smooth_mm:g} mm; average {dev.mean() * 1000:.2f} mm)")
    else:
        say("no rounding (--smooth-mm 0): the replay is exactly the recording")

    # ---- timing (curvature on loop centre only)
    sg = arc(Ys)
    ok = np.concatenate([[True], np.diff(sg) > 1e-9])
    Qs, Ys, sg, dev = Qs[ok], Ys[ok], sg[ok], dev[ok]
    v_max, a_lat, a_tan, v_min = a.speed_mm_s / 1000.0, a.bend_acc_mm_s2 / 1000.0, a.acc_mm_s2 / 1000.0, 0.0005
    L = loop_only(Ys)
    dL = np.gradient(L, sg, axis=0)
    ddL = np.gradient(dL, sg, axis=0)
    dQ = np.gradient(Qs, sg, axis=0)
    v = np.full(len(sg), v_max)
    v = np.minimum(v, np.sqrt(a_lat / np.maximum(np.linalg.norm(ddL, axis=1), 1e-9)))
    v = np.minimum(v, a.max_joint_vel / np.maximum(np.abs(dQ).max(axis=1), 1e-9))
    v = np.clip(v, v_min, v_max)
    v[0] = v[-1] = v_min
    ds = np.diff(sg)
    for i in range(len(v) - 1):                       # accelerate
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2 * a_tan * ds[i]))
    for i in range(len(v) - 2, -1, -1):               # brake
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_tan * ds[i]))
    t = np.concatenate([[0.0], np.cumsum(2 * ds / (v[:-1] + v[1:]))])
    qd = np.diff(Qs, axis=0) / np.diff(t)[:, None]
    say(f"replay: {sg[-1] * 100:.0f} cm, {t[-1]:.0f} s at up to {a.speed_mm_s:g} mm/s "
        f"(average {sg[-1] / t[-1] * 1000:.1f} mm/s), fastest joint {np.degrees(np.abs(qd).max()):.0f} deg/s")
    return dict(t=t, q=Qs, qd=qd, Y=Ys, s=sg, dev=dev, Y_taught=Yf, Q_taught=Qf, loop=loop, side=side,
                Y_raw=Y, keep=keep)


def plot_plan(plan, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    INK, INK2, GRID, SURF, C0, C1 = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29c", "#2a78d6"
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), facecolor=SURF, gridspec_kw=dict(height_ratios=[1.7, 0.7]))
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, color=GRID, lw=0.8)
    L0, L1 = plan["Y_raw"][:, 3:6] * 100, plan["Y"][:, 3:6] * 100
    ax = axs[0]
    ax.plot(L0[:, 0], L0[:, 2], color=C0, lw=0, marker="o", ms=2.2, label="taught (every recorded point)")
    ax.plot(L1[:, 0], L1[:, 2], color=C1, lw=1.6, label="replay")
    ax.annotate("start", (L1[0, 0], L1[0, 2]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.annotate("end", (L1[-1, 0], L1[-1, 2]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("loop x (cm), reversed to match the camera view", color=INK2)
    ax.set_ylabel("loop z (cm)", color=INK2)
    ax.set_title("Loop centre: replay tracks the taught path (planning uses position only)", loc="left", color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
    ax = axs[1]
    ax.plot(plan["t"], plan["dev"] * 1000, color=C1, lw=1.2)
    ax.set_xlabel("time in the replay (s)", color=INK2)
    ax.set_ylabel("mm", color=INK2)
    ax.set_title("Distance of the replay from the nearest taught loop pose", loc="left", color=INK)
    ax.set_ylim(0, max(0.6, float(plan["dev"].max()) * 1300))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    return path


# ============================================================================== kinematics (optional, for checks)
def make_fk(urdf_xml, base, tip, csv_names):
    """FK (+ Jacobian as fk.jac) of the tip link from the URDF, joints in CSV order. None if KDL is missing."""
    try:
        import PyKDL as kdl
        from urdf_parser_py.urdf import URDF
        from kdl_parser_py.urdf import treeFromUrdfModel
        robot = URDF.from_xml_string(urdf_xml)
        ok, tree = treeFromUrdfModel(robot)
        chain = tree.getChain(base, tip)
        names = [chain.getSegment(i).getJoint().getName() for i in range(chain.getNrOfSegments())
                 if chain.getSegment(i).getJoint().getTypeName() not in ("None", "Fixed")]
        if not ok or sorted(names) != sorted(csv_names):
            return None
        order = [csv_names.index(n) for n in names]
        n = len(order)
        solver = kdl.ChainFkSolverPos_recursive(chain)
        jsolver = kdl.ChainJntToJacSolver(chain)
        jm = {j.name: j for j in robot.joints}
        lim = [(jm[nm].limit.lower, jm[nm].limit.upper) if jm[nm].type != "continuous" and jm[nm].limit is not None
               else (-np.inf, np.inf) for nm in csv_names]

        def arr_of(q):
            arr = kdl.JntArray(n)
            for i, k in enumerate(order):
                arr[i] = float(q[k])
            return arr

        def fk(q):
            f = kdl.Frame()
            solver.JntToCart(arr_of(q), f)
            return (np.array([f.p[0], f.p[1], f.p[2]]),
                    np.array([[f.M[i, j] for j in range(3)] for i in range(3)]))

        def jac(q):
            Jk = kdl.Jacobian(n)
            jsolver.JntToJac(arr_of(q), Jk)
            J = np.zeros((6, n))
            for i, k in enumerate(order):
                for r in range(6):
                    J[r, k] = Jk[r, i]
            return J
        fk.keep = (chain, solver, jsolver)          # KDL solvers only hold references
        fk.limits = np.array(lim)
        fk.jac = jac
        return fk
    except Exception:
        return None


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def tcp_state(fk, q, tcp):
    """Loop (TCP) position, rotation and Jacobian at q."""
    p, Rm = fk(q)
    r = Rm @ tcp
    J = fk.jac(q)
    J[:3] -= skew(r) @ J[3:]
    return p + r, Rm, J


def dls_ik_step(fk, q, q_d, qd_d, tcp, a):
    """Feed-forward DLS IK.  xdot = xdot_f + K e ;  qdot = J^T (J J^T + lam I)^-1 xdot.
    Desired pose / twist = FK and Jacobian of the taught path point (q_d, qd_d).
    Returns qdot, position error [m], orientation error [rad], sigma_min."""
    p_d, R_d, J_d = tcp_state(fk, q_d, tcp)
    xdot_f = J_d @ qd_d                                   # feed-forward twist of the taught path
    p, Rm, J = tcp_state(fk, q, tcp)
    e_p = p_d - p
    e_o = Rotation.from_matrix(R_d @ Rm.T).as_rotvec()
    xdot = xdot_f + np.concatenate([a.kp * e_p, a.ko * e_o])
    smin = float(np.linalg.svd(J, compute_uv=False)[-1])
    lam = a.lam_min
    if smin < a.sigma_eps:
        lam += a.lam_max * (1.0 - (smin / a.sigma_eps) ** 2)
    qdot = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), xdot)
    return qdot, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o)), smin


def ik_dry_run(fk, plan, a, dt=0.02):
    """Integrate the DLS IK along the whole plan (ideal arm). Returns worst errors."""
    t_arr, q_arr, qd_arr = plan["t"], plan["q"], plan["qd"]
    q = q_arr[0].copy()
    tcp = plan["loop"]
    worst = dict(mm=0.0, deg=0.0, sigma=np.inf, cfg=0.0, t_sigma=0.0)
    for tau in np.arange(0.0, t_arr[-1], dt):
        i = int(np.clip(np.searchsorted(t_arr, tau, side="right") - 1, 0, len(t_arr) - 2))
        f = (tau - t_arr[i]) / (t_arr[i + 1] - t_arr[i])
        q_d = q_arr[i] + f * (q_arr[i + 1] - q_arr[i])
        qdot, ep, eo, smin = dls_ik_step(fk, q, q_d, qd_arr[i], tcp, a)
        m = float(np.abs(qdot).max())
        if m > a.max_joint_vel:
            qdot *= a.max_joint_vel / m
        q = q + qdot * dt
        if smin < worst["sigma"]:
            worst["sigma"], worst["t_sigma"] = smin, tau
        worst["mm"] = max(worst["mm"], ep * 1000)
        worst["deg"] = max(worst["deg"], math.degrees(eo))
        worst["cfg"] = max(worst["cfg"], math.degrees(float(np.abs(q - q_d).max())))
    return worst


def fk_points(fk, q, loop, side):
    p, Rm = fk(q)
    return np.concatenate([p, p + Rm @ loop, p + Rm @ (loop + side)])


def checks(plan, Q, P, a, fk, q_now):
    """Checks before moving. Returns True if the replay may start."""
    ok = True
    print("\nCHECKS")
    if fk is not None:
        # 1. recording vs robot model
        ii = np.linspace(0, len(Q) - 1, min(len(Q), 150)).astype(int)
        e = max(np.linalg.norm(fk(Q[i])[0] - P[i]) for i in ii) * 1000
        print(f"  1. recorded joints -> FK -> recorded position: max {e:.2f} mm "
              + ("(same robot model as while teaching)" if e < 1.0 else "!! differs: other tip link / URDF than while teaching"))
        # 2. true distance of the replay from the taught poses (loop-centric)
        Yt = np.array([fk_points(fk, q, plan["loop"], plan["side"]) for q in plan["Q_taught"][::3]])
        Yp = np.array([fk_points(fk, q, plan["loop"], plan["side"]) for q in plan["q"][::3]])
        # Compare on loop centre only
        d = cKDTree(Yt[:, 3:6]).query(Yp[:, 3:6])[0] * 1000
        lim_mm = a.smooth_mm + 0.6          # modestly relaxed
        good = d.max() <= lim_mm
        ok &= good
        print(f"  2. replay vs taught loop poses (robot FK): max {d.max():.2f} mm "
              + (f"(allowed {lim_mm:.2f})" if good else f"!! above {lim_mm:.2f} mm"))
        # 3. joint limits
        mar = np.degrees(np.min(np.minimum(plan["q"] - fk.limits[:, 0], fk.limits[:, 1] - plan["q"])))
        good = mar > 1.5
        ok &= good
        print(f"  3. closest joint limit on the path: {mar:.1f} deg" + ("" if good else "  !! too close"))
        if not a.joint_mode:
            w = ik_dry_run(fk, plan, a)
            # Relaxed acceptance: practical tracking, not theoretical zero-error
            good = w["mm"] <= 1.2 and w["deg"] <= 1.5 and w["cfg"] <= 2.5 and w["sigma"] >= 0.008
            ok &= good
            print(f"  4. DLS IK dry run of the whole path: pose error max {w['mm']:.3f} mm / {w['deg']:.3f} deg, "
                  f"sigma_min {w['sigma']:.3f} (at {w['t_sigma']:.0f} s), IK joints vs taught joints max "
                  f"{w['cfg']:.2f} deg" + ("" if good else "  !! IK does not track the taught path"))
    else:
        print("  (robot model not available: FK checks skipped, the replay itself does not need it)")
    if q_now is not None:
        dq = np.degrees(np.abs(q_now - plan["q"][0]))
        good = dq.max() <= a.max_start_deg
        ok &= good
        print(f"  5. arm now vs taught start: largest joint difference {dq.max():.1f} deg "
              f"(joint{int(np.argmax(dq)) + 1})" + ("" if good else f"  !! above {a.max_start_deg:g} deg"))
        if not good:
            print("     Move the arm near the taught start first (teleop or heal_init). Taught start joints [rad]: "
                  + ", ".join(f"{v:.4f}" for v in plan["q"][0]))
        elif fk is not None:
            d0 = np.linalg.norm(fk(q_now)[0] - fk(plan["q"][0])[0]) * 1000
            print(f"     the arm first moves {d0:.0f} mm in a straight joint move to the taught start - "
                  "make sure that way is free (best: start with the loop already at the wire start)")
    return bool(ok)


# ============================================================================== live
def run_live(a, plan, Q, P, names):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    class RunNode(Node):
        def __init__(self):
            super().__init__("prmth_run")
            self.js, self.js_time, self.urdf, self.pub = None, 0.0, None, None
            self.create_subscription(JointState, JOINT_STATES_TOPIC, self._js, 50)
            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(String, ROBOT_DESCRIPTION_TOPIC, self._urdf, qos)

        def _js(self, msg):
            if set(names).issubset(msg.name):                 # only the arm's own full messages
                self.js = dict(zip(msg.name, msg.position))
                self.js_time = time.monotonic()

        def _urdf(self, msg):
            self.urdf = msg.data

        def q(self):
            return np.array([self.js[n] for n in names], float)

        def send(self, qdot):
            by = dict(zip(names, qdot))
            m = Float64MultiArray()
            m.data = [float(by.get(j, 0.0)) for j in CMD_JOINT_ORDER]
            self.pub.publish(m)

        def zero(self):
            if self.pub is None:
                return
            for _ in range(3):
                try:
                    m = Float64MultiArray()
                    m.data = [0.0] * len(CMD_JOINT_ORDER)
                    self.pub.publish(m)
                except Exception:
                    pass
                time.sleep(0.01)

    try:   # keep ROS alive on Ctrl+C so the zero velocity still goes out
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    node = RunNode()
    log, result = [], None
    try:
        t0 = time.monotonic()
        while (node.js is None or node.urdf is None) and time.monotonic() - t0 < 8.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.js is None:
            print(f"ERROR: no arm message on {JOINT_STATES_TOPIC} - is the robot launch running?")
            return 1
        fk = make_fk(node.urdf, a.base_link, a.tip_link, names) if node.urdf else None
        q_now = node.q()
        if fk is None and not a.joint_mode:
            print(f"ERROR: DLS IK needs the robot model: no usable URDF on {ROBOT_DESCRIPTION_TOPIC} "
                  f"(or PyKDL / kdl_parser_py missing, or joints differ from {names}).")
            return 1
        print("controller: " + ("joint-space feed-forward (no IK)" if a.joint_mode else
                                "feed-forward DLS IK   xdot = xdot_f + K e,  qdot = J^T (J J^T + lam I)^-1 xdot"))
        if not checks(plan, Q, P, a, fk, q_now) and not a.force:
            print("\nCHECKS FAILED (see !!). Not moving.")
            return 1
        if a.dry_run:
            print("\nAll checks passed.  --dry-run: nothing moved.")
            return 0
        others = node.get_publishers_info_by_topic(a.cmd_topic)
        if others:
            print(f"\nERROR: {sorted({o.node_name for o in others})} already publish(es) to {a.cmd_topic} "
                  "(teleop still open?). Close it first. Not moving.")
            return 1
        node.pub = node.create_publisher(Float64MultiArray, a.cmd_topic, 10)
        input(f"\nAll checks passed. Replay takes about {plan['t'][-1]:.0f} s. "
              "Keep the e-stop in reach. Press Enter to start (Ctrl+C to cancel) ")

        T, t_arr, q_arr, qd_arr = plan["t"][-1], plan["t"], plan["q"], plan["qd"]
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.0)
        q0 = node.q()
        d0 = np.abs(q_arr[0] - q0).max()
        T_a = max(2.0, 1.875 * d0 / a.approach_rad_s) if d0 > math.radians(0.05) else 0.0
        dt = 1.0 / a.rate_hz
        e0, e1 = math.radians(a.wait_deg), math.radians(a.wait_deg * 4)
        ta = tau = 0.0
        last = time.monotonic()
        next_print, hold_since, slow_since = 0.0, None, None
        max_e = max_mm = 0.0
        result = "done"
        while True:
            now = time.monotonic()
            step = min(now - last, 3 * dt)
            last = now
            rclpy.spin_once(node, timeout_sec=0.0)
            if now - node.js_time > a.js_timeout:
                result = f"ABORT: /joint_states stopped (no data for {now - node.js_time:.2f} s)"
                break
            q = node.q()
            if ta < T_a:                                           # move to the taught start
                ta = min(ta + step, T_a)
                u = ta / T_a
                q_d = q0 + (q_arr[0] - q0) * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
                ff = (q_arr[0] - q0) * (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T_a
                if ta >= T_a:
                    ff = ff * 0.0
                scale, phase = 1.0, "to start"
            else:                                                  # the taught path
                i = int(np.clip(np.searchsorted(t_arr, tau, side="right") - 1, 0, len(t_arr) - 2))
                f = (tau - t_arr[i]) / (t_arr[i + 1] - t_arr[i])
                q_d = q_arr[i] + f * (q_arr[i + 1] - q_arr[i])
                e_now = np.abs(q_d - q).max()
                scale = float(np.clip(1.0 - (e_now - e0) / (e1 - e0), 0.05, 1.0)) if tau < T else 0.0
                ff = qd_arr[i] * scale
                tau = min(tau + step * scale, T)
                phase = "path"
            e = q_d - q
            e_max = float(np.abs(e).max())
            max_e = max(max_e, e_max)
            if e_max > math.radians(a.abort_deg):
                result = (f"ABORT: joint{int(np.argmax(np.abs(e))) + 1} is {math.degrees(e_max):.1f} deg off "
                          f"({phase}, {tau:.1f} s) - the arm is not following the commands")
                break
            if phase == "path" and tau < T and scale < 0.3:      # the path keeps waiting for the arm
                slow_since = slow_since or now
                if now - slow_since > a.stall_s:
                    result = (f"ABORT: joint{int(np.argmax(np.abs(e))) + 1} stays {math.degrees(e_max):.2f} deg behind "
                              f"for {a.stall_s:g} s ({tau:.1f} s into the path) - the arm is not following the commands")
                    break
            else:
                slow_since = None
            smin = float("nan")
            if a.joint_mode:
                qdot = ff + a.kp * e
            else:
                qdot, ep_, eo_, smin = dls_ik_step(fk, q, q_d, ff, plan["loop"], a)
                if ep_ * 1000 > a.abort_mm or math.degrees(eo_) > a.abort_rot_deg:
                    result = (f"ABORT: loop {ep_ * 1000:.1f} mm / {math.degrees(eo_):.1f} deg off the path "
                              f"({phase}, {tau:.1f} s)")
                    break
            m = float(np.abs(qdot).max())
            if m > a.max_joint_vel:
                qdot *= a.max_joint_vel / m
            node.send(qdot)
            mm = float("nan")
            if fk is not None:
                mm = float(np.linalg.norm(fk_points(fk, q, plan["loop"], plan["side"])[3:6]
                                          - fk_points(fk, q_d, plan["loop"], plan["side"])[3:6])) * 1000
                max_mm = max(max_mm, mm)
            log.append([now - t0, tau, scale, mm, math.degrees(e_max), smin, *q_d, *q, *qdot])
            if now >= next_print:
                next_print = now + 0.5
                print(f"\r  {phase:8s} {tau:6.1f} / {T:.0f} s   joint error {math.degrees(e_max):5.2f} deg"
                      + (f"   loop {mm:4.2f} mm" if fk is not None else "")
                      + (f"   waiting for the arm x{scale:.2f}" if phase == "path" and scale < 0.95 and tau < T else " " * 28),
                      end="", flush=True)
            if ta >= T_a and tau >= T:
                hold_since = hold_since or now
                if e_max < math.radians(0.03) or now - hold_since > 1.5:
                    break
            time.sleep(max(0.0, dt - (time.monotonic() - now)))
        node.zero()
        print(f"\n\n{result}")
        print(f"tracking: largest joint error {math.degrees(max_e):.2f} deg"
              + (f", loop at most {max_mm:.2f} mm from the plan" if fk is not None else ""))
        return 0 if result == "done" else 2
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        result = "cancelled"
        return 130
    finally:
        node.zero()
        if log:
            name = f"run_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
            with open(name, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "path_t", "speed_scale", "loop_err_mm", "joint_err_deg", "sigma_min"] + [f"qd_{n}" for n in names]
                           + [f"q_{n}" for n in names] + [f"cmd_{n}" for n in names])
                w.writerows([[f"{v:.6f}" for v in r] for r in log])
            print(f"log: {os.path.abspath(name)}  ({result})")
        time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description="team_rocket_run: exact replay of taught buzz-wire path (reduced overconstraint planning)")
    ap.add_argument("csv", nargs="+", help="RAW teleop recording(s): buzzwire_waypoints_<date>_<time>.csv")
    ap.add_argument("--dry-run", action="store_true", help="checks only, the robot does not move")
    ap.add_argument("--plan-only", action="store_true", help="no robot needed: plan + picture only")
    ap.add_argument("--speed-mm-s", type=float, default=10.0, help="top speed of the loop")
    ap.add_argument("--smooth-mm", type=float, default=0.5, help="rounding: max distance from a taught loop pose (0 = none)")
    ap.add_argument("--retrace-mm", type=float, default=1.0, help="loop centres this close count as the same (retrace removal; 0 = keep all)")
    ap.add_argument("--retrace-horizon-mm", type=float, default=400.0, help="longest retrace that is removed")
    ap.add_argument("--loop-mm", default="0,0,-200", help="loop centre in the end-effector frame (for speed / checks only)")
    ap.add_argument("--side-mm", type=float, default=30.0, help="side point for FK checks (planning ignores orientation constraint)")
    ap.add_argument("--bend-acc-mm-s2", type=float, default=30.0, help="slow-down in bends")
    ap.add_argument("--acc-mm-s2", type=float, default=20.0, help="speeding up / braking")
    ap.add_argument("--max-joint-vel", type=float, default=0.4, help="rad/s")
    ap.add_argument("--kp", type=float, default=3.0, help="K on the position error [1/s]")
    ap.add_argument("--ko", type=float, default=2.5, help="K on the orientation error [1/s] (slightly reduced)")
    ap.add_argument("--lam-min", type=float, default=1e-6, help="DLS damping lambda^2 away from singularities")
    ap.add_argument("--lam-max", type=float, default=1e-3, help="extra damping at a singularity")
    ap.add_argument("--sigma-eps", type=float, default=0.02, help="damping grows below this sigma_min")
    ap.add_argument("--abort-mm", type=float, default=5.0, help="loop position error that stops the run")
    ap.add_argument("--abort-rot-deg", type=float, default=4.0, help="orientation error that stops the run")
    ap.add_argument("--joint-mode", action="store_true", help="fallback: joint-space feed-forward, no IK")
    ap.add_argument("--wait-deg", type=float, default=0.15, help="joint error above which the path slows down for the arm")
    ap.add_argument("--abort-deg", type=float, default=1.2, help="joint difference to the taught joints that stops the run")
    ap.add_argument("--stall-s", type=float, default=2.5, help="stop if the arm keeps lagging this long")
    ap.add_argument("--max-start-deg", type=float, default=45.0, help="arm must be this close to the taught start")
    ap.add_argument("--approach-rad-s", type=float, default=0.15, help="joint speed of the move to the taught start")
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--js-timeout", type=float, default=0.1)
    ap.add_argument("--force", action="store_true", help="move even if a check failed")
    ap.add_argument("--cmd-topic", default=CMD_TOPIC)
    ap.add_argument("--base-link", default="base_link")
    ap.add_argument("--tip-link", default="end-effector")
    a = ap.parse_args()

    Q, P, R, names = load_recordings(a.csv)
    if a.retrace_mm <= 0:
        a.retrace_mm = 1e-6
    plan = make_plan(Q, P, R, a)
    png = plot_plan(plan, "team_rocket_path.png")
    if png:
        print(f"picture: {os.path.abspath(png)}")
    if a.plan_only:
        return 0
    return run_live(a, plan, Q, P, names)


if __name__ == "__main__":
    sys.exit(main())
