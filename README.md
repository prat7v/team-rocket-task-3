# Team Rocket — Task 3: HEAL Buzz-Wire Path Teaching and Replay

ME 639: Introduction to Robotics · Project Task 3

## Overview

This project implements a teach-and-replay workflow for guiding a loop along a buzz-wire course using the HEAL robotic arm. Keyboard teleoperation is used to teach Cartesian poses and record joint configurations. A replay script processes a taught path and tracks it using feed-forward motion and damped least-squares (DLS) inverse kinematics.

The implementation uses Python, ROS 2, NumPy, SciPy, PyKDL, URDF/KDL parsers, Pygame, and Matplotlib.

## Approach and Methodology

1. **Robot model and feedback:** Read the URDF from `/robot_description` and joint feedback from `/joint_states`. Build a KDL chain from `base_link` to `end-effector` to compute forward kinematics and the Jacobian.
2. **Teaching:** Convert keyboard inputs into Cartesian linear and angular velocities. Map these to joint velocities through a damped Jacobian inverse and publish to `/velocity_controller/commands`. Save selected poses and joint positions as CSV waypoints.
3. **Path preparation:** Load raw trajectory recordings, remove waiting/retraced sections, and resample the path. Optional smoothing limits deviation of the loop centre from the taught path. Planning metrics use loop-centre position; the controller still tracks position and orientation.
4. **Time parameterization:** Limit speed using loop curvature, tangential acceleration, and joint velocity constraints.
5. **Replay:** At 100 Hz by default, combine the desired Cartesian feed-forward twist with position and orientation error feedback. Compute joint commands using a DLS Jacobian inverse, with increased damping near singularities. Slow path progress when the arm lags behind.
6. **Checks and logging:** Check model consistency, joint limits, start configuration, and simulated IK tracking before execution. Log tracking errors, joint positions, and velocity commands during a run.

The replay controller uses the form:

```text
v_command = v_feedforward + K * pose_error
q_dot = J.T * inverse(J * J.T + damping * I) * v_command
```

## Source Files

| File | Purpose |
| --- | --- |
| `team_rocket_record_code_HEAL.py` | Keyboard Cartesian teleoperation and manual waypoint recording |
| `teamrocket_datatrace.py` | NumPad teleoperation with a precision mode and manual waypoint recording |
| `team_rocket_run.py` | Path preparation, plotting, preflight checks, and trajectory replay |

These scripts and the raw trajectory CSV must be included in the repository alongside this README. The lab robot driver and robot description are external dependencies.

## Setup

Use a Linux environment with the lab-compatible ROS 2 distribution, the HEAL driver workspace, and Python 3.10 or newer. Source ROS 2 and the robot workspace in each terminal:

```bash
source /opt/ros/<your_ros_distro>/setup.bash
source <path_to_addverb_ws>/install/setup.bash
```

Required Python modules:

```text
rclpy, sensor_msgs, std_msgs, numpy, scipy, pygame, matplotlib,
PyKDL, urdf_parser_py, kdl_parser_py
```

Install the ROS/KDL components using packages compatible with your ROS distribution. Install the remaining libraries in the Python environment used by ROS:

```bash
python3 -m pip install numpy scipy pygame matplotlib
```

Clone the project:

```bash
git clone https://github.com/prat7v/team-rocket-task-3.git
cd team-rocket-task-3
```

Start the HEAL server using the supplied lab procedure. With the server running, launch the velocity controller in another sourced terminal:

```bash
ros2 launch addverb_cobot_control velocity_control.launch.py
```

Confirm that the robot description and joint feedback are available:

```bash
ros2 topic list
ros2 topic echo /joint_states --once
```

## Teaching Waypoints

```bash
python3 team_rocket_record_code_HEAL.py --ros-args -p tip_link:=end-effector
```

Focus the Pygame window before using the keyboard.

| Keys | Action |
| --- | --- |
| `Q / A` | Positive / negative X translation |
| `W / S` | Positive / negative Y translation |
| `E / D` | Positive / negative Z translation |
| `R / F` | Positive / negative roll velocity |
| `T / G` | Positive / negative pitch velocity |
| `Y / H` | Positive / negative yaw velocity |
| `P` | Save a waypoint |
| `Esc` | Clear motion commands |
| `X` | Exit and close the recording |

The recorder saves a waypoint on each press of `P`; it does not continuously record every control cycle. The NumPad alternative is:

```bash
python3 teamrocket_datatrace.py
```

Its window displays the key bindings, including Shift for precision motion.

### Recording Format

The replay script expects a raw CSV with numeric columns:

```text
joint1,joint2,joint3,joint4,joint5,joint6,x,y,z,qx,qy,qz,qw
```

Joint positions are in radians, Cartesian positions in metres, and quaternion order is XYZW. A recording must contain at least five valid rows.

The manual waypoint recorders currently save positions as `x_m,y_m,z_m` and joint data in `joint_names_json` and `joint_positions_rad_json`. Their output is **not directly compatible** with the replay loader. Before replay, convert it by mapping the joint names to `joint1` through `joint6`, expanding the corresponding joint values into numeric columns, and renaming the position columns to `x,y,z`. Use a sufficiently dense, validated taught path; sparse waypoint interpolation can cut across the wire course.

## Planning and Replay

Replace `recording.csv` with the compatible raw trajectory file.

### Offline plan and plot

This mode does not require a robot connection and does not publish motion commands:

```bash
python3 team_rocket_run.py recording.csv --plan-only
```

### Preflight checks

With the robot server, description, joint feedback, and controller available:

```bash
python3 team_rocket_run.py recording.csv --dry-run
```

### Execute the replay

Stop the teleoperation node before starting replay so that only one node commands the velocity controller. Verify the loop offset, taught start configuration, and approach route for the actual setup. The script can move toward the taught start before replaying the path.

```bash
python3 team_rocket_run.py recording.csv --speed-mm-s 10 --smooth-mm 0.5
```

The script requests Enter before motion. Use `Ctrl+C` to cancel. Software checks include stale joint-state detection, excessive tracking error, persistent lag, and competing command publishers; these do not provide collision detection or guarantee wire clearance.

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--speed-mm-s` | `10` | Maximum loop-centre speed |
| `--smooth-mm` | `0.5` | Allowed planning deviation from taught loop positions |
| `--retrace-mm` | `1.0` | Proximity threshold for removing retraced motion |
| `--loop-mm` | `0,0,-200` | Loop-centre offset in the end-effector frame; calibrate for the tool |
| `--max-joint-vel` | `0.4` | Joint speed limit in rad/s |
| `--rate-hz` | `100` | Replay control frequency |

To disable smoothing and retrace removal together:

```bash
python3 team_rocket_run.py recording.csv --smooth-mm 0 --retrace-mm 0 --plan-only
```

For the full option list:

```bash
python3 team_rocket_run.py --help
```

## Outputs

- `team_rocket_path.png`: taught and planned loop-centre paths, with planning deviation.
- `run_log_<date>_<time>.csv`: execution timing, speed scaling, tracking errors, singularity metric, desired/measured joint positions, and commanded velocities.

Hardware performance and successful wire clearance should be assessed from the actual run log and demonstration; no measured performance figures are claimed here.

## Demo Video

**[Watch the project demonstration](REPLACE_WITH_UNLISTED_DEMO_URL)**

Replace this placeholder with the unlisted video link before submission. The Task 3 demonstration must be no longer than **1.5 minutes** and show the project running with a brief walkthrough of its main functionality.
