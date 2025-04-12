import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
dt = 0.1
T = 20
N = int(T / dt)

# Robot state
state = np.array([0.0, 0.0, 0.0])
states = [state.copy()]

# Reference trajectory (circle)
radius = 5
omega_ref = 0.2
v_ref = radius * omega_ref
trajectory = np.array([
    [radius * np.cos(omega_ref * t * dt),
     radius * np.sin(omega_ref * t * dt),
     omega_ref * t * dt + np.pi / 2] for t in range(N)
])

# PID gains
Kp_v = 1.5
Kp_w = 6.0

# Simulate and store states
for t in range(N):
    x, y, theta = state
    x_ref, y_ref, theta_ref = trajectory[t]

    dx = x_ref - x
    dy = y_ref - y
    error_x = np.cos(theta) * dx + np.sin(theta) * dy
    error_y = -np.sin(theta) * dx + np.cos(theta) * dy

    v = v_ref + Kp_v * error_x
    w = Kp_w * error_y

    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += w * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    state = np.array([x, y, theta])
    states.append(state.copy())

states = np.array(states)

# --- Animation ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-radius * 1.5, radius * 1.5)
ax.set_ylim(-radius * 1.5, radius * 1.5)
ax.set_title("Differential Drive Robot (PID Tracking)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
ax.plot(trajectory[:, 0], trajectory[:, 1], 'r--', label='Reference Path')
robot_path, = ax.plot([], [], 'b-', label='Robot Path')
robot_dot, = ax.plot([], [], 'bo')  # Robot position
ax.legend()

def update(frame):
    robot_path.set_data(states[:frame+1, 0], states[:frame+1, 1])
    robot_dot.set_data(states[frame, 0], states[frame, 1])
    return robot_path, robot_dot

ani = FuncAnimation(fig, update, frames=len(states), interval=100)

# To display the animation
plt.show()

# Optional: Save to MP4 (requires ffmpeg)
# ani.save("robot_tracking.mp4", writer="ffmpeg", fps=10)
