import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1  # time step
T = 20    # total simulation time (s)
N = int(T / dt)

# Robot state: [x, y, theta]
state = np.array([0.0, 0.0, 0.0])  # initial position
states = [state.copy()]

# Reference trajectory (circular path)
radius = 5
omega_ref = 0.2  # angular velocity of the reference
v_ref = radius * omega_ref

trajectory = np.array([
    [radius * np.cos(omega_ref * t * dt),
     radius * np.sin(omega_ref * t * dt),
     omega_ref * t * dt + np.pi/2] for t in range(N)
])

# PID control gains
Kp_v = 1.5   # linear velocity proportional gain
Kp_w = 1100   # angular velocity proportional gain

# Main simulation loop
for t in range(N):
    x, y, theta = state
    x_ref, y_ref, theta_ref = trajectory[t]

    # Position error in global frame
    dx = x_ref - x
    dy = y_ref - y

    # Transform error into robot's frame
    error_x = np.cos(theta) * dx + np.sin(theta) * dy
    error_y = -np.sin(theta) * dx + np.cos(theta) * dy

    # Control law (P only for simplicity)
    v = v_ref + Kp_v * error_x
    w = Kp_w * error_y

    # Update robot state using unicycle model
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += w * dt

    # Normalize theta to [-π, π]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    state = np.array([x, y, theta])
    states.append(state.copy())

states = np.array(states)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r--', label='Reference Path')
plt.plot(states[:, 0], states[:, 1], 'b-', label='Robot Path (PID)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Differential Drive Robot Trajectory Tracking (PID)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
