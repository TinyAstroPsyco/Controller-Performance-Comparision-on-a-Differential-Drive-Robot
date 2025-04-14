import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from matplotlib.widgets import Button, RadioButtons
from matplotlib import gridspec
from scipy.spatial.distance import cdist
import pandas as pd


# Loading the trajectory from a csv:
def load_trajectory_from_csv(csv_path, num_points_between=10):
    """
    Load x,y coordinates from a CSV file and generate a trajectory.
    
    Args:
        csv_path: Path to the CSV file (with x in column A, y in column B)
        num_points_between: Number of interpolation points between each waypoint
        
    Returns:
        List of (x,y) tuples representing the trajectory
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract x and y columns (assuming they're in columns A and B)
    # Convert column names to match pandas convention if needed
    if 'A' in df.columns and 'B' in df.columns:
        x_col, y_col = 'A', 'B'
    else:
        # Assume first and second columns
        x_col, y_col = df.columns[0], df.columns[1]
    
    # Get the waypoints as a list of (x,y) tuples
    waypoints = list(zip(df[x_col], df[y_col]))
    
    # Generate the trajectory using the existing custom function
    trajectory = TrajectoryGenerator.custom(waypoints, csv_path, num_points_between)
    
    return trajectory


class DifferentialDriveRobot:
    """Differential drive robot model."""
    def __init__(self, x=0, y=0, theta=0, wheel_radius=0.05, wheel_base=0.2, color='blue'): # Everything is in meters
        # Robot state [x, y, theta]
        self.state = np.array([x, y, theta])
        
        # Robot physical parameters
        self.wheel_radius = wheel_radius  # wheel radius in meters
        self.wheel_base = wheel_base      # distance between wheels in meters
        
        # Robot dimensions for visualization
        self.length = 0.3 # 0.3 Meters
        self.width = 0.2 # 0.2 Meters
        
        # Robot color for visualization
        self.color = color
        
        # Control inputs [v, omega] (linear and angular velocity)
        self.control_inputs = np.array([0.0, 0.0])
        
    def update(self, dt, v, omega):
        """Update robot state based on control inputs."""
        # Store control inputs
        self.control_inputs = np.array([v, omega])
        
        # Current state
        x, y, theta = self.state
        
        # State update equations (kinematics model)
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        
        # Update state
        self.state = np.array([x_new, y_new, theta_new])
        
        return self.state
    
    def get_wheel_velocities(self, v, omega):
        """Calculate wheel velocities from v and omega."""
        vr = (2*v + omega*self.wheel_base) / (2*self.wheel_radius)  # right wheel
        vl = (2*v - omega*self.wheel_base) / (2*self.wheel_radius)  # left wheel
        return vl, vr

# Generates Basic Trajectories such as circle, square and also a Custom Trajectory from the check points.
class TrajectoryGenerator:
    """Generates various trajectories for testing."""
    @staticmethod
    def circle(radius=3.0, num_points=1000):
        """Generate a circular trajectory."""
        t = np.linspace(0, 2*np.pi, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        return list(zip(x, y))
    
    @staticmethod
    def figure_eight(a=2.0, b=2.0, num_points=1000):
        """Generate a figure-eight trajectory."""
        t = np.linspace(0, 2*np.pi, num_points)
        x = a * np.sin(t)
        y = b * np.sin(2*t)
        return list(zip(x, y))
    
    @staticmethod
    def square(side=3.0, num_points_per_side=25):
        """Generate a square trajectory."""
        trajectory = []
        
        # Create points for each side of the square
        # Bottom side (left to right)
        for i in range(num_points_per_side):
            t = i / (num_points_per_side - 1)
            x = -side/2 + side * t
            y = -side/2
            trajectory.append((x, y))
        
        # Right side (bottom to top)
        for i in range(num_points_per_side):
            t = i / (num_points_per_side - 1)
            x = side/2
            y = -side/2 + side * t
            trajectory.append((x, y))
        
        # Top side (right to left)
        for i in range(num_points_per_side):
            t = i / (num_points_per_side - 1)
            x = side/2 - side * t
            y = side/2
            trajectory.append((x, y))
        
        # Left side (top to bottom)
        for i in range(num_points_per_side):
            t = i / (num_points_per_side - 1)
            x = -side/2
            y = side/2 - side * t
            trajectory.append((x, y))
            
        return trajectory
    
    @staticmethod
    def custom(points=None, csv_path='Trajectories/closureless_trajectory.csv', num_points_between=10):
        """Generate a trajectory from a list of waypoints or a CSV file.
        
        Args:
            points: List of (x,y) tuples as waypoints, or None if using CSV
            csv_path: Path to a CSV file with x,y coordinates, or None if using points
            num_points_between: Number of interpolation points between waypoints
            
        Returns:
            List of (x,y) tuples representing the trajectory
        """
        # If CSV path is provided and points not provided, load waypoints from the file
        if points is None and csv_path is not None:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Extract x and y columns (assuming they're in columns A and B)
            # Convert column names to match pandas convention if needed
            if 'A' in df.columns and 'B' in df.columns:
                x_col, y_col = 'A', 'B'
            else:
                # Assume first and second columns
                x_col, y_col = df.columns[0], df.columns[1]
            
            # Get the waypoints as a list of (x,y) tuples
            points = list(zip(df[x_col], df[y_col]))
            print(f'Loaded {len(points)} waypoints from {csv_path}')
        
        # Make sure we have points to work with
        if points is None or len(points) < 2:
            raise ValueError("Either provide points or a valid CSV file with at least 2 waypoints")
        
        # Generate trajectory by interpolating between waypoints
        trajectory = []
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            for j in range(num_points_between):
                t = j / num_points_between
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                trajectory.append((x, y))
                
        # Add the last point
        trajectory.append(points[-1])
        
        print(f'Generated trajectory with {len(trajectory)} points')
        return trajectory

# Controllers from the original file
class PIDController:
    def __init__(self):
        self.prev_dist_error = 0.0
        self.prev_angle_error = 0.0
        self.dist_error_sum = 0.0
        self.angle_error_sum = 0.0
        
    def control(self, robot_state, target_point, dt):
        x, y, theta = robot_state # Has the current position of the robot and the heading
        x_target, y_target = target_point # This is the a (x,y) point on the trajectory that is looking ahead and changes continunously
        
        # Calculate the distance to target
        dist_to_target = np.sqrt((x_target - x)**2 + (y_target - y)**2) # I am computing my distance to the target point from the current state
        
        # Calculate the angle to the target
        angle_to_target = np.arctan2(y_target - y, x_target - x) # Computing the heading
        
        # Calculate the angle error (normalized between -pi and pi)
        angle_error = np.arctan2(np.sin(angle_to_target - theta), np.cos(angle_to_target - theta))
        
        # Calculate error derivatives
        dist_error_deriv = (dist_to_target - self.prev_dist_error) / dt
        angle_error_deriv = (angle_error - self.prev_angle_error) / dt
        
        # Update error integrals (with anti-windup)
        self.dist_error_sum += dist_to_target * dt
        self.dist_error_sum = np.clip(self.dist_error_sum, -1.0, 1.0)  # Anti-windup
        
        self.angle_error_sum += angle_error * dt
        self.angle_error_sum = np.clip(self.angle_error_sum, -1.0, 1.0)  # Anti-windup
        
        # Store current errors for next iteration
        self.prev_dist_error = dist_to_target
        self.prev_angle_error = angle_error
        
        # PID gains for linear velocity
        kp_v = 0.8   # Proportional gain
        ki_v = 1.0   # Integral gain
        kd_v = 1   # Derivative gain
        
        # PID gains for angular velocity
        kp_omega = 2.0   # Proportional gain
        ki_omega = 1.0   # Integral gain
        kd_omega = 0.6   # Derivative gain
        
        # Calculate linear velocity using PID
        v = (kp_v * dist_to_target + 
             ki_v * self.dist_error_sum - 
             kd_v * dist_error_deriv)
        
        # Calculate angular velocity using PID
        omega = (kp_omega * angle_error + 
                 ki_omega * self.angle_error_sum + 
                 kd_omega * angle_error_deriv)
        
        # Apply limits
        v = max(0.2, min(v, 5.0))  # Clamp between 0.2 and 5.0
        omega = np.clip(omega, -np.pi, np.pi)  # Limit angular velocity
        
        return v, omega

class MPCController:
    def __init__(self):
        self.last_v = 0.0
        self.last_omega = 0.0
        
    def control(self, robot_state, target_point, dt):
        # MPC parameters
        N = 8  # Prediction horizon
        
        # Current state
        x, y, theta = robot_state
        x_target, y_target = target_point
        v_desired = 5 # 5 m/s
        
        # Cost function weights
        w_pos = 3      # Position error weight
        w_theta = 3   # Heading error weight
        w_v = 0.05        # Velocity smoothness weight
        w_omega = 0.2    # Angular velocity smoothness weight
        w_speed_tracking = 1
        
        # Control constraints
        v_min, v_max = 0.2, 5.0
        omega_min, omega_max = -np.pi, np.pi
        
        # Simple MPC approach: evaluate multiple control sequences and pick the best
        best_cost = float('inf')
        best_v = self.last_v
        best_omega = self.last_omega
        
        # Discretized control space to search
        # v_options = np.linspace(max(v_min, self.last_v - 0.5), 
        #                        min(v_max, self.last_v + 0.5), 5)
        v_options = np.linspace(v_min, v_max, 10)  # Search full range!

        omega_options = np.linspace(max(omega_min, self.last_omega - 0.5), 
                                   min(omega_max, self.last_omega + 0.5), 7)
        
        # Evaluate each control sequence
        for v in v_options:
            for omega in omega_options:
                # Predict future states using simplified model
                pred_x, pred_y, pred_theta = x, y, theta
                total_cost = 0
                
                for i in range(N):
                    # Simulate the robot model forward
                    pred_x += v * np.cos(pred_theta) * dt
                    pred_y += v * np.sin(pred_theta) * dt
                    pred_theta += omega * dt
                    
                    # Position error to target
                    pos_error = np.sqrt((x_target - pred_x)**2 + (y_target - pred_y)**2)
                    
                    # Heading error to target
                    desired_theta = np.arctan2(y_target - pred_y, x_target - pred_x)
                    theta_error = np.arctan2(np.sin(desired_theta - pred_theta), 
                                            np.cos(desired_theta - pred_theta))
                    
                    # Control smoothness cost
                    v_change = abs(v - self.last_v)
                    omega_change = abs(omega - self.last_omega)

                    speed_tracking_cost = w_speed_tracking * abs(v - v_desired)
                    # Weighted cost for this step
                    step_cost = (w_pos * pos_error + 
                                w_theta * abs(theta_error) + 
                                w_v * v_change + 
                                w_omega * omega_change + speed_tracking_cost)
                    
                    # Discount future costs
                    discount = 0.9 ** i
                    total_cost += discount * step_cost
                
                # If this control sequence has lower cost, select it
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_omega = omega
        
        # Save the chosen controls for next iteration
        self.last_v = best_v
        self.last_omega = best_omega
        
        return best_v, best_omega

# ADDITIONAL CONTROLLERS

class PDController:
    """Proportional-Derivative controller for trajectory tracking."""
    def __init__(self):
        self.prev_angle_error = 0.0
        self.prev_distance_error = 0.0
        
    def control(self, robot_state, target_point, dt):
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Calculate the distance to target
        dist_to_target = np.sqrt((x_target - x)**2 + (y_target - y)**2)
        
        # calculating the distance error rate
        dist_error_der = (dist_to_target - self.prev_distance_error) / dt
        self.prev_distance_error = dist_to_target
        # Calculate the angle to the target
        angle_to_target = np.arctan2(y_target - y, x_target - x)
        
        # Calculate the angle error (normalized between -pi and pi)
        angle_error = np.arctan2(np.sin(angle_to_target - theta), np.cos(angle_to_target - theta))
        
        # Calculate error derivative for the angle
        angle_error_deriv = (angle_error - self.prev_angle_error) / dt
        
        # Store current error for next iteration
        self.prev_angle_error = angle_error
        
        # PD gains
        kp_v = 2    # Proportional gain for velocity
        kd_v = 1
        kp_omega = 2.0  # Proportional gain for angular velocity
        kd_omega = 1.0  # Derivative gain for angular velocity
        
        # Calculate linear velocity - proportional to distance
        v = kp_v * dist_to_target + kd_v * dist_error_der
        v = max(0.2, min(v, 5.0))  # Clamp between 0.2 and 5.0
        
        # Calculate angular velocity - PD control on the heading error
        omega = kp_omega * angle_error + kd_omega * angle_error_deriv
        omega = np.clip(omega, -np.pi, np.pi)  # Limit angular velocity
        
        return v, omega

class LQRController:
    """Linear Quadratic Regulator controller for trajectory tracking."""
    def __init__(self):
        # LQR gain matrices
        self.K = np.array([
            [1.5, 0.0, 0.0],  # Gain for x position error
            [0.0, 1.5, 2.0]   # Gain for y position and heading errors
        ])
        
    def control(self, robot_state, target_point, dt):
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Calculate error in robot frame
        error_global = np.array([
            x_target - x,
            y_target - y
        ])
        
        # Rotate error to robot frame
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, s],
            [-s, c]
        ])
        error_robot = R @ error_global
        
        # Create error state vector [x_error, y_error, theta_error]
        # For theta error, use the angle to the target
        angle_to_target = np.arctan2(error_global[1], error_global[0])
        theta_error = np.arctan2(np.sin(angle_to_target - theta), np.cos(angle_to_target - theta))
        error_state = np.array([error_robot[0], error_robot[1], theta_error])
        
        # Calculate control inputs using LQR gains
        v_r = self.K[0] @ error_state  # Right wheel velocity
        v_l = self.K[1] @ error_state  # Left wheel velocity
        
        # Convert wheel velocities to v and omega
        v = (v_r + v_l) / 2
        omega = (v_r - v_l) / 0.2  # 0.2 is the wheel base
        
        # Apply limits
        v = max(0.2, min(v, 5.0))
        omega = np.clip(omega, -np.pi, np.pi)
        
        return v, omega

class OpenLoopController:
    """Open loop controller that ignores target point and just drives forward."""
    def __init__(self):
        self.v = 1.0  # Constant forward velocity
        self.omega = 0.0  # Initial angular velocity
        self.t = 0.0  # Time tracking
        
    def control(self, robot_state, target_point, dt):
        # Increment time
        self.t += dt
        
        # Periodically change direction to create an interesting path
        if int(self.t / 3) % 2 == 0:
            self.omega = 0.3
        else:
            self.omega = -0.3
            
        return self.v, self.omega

class FastMPCController:
    """MPC controller optimized for speed."""
    def __init__(self):
        self.last_v = 0.0
        self.last_omega = 0.0
        
    def control(self, robot_state, target_point, dt):
        # MPC parameters
        N = 20  # Prediction horizon
        
        # Current state
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Cost function weights with priority on speed
        w_pos = 0.5      # Position error weight (lower to prioritize speed)
        w_theta = 1    # Heading error weight (lower to prioritize speed)
        w_v = 0.05       # Velocity smoothness weight (lower to allow faster acceleration)
        w_omega = 0.1    # Angular velocity smoothness weight
        
        # Control constraints with higher minimum velocity
        v_min, v_max = 0, 5.0  # Minimum velocity of 0.5 instead of 0
        omega_min, omega_max = -np.pi, np.pi
        
        # Velocity bias to encourage higher speeds
        velocity_bias = 0.3
        
        # Simple MPC approach: evaluate multiple control sequences and pick the best
        best_cost = float('inf')
        best_v = self.last_v
        best_omega = self.last_omega
        
        # Discretized control space to search with bias toward acceleration
        v_range_min = max(v_min, self.last_v - 0.3)  # Less deceleration
        v_range_max = min(v_max, self.last_v + 0.7)  # More acceleration
        v_options = np.linspace(v_range_min, v_range_max, 5)
        
        omega_options = np.linspace(max(omega_min, self.last_omega - 0.5), 
                                   min(omega_max, self.last_omega + 0.5), 7)
        
        # Evaluate each control sequence
        for v in v_options:
            for omega in omega_options:
                # Predict future states using simplified model
                pred_x, pred_y, pred_theta = x, y, theta
                total_cost = 0
                
                for i in range(N):
                    # Simulate the robot model forward
                    pred_x += v * np.cos(pred_theta) * dt
                    pred_y += v * np.sin(pred_theta) * dt
                    pred_theta += omega * dt
                    
                    # Position error to target
                    pos_error = np.sqrt((x_target - pred_x)**2 + (y_target - pred_y)**2)
                    
                    # Heading error to target
                    desired_theta = np.arctan2(y_target - pred_y, x_target - pred_x)
                    theta_error = np.arctan2(np.sin(desired_theta - pred_theta), 
                                            np.cos(desired_theta - pred_theta))
                    
                    # Control smoothness cost
                    v_change = abs(v - self.last_v)
                    omega_change = abs(omega - self.last_omega)
                    
                    # Add velocity reward (negative cost for higher velocities)
                    velocity_reward = -velocity_bias * v / v_max
                    
                    # Weighted cost for this step
                    step_cost = (w_pos * pos_error + 
                                w_theta * abs(theta_error) + 
                                w_v * v_change + 
                                w_omega * omega_change +
                                velocity_reward)  # Adding velocity reward term
                    
                    # Discount future costs
                    discount = 0.9 ** i
                    total_cost += discount * step_cost
                
                # If this control sequence has lower cost, select it
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_omega = omega
        
        # Save the chosen controls for next iteration
        self.last_v = best_v
        self.last_omega = best_omega
        
        return best_v, best_omega

# MULTI-ROBOT SIMULATOR WITH UPDATED CONTROLLERS

class MultiRobotSimulator:
    """Simulator for multiple differential drive robots with different controllers."""
    def __init__(self, dt=0.05, lookahead_distance=0.5):
        
        # Simulation parameters
        self.dt = dt  # time step in seconds
        self.t = 0    # current time
        self.lookahead_distance = lookahead_distance  # lookahead distance for path following
        
        # Define robot colors 
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # Initialize robots with different controllers
        self.robots = []
        self.controllers = []
        self.controller_names = []
        
        # Robot 1 - PID controller (blue)
        self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[0]))
        self.controllers.append(PIDController())
        self.controller_names.append("PID")
        
        # Robot 2 - MPC controller (red)
        self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[1]))
        self.controllers.append(MPCController())
        self.controller_names.append("MPC")
        
        # Robot 3 - PD controller (green)
        self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[2]))
        self.controllers.append(PDController())
        self.controller_names.append("PD")
        
        # Robot 4 - LQR controller (orange)
        # self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[3]))
        # self.controllers.append(LQRController())
        # self.controller_names.append("LQR")
        
        # Robot 5 - Fast MPC controller (purple)
        self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[4]))
        self.controllers.append(FastMPCController())
        self.controller_names.append("Fast MPC")
        
        # Robot 6 - Open Loop controller (brown)
        # self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[5]))
        # self.controllers.append(OpenLoopController())
        # self.controller_names.append("Open Loop")
        
        # Number of robots
        self.num_robots = len(self.robots)
        
        # Trajectory parameters and data
        self.trajectory_type = "custom"
        self.trajectory = TrajectoryGenerator.custom(csv_path="Trajectories/closureless_trajectory.csv")
        self.trajectory_array = np.array(self.trajectory)  # For efficient distance calculations
        
        # Robot tracking data
        self.closest_point_idx = [0] * self.num_robots
        self.target_point_idx = [0] * self.num_robots
        self.tracking_errors = [[] for _ in range(self.num_robots)]
        self.control_inputs = [[] for _ in range(self.num_robots)]
        self.robot_paths = [[] for _ in range(self.num_robots)]
        self.projected_points = [None] * self.num_robots
        
        # Common data
        self.timestamps = []
        
        # Visualization elements
        self.robot_bodies = []
        self.directions = []
        self.path_lines = []
        self.closest_point_circles = []
        self.target_point_circles = []
        self.cross_track_lines = []
        self.error_lines = []
        self.v_lines = []
        self.omega_lines = []
        
        # Setup visualization
        self.setup_visualization()
        
    def reset(self):

        # Reset robots
        colors = [r.color for r in self.robots]  # Save colors
        self.robots = []
        for i in range(self.num_robots):
            self.robots.append(DifferentialDriveRobot(x=0, y=0, theta=0, color=colors[i]))
        
        # Reset controllers
        controller_classes = [c.__class__ for c in self.controllers]
        self.controllers = []
        for controller_class in controller_classes:
            self.controllers.append(controller_class())
        
        # Reset time and indices
        self.t = 0
        self.closest_point_idx = [0] * self.num_robots
        self.target_point_idx = [0] * self.num_robots
        
        # Reset metrics
        self.tracking_errors = [[] for _ in range(self.num_robots)]
        self.control_inputs = [[] for _ in range(self.num_robots)]
        self.robot_paths = [[] for _ in range(self.num_robots)]
        self.projected_points = [None] * self.num_robots
        self.timestamps = []
        
        # Reset plots
        for i in range(self.num_robots):
            self.path_lines[i].set_data([], [])
            self.error_lines[i].set_data([], [])
            self.v_lines[i].set_data([], [])
            self.omega_lines[i].set_data([], [])
            self.cross_track_lines[i].set_data([], [])
            
            # Reset target visualization
            if self.trajectory:
                self.closest_point_circles[i].center = self.trajectory[0]
                self.target_point_circles[i].center = self.trajectory[0]
        
        # Restart animation if stopped
        try:
            if hasattr(self, 'anim') and not getattr(self, 'anim_running', True):
                self.anim.event_source.start()
                self.anim_running = True
        except:
            # If there's an issue with the animation, restart it
            if hasattr(self, 'fig'):
                self.anim = animation.FuncAnimation(
                    self.fig, self.update_frame, interval=int(self.dt * 1000),
                    blit=False
                )
                self.anim_running = True

    def set_trajectory(self, trajectory_type):
        """Set the trajectory type and generate the trajectory."""
        self.trajectory_type = trajectory_type
        
        if trajectory_type == "circle":
            self.trajectory = TrajectoryGenerator.circle()
        elif trajectory_type == "eight":
            self.trajectory = TrajectoryGenerator.figure_eight()
        elif trajectory_type == "square":
            self.trajectory = TrajectoryGenerator.square()
        elif trajectory_type == "custom":
            # Default custom path from CSV
            self.trajectory = TrajectoryGenerator.custom(csv_path="Trajectories/closureless_trajectory.csv")
        
        # Update trajectory array for distance calculations
        self.trajectory_array = np.array(self.trajectory)
        
        # Update trajectory visualization
        traj_x = [point[0] for point in self.trajectory]
        traj_y = [point[1] for point in self.trajectory]
        self.traj_line.set_data(traj_x, traj_y)
        
        # Auto-adjust the view to fit the trajectory with some margin
        margin = 1.0  # Add 1 meter margin around trajectory
        x_min, x_max = min(traj_x) - margin, max(traj_x) + margin
        y_min, y_max = min(traj_y) - margin, max(traj_y) + margin
        
        # Keep aspect ratio square
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range > y_range:
            y_center = (y_min + y_max) / 2
            y_min = y_center - x_range / 2
            y_max = y_center + x_range / 2
        else:
            x_center = (x_min + x_max) / 2
            x_min = x_center - y_range / 2
            x_max = x_center + y_range / 2
            
        self.ax_sim.set_xlim(x_min, x_max)
        self.ax_sim.set_ylim(y_min, y_max)
        
        # Reset simulation
        self.reset()

    def find_closest_point(self, robot_state):
        """Find the closest point on the trajectory to the current robot position."""
        if not self.trajectory:
            return 0
            
        robot_x, robot_y, _ = robot_state
        robot_position = np.array([[robot_x, robot_y]])
        
        # Calculate distances to all points in the trajectory
        distances = cdist(robot_position, self.trajectory_array).flatten()
        
        # Find the closest point
        closest_idx = np.argmin(distances)
        
        return closest_idx

    def find_target_point(self, closest_idx):
        """Find a target point on the trajectory ahead of the closest point."""
        if not self.trajectory:
            return 0
            
        # Start from the closest point
        target_idx = closest_idx
        cumulative_distance = 0.0
        
        # Move along the trajectory until we reach the lookahead distance
        while cumulative_distance < self.lookahead_distance and target_idx < len(self.trajectory) - 1:
            current_point = np.array(self.trajectory[target_idx])
            next_point = np.array(self.trajectory[target_idx + 1])
            
            segment_distance = np.linalg.norm(next_point - current_point)
            cumulative_distance += segment_distance
            target_idx += 1
        
        return target_idx

    def calculate_cross_track_error(self, robot_state, closest_idx, robot_index):
        """Calculate the cross-track error (perpendicular distance to the path)."""
        if closest_idx >= len(self.trajectory) - 1:
            return self.tracking_errors[robot_index][-1] if self.tracking_errors[robot_index] else 0
            
        # Get the robot position
        robot_x, robot_y, _ = robot_state
        robot_pos = np.array([robot_x, robot_y])
        
        # Get the closest point and the next point on the trajectory
        p1 = np.array(self.trajectory[closest_idx])
        p2 = np.array(self.trajectory[closest_idx + 1])
        
        # Calculate the direction vector of the path segment
        path_vector = p2 - p1
        path_length = np.linalg.norm(path_vector)
        
        if path_length < 1e-6:  # Avoid division by zero
            return np.linalg.norm(robot_pos - p1)
            
        # Normalize the path vector
        path_direction = path_vector / path_length
        
        # Calculate the vector from the closest point to the robot
        robot_vector = robot_pos - p1
        
        # Project the robot vector onto the path direction
        projection_length = np.dot(robot_vector, path_direction)
        
        # Calculate the projected point on the path
        projected_point = p1 + projection_length * path_direction
        
        # Calculate the cross-track error
        cross_track_error = np.linalg.norm(robot_pos - projected_point)
        
        # Store the projected point for visualization
        self.projected_points[robot_index] = projected_point
        
        return cross_track_error

    def step(self):
        """Perform one simulation step for all robots."""
        # Check if we've reached the end of the trajectory
        if not self.trajectory or all(idx >= len(self.trajectory) - 1 for idx in self.target_point_idx):
            return False
        
        # Process each robot
        for i in range(self.num_robots):
            # Skip robots that have reached the end
            # if self.target_point_idx[i] >= len(self.trajectory) - 1 and self.robots[i].state == self.trajectory[-1]:
            cross_track_error_forend = self.calculate_cross_track_error(
                self.robots[i].state, self.closest_point_idx[i], i
            )
            if self.target_point_idx[i] >= len(self.trajectory) - 1 and  cross_track_error_forend <= 0.01:
                # Keep the robot stationary at its current position
                v, omega = 0.0, 0.0
                
                # Record zero control inputs
                self.control_inputs[i].append((v, omega))
                
                # Add the same position to path (robot stays in place)
                self.robot_paths[i].append((self.robots[i].state[0], self.robots[i].state[1]))
                
                # Use the last error value (or 0 if none exists)
                last_error = self.tracking_errors[i][-1] if self.tracking_errors[i] else 0
                self.tracking_errors[i].append(last_error)
                
                # Continue to next robot
                continue
                
            # Find the closest point on the trajectory to the robot
            self.closest_point_idx[i] = self.find_closest_point(self.robots[i].state)
            
            # Find the target point ahead on the trajectory
            self.target_point_idx[i] = self.find_target_point(self.closest_point_idx[i])
            
            # Get the target point
            target_point = self.trajectory[self.target_point_idx[i]]
            
            # Calculate the cross-track error
            cross_track_error = self.calculate_cross_track_error(
                self.robots[i].state, self.closest_point_idx[i], i
            )
            
            # Call the controller function to get control inputs
            v, omega = self.controllers[i].control(
                self.robots[i].state, target_point, self.dt
            )
            
            # Update robot state
            self.robots[i].update(self.dt, v, omega)
            
            # Record metrics
            self.tracking_errors[i].append(cross_track_error)
            self.control_inputs[i].append((v, omega))
            self.robot_paths[i].append((self.robots[i].state[0], self.robots[i].state[1]))
        
        # Update time
        self.timestamps.append(self.t)
        self.t += self.dt
        
        return True

    def setup_visualization(self):
        """Set up the visualization environment for multiple robots."""
        # Create figure and axes
        self.fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(3, 4)
        
        # Main simulation plot
        self.ax_sim = self.fig.add_subplot(gs[:2, :2])
        self.ax_sim.set_aspect('equal')
        window_val = 20
        self.ax_sim.set_xlim(-0.5 * window_val, window_val)
        self.ax_sim.set_ylim(-1 * window_val, window_val)
        self.ax_sim.set_xlabel('X (m)')
        self.ax_sim.set_ylabel('Y (m)')
        self.ax_sim.set_title('Differential Drive Robots - Controller Comparison')
        self.ax_sim.grid(False)
        
        # Error plot
        self.ax_error = self.fig.add_subplot(gs[0, 2:])
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error (m)')
        self.ax_error.set_title('Cross-Track Error Comparison')
        self.ax_error.grid(True)
        
        # Control input plots (two separate plots for v and omega)
        self.ax_v = self.fig.add_subplot(gs[1, 2])
        self.ax_v.set_xlabel('Time (s)')
        self.ax_v.set_ylabel('v (m/s)')
        self.ax_v.set_title('Linear Velocity')
        self.ax_v.grid(True)
        
        self.ax_omega = self.fig.add_subplot(gs[1, 3])
        self.ax_omega.set_xlabel('Time (s)')
        self.ax_omega.set_ylabel('ω (rad/s)')
        self.ax_omega.set_title('Angular Velocity')
        self.ax_omega.grid(True)
        
        # Trajectory selection
        self.ax_traj = self.fig.add_subplot(gs[2, 0])
        self.ax_traj.set_title('Trajectory Selection')
        self.ax_traj.axis('off')
        
        # Controls
        self.ax_controls = self.fig.add_subplot(gs[2, 1:3])
        self.ax_controls.set_title('Controls')
        self.ax_controls.axis('off')
        
        # Performance metrics
        self.ax_metrics = self.fig.add_subplot(gs[2, 3])
        self.ax_metrics.set_title('Performance Metrics')
        self.ax_metrics.axis('off')
        
        # Trajectory visualization
        traj_x = [point[0] for point in self.trajectory]
        traj_y = [point[1] for point in self.trajectory]
        self.traj_line, = self.ax_sim.plot(traj_x, traj_y, 'k-', alpha=0.7, linewidth=2, label='Reference Path')
        
        # Initialize visualization elements for each robot
        for i in range(self.num_robots):
            robot = self.robots[i]
            color = robot.color
            controller_name = self.controller_names[i]
            
            # Robot body
            robot_body = Rectangle(
                (0, 0), robot.width, robot.length, 
                fill=True, color=color, alpha=0.7
            )
            self.robot_bodies.append(self.ax_sim.add_patch(robot_body))
            
            # Direction indicator
            direction = plt.Line2D([0, 0], [0, 0], color='black', lw=2)
            self.directions.append(direction)
            self.ax_sim.add_line(direction)
            
            # Path line
            path_line, = self.ax_sim.plot([], [], linestyle='--', color=color, alpha=0.7, 
                                        label=f'{controller_name}')
            self.path_lines.append(path_line)
            
            # Closest point
            closest_point = Circle((0, 0), radius=0.06, fill=True, color=color, alpha=0.3)
            self.closest_point_circles.append(self.ax_sim.add_patch(closest_point))
            
            # Target point
            target_point = Circle((0, 0), radius=0.1, fill=True, color=color, alpha=0.5)
            self.target_point_circles.append(self.ax_sim.add_patch(target_point))
            
            # Cross-track error line
            cross_track_line, = self.ax_sim.plot([], [], color=color, lw=1, alpha=0.5)
            self.cross_track_lines.append(cross_track_line)
            
            # Error plot line
            error_line, = self.ax_error.plot([], [], color=color, 
                                            label=f'{controller_name}')
            self.error_lines.append(error_line)
            
            # Velocity plot lines
            v_line, = self.ax_v.plot([], [], color=color, 
                                    label=f'{controller_name}')
            self.v_lines.append(v_line)
            
            omega_line, = self.ax_omega.plot([], [], color=color, 
                                            label=f'{controller_name}')
            self.omega_lines.append(omega_line)
            
            # Add text annotation for robot type
            self.ax_sim.text(0.02, 0.95 - i*0.05, f'{color.capitalize()}: {controller_name}', 
                            transform=self.ax_sim.transAxes, color=color)
        
        # Add legends
        self.ax_sim.legend(loc='upper left')
        self.ax_error.legend(loc='upper left')
        self.ax_v.legend(loc='upper left')
        self.ax_omega.legend(loc='upper left')
        
        # Trajectory selection radio buttons
        self.radio_traj = RadioButtons(
            self.ax_traj, ('circle', 'eight', 'square', 'custom'),
            active=3  # Default to custom
        )
        self.radio_traj.on_clicked(self.set_trajectory)
        
        # Reset button
        self.ax_reset = plt.axes([0.82, 0.01, 0.1, 0.04])
        self.button_reset = Button(self.ax_reset, 'Reset')
        self.button_reset.on_clicked(lambda event: self.reset())
        
        # Lookahead distance slider
        self.ax_lookahead = plt.axes([0.55, 0.15, 0.3, 0.03])
        from matplotlib.widgets import Slider
        self.slider_lookahead = Slider(
            self.ax_lookahead, 'Lookahead', 0.1, 2.0, 
            valinit=self.lookahead_distance
        )
        self.slider_lookahead.on_changed(self.update_lookahead)
        
        # Animation
        self.anim_running = True
        self.anim = animation.FuncAnimation(
            self.fig, self.update_frame, interval=int(self.dt * 1000),
            blit=False, save_count= 100 # Limit frame caching
        )
        
        # plt.tight_layout()

    def update_lookahead(self, val):
        """Update the lookahead distance."""
        self.lookahead_distance = val

    def update_frame(self, frame):
        """Update animation frame."""
        # Perform simulation step
        active = self.step()
        
        if not active:
            try:
                self.anim.event_source.stop()
                self.anim_running = False
            except:
                pass
            return
        
        # Update visualizations for each robot
        for i in range(self.num_robots):
            # Update robot visualization
            self.update_robot_visualization(self.robots[i], self.robot_bodies[i], self.directions[i])
            
            # Update closest and target point visualization
            if self.closest_point_idx[i] < len(self.trajectory):
                self.closest_point_circles[i].center = self.trajectory[self.closest_point_idx[i]]
            
            if self.target_point_idx[i] < len(self.trajectory):
                self.target_point_circles[i].center = self.trajectory[self.target_point_idx[i]]
            
            # Update cross-track error line visualization
            if self.projected_points[i] is not None:
                robot_x, robot_y, _ = self.robots[i].state
                self.cross_track_lines[i].set_data(
                    [robot_x, self.projected_points[i][0]],
                    [robot_y, self.projected_points[i][1]]
                )
            
            # Update path visualization
            if self.robot_paths[i]:
                path_x = [point[0] for point in self.robot_paths[i]]
                path_y = [point[1] for point in self.robot_paths[i]]
                self.path_lines[i].set_data(path_x, path_y)
        

        # Update error plot
        if self.timestamps:
            for i in range(self.num_robots):
                if self.tracking_errors[i]:
                    self.error_lines[i].set_data(self.timestamps, self.tracking_errors[i])
            
            self.ax_error.relim()
            self.ax_error.autoscale_view()
            
            # Calculate average errors for metrics display
            avg_errors = []
            for i in range(self.num_robots):
                avg_error = np.mean(self.tracking_errors[i]) if self.tracking_errors[i] else 0
                avg_errors.append(avg_error)
            
            # Update metrics text with sorted errors
            sorted_indices = np.argsort(avg_errors)
            metrics_text = "Average Errors (Best to Worst):\n"
            for rank, idx in enumerate(sorted_indices):
                controller_name = self.controller_names[idx]
                metrics_text += f"{rank+1}. {controller_name}: {avg_errors[idx]:.4f} m\n"
            
            # Clear and update the metrics axis
            self.ax_metrics.clear()
            self.ax_metrics.text(0.1, 0.5, metrics_text, verticalalignment='center')
            self.ax_metrics.set_title('Performance Metrics')
            self.ax_metrics.axis('off')
        
        # Update velocity plots
        if self.timestamps:
            for i in range(self.num_robots):
                if self.control_inputs[i]:
                    v_values = [inputs[0] for inputs in self.control_inputs[i]]
                    omega_values = [inputs[1] for inputs in self.control_inputs[i]]
                    
                    self.v_lines[i].set_data(self.timestamps, v_values)
                    self.omega_lines[i].set_data(self.timestamps, omega_values)
            
            self.ax_v.relim()
            self.ax_v.autoscale_view()
            self.ax_omega.relim()
            self.ax_omega.autoscale_view()

    def update_robot_visualization(self, robot, body, direction):
        """Update the robot visualization based on current state."""
        x, y, theta = robot.state
        
        # Create a transformation matrix
        t = transforms.Affine2D().rotate(theta).translate(x, y)
        
        # Apply the transformation to the robot body
        x_offset = -robot.width / 2
        y_offset = -robot.length / 2
        body.set_xy((x_offset, y_offset))
        body.set_transform(t + self.ax_sim.transData)
        
        # Update the direction indicator
        dir_length = robot.length / 2
        direction.set_data(
            [x, x + dir_length * np.cos(theta)],
            [y, y + dir_length * np.sin(theta)]
        )

    def run(self):
        """Run the simulation (show the plot)."""
        plt.show()

    def get_data(self):
        """Return the simulation data for analysis."""
        data = {
            'timestamps': self.timestamps,
            'trajectory': self.trajectory
        }
        
        for i in range(self.num_robots):
            controller_name = self.controller_names[i]
            data[f'tracking_errors_{controller_name}'] = self.tracking_errors[i]
            data[f'control_inputs_{controller_name}'] = self.control_inputs[i]
            data[f'robot_path_{controller_name}'] = self.robot_paths[i]
        
        return data
    

if __name__ == "__main__":
    # Create the multi-robot simulator
    sim = MultiRobotSimulator()

    # Run the simulation
    sim.run()