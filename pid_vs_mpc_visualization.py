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
    trajectory = TrajectoryGenerator.custom(waypoints, "trajectory.csv",num_points_between)
    
    return trajectory






class DifferentialDriveRobot:
    """Differential drive robot model."""
    def __init__(self, x=0, y=0, theta=0, wheel_radius=0.05, wheel_base=0.2, color='blue'):
        # Robot state [x, y, theta]
        self.state = np.array([x, y, theta])
        
        # Robot physical parameters
        self.wheel_radius = wheel_radius  # wheel radius in meters
        self.wheel_base = wheel_base      # distance between wheels in meters
        
        # Robot dimensions for visualization
        self.length = 0.3
        self.width = 0.2
        
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
    def custom(points=None, csv_path='trajectory.csv', num_points_between=10):
        """Generate a trajectory from a list of waypoints or a CSV file.
        
        Args:
            points: List of (x,y) tuples as waypoints, or None if using CSV
            csv_path: Path to a CSV file with x,y coordinates, or None if using points
            num_points_between: Number of interpolation points between waypoints
            
        Returns:
            List of (x,y) tuples representing the trajectory
        """
        # If CSV path is provided, load waypoints from the file
        if csv_path is not None:
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
def pid_controller(robot_state, target_point, dt):
    """
    PID controller for adaptive path following.
    
    Args:
        robot_state: [x, y, theta] state of the robot
        target_point: (x, y) target position on the path ahead
        dt: time step in seconds
        
    Returns:
        v: linear velocity
        omega: angular velocity
    """
    # Static variables to store previous errors and integrals
    if not hasattr(pid_controller, "prev_dist_error"):
        pid_controller.prev_dist_error = 0.0
        pid_controller.prev_angle_error = 0.0
        pid_controller.dist_error_sum = 0.0
        pid_controller.angle_error_sum = 0.0
    
    x, y, theta = robot_state
    x_target, y_target = target_point
    
    # Calculate the distance to target
    dist_to_target = np.sqrt((x_target - x)**2 + (y_target - y)**2)
    
    # Calculate the angle to the target
    angle_to_target = np.arctan2(y_target - y, x_target - x)
    
    # Calculate the angle error (normalized between -pi and pi)
    angle_error = np.arctan2(np.sin(angle_to_target - theta), np.cos(angle_to_target - theta))
    
    # Calculate error derivatives
    dist_error_deriv = (dist_to_target - pid_controller.prev_dist_error) / dt
    angle_error_deriv = (angle_error - pid_controller.prev_angle_error) / dt
    
    # Update error integrals (with anti-windup)
    pid_controller.dist_error_sum += dist_to_target * dt
    pid_controller.dist_error_sum = np.clip(pid_controller.dist_error_sum, -1.0, 1.0)  # Anti-windup
    
    pid_controller.angle_error_sum += angle_error * dt
    pid_controller.angle_error_sum = np.clip(pid_controller.angle_error_sum, -1.0, 1.0)  # Anti-windup
    
    # Store current errors for next iteration
    pid_controller.prev_dist_error = dist_to_target
    pid_controller.prev_angle_error = angle_error
    
    # PID gains for linear velocity
    kp_v = 2.0   # Proportional gain
    ki_v = 1  # Integral gain
    kd_v = 0.6   # Derivative gain
    
    # PID gains for angular velocity
    kp_omega = 2.0   # Proportional gain
    ki_omega = 1   # Integral gain
    kd_omega = 0.6   # Derivative gain
    
    # Calculate linear velocity using PID
    v = (kp_v * dist_to_target + 
         ki_v * pid_controller.dist_error_sum + 
         kd_v * dist_error_deriv)
    
    # Calculate angular velocity using PID
    omega = (kp_omega * angle_error + 
             ki_omega * pid_controller.angle_error_sum + 
             kd_omega * angle_error_deriv)
    
    # Apply limits
    v = max(0.2, min(v, 5.0))  # Clamp between 0.2 and 5.0
    omega = np.clip(omega, -np.pi, np.pi)  # Limit angular velocity
    
    return v, omega

def mpc_controller(robot_state, target_point, dt):
    """
    Simple MPC controller for path following.
    
    Args:
        robot_state: [x, y, theta] state of the robot
        target_point: (x, y) target position on the path ahead
        dt: time step in seconds
        
    Returns:
        v: linear velocity
        omega: angular velocity
    """
    # MPC parameters
    N = 10  # Prediction horizon
    
    # Initialize storage for MPC if not already done
    if not hasattr(mpc_controller, "initialized"):
        mpc_controller.initialized = True
        mpc_controller.last_v = 0.0
        mpc_controller.last_omega = 0.0
    
    # Current state
    x, y, theta = robot_state
    x_target, y_target = target_point
    
    # Cost function weights
    w_pos = 1.0      # Position error weight
    w_theta = 0.8    # Heading error weight
    w_v = 0.1        # Velocity smoothness weight
    w_omega = 0.2    # Angular velocity smoothness weight
    
    # Control constraints
    v_min, v_max = 0.0, 5.0
    omega_min, omega_max = -np.pi, np.pi
    
    # Simple MPC approach: evaluate multiple control sequences and pick the best
    best_cost = float('inf')
    best_v = mpc_controller.last_v
    best_omega = mpc_controller.last_omega
    
    # Discretized control space to search
    v_options = np.linspace(max(v_min, mpc_controller.last_v - 0.5), 
                           min(v_max, mpc_controller.last_v + 0.5), 5)
    omega_options = np.linspace(max(omega_min, mpc_controller.last_omega - 0.5), 
                               min(omega_max, mpc_controller.last_omega + 0.5), 7)
    
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
                v_change = abs(v - mpc_controller.last_v)
                omega_change = abs(omega - mpc_controller.last_omega)
                
                # Weighted cost for this step
                step_cost = (w_pos * pos_error + 
                            w_theta * abs(theta_error) + 
                            w_v * v_change + 
                            w_omega * omega_change)
                
                # Discount future costs
                discount = 0.9 ** i
                total_cost += discount * step_cost
            
            # If this control sequence has lower cost, select it
            if total_cost < best_cost:
                best_cost = total_cost
                best_v = v
                best_omega = omega
    
    # Save the chosen controls for next iteration
    mpc_controller.last_v = best_v
    mpc_controller.last_omega = best_omega
    
    return best_v, best_omega

# Create independent instances of controllers to avoid shared state
class PIDController:
    def __init__(self):
        self.prev_dist_error = 0.0
        self.prev_angle_error = 0.0
        self.dist_error_sum = 0.0
        self.angle_error_sum = 0.0
        
    def control(self, robot_state, target_point, dt):
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Calculate the distance to target
        dist_to_target = np.sqrt((x_target - x)**2 + (y_target - y)**2)
        
        # Calculate the angle to the target
        angle_to_target = np.arctan2(y_target - y, x_target - x)
        
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
        kp_v = 2.0   # Proportional gain
        ki_v = 1.0   # Integral gain
        kd_v = 0.6   # Derivative gain
        
        # PID gains for angular velocity
        kp_omega = 2.0   # Proportional gain
        ki_omega = 1.0   # Integral gain
        kd_omega = 0.6   # Derivative gain
        
        # Calculate linear velocity using PID
        v = (kp_v * dist_to_target + 
             ki_v * self.dist_error_sum + 
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
        N = 10  # Prediction horizon
        
        # Current state
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Cost function weights
        w_pos = 1.0      # Position error weight
        w_theta = 1.5    # Heading error weight
        w_v = 0.01        # Velocity smoothness weight
        w_omega = 0.1    # Angular velocity smoothness weight
        
        # Control constraints
        v_min, v_max = 0.5, 5.0
        omega_min, omega_max = -np.pi, np.pi
        
        # Simple MPC approach: evaluate multiple control sequences and pick the best
        best_cost = float('inf')
        best_v = self.last_v
        best_omega = self.last_omega
        
        # Discretized control space to search
        v_options = np.linspace(max(v_min, self.last_v - 0.5), 
                               min(v_max, self.last_v + 0.5), 5)
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
                    
                    # Weighted cost for this step
                    step_cost = (w_pos * pos_error + 
                                w_theta * abs(theta_error) + 
                                w_v * v_change + 
                                w_omega * omega_change)
                    
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

class MultiRobotSimulator:
    """Simulator for multiple differential drive robots with different controllers."""
    def __init__(self, dt=0.05, lookahead_distance=0.5):
        # Simulation parameters
        self.dt = dt  # time step in seconds
        self.t = 0    # current time
        self.lookahead_distance = lookahead_distance  # lookahead distance for path following
        
        # Initialize robots with different controllers
        # Robot 1 - PID controller (blue)
        self.robot1 = DifferentialDriveRobot(x=0, y=0, theta=0, color='blue')
        self.controller1 = PIDController()
        
        # Robot 2 - MPC controller (red)
        self.robot2 = DifferentialDriveRobot(x=0, y=0, theta=0, color='red')
        self.controller2 = MPCController()
        
        # Trajectory parameters and data
        self.trajectory_type = "custom"
        self.trajectory = TrajectoryGenerator.circle()
        self.trajectory_array = np.array(self.trajectory)  # For efficient distance calculations
        
        # Robot 1 tracking data
        self.closest_point_idx1 = 0
        self.target_point_idx1 = 0
        self.tracking_errors1 = []
        self.control_inputs1 = []
        self.robot_path1 = []
        
        # Robot 2 tracking data
        self.closest_point_idx2 = 0
        self.target_point_idx2 = 0
        self.tracking_errors2 = []
        self.control_inputs2 = []
        self.robot_path2 = []
        
        # Common data
        self.timestamps = []
        
        # Setup visualization
        self.setup_visualization()
        
    def reset(self):
        """Reset the simulation."""
        # Reset robots
        self.robot1 = DifferentialDriveRobot(x=0, y=0, theta=0, color='blue')
        self.robot2 = DifferentialDriveRobot(x=0, y=0, theta=0, color='red')
        
        # Reset controllers
        self.controller1 = PIDController()
        self.controller2 = MPCController()
        
        # Reset time and indices
        self.t = 0
        self.closest_point_idx1 = 0
        self.target_point_idx1 = 0
        self.closest_point_idx2 = 0
        self.target_point_idx2 = 0
        
        # Reset metrics
        self.tracking_errors1 = []
        self.control_inputs1 = []
        self.robot_path1 = []
        self.tracking_errors2 = []
        self.control_inputs2 = []
        self.robot_path2 = []
        self.timestamps = []
        
        # Reset plots
        self.path_line1.set_data([], [])
        self.path_line2.set_data([], [])
        self.error_line1.set_data([], [])
        self.error_line2.set_data([], [])
        self.v_line1.set_data([], [])
        self.v_line2.set_data([], [])
        self.omega_line1.set_data([], [])
        self.omega_line2.set_data([], [])
        
        # Reset target visualization
        if self.trajectory:
            self.closest_point_circle1.center = self.trajectory[0]
            self.target_point_circle1.center = self.trajectory[0]
            self.closest_point_circle2.center = self.trajectory[0]
            self.target_point_circle2.center = self.trajectory[0]
            if hasattr(self, 'cross_track_line1'):
                self.cross_track_line1.set_data([], [])
            if hasattr(self, 'cross_track_line2'):
                self.cross_track_line2.set_data([], [])
        
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
            # Default custom path
            custom_points = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]
            self.trajectory = TrajectoryGenerator.custom(custom_points)
        
        # Update trajectory array for distance calculations
        self.trajectory_array = np.array(self.trajectory)
        
        # Update trajectory visualization
        traj_x = [point[0] for point in self.trajectory]
        traj_y = [point[1] for point in self.trajectory]
        self.traj_line.set_data(traj_x, traj_y)
        self.ax_sim.relim()
        self.ax_sim.autoscale_view()
        
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
    
    def calculate_cross_track_error(self, robot_state, closest_idx, robot_number):
        """Calculate the cross-track error (perpendicular distance to the path)."""
        if closest_idx >= len(self.trajectory) - 1:
            if robot_number == 1:
                return self.tracking_errors1[-1] if self.tracking_errors1 else 0
            else:
                return self.tracking_errors2[-1] if self.tracking_errors2 else 0
            
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
        if robot_number == 1:
            self.projected_point1 = projected_point
        else:
            self.projected_point2 = projected_point
        
        return cross_track_error
    
    def step(self):
        """Perform one simulation step for both robots."""
        # Check if we've reached the end of the trajectory
        if not self.trajectory or (self.target_point_idx1 >= len(self.trajectory) - 1 and 
                                  self.target_point_idx2 >= len(self.trajectory) - 1):
            return False
        
        # Robot 1 (PID Controller)
        # Find the closest point on the trajectory to the robot
        self.closest_point_idx1 = self.find_closest_point(self.robot1.state)
        
        # Find the target point ahead on the trajectory
        self.target_point_idx1 = self.find_target_point(self.closest_point_idx1)
        
        # Get the target point
        target_point1 = self.trajectory[self.target_point_idx1]
        
        # Calculate the cross-track error
        cross_track_error1 = self.calculate_cross_track_error(self.robot1.state, self.closest_point_idx1, 1)
        
        # Call the controller function to get control inputs
        v1, omega1 = self.controller1.control(self.robot1.state, target_point1, self.dt)
        
        # Update robot state
        self.robot1.update(self.dt, v1, omega1)
        
        # Record metrics
        self.tracking_errors1.append(cross_track_error1)
        self.control_inputs1.append((v1, omega1))
        self.robot_path1.append((self.robot1.state[0], self.robot1.state[1]))
        
        # Robot 2 (MPC Controller)
        # Find the closest point on the trajectory to the robot
        self.closest_point_idx2 = self.find_closest_point(self.robot2.state)
        
        # Find the target point ahead on the trajectory
        self.target_point_idx2 = self.find_target_point(self.closest_point_idx2)
        
        # Get the target point
        target_point2 = self.trajectory[self.target_point_idx2]
        
        # Calculate the cross-track error
        cross_track_error2 = self.calculate_cross_track_error(self.robot2.state, self.closest_point_idx2, 2)
        
        # Call the controller function to get control inputs
        v2, omega2 = self.controller2.control(self.robot2.state, target_point2, self.dt)
        
        # Update robot state
        self.robot2.update(self.dt, v2, omega2)
        
        # Record metrics
        self.tracking_errors2.append(cross_track_error2)
        self.control_inputs2.append((v2, omega2))
        self.robot_path2.append((self.robot2.state[0], self.robot2.state[1]))
        
        # Update time
        self.timestamps.append(self.t)
        self.t += self.dt
        
        return True
    
    def setup_visualization(self):
        """Set up the visualization environment for two robots."""
        # Create figure and axes
        self.fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(3, 4)
        
        # Main simulation plot
        self.ax_sim = self.fig.add_subplot(gs[:2, :2])
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_xlim(-10, 10)
        self.ax_sim.set_ylim(-10, 10)
        self.ax_sim.set_xlabel('X (m)')
        self.ax_sim.set_ylabel('Y (m)')
        self.ax_sim.set_title('Differential Drive Robots - PID vs MPC')
        self.ax_sim.grid(True)
        
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
        
        # Robot 1 visualization (PID - Blue)
        self.robot1_body = Rectangle(
            (0, 0), self.robot1.width, self.robot1.length, 
            fill=True, color=self.robot1.color, alpha=0.7
        )
        self.robot1_body_patch = self.ax_sim.add_patch(self.robot1_body)
        
        # Robot 1 direction indicator
        self.direction1 = plt.Line2D([0, 0], [0, 0], color='black', lw=2)
        self.ax_sim.add_line(self.direction1)
        
        # Robot 2 visualization (MPC - Red)
        self.robot2_body = Rectangle(
            (0, 0), self.robot2.width, self.robot2.length, 
            fill=True, color=self.robot2.color, alpha=0.7
        )
        self.robot2_body_patch = self.ax_sim.add_patch(self.robot2_body)
        
        # Robot 2 direction indicator
        self.direction2 = plt.Line2D([0, 0], [0, 0], color='black', lw=2)
        self.ax_sim.add_line(self.direction2)
        
        # Trajectory visualization
        traj_x = [point[0] for point in self.trajectory]
        traj_y = [point[1] for point in self.trajectory]
        self.traj_line, = self.ax_sim.plot(traj_x, traj_y, 'g-', alpha=0.5, label='Reference Path')
        
        # Robot paths
        self.path_line1, = self.ax_sim.plot([], [], 'b--', alpha=0.7, label='PID Robot Path')
        self.path_line2, = self.ax_sim.plot([], [], 'r--', alpha=0.7, label='MPC Robot Path')
        
        # Closest points on trajectory
        self.closest_point_circle1 = Circle((0, 0), radius=0.06, fill=True, color='blue', alpha=0.5)
        self.ax_sim.add_patch(self.closest_point_circle1)
        
        self.closest_point_circle2 = Circle((0, 0), radius=0.06, fill=True, color='red', alpha=0.5)
        self.ax_sim.add_patch(self.closest_point_circle2)
        
        # Target points ahead on trajectory
        self.target_point_circle1 = Circle((0, 0), radius=0.1, fill=True, color='blue', alpha=0.5, label='PID Target')
        self.ax_sim.add_patch(self.target_point_circle1)
        
        self.target_point_circle2 = Circle((0, 0), radius=0.1, fill=True, color='red', alpha=0.5, label='MPC Target')
        self.ax_sim.add_patch(self.target_point_circle2)
        
        # Cross-track error line visualization
        self.cross_track_line1, = self.ax_sim.plot([], [], 'b-', lw=1, alpha=0.5)
        self.cross_track_line2, = self.ax_sim.plot([], [], 'r-', lw=1, alpha=0.5)
        
        # Performance plots
        self.error_line1, = self.ax_error.plot([], [], 'b-', label='PID Error')
        self.error_line2, = self.ax_error.plot([], [], 'r-', label='MPC Error')
        self.ax_error.legend()
        
        self.v_line1, = self.ax_v.plot([], [], 'b-', label='PID')
        self.v_line2, = self.ax_v.plot([], [], 'r-', label='MPC')
        self.ax_v.legend()
        
        self.omega_line1, = self.ax_omega.plot([], [], 'b-', label='PID')
        self.omega_line2, = self.ax_omega.plot([], [], 'r-', label='MPC')
        self.ax_omega.legend()
        
        # Trajectory selection radio buttons
        self.radio_traj = RadioButtons(
            self.ax_traj, ('circle', 'eight', 'square', 'custom'),
            active=0  # Default to circle
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
        
        # Add legend for the robots
        self.ax_sim.legend(loc='upper left')
        
        # Add text annotation for robot types
        self.robot1_text = self.ax_sim.text(0.02, 0.95, 'Blue: PID Controller', 
                                          transform=self.ax_sim.transAxes, color='blue')
        self.robot2_text = self.ax_sim.text(0.02, 0.90, 'Red: MPC Controller', 
                                          transform=self.ax_sim.transAxes, color='red')
        
        # Animation
        self.anim_running = True
        self.anim = animation.FuncAnimation(
            self.fig, self.update_frame, interval=int(self.dt * 1000),
            blit=False
        )
        
        plt.tight_layout()
    
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
        
        # Update robot visualizations
        self.update_robot_visualization(self.robot1, self.robot1_body, self.direction1)
        self.update_robot_visualization(self.robot2, self.robot2_body, self.direction2)
        
        # Update closest and target point visualization
        if self.closest_point_idx1 < len(self.trajectory):
            self.closest_point_circle1.center = self.trajectory[self.closest_point_idx1]
        
        if self.target_point_idx1 < len(self.trajectory):
            self.target_point_circle1.center = self.trajectory[self.target_point_idx1]
            
        if self.closest_point_idx2 < len(self.trajectory):
            self.closest_point_circle2.center = self.trajectory[self.closest_point_idx2]
        
        if self.target_point_idx2 < len(self.trajectory):
            self.target_point_circle2.center = self.trajectory[self.target_point_idx2]
        
        # Update cross-track error line visualization
        if hasattr(self, 'projected_point1'):
            robot1_x, robot1_y, _ = self.robot1.state
            self.cross_track_line1.set_data(
                [robot1_x, self.projected_point1[0]],
                [robot1_y, self.projected_point1[1]]
            )
            
        if hasattr(self, 'projected_point2'):
            robot2_x, robot2_y, _ = self.robot2.state
            self.cross_track_line2.set_data(
                [robot2_x, self.projected_point2[0]],
                [robot2_y, self.projected_point2[1]]
            )
        
        # Update plots
        # Robot paths
        path1_x = [point[0] for point in self.robot_path1]
        path1_y = [point[1] for point in self.robot_path1]
        self.path_line1.set_data(path1_x, path1_y)
        
        path2_x = [point[0] for point in self.robot_path2]
        path2_y = [point[1] for point in self.robot_path2]
        self.path_line2.set_data(path2_x, path2_y)
        
        # Error plot
        if self.timestamps:
            if self.tracking_errors1:
                self.error_line1.set_data(self.timestamps, self.tracking_errors1)
            if self.tracking_errors2:
                self.error_line2.set_data(self.timestamps, self.tracking_errors2)
            self.ax_error.relim()
            self.ax_error.autoscale_view()
            
            # Calculate average errors
            avg_error1 = np.mean(self.tracking_errors1) if self.tracking_errors1 else 0
            avg_error2 = np.mean(self.tracking_errors2) if self.tracking_errors2 else 0
            
            # Update metrics text
            metrics_text = f"Average Errors:\nPID: {avg_error1:.4f} m\nMPC: {avg_error2:.4f} m"
            
            # Add current error to title
            if self.tracking_errors1 and self.tracking_errors2:
                current_error1 = self.tracking_errors1[-1]
                current_error2 = self.tracking_errors2[-1]
                self.ax_error.set_title(f'Cross-Track Error: PID={current_error1:.3f}m, MPC={current_error2:.3f}m')
                
                # Clear and update the metrics axis
                self.ax_metrics.clear()
                self.ax_metrics.text(0.1, 0.5, metrics_text, verticalalignment='center')
                self.ax_metrics.set_title('Performance Metrics')
                self.ax_metrics.axis('off')
        
        # Control input plots
        if self.timestamps and self.control_inputs1 and self.control_inputs2:
            v1_values = [inputs[0] for inputs in self.control_inputs1]
            omega1_values = [inputs[1] for inputs in self.control_inputs1]
            
            v2_values = [inputs[0] for inputs in self.control_inputs2]
            omega2_values = [inputs[1] for inputs in self.control_inputs2]
            
            self.v_line1.set_data(self.timestamps, v1_values)
            self.v_line2.set_data(self.timestamps, v2_values)
            self.ax_v.relim()
            self.ax_v.autoscale_view()
            
            self.omega_line1.set_data(self.timestamps, omega1_values)
            self.omega_line2.set_data(self.timestamps, omega2_values)
            self.ax_omega.relim()
            self.ax_omega.autoscale_view()
            
            # Add current values to titles
            current_v1 = v1_values[-1]
            current_omega1 = omega1_values[-1]
            current_v2 = v2_values[-1]
            current_omega2 = omega2_values[-1]
            
            self.ax_v.set_title(f'v: PID={current_v1:.2f}, MPC={current_v2:.2f} m/s')
            self.ax_omega.set_title(f'ω: PID={current_omega1:.2f}, MPC={current_omega2:.2f} rad/s')
    
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
        return {
            'timestamps': self.timestamps,
            'tracking_errors1': self.tracking_errors1,
            'tracking_errors2': self.tracking_errors2,
            'control_inputs1': self.control_inputs1,
            'control_inputs2': self.control_inputs2,
            'robot_path1': self.robot_path1,
            'robot_path2': self.robot_path2,
            'trajectory': self.trajectory
        }

# Main
if __name__ == "__main__":
    # Create the multi-robot simulator
    sim = MultiRobotSimulator()
    
    # Run the simulation
    sim.run()