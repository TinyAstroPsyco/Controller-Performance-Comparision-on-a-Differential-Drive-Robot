import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib import gridspec

from robot_model import DifferentialDriveRobot, TrajectoryGenerator
from controllers import PIDController, MPCController, FastMPCController, AdaptiveController

class MultiRobotSimulator:
    """Simulator for multiple differential drive robots with different controllers."""
    def __init__(self, controller_list=None, dt=0.05, lookahead_distance=0.5):
        # Default controllers if none provided
        if controller_list is None:
            controller_list = [
                {"controller": PIDController(), "color": "blue"},
                {"controller": MPCController(), "color": "red"}
            ]
            
        self.controllers = controller_list
        self.num_robots = len(controller_list)
        
        # Simulation parameters
        self.dt = dt  # time step in seconds
        self.t = 0    # current time
        self.lookahead_distance = lookahead_distance  # lookahead distance for path following
        
        # Initialize robots with different controllers
        self.robots = []
        for i, ctrl_info in enumerate(self.controllers):
            robot = DifferentialDriveRobot(x=0, y=0, theta=0, color=ctrl_info["color"])
            self.robots.append(robot)
        
        # Trajectory parameters and data
        self.trajectory_type = "circle"
        self.trajectory = TrajectoryGenerator.circle()
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
        
        # Setup visualization
        self.setup_visualization()
        
    def reset(self):
        """Reset the simulation."""
        # Reset robots
        self.robots = []
        for i, ctrl_info in enumerate(self.controllers):
            robot = DifferentialDriveRobot(x=0, y=0, theta=0, color=ctrl_info["color"])
            self.robots.append(robot)
        
        # Reset controllers
        for i, ctrl_info in enumerate(self.controllers):
            if hasattr(ctrl_info["controller"], "__init__"):
                ctrl_class = ctrl_info["controller"].__class__
                self.controllers[i]["controller"] = ctrl_class()
        
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
        
        # Reset target visualization
        if self.trajectory:
            for i in range(self.num_robots):
                self.closest_point_circles[i].center = self.trajectory[0]
                self.target_point_circles[i].center = self.trajectory[0]
                self.cross_track_lines[i].set_data([], [])
        
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
        distances = dict(robot_position, self.trajectory_array).flatten()
        
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
            if self.target_point_idx[i] >= len(self.trajectory) - 1:
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
            v, omega = self.controllers[i]["controller"].control(
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
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4)
        
        # Main simulation plot
        self.ax_sim = self.fig.add_subplot(gs[:2, :2])
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_xlim(-5, 5)
        self.ax_sim.set_ylim(-5, 5)
        self.ax_sim.set_xlabel('X (m)')
        self.ax_sim.set_ylabel('Y (m)')
        self.ax_sim.set_title('Differential Drive Robots - Controller Comparison')
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
        
        # Initialize visualization elements for each robot
        self.robot_bodies = []
        self.direction_indicators = []
        self.path_lines = []
        self.closest_point_circles = []
        self.target_point_circles = []
        self.cross_track_lines = []
        self.error_lines = []
        self.v_lines = []
        self.omega_lines = []
        
        for i, robot in enumerate(self.robots):
            color = robot.color
            
            # Robot body
            robot_body = Rectangle(
                (0, 0), robot.width, robot.length, 
                fill=True, color=color, alpha=0.7
            )
            self.robot_bodies.append(self.ax_sim.add_patch(robot_body))
            
            # Direction indicator
            direction = plt.Line2D([0, 0], [0, 0], color='black', lw=2)
            self.direction_indicators.append(direction)
            self.ax_sim.add_line(direction)
            
            # Path line
            path_line, = self.ax_sim.plot([], [], linestyle='--', color=color, alpha=0.7, 
                                        label=f'{self.controllers[i]["controller"].name}')
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
                                           label=f'{self.controllers[i]["controller"].name}')
            self.error_lines.append(error_line)
            
            # Velocity plot lines
            v_line, = self.ax_v.plot([], [], color=color, 
                                   label=f'{self.controllers[i]["controller"].name}')
            self.v_lines.append(v_line)
            
            omega_line, = self.ax_omega.plot([], [], color=color, 
                                          label=f'{self.controllers[i]["controller"].name}')
            self.omega_lines.append(omega_line)
        
        # Trajectory visualization
        traj_x = [point[0] for point in self.trajectory]
        traj_y = [point[1] for point in self.trajectory]
        self.traj_line, = self.ax_sim.plot(traj_x, traj_y, 'g-', alpha=0.5, label='Reference Path')
        
        # Add legends
        self.ax_sim.legend(loc='upper left')
        self.ax_error.legend()
        self.ax_v.legend()
        self.ax_omega.legend()
        
        # Add text annotations for robot types
        for i, ctrl_info in enumerate(self.controllers):
            controller_name = ctrl_info["controller"].name
            color = ctrl_info["color"]
            self.ax_sim.text(0.02, 0.95 - i*0.05, f'{color.capitalize()}: {controller_name}', 
                          transform=self.ax_sim.transAxes, color=color)
        
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
        self.slider_lookahead = Slider(
            self.ax_lookahead, 'Lookahead', 0.1, 2.0, 
            valinit=self.lookahead_distance
        )
        self.slider_lookahead.on_changed(self.update_lookahead)
        
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
        
        # Update visualizations for each robot
        for i, robot in enumerate(self.robots):
            # Update robot visualization
            self.update_robot_visualization(robot, self.robot_bodies[i], self.direction_indicators[i])
            
            # Update closest and target point visualization
            if self.closest_point_idx[i] < len(self.trajectory):
                self.closest_point_circles[i].center = self.trajectory[self.closest_point_idx[i]]
            
            if self.target_point_idx[i] < len(self.trajectory):
                self.target_point_circles[i].center = self.trajectory[self.target_point_idx[i]]
            
            # Update cross-track error line visualization
            if self.projected_points[i] is not None:
                robot_x, robot_y, _ = robot.state
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
            
            # Update metrics text
            metrics_text = "Average Errors:\n"
            for i in range(self.num_robots):
                ctrl_name = self.controllers[i]["controller"].name
                metrics_text += f"{ctrl_name}: {avg_errors[i]:.4f} m\n"
            
            # Add current error to title
            current_errors = []
            for i in range(self.num_robots):
                if self.tracking_errors[i]:
                    current_errors.append(f"{self.controllers[i]['controller'].name}={self.tracking_errors[i][-1]:.3f}m")
            
            self.ax_error.set_title(f'Cross-Track Error: {", ".join(current_errors)}')
            
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
            controller_name = self.controllers[i]["controller"].name
            data[f'tracking_errors_{controller_name}'] = self.tracking_errors[i]
            data[f'control_inputs_{controller_name}'] = self.control_inputs[i]
            data[f'robot_path_{controller_name}'] = self.robot_paths[i]
        
        return data
