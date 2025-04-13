import numpy as np
from scipy.spatial.distance import cdist

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
    def custom(points, num_points_between=10):
        """Generate a trajectory from a list of waypoints."""
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
        
        return trajectory
