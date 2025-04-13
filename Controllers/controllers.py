import numpy as np

class Controller:
    """Base controller class for trajectory following."""
    def __init__(self, name="Base Controller"):
        self.name = name
    
    def control(self, robot_state, target_point, dt):
        """
        Base control method to be overridden by subclasses.
        
        Args:
            robot_state: [x, y, theta] state of the robot
            target_point: (x, y) target position on the path ahead
            dt: time step in seconds
            
        Returns:
            v: linear velocity
            omega: angular velocity
        """
        raise NotImplementedError("Subclass must implement abstract method")

class PIDController(Controller):
    def __init__(self, name="PID Controller"):
        super().__init__(name)
        self.prev_dist_error = 0.0
        self.prev_angle_error = 0.0
        self.dist_error_sum = 0.0
        self.angle_error_sum = 0.0
        
        # Default PID gains for linear velocity
        self.kp_v = 2.0    # Proportional gain
        self.ki_v = 1.0    # Integral gain
        self.kd_v = 0.6    # Derivative gain
        
        # Default PID gains for angular velocity
        self.kp_omega = 2.0    # Proportional gain
        self.ki_omega = 1.0    # Integral gain
        self.kd_omega = 0.6    # Derivative gain
        
    def control(self, robot_state, target_point, dt):
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
        
        # Calculate linear velocity using PID
        v = (self.kp_v * dist_to_target + 
             self.ki_v * self.dist_error_sum + 
             self.kd_v * dist_error_deriv)
        
        # Calculate angular velocity using PID
        omega = (self.kp_omega * angle_error + 
                 self.ki_omega * self.angle_error_sum + 
                 self.kd_omega * angle_error_deriv)
        
        # Apply limits
        v = max(0.2, min(v, 5.0))  # Clamp between 0.2 and 5.0
        omega = np.clip(omega, -np.pi, np.pi)  # Limit angular velocity
        
        return v, omega
    
    def set_gains(self, kp_v=None, ki_v=None, kd_v=None, kp_omega=None, ki_omega=None, kd_omega=None):
        """Set the PID controller gains."""
        if kp_v is not None: self.kp_v = kp_v
        if ki_v is not None: self.ki_v = ki_v
        if kd_v is not None: self.kd_v = kd_v
        if kp_omega is not None: self.kp_omega = kp_omega
        if ki_omega is not None: self.ki_omega = ki_omega
        if kd_omega is not None: self.kd_omega = kd_omega

class MPCController(Controller):
    def __init__(self, name="MPC Controller"):
        super().__init__(name)
        self.last_v = 0.0
        self.last_omega = 0.0
        
        # MPC parameters
        self.N = 10  # Prediction horizon
        
        # Default cost function weights
        self.w_pos = 1.0      # Position error weight
        self.w_theta = 0.8    # Heading error weight
        self.w_v = 0.1        # Velocity smoothness weight
        self.w_omega = 0.2    # Angular velocity smoothness weight
        
        # Default control constraints
        self.v_min, self.v_max = 0.0, 5.0
        self.omega_min, self.omega_max = -np.pi, np.pi
        
        # Speed bias (0 = no bias, higher values encourage higher speeds)
        self.velocity_bias = 0.0
        
    def control(self, robot_state, target_point, dt):
        """
        MPC controller for path following.
        
        Args:
            robot_state: [x, y, theta] state of the robot
            target_point: (x, y) target position on the path ahead
            dt: time step in seconds
            
        Returns:
            v: linear velocity
            omega: angular velocity
        """
        # Current state
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Simple MPC approach: evaluate multiple control sequences and pick the best
        best_cost = float('inf')
        best_v = self.last_v
        best_omega = self.last_omega
        
        # Discretized control space to search
        v_range_min = max(self.v_min, self.last_v - 0.5)
        v_range_max = min(self.v_max, self.last_v + 0.5)
        v_options = np.linspace(v_range_min, v_range_max, 5)
        
        omega_options = np.linspace(max(self.omega_min, self.last_omega - 0.5), 
                                   min(self.omega_max, self.last_omega + 0.5), 7)
        
        # Evaluate each control sequence
        for v in v_options:
            for omega in omega_options:
                # Predict future states using simplified model
                pred_x, pred_y, pred_theta = x, y, theta
                total_cost = 0
                
                for i in range(self.N):
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
                    
                    # Optional velocity reward (negative cost for higher velocities)
                    velocity_reward = -self.velocity_bias * v / self.v_max if self.velocity_bias > 0 else 0
                    
                    # Weighted cost for this step
                    step_cost = (self.w_pos * pos_error + 
                                self.w_theta * abs(theta_error) + 
                                self.w_v * v_change + 
                                self.w_omega * omega_change +
                                velocity_reward)
                    
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
    
    def set_params(self, w_pos=None, w_theta=None, w_v=None, w_omega=None, 
                  v_min=None, v_max=None, omega_min=None, omega_max=None,
                  velocity_bias=None, prediction_horizon=None):
        """Set the MPC controller parameters."""
        if w_pos is not None: self.w_pos = w_pos
        if w_theta is not None: self.w_theta = w_theta
        if w_v is not None: self.w_v = w_v
        if w_omega is not None: self.w_omega = w_omega
        if v_min is not None: self.v_min = v_min
        if v_max is not None: self.v_max = v_max
        if omega_min is not None: self.omega_min = omega_min
        if omega_max is not None: self.omega_max = omega_max
        if velocity_bias is not None: self.velocity_bias = velocity_bias
        if prediction_horizon is not None: self.N = prediction_horizon

class FastMPCController(MPCController):
    """MPC controller optimized for speed."""
    def __init__(self, name="Fast MPC Controller"):
        super().__init__(name)
        # Override default parameters to favor speed
        self.w_pos = 0.8      # Position error weight (decreased)
        self.w_theta = 0.7    # Heading error weight (decreased)
        self.w_v = 0.05       # Velocity smoothness weight (decreased)
        self.v_min = 0.5      # Minimum velocity (increased)
        self.velocity_bias = 0.3  # Add bias toward higher velocities
    
    def control(self, robot_state, target_point, dt):
        """Override to use asymmetric velocity search range."""
        # Current state
        x, y, theta = robot_state
        x_target, y_target = target_point
        
        # Simple MPC approach: evaluate multiple control sequences and pick the best
        best_cost = float('inf')
        best_v = self.last_v
        best_omega = self.last_omega
        
        # Discretized control space to search with asymmetric range
        # Smaller deceleration, larger acceleration
        v_range_min = max(self.v_min, self.last_v - 0.3)  
        v_range_max = min(self.v_max, self.last_v + 0.7)
        v_options = np.linspace(v_range_min, v_range_max, 5)
        
        omega_options = np.linspace(max(self.omega_min, self.last_omega - 0.5), 
                                   min(self.omega_max, self.last_omega + 0.5), 7)
        
        # Rest of the control method same as parent class
        # Evaluate each control sequence
        for v in v_options:
            for omega in omega_options:
                # Predict future states using simplified model
                pred_x, pred_y, pred_theta = x, y, theta
                total_cost = 0
                
                for i in range(self.N):
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
                    
                    # Velocity reward (negative cost for higher velocities)
                    velocity_reward = -self.velocity_bias * v / self.v_max
                    
                    # Weighted cost for this step
                    step_cost = (self.w_pos * pos_error + 
                                self.w_theta * abs(theta_error) + 
                                self.w_v * v_change + 
                                self.w_omega * omega_change +
                                velocity_reward)
                    
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

class AdaptiveController(Controller):
    """
    Adaptive controller that adjusts parameters based on estimated robot dynamics.
    This is a placeholder for future implementation of robust adaptive control.
    """
    def __init__(self, name="Adaptive Controller"):
        super().__init__(name)
        self.base_controller = PIDController()
        
        # Parameter estimation variables
        self.estimated_params = {
            'wheel_radius': 0.05,
            'wheel_base': 0.2,
            'mass': 1.0,
            'friction': 0.1
        }
        
        # Adaptation rates
        self.adaptation_rate = 0.01
        
        # History for parameter estimation
        self.state_history = []
        self.control_history = []
        self.error_history = []
        
    def control(self, robot_state, target_point, dt):
        """
        Adaptive control that updates parameters based on observations.
        Currently a placeholder implementation.
        """
        # Store history for parameter estimation
        self.state_history.append(robot_state)
        
        # For now, just use the base controller
        v, omega = self.base_controller.control(robot_state, target_point, dt)
        
        # Store control for parameter estimation
        self.control_history.append((v, omega))
        
        # Here we would update parameter estimates based on observed behavior
        # This is where the actual parameter estimation would happen
        self._update_parameters(robot_state, (v, omega), dt)
        
        # Update controller gains based on parameter estimates
        self._adapt_controller_gains()
        
        return v, omega
    
    def _update_parameters(self, current_state, control_input, dt):
        """
        Update parameter estimates based on observed behavior.
        This would implement parameter estimation algorithms (e.g., recursive least squares).
        For now, this is a placeholder.
        """
        # TODO: Implement parameter estimation
        pass
    
    def _adapt_controller_gains(self):
        """
        Adapt controller gains based on updated parameter estimates.
        For now, this is a placeholder.
        """
        # TODO: Implement control gain adaptation
        pass

# Add more controllers here as needed
