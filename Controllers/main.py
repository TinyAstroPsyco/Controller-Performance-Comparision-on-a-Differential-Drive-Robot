import numpy as np
import matplotlib.pyplot as plt
from controllers import PIDController, MPCController, FastMPCController, AdaptiveController
from simulation import MultiRobotSimulator

def main():
    """Run the simulation with multiple controllers."""
    
    # Define the controllers to use in the simulation
    controllers = [
        {"controller": PIDController(name="PID Controller"), "color": "blue"},
        {"controller": MPCController(name="MPC Controller"), "color": "red"}
    ]
    
    # Create and run the simulator
    sim = MultiRobotSimulator(controller_list=controllers)
    sim.run()
    
def compare_controllers():
    """Run a simulation comparing multiple controller types."""
    
    # Define several controllers with different parameters
    controllers = [
        {"controller": PIDController(name="PID"), "color": "blue"},
        {"controller": MPCController(name="MPC"), "color": "red"},
        {"controller": FastMPCController(name="Fast MPC"), "color": "green"}
    ]
    
    # Create and run the simulator
    sim = MultiRobotSimulator(controller_list=controllers)
    sim.run()
    
def compare_pid_parameters():
    """Compare different PID parameter settings."""
    
    # Create PID controllers with different parameter sets
    pid1 = PIDController(name="Default PID")
    
    pid2 = PIDController(name="Aggressive PID")
    pid2.set_gains(kp_v=3.0, ki_v=1.5, kd_v=0.8, kp_omega=3.0, ki_omega=1.5, kd_omega=0.8)
    
    pid3 = PIDController(name="Smooth PID")
    pid3.set_gains(kp_v=1.5, ki_v=0.5, kd_v=1.0, kp_omega=1.5, ki_omega=0.5, kd_omega=1.0)
    
    controllers = [
        {"controller": pid1, "color": "blue"},
        {"controller": pid2, "color": "red"},
        {"controller": pid3, "color": "green"}
    ]
    
    # Create and run the simulator
    sim = MultiRobotSimulator(controller_list=controllers)
    sim.run()
    
def compare_mpc_parameters():
    """Compare different MPC parameter settings."""
    
    # Create MPC controllers with different parameter sets
    mpc1 = MPCController(name="Default MPC")
    
    mpc2 = MPCController(name="Speed-focused MPC")
    mpc2.set_params(w_pos=0.8, w_theta=0.7, w_v=0.05, velocity_bias=0.3)
    
    mpc3 = MPCController(name="Precision-focused MPC")
    mpc3.set_params(w_pos=1.5, w_theta=1.2, w_v=0.2, w_omega=0.4)
    
    controllers = [
        {"controller": mpc1, "color": "blue"},
        {"controller": mpc2, "color": "red"},
        {"controller": mpc3, "color": "green"}
    ]
    
    # Create and run the simulator
    sim = MultiRobotSimulator(controller_list=controllers)
    sim.run()

if __name__ == "__main__":
    # Run one of the demonstration functions
    main()  # Basic PID vs MPC comparison
    
    # Uncomment one of these to run different comparisons
    # compare_controllers()  # Compare PID, MPC, and Fast MPC
    # compare_pid_parameters()  # Compare different PID parameter settings
    # compare_mpc_parameters()  # Compare different MPC parameter settings
