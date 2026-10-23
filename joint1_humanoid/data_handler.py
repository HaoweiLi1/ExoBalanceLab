import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DataHandler:
    """
    Integrated class for data logging and plotting in ankle exoskeleton simulation.
    Handles creation, collection, saving, and plotting of simulation data.
    """
    
    def __init__(self, output_dir='data'):
        """
        Initialize the data handler.
        
        Args:
            output_dir: Base directory for saving data (default: 'data')
        """
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory with timestamp
        self.run_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Dictionary to store all data arrays
        self.data_arrays = {}
        
        # Configure matplotlib settings
        plt.rcParams.update({
            'font.family': 'serif',
            'axes.labelsize': 12,
            'font.size': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
        
        print(f"Data handler initialized. Saving data to: {self.run_dir}")
        
        # Initialize data arrays
        self.create_standard_datasets()
    
    def create_dataset(self, name, columns):
        """
        Create a new dataset for logging.
        
        Args:
            name: Name of the dataset
            columns: Number of columns in the dataset
        """
        self.data_arrays[name] = np.empty((0, columns))
    
    def log_data(self, name, data_row):
        """
        Log a row of data to a specific dataset.
        
        Args:
            name: Name of the dataset to log to
            data_row: Numpy array containing the data row
        """
        if name not in self.data_arrays:
            # If dataset doesn't exist, create it with the right number of columns
            self.create_dataset(name, len(data_row))
        
        self.data_arrays[name] = np.vstack((self.data_arrays[name], data_row))
    
    def create_standard_datasets(self):
        """
        Create standard datasets used in ankle exo simulation.
        """
        # State data
        self.create_dataset("joint_position", 2)  # time, position
        self.create_dataset("joint_velocity", 2)  # time, velocity
        self.create_dataset("body_com", 4)        # time, x, y, z
        
        # Torque data
        self.create_dataset("human_torque", 2)   # time, torque (renamed from ankle_torque to human_torque)
        self.create_dataset("exo_torque", 2)     # time, torque
        self.create_dataset("gravity_torque", 2) # time, torque
        self.create_dataset("human_rtd", 3)      # time, rtd_value, rtd_limit
        
        # Force data
        self.create_dataset("contact_force", 3)    # time, front, back
        self.create_dataset("constraint_force", 5) # time, f1, f2, f3, f4
    
    def save_data(self):
        """
        Save all collected data to CSV files with organized structure.
        """
        # Save state data (joint position, velocity, body COM)
        if all(name in self.data_arrays for name in ["joint_position", "joint_velocity", "body_com"]):
            pos_data = self.data_arrays["joint_position"][1:]  # Skip the first empty row
            vel_data = self.data_arrays["joint_velocity"][1:]
            com_data = self.data_arrays["body_com"][1:]
            
            # Combine data with matching timestamps
            time_values = pos_data[:, 0]
            state_data = np.zeros((len(time_values), 6))
            state_data[:, 0] = time_values
            state_data[:, 1] = pos_data[:, 1]  # ankle angle
            state_data[:, 2] = vel_data[:, 1]  # ankle velocity
            state_data[:, 3:6] = com_data[:, 1:4]  # body com x, y, z
            
            # Create header with column names and units
            header = "time(s),ankle_angle(deg),ankle_velocity(deg/s),body_com_x(m),body_com_y(m),body_com_z(m)"
            
            # Save to CSV
            state_path = os.path.join(self.run_dir, "state.csv")
            np.savetxt(state_path, state_data, delimiter=",", fmt="%.6f", header=header, comments='')
            print(f"Saved state data to {state_path}")
        
        # Save torque data (human torque, exo torque, gravity torque, RTD)
        if all(name in self.data_arrays for name in ["human_torque", "exo_torque", "gravity_torque", "human_rtd"]):
            human_data = self.data_arrays["human_torque"][1:]
            exo_data = self.data_arrays["exo_torque"][1:]
            gravity_data = self.data_arrays["gravity_torque"][1:]
            rtd_data = self.data_arrays["human_rtd"][1:]
            
            # Combine data with matching timestamps
            time_values = human_data[:, 0]
            torque_data = np.zeros((len(time_values), 5))
            torque_data[:, 0] = time_values
            torque_data[:, 1] = human_data[:, 1]    # human ankle torque
            torque_data[:, 2] = exo_data[:, 1]      # exo torque
            torque_data[:, 3] = gravity_data[:, 1]  # gravity torque
            torque_data[:, 4] = rtd_data[:, 1]      # ankle RTD
            
            # Create header with column names and units
            header = "time(s),ankle_torque(Nm),exo_torque(Nm),gravity_torque(Nm),ankle_rtd(Nm/s)"
            
            # Save to CSV
            torque_path = os.path.join(self.run_dir, "torque.csv")
            np.savetxt(torque_path, torque_data, delimiter=",", fmt="%.6f", header=header, comments='')
            print(f"Saved torque data to {torque_path}")
        
        # Save force data (contact forces and constraint forces)
        if all(name in self.data_arrays for name in ["contact_force", "constraint_force"]):
            contact_data = self.data_arrays["contact_force"][1:]
            constraint_data = self.data_arrays["constraint_force"][1:]
            
            # Combine data with matching timestamps
            time_values = contact_data[:, 0]
            force_data = np.zeros((len(time_values), 7))
            force_data[:, 0] = time_values
            force_data[:, 1] = contact_data[:, 1]     # front contact force
            force_data[:, 2] = contact_data[:, 2]     # back contact force
            force_data[:, 3:7] = constraint_data[:, 1:5]  # constraint forces
            
            # Create header with column names and units
            header = "time(s),front_contact_force(N),back_contact_force(N),foot_rotation_constraint(N),slide_x_constraint(N),slide_z_constraint(N),ankle_hinge_constraint(N)"
            
            # Save to CSV
            force_path = os.path.join(self.run_dir, "force.csv")
            np.savetxt(force_path, force_data, delimiter=",", fmt="%.6f", header=header, comments='')
            print(f"Saved force data to {force_path}")
    
    def create_plots(self):
        """
        Create a 3x1 plot with ankle angle, velocity, and torques (human and exo only).
        """
        # Check if required datasets exist
        required_datasets = ["joint_position", "joint_velocity", "human_torque", 
                           "exo_torque"]
        
        for name in required_datasets:
            if name not in self.data_arrays or len(self.data_arrays[name]) <= 1:
                print(f"Error: Required dataset '{name}' not found or empty.")
                return
        
        # Get data (skip first empty row)
        pos_data = self.data_arrays["joint_position"][1:]
        vel_data = self.data_arrays["joint_velocity"][1:]
        human_torque_data = self.data_arrays["human_torque"][1:]
        exo_torque_data = self.data_arrays["exo_torque"][1:]
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
        
        # Plot 1: Ankle angle
        axs[0].plot(pos_data[:, 0], pos_data[:, 1], 'b-', linewidth=1.5)
        axs[0].set_ylabel('Ankle Angle (deg)')
        axs[0].set_title('Ankle Angle vs Time')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Ankle velocity
        axs[1].plot(vel_data[:, 0], vel_data[:, 1], 'g-', linewidth=1.5)
        axs[1].set_ylabel('Angular Velocity (deg/s)')
        axs[1].set_title('Ankle Angular Velocity vs Time')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Torques (human and exo only)
        axs[2].plot(human_torque_data[:, 0], human_torque_data[:, 1], 'r-', linewidth=1.5, label='Human Ankle Torque')
        axs[2].plot(exo_torque_data[:, 0], exo_torque_data[:, 1], 'b-', linewidth=1.5, label='Exo Torque')
        axs[2].set_ylabel('Torque (Nm)')
        axs[2].set_title('Joint Torques vs Time')
        axs[2].legend()
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].set_xlabel('Time (s)')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "ankle_plots.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Plots saved to {os.path.join(self.run_dir, 'ankle_plots.png')}")
    
    def finalize(self):
        """
        Save all data and generate plots.
        """
        self.save_data()
        self.create_plots()
        print(f"Data processing complete. Results saved to: {self.run_dir}")