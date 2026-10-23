import mujoco as mj
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from queue import Queue
import xml.etree.ElementTree as ET
import shutil
import yaml

from renderer import MujocoRenderer
from data_handler import DataHandler
from controllers import create_human_controller, create_exo_controller, create_hip_controller, HumanLQRController

plt.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

control_log_queue = Queue()
counter_queue = Queue()

class AnkleHipExoSimulation:
    """
    Main simulation class for ankle-hip exoskeleton.
    Handles physics simulation, controllers, and data logging.
    """
    
    def __init__(self, config_file='config.yaml'):
        """
        Initialize the simulation.
        
        Args:
            config_file: Path to configuration YAML file
        """
        # print('Simulation initialized')
        self.config_file = config_file
        self.params = self.load_params_from_yaml(config_file)
        self.config = self.params['config']

        # Controllers
        self.human_controller = None
        self.exo_controller = None
        self.hip_controller = None
        self.show_exoskeleton = True  # Add flag for exoskeleton visibility
        
        # Renderer
        self.renderer = None
        self.data_handler = DataHandler()
        self.visualization_flag = self.config.get('visualization_flag', True)

        # MuJoCo objects
        self.model = None
        self.data = None
        self.ankle_joint_id = None
        self.hip_joint_id = None
        self.human_body_id = None
        
        # Setpoints
        self.ankle_position_setpoint = self.config['ankle_position_setpoint_radians']
        self.hip_position_setpoint = self.config['hip_position_setpoint_radians']

    def load_params_from_yaml(self, file_path):
        """Load configuration from YAML file."""
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
        
    def copy_config_to_result_dir(self, file_path, result_dir):
        """Copy the config file directly to the result directory."""
        # If the result directory doesn't exist, create it
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        shutil.copy2(file_path, os.path.join(result_dir, 'config.yaml'))

    def update_renderer_controller_params(self):
        """Update the renderer with current controller parameters."""
        if not self.renderer or not self.visualization_flag:
            return
            
        updated_params = {}
        
        # Get LQR gains if using LQR controller
        if isinstance(self.human_controller, HumanLQRController):
            # Format the gain matrix for display
            gain_matrix = self.human_controller.K
            gain_str = f"[{gain_matrix[0,0]:.2f}, {gain_matrix[0,1]:.2f}]"
            updated_params['LQR Gain'] = gain_str
        
        # Get dynamic exo parameters if using PD controller with dynamic gains
        exo_config = self.config['controllers']['exo']
        if exo_config['type'] == 'PD' and exo_config.get('pd_params', {}).get('use_dynamic_gains', False):
            # Get dynamic gains from the controller
            if hasattr(self.human_controller, 'dynamic_gains') and self.human_controller.dynamic_gains:
                kp = self.human_controller.dynamic_gains.get('kp', 0)
                kd = self.human_controller.dynamic_gains.get('kd', 0)
                updated_params['Exo Kp'] = f"{kp:.2f}"
                updated_params['Exo Kd'] = f"{kd:.2f}"
        
        # Update renderer with new parameters
        if updated_params:
            self.renderer.update_controller_params(updated_params)

    def initialize_controllers(self, model, data):
        """Initialize controllers based on configuration."""
        # Get human controller configuration
        human_config = self.config['controllers']['human']
        human_type = human_config['type']

        # Get exo controller configuration
        exo_config = self.config['controllers']['exo']
        exo_type = exo_config['type']

        # Get hip controller configuration
        hip_config = self.config['controllers']['hip']
        hip_type = hip_config['type']

        # Prepare parameters for human controller
        human_params = {
            'mass': self.config['M_total'] - 2*0.0145*self.config['M_total'],
            'leg_length': 0.575 * self.config['H_total'],
            # Get MRTD parameters from main config section
            'mrtd_df': self.config.get('mrtd_df'),
            'mrtd_pf': self.config.get('mrtd_pf')
        }

        # Debug print for MRTD params
        print(f"MRTD from config - DF: {self.config.get('mrtd_df')}, PF: {self.config.get('mrtd_pf')}")
        print(f"Human params for controller: {human_params}")
        
        # Add controller-specific parameters
        if human_type == "LQR":
            lqr_params = human_config['lqr_params']
            human_params.update({
                'Q_angle': lqr_params['Q_angle'],
                'Q_velocity': lqr_params['Q_velocity'],
                'R': lqr_params['R']
            })
            # For LQR, also pass exo configuration if exo is enabled
            if exo_type != "None":
                human_params['exo_config'] = exo_config
                
                # Get the pre-calculated values from xml_utilities for dynamic gains
                if exo_type == "PD" and exo_config.get('pd_params', {}).get('use_dynamic_gains', False):
                    _, m_feet, m_body, l_COM, _, _, K_p, *_ = MujocoRenderer.calculate_kp_and_geom(
                        self.config['M_total'], 
                        self.config['H_total']
                    )
                    
                    # Calculate dynamic gains
                    kp = K_p  # Already calculated as m_body * g * l_COM
                    kd = 0.3 * np.sqrt(m_body * l_COM**2 * kp)
                    
                    # Pass dynamic gains to LQR controller
                    human_params['dynamic_gains'] = {
                        'kp': kp,
                        'kd': kd
                    }
                    print(f"Passing exo dynamic gains to LQR - Kp: {kp:.2f}, Kd: {kd:.2f}")
        elif human_type == "PD":
            pd_params = human_config['pd_params']
            human_params.update({
                'kp': pd_params['kp'],
                'kd': pd_params['kd']
            })
        elif human_type == "Precomputed":
            human_params['precomputed_params'] = human_config.get('precomputed_params', {})
            # Log information about the trajectory file
            trajectory_file = human_params['precomputed_params'].get('trajectory_file')
            if trajectory_file:
                print(f"Using precomputed trajectory from: {trajectory_file}")
            
        # Create human controller using factory function
        self.human_controller = create_human_controller(
            human_type, model, data, human_params
        )

        # Set show_exoskeleton flag based on controller type
        self.show_exoskeleton = exo_type != "None"
        
        # Get the pre-calculated values from MujocoRenderer
        _, m_feet, m_body, l_COM, _, _, K_p, *_ = MujocoRenderer.calculate_kp_and_geom(
            self.config['M_total'], 
            self.config['H_total']
        )

        # Prepare parameters for exo controller
        exo_params = {}
        
        # Add controller-specific parameters
        if exo_type == "PD" and exo_config.get('pd_params', {}).get('use_dynamic_gains', False):
            # Calculate Kp and Kd dynamically using the pre-calculated values
            kp = K_p  # Already calculated as m_body * g * l_COM
            kd = 0.3 * np.sqrt(m_body * l_COM**2 * kp)
            
            exo_params.update({
                'kp': kp,
                'kd': kd
            })
            
            print(f"Exo PD controller configured with dynamic gains - Kp: {kp:.2f}, Kd: {kd:.2f}")

        elif exo_type == "PD":
            # Use the values from config if dynamic gains are not enabled
            pd_params = exo_config['pd_params']
            exo_params.update({
                'kp': pd_params.get('kp', 400),
                'kd': pd_params.get('kd', 10)
            })

        # Create exo controller using factory function
        self.exo_controller = create_exo_controller(
            exo_type, model, data, exo_params
        )
        
        # Prepare parameters for hip controller
        hip_params = {
            'mrtd': self.config.get('hip_mrtd')
        }
        
        # Add hip controller-specific parameters
        if hip_type == "PD":
            pd_params = hip_config['pd_params']
            hip_params.update({
                'kp': pd_params.get('kp', 700),
                'kd': pd_params.get('kd', 50)
            })
        
        # Create hip controller using factory function
        self.hip_controller = create_hip_controller(
            hip_type, model, data, hip_params
        )
        
        # Set exoskeleton visibility based on controller type
        self.toggle_exoskeleton_visibility(model, self.show_exoskeleton)

        if self.visualization_flag and self.renderer:
            self.update_renderer_controller_params()

    def toggle_exoskeleton_visibility(self, model, show_exoskeleton):
        """
        Toggle the visibility of exoskeleton components in the model.
        
        Args:
            model: MuJoCo model
            show_exoskeleton: Boolean indicating whether to show the exoskeleton
        """
        # List of all exoskeleton component names
        exo_geom_names = [
            "exo_heel_attachment", 
            "exo_housing", 
            "exo_connecting_rod",
            "exo_joint_upper", 
            "exo_joint_lower",
            "exo_calf_strap_left",
            "exo_calf_strap_right",
            "exo_calf_strap_top"
        ]
        
        # Set the visibility of each component
        for name in exo_geom_names:
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:  # Check if the geom exists
                # Set alpha to 0 (invisible) or 1 (visible)
                model.geom_rgba[geom_id][3] = 1.0 if show_exoskeleton else 0.0
                
                # Additionally, if not showing, move the geom far away to ensure no interactions
                if not show_exoskeleton:
                    model.geom_pos[geom_id] = np.array([1000.0, 1000.0, 1000.0])

    def controller(self, model, data):
        """Controller function for the leg and hip joints."""
        # Get current ankle state
        ankle_state = np.array([
            data.sensordata[0],  # Ankle joint angle
            data.qvel[3]         # Ankle joint velocity
        ])
        
        # Get current hip state
        hip_state = np.array([
            data.sensordata[3],  # Hip joint angle
            data.qvel[4]         # Hip joint velocity
        ])
        
        # Compute human ankle control
        human_torque = self.human_controller.compute_control(
            state=ankle_state,
            target=self.ankle_position_setpoint
        )
        data.ctrl[0] = human_torque
        
        # Compute exo ankle control
        exo_torque = self.exo_controller.compute_control(
            state=ankle_state,
            target=self.ankle_position_setpoint
        )
        data.ctrl[1] = exo_torque
        
        # Compute hip control
        hip_torque = self.hip_controller.compute_control(
            state=hip_state,
            target=self.hip_position_setpoint
        )
        data.ctrl[2] = hip_torque

    def initialize_model(self):
        """Initialize MuJoCo model and data."""
        # Get model parameters
        xml_path = 'initial_humanoid.xml'
        translation_friction_constant = self.config['translation_friction_constant']
        rolling_friction_constant = self.config['rolling_friction_constant']
        M_total = self.config['M_total']
        H_total = self.config['H_total']

        result_model_path = os.path.join(self.data_handler.run_dir, 'literature_humanoid.xml')
        # Prepare model parameters
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Use MujocoRenderer.calculate_kp_and_geom
        h_f, m_feet, m_body, l_COM, l_foot, a, self.K_p, lower_leg_length, upper_body_length, \
        lower_leg_mass, upper_body_mass, lower_leg_com, upper_body_com = MujocoRenderer.calculate_kp_and_geom(M_total, H_total)
        
        self.copy_config_to_result_dir(self.config_file, self.data_handler.run_dir)
        
        # Use MujocoRenderer.set_geometry_params
        MujocoRenderer.set_geometry_params(
            root, 
            m_feet, 
            m_body, 
            l_COM, 
            l_foot, 
            a, 
            H_total, 
            h_f, 
            translation_friction_constant, 
            rolling_friction_constant,
            lower_leg_length,
            upper_body_length,
            lower_leg_mass,
            upper_body_mass,
            lower_leg_com,
            upper_body_com
        )

        # Write modified model
        tree.write(result_model_path)
        
        # Load model and create data
        self.model = mj.MjModel.from_xml_path(result_model_path)
        self.data = mj.MjData(self.model)
        
        # Set simulation parameters
        self.model.opt.timestep = self.config['simulation_timestep']
        self.model.opt.gravity = np.array([0, 0, -9.81])
        
        # Set initial conditions - foot
        self.data.qvel[0] = self.config['foot_rotation_initial_velocity']  # hinge joint at top of body
        self.data.qvel[1] = self.config['foot_x_initial_velocity']         # slide joint in x direction
        self.data.qvel[2] = self.config['foot_z_initial_velocity']         # slide joint in z direction
        self.data.qpos[0] = self.config['foot_angle_initial_position_radians']
        
        # Set initial conditions - ankle
        self.data.qvel[3] = self.config['ankle_initial_velocity']          # hinge joint at ankle
        self.data.qpos[3] = self.config['ankle_initial_position_radians']
        
        # Set initial conditions - hip
        self.data.qvel[4] = self.config['hip_initial_velocity']            # hinge joint at hip
        self.data.qpos[4] = self.config['hip_initial_position_radians']

        # Call forward to update derived quantities
        mj.mj_forward(self.model, self.data)

        print(f"Initial ankle position: {self.data.qpos[3]}, velocity: {self.data.qvel[3]}")
        print(f"Initial hip position: {self.data.qpos[4]}, velocity: {self.data.qvel[4]}")
        
        # Get important joint and body IDs
        self.ankle_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ankle_hinge")
        self.hip_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "hip_hinge")
        self.human_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "long_link_body")
        
        # Initialize controllers
        self.initialize_controllers(self.model, self.data)
  
    def initialize(self):
        """Initialize all simulation components."""
        # First, set up the data handler
        self.data_handler.create_standard_datasets()
        
        # Initialize renderer if visualization is enabled
        if self.visualization_flag:
            self.renderer = MujocoRenderer()
        
        # Initialize model (now uses renderer for XML configuration)
        self.initialize_model()
        
        # Set initial state
        original_time = self.data.time
        self.data.time = 0.0
        
        # Log initial state at t=0 before any physics steps
        self._log_simulation_data()

        # Restore original time
        self.data.time = original_time

        # Complete renderer setup with the model if visualization is enabled
        if self.visualization_flag:
            self.renderer.setup_visualization(self.model, self.config)
            self.renderer.start_recording()
        
        print(f"Simulation duration: {self.config['simend']} seconds")

    def _log_simulation_data(self, custom_time=None):
        """Log all simulation data for the current timestep."""

        # Use custom time if provided, otherwise use simulation time
        time_value = custom_time if custom_time is not None else self.data.time
        
        # Extract actuator IDs
        human_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
        exo_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
        hip_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "hip_actuator")

        # Get ankle torque values
        human_ankle_torque = self.data.actuator_force[human_actuator_id] 
        exo_ankle_torque = self.data.actuator_force[exo_actuator_id] 
        ankle_torque_executed = self.data.qfrc_actuator[self.ankle_joint_id]
        ankle_gravity_torque = self.data.qfrc_bias[self.ankle_joint_id]
        
        # Get hip torque values
        hip_torque = self.data.actuator_force[hip_actuator_id]
        hip_gravity_torque = self.data.qfrc_bias[self.hip_joint_id]
        
        # Log ankle torque data
        self.data_handler.log_data("human_ankle_torque", np.array([time_value, human_ankle_torque]))
        self.data_handler.log_data("exo_ankle_torque", np.array([time_value, exo_ankle_torque]))
        self.data_handler.log_data("ankle_gravity_torque", np.array([time_value, ankle_gravity_torque]))
        
        # Log hip torque data
        self.data_handler.log_data("hip_torque", np.array([time_value, hip_torque]))
        self.data_handler.log_data("hip_gravity_torque", np.array([time_value, hip_gravity_torque]))
        
        # Log ankle joint data
        self.data_handler.log_data("ankle_position", np.array([time_value, 180/np.pi*self.data.qpos[self.ankle_joint_id]]))
        self.data_handler.log_data("ankle_velocity", np.array([time_value, 180/np.pi*self.data.qvel[self.ankle_joint_id]]))
        
        # Log hip joint data
        self.data_handler.log_data("hip_position", np.array([time_value, 180/np.pi*self.data.qpos[self.hip_joint_id]]))
        self.data_handler.log_data("hip_velocity", np.array([time_value, 180/np.pi*self.data.qvel[self.hip_joint_id]]))
        
        # Constraint and contact forces
        self.data_handler.log_data("constraint_force", np.array([
            time_value, 
            self.data.qfrc_constraint[0], 
            self.data.qfrc_constraint[1], 
            self.data.qfrc_constraint[2], 
            self.data.qfrc_constraint[3]
        ]))
        self.data_handler.log_data("contact_force", np.array([
            time_value, 
            self.data.sensordata[1], 
            self.data.sensordata[2]
        ]))
        
        # COM data
        com = self.data.xipos[self.human_body_id]
        self.data_handler.log_data("body_com", np.array([time_value, com[0], com[1], com[2]]))

        # Log ankle RTD data if available
        if hasattr(self.human_controller, 'current_rtd'):
            self.data_handler.log_data("ankle_rtd", np.array([
                time_value, 
                self.human_controller.current_rtd, 
                self.human_controller.current_rtd_limit
            ]))
            
        # Log hip RTD data if available
        if hasattr(self.hip_controller, 'current_rtd'):
            self.data_handler.log_data("hip_rtd", np.array([
                time_value, 
                self.hip_controller.current_rtd, 
                self.hip_controller.current_rtd_limit
            ]))

    def _simulation_step(self):
        """Execute one step of the simulation."""       
        self.controller(self.model, self.data)
        mj.mj_step(self.model, self.data)
        self._log_simulation_data()
            
    def run(self):
        """Run the simulation."""
        simend = self.config['simend']

        # Record initial state at t=0 before any physics steps
        original_time = self.data.time  # Save current time
        self.data.time = 0.0           # Explicitly set to t=0

        self.controller(self.model, self.data)
        
        # Do a forward pass so MuJoCo sees the new ctrl
        mj.mj_forward(self.model, self.data)

        self._log_simulation_data()    # Log initial state
        self.data.time = original_time # Restore original time
        
        start_time = time.time()
        
        # Main simulation loop with visualization
        if self.renderer:
            self.update_renderer_controller_params()

            while not self.renderer.window_should_close():
                simstart = self.data.time
                
                # Run physics at a higher rate than rendering (60 fps)
                while (self.data.time - simstart < 1.0/60.0):
                    # Step physics and record data
                    self._simulation_step()
                    
                    # Check if simulation time exceeded
                    if self.data.time >= simend:
                        break
                
                # Check if simulation time exceeded
                if self.data.time >= simend:
                    break
                    
                # Render the current state
                self.renderer.render(self.model, self.data)
        
        # Without visualization - run the simulation faster
        else:
            while self.data.time < simend:
                self._simulation_step()
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
            
        if self.renderer:
            video_filename = 'simulation.mp4'
            video_path = os.path.join(self.data_handler.run_dir, video_filename)
            self.renderer.save_video(video_path, 60)  # 60 fps
            self.renderer.close()
        
        self.data_handler.finalize()
        print(f"Results saved to: {self.data_handler.run_dir}")

if __name__ == "__main__":
    simulation = AnkleHipExoSimulation('config.yaml')
    simulation.initialize()
    simulation.run()