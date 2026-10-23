import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import imageio
import cv2  # Add OpenCV import for text rendering
import xml.etree.ElementTree as ET

# XML utilities functions integrated from xml_utilities.py
def calculate_kp_and_geom(weight, height):
    """
    Calculate mass, geometry, and controller gain data from literature equations.
    
    Args:
        weight: Total body mass in kg
        height: Total body height in meters
        
    Returns:
        Tuple of (h_f, m_feet, m_body, l_COM, l_foot, a, K_p)
    """
    M_total = weight # [kg]
    H_total = height # [meters]
    m_feet = 2*0.0145 * M_total
    m_body = M_total - m_feet
    l_COM = 0.575*H_total
    l_foot = 0.152*H_total
    h_f = 0.039*H_total
    a = 0.19*l_foot
    K_p = m_body * 9.81 * l_COM 

    return h_f, m_feet, m_body, l_COM, l_foot, a, K_p


def set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric):
    """
    Set geometry parameters in the XML model.
    
    Args:
        root: XML root element
        m_feet: Mass of feet
        m_body: Mass of body
        l_COM: Length to center of mass
        l_foot: Length of foot
        a: Ankle position
        H_total: Total height
        h_f: Height of foot
        trans_fric: Translation friction constant
        roll_fric: Rolling friction constant
    """
    for geom in root.iter('geom'):
        if geom.get('name') == "long_link_geom":
            geom.set('mass', "0")
            geom.set('fromto', f'0 0 {H_total} 0 0 0') # this from (x,y,z)_1 to (x,y,z)_2 is w.r.t the long_link_body frame
            # geom.set('pos', f'0 0 {H_total-l_COM}')

        elif geom.get('name') == "m_body":    
            geom.set('mass', str(m_body))
            geom.set('pos', f"0 0 {l_COM}") # this x,y,z position is w.r.t the long_link_body frame
            # geom.set('size', 0.05)
        
        elif geom.get('name') == "foot":
            geom.set('pos', f'0 0 0') # this x,y,z position is w.r.t the foot_body frame

    for body in root.iter('body'):
            if body.get('name') == "foot":
                # poo = body.get('pos')
                # print(f'pos: {poo}')
                body.set('pos',  f'0. 0 0') # this x,y,z position is w.r.t the global frame
                body.set('quat', f'0 0 0 1') # unit quaternion for pi radians rotation about global z-axis


            elif body.get('name') == "long_link_body":
                # size = float(body.get('size'))
                body.set('pos', f'{-l_foot/2+a} 0 {h_f}') # this x,y,z position is w.r.t the foot reference frame

    for joint in root.iter('joint'):
            if joint.get('name') == "ankle_hinge":
                joint.set("pos", f"0 0 0") # this x,y,z position is w.r.t the long_link_body reference frame

            elif joint.get('name') == "rotation_dof":
                joint.set('pos', f'{-l_foot/2+a} 0 {h_f}') # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

            elif joint.get('name') == "joint_slide_x":
                joint.set('pos', f"{-l_foot/2+a} 0 0.035") # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

            elif joint.get('name') == "joint_slide_z":
                joint.set('pos', f"{-l_foot/2+a} 0 0.035") # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

    for pair in root.iter('pair'):
        if pair.get('name') == "foot_ground_friction":
            pair.set('friction', f"{trans_fric} {trans_fric} 0.99 {roll_fric} {roll_fric}")

    for mesh in root.iter('mesh'):
        if mesh.get('name') == "foot_mesh":
            mesh.set('vertex', f"{-l_foot/2} -0.045 0   {-l_foot/2} 0.045 0   {l_foot/2} -0.045 0   {l_foot/2} 0.045 0  {-l_foot/2+a} -0.045 {h_f} {-l_foot/2+a} 0.045 {h_f}")

    for site in root.iter('site'):
        if site.get('name') == "front_foot_site":
            # site.set('pos', f"{-l_foot/2} 0 0.05")
            site.set('fromto', f"{-l_foot/2} 0 0.0 {-l_foot/2} 0 0.1")
            # site.set('fromto', f"0 0 0.0 0 0 0.1")

        elif site.get('name') == "back_foot_site":
            # site.set('pos', f"{l_foot/2} 0 0.05")
            site.set('fromto', f"{l_foot/2} 0 0.0 {l_foot/2} 0 0.1")


class MujocoRenderer:
    """
    Class for handling MuJoCo rendering and visualization separately from simulation.
    This handles window creation, scene rendering, and video recording.
    Also includes XML model processing functionality.
    """
    
    def __init__(self, window_size=(1200, 912), title="Ankle Exoskeleton Simulation"):
        """
        Initialize the renderer.
        
        Args:
            window_size: Tuple of (width, height) for the window
            title: Window title
        """
        self.window_size = window_size
        self.title = title
        self.frames = []  # For recording
        self.is_recording = False
        self.window = None
        self.scene = None
        self.context = None
        self.cam = None
        self.opt = None
        
        # Simulation parameters to display
        self.controller_params = {}
        self.initial_conditions = {}
        
        # Initialize GLFW and create window
        self._init_glfw()
        
    def _init_glfw(self):
        """Initialize GLFW and create a window."""
        glfw.init()
        self.window = glfw.create_window(
            self.window_size[0], 
            self.window_size[1], 
            self.title, 
            None, 
            None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Create camera and options objects
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        
        # Set default camera and option parameters
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
    
    def set_controller_params(self, params):
        """
        Set controller parameters to display on video.
        
        Args:
            params: Dictionary of controller parameters
        """
        self.controller_params = params
        
    def update_controller_params(self, params):
        """
        Update specific controller parameters (keeps existing ones).
        
        Args:
            params: Dictionary of controller parameters to update
        """
        for key, value in params.items():
            self.controller_params[key] = value
    
    def set_initial_conditions(self, conditions):
        """
        Set initial conditions to display on video.
        
        Args:
            conditions: Dictionary of initial conditions
        """
        self.initial_conditions = conditions
    
    def setup_visualization(self, model, config):
        """
        Configure visualization settings based on config parameters.
        
        Args:
            model: MuJoCo model object
            config: Configuration dictionary with visualization settings
        """
        # Configure visualization flags
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = config.get('visualize_contact_force', False)
        # self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = config.get('visualize_perturbation_force', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = config.get('visualize_joints', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = config.get('visualize_actuators', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_COM] = config.get('visualize_center_of_mass', False)
        
        # Set camera configuration
        self.cam.azimuth = config.get('camera_azimuth', 90)
        self.cam.distance = config.get('camera_distance', 3.0)
        self.cam.elevation = config.get('camera_elevation', -5)
        
        # Parse and set lookat coordinates
        lookat_string = config.get('camera_lookat_xyz', '0, 0, 1')
        lookat_coords = tuple(map(float, lookat_string.split(', ')))
        self.cam.lookat = np.array([lookat_coords[0], lookat_coords[1], lookat_coords[2]])
        
        # Configure model visualization parameters
        model.vis.map.force = 0.25  # scaling parameter for force vector's length
        model.vis.map.torque = 0.1  # scaling parameter for control torque
        
        # Configure scale parameters
        model.vis.scale.contactwidth = 0.05   # width of the floor contact point
        model.vis.scale.contactheight = 0.01  # height of the floor contact point
        model.vis.scale.forcewidth = 0.03     # width of the force vector
        model.vis.scale.com = 0.2             # com radius
        model.vis.scale.actuatorwidth = 0.1   # diameter of visualized actuator
        model.vis.scale.actuatorlength = 0.1  # thickness of visualized actuator
        model.vis.scale.jointwidth = 0.025    # diameter of joint arrows
        model.vis.scale.framelength = 0.25
        model.vis.scale.framewidth = 0.05
        
        # Configure color parameters
        model.vis.rgba.contactforce = np.array([0.7, 0., 0., 0.5], dtype=np.float32)
        model.vis.rgba.force = np.array([0., 0.7, 0., 0.5], dtype=np.float32)
        model.vis.rgba.joint = np.array([0.2, 1, 0.1, 0.8])
        model.vis.rgba.actuatorpositive = np.array([0., 0.9, 0., 0.5])  # color when actuator is exerting positive force
        model.vis.rgba.actuatornegative = np.array([0.9, 0., 0., 0.5])  # color when actuator is exerting negative force
        model.vis.rgba.com = np.array([1., 0.647, 0., 0.5])  # center of mass color
        
        # Create scene and context
        self.scene = mj.MjvScene(model, maxgeom=10000)
        self.context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
        
        # Extract controller parameters and initial conditions from config
        self._extract_config_params(config)
    
    def _extract_config_params(self, config):
        """
        Extract parameters from config to display on video.
        
        Args:
            config: Configuration dictionary
        """
        # Extract controller parameters
        if 'controllers' in config:
            if 'human' in config['controllers']:
                human_controller = config['controllers']['human']
                self.controller_params['Human Controller'] = human_controller['type']
                
                if human_controller['type'] == 'LQR':
                    # For LQR, we want to show gains instead of Q and R values
                    # We'll update these values later from the controller itself
                    # But keep the Q and R values for now
                    if 'lqr_params' in human_controller:
                        lqr = human_controller['lqr_params']
                        self.controller_params['Q_angle'] = lqr.get('Q_angle', 'N/A')
                        self.controller_params['Q_velocity'] = lqr.get('Q_velocity', 'N/A')
                        self.controller_params['R'] = lqr.get('R', 'N/A')
                        # Add a placeholder for the LQR gain matrix
                        self.controller_params['LQR Gain'] = "Not yet initialized"
                
                elif human_controller['type'] == 'PD' and 'pd_params' in human_controller:
                    pd = human_controller['pd_params']
                    self.controller_params['Kp'] = pd.get('kp', 'N/A')
                    self.controller_params['Kd'] = pd.get('kd', 'N/A')
            
            if 'exo' in config['controllers']:
                exo_controller = config['controllers']['exo']
                self.controller_params['Exo Controller'] = exo_controller['type']
                
                if exo_controller['type'] == 'PD' and 'pd_params' in exo_controller:
                    pd = exo_controller['pd_params']
                    
                    # Check if using dynamic gains
                    use_dynamic_gains = pd.get('use_dynamic_gains', False)
                    if use_dynamic_gains:
                        self.controller_params['Exo Kp'] = "Dynamic (calculating...)"
                        self.controller_params['Exo Kd'] = "Dynamic (calculating...)"
                    else:
                        self.controller_params['Exo Kp'] = pd.get('kp', 'N/A')
                        self.controller_params['Exo Kd'] = pd.get('kd', 'N/A')
        
        # Extract initial conditions - without the label "Initial"
        self.initial_conditions['Ankle Angle'] = f"{config.get('ankle_initial_position_radians', 0):.3f} rad"
        self.initial_conditions['Ankle Velocity'] = f"{config.get('ankle_initial_velocity', 0):.3f} rad/s"
        self.initial_conditions['Ankle Setpoint'] = f"{config.get('ankle_position_setpoint_radians', 0):.3f} rad"
        self.initial_conditions['Human Height'] = f"{config.get('H_total', 0):.2f} m"
        self.initial_conditions['Human Weight'] = f"{config.get('M_total', 0):.1f} kg"
    
    def render(self, model, data):
        """
        Render a single frame of the simulation.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            
        Returns:
            bool: True if rendering should continue, False if window should close
        """
        if self.window_should_close():
            return False
            
        # Get framebuffer viewport size
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Update scene and render
        mj.mjv_updateScene(
            model, 
            data, 
            self.opt, 
            None, 
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value, 
            self.scene
        )
        mj.mjr_render(viewport, self.scene, self.context)
        
        # Record frame if enabled
        if self.is_recording:
            self._capture_frame(viewport, viewport_height, viewport_width, data.time)
            
        # Swap buffers and process events
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
        return True
    
    def _capture_frame(self, viewport, height, width, sim_time):
        """
        Capture a frame for video recording and add text overlays.
        
        Args:
            viewport: MjrRect viewport object
            height: Viewport height
            width: Viewport width
            sim_time: Current simulation time
        """
        rgb_array = np.empty((height, width, 3), dtype=np.uint8)
        depth_array = np.empty((height, width), dtype=np.float32)
        
        mj.mjr_readPixels(
            rgb=rgb_array, 
            depth=depth_array, 
            viewport=viewport, 
            con=self.context
        )
        
        # Flip the image vertically since OpenGL renders with inverted Y axis
        rgb_array = np.flipud(rgb_array)
        
        # Add parameter overlays using OpenCV
        img = self._add_text_overlays(rgb_array, sim_time)
        
        self.frames.append(img)
    
    def _add_text_overlays(self, img, sim_time):
        """
        Add text overlays to the frame using OpenCV.
        
        Args:
            img: RGB image array
            sim_time: Current simulation time
            
        Returns:
            Modified image with text overlays
        """
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get image dimensions
        height, width = img_bgr.shape[:2]
        
        # Define font and text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Add simulation time
        cv2.putText(img_bgr, f"Time: {sim_time:.2f}s", (10, 30), 
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Add controller parameters (upper left) without the header
        y_pos = 70
        
        for key, value in self.controller_params.items():
            cv2.putText(img_bgr, f"{key}: {value}", (10, y_pos), 
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_pos += 25
        
        # Add initial conditions (upper right) without the header
        y_pos = 70
        x_pos = width - 300
        
        for key, value in self.initial_conditions.items():
            cv2.putText(img_bgr, f"{key}: {value}", (x_pos, y_pos), 
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_pos += 25
        
        # Convert back to RGB
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    def start_recording(self):
        """Start recording frames for video."""
        self.is_recording = True
        self.frames = []
    
    def stop_recording(self):
        """Stop recording frames."""
        self.is_recording = False
    
    def save_video(self, filename, fps=60):
        """
        Save recorded frames as a video.
        
        Args:
            filename: Output video filename
            fps: Frames per second
        """
        if self.frames:
            print(f"Saving video to {filename} ({len(self.frames)} frames at {fps} fps)")
            imageio.mimwrite(filename, self.frames, fps=fps)
            self.frames = []
        else:
            print("No frames to save")
    
    def window_should_close(self):
        """Check if the window should close."""
        return glfw.window_should_close(self.window)
    
    def close(self):
        """Clean up resources."""
        # MuJoCo Python bindings handle freeing objects automatically
        # when they go out of scope, so explicit freeing is not needed
            
        glfw.terminate()
        print("Renderer resources released")
    
    # Added XML utility methods
    @staticmethod
    def calculate_kp_and_geom(weight, height):
        """
        Calculate mass, geometry, and controller gain data from literature equations.
        Wrapper for external function
        """
        return calculate_kp_and_geom(weight, height)
    
    @staticmethod
    def set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric):
        """
        Set geometry parameters in the XML model.
        Wrapper for external function
        """
        set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric)