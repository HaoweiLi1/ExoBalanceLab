> Historical workflow. Install the unified package from the repository root first (`pip install -e .`).
> For maintained commands and verified experiments, see the [root README](../README.md).
> Historical result claims below retain their original scope; they are not new validation.

# Double-Joint Humanoid MuJoCo Simulation

This readme provides a detailed guide on using the MuJoCo simulation for the double-joint (ankle + hip) humanoid model and explains how it was developed by extending the single-joint (ankle only) version.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How to Run a Simulation](#how-to-run-a-simulation)
- [Understanding the Model](#understanding-the-model)
- [Extending from Single-Joint to Double-Joint](#extending-from-single-joint-to-double-joint)
- [Visualization and Data Analysis](#visualization-and-data-analysis)
- [Troubleshooting](#troubleshooting)

## Overview

This project provides a biomechanical simulation of a human leg with two controlled joints (ankle and hip) using the MuJoCo physics engine. The model incorporates realistic anthropometric parameters, various controller options, and an optional exoskeleton component for the ankle. The simulation allows for detailed analysis of joint dynamics, controller performance, and biomechanical interactions.

## Project Structure

```
joint2_humanoid/
├── simulation.py            # Main simulation class and entry point
├── controllers.py           # Controller implementations (PD, LQR, Precomputed)
├── data_handler.py          # Data logging, saving, and plotting functionality
├── renderer.py              # Visualization and XML utilities
├── initial_humanoid.xml     # Base MuJoCo model definition
├── config.yaml              # Simulation configuration parameters
├── pre_traj/                # Directory containing precomputed trajectories
│   └── ankle_torque_trajectory.csv   # Sample precomputed ankle trajectory
├── data/                    # Directory for simulation output (auto-created)
│   └── YYYYMMDD_HHMMSS/     # Timestamped run directories
│       ├── ankle_plots.png  # Generated ankle joint plots
│       ├── hip_plots.png    # Generated hip joint plots
│       ├── ankle_state.csv  # Ankle state data
│       ├── hip_state.csv    # Hip state data
│       ├── ankle_torque.csv # Ankle torque data
│       ├── hip_torque.csv   # Hip torque data
│       ├── force.csv        # Contact and constraint force data
│       ├── config.yaml      # Copy of the run configuration
│       └── simulation.mp4   # Video recording of the simulation
└── README.md                # This documentation file
```

## Installation

### Prerequisites

1. Python 3.8 or newer
2. MuJoCo 2.1.0 or newer

### Dependencies

Install the required Python packages:

```bash
pip install numpy scipy matplotlib mujoco imageio imageio-ffmpeg opencv-python pyyaml
```

## Usage

The simulation can be run with different controller configurations for both ankle and hip joints. The model supports human controllers, an optional exoskeleton controller for the ankle, and a hip controller with adjustable parameters.

### Controller Types

#### Human (Ankle) Controllers:
- **LQR**: Linear Quadratic Regulator for optimal control
- **PD**: Proportional-Derivative controller with configurable gains
- **Precomputed**: Uses pre-defined torque trajectories from CSV files

#### Exoskeleton (Ankle) Controllers:
- **PD**: Proportional-Derivative controller
- **None**: Exoskeleton disabled

#### Hip Controllers:
- **PD**: Proportional-Derivative controller
- **None**: Hip controller disabled

## Configuration

The `config.yaml` file contains all simulation parameters including:

1. **Simulation Parameters**: Duration, timestep, visualization settings
2. **Body Parameters**: Mass and height of the simulated human
3. **Joint Parameters**: Initial conditions, setpoints, torque limits for both ankle and hip joints
4. **Controller Settings**: Type and parameters for human, exoskeleton, and hip controllers
5. **Visualization Settings**: Camera position, visual elements to display

Example configuration snippet:
```yaml
config:
  # Simulation parameters
  simend: 6                                     # Simulation duration (seconds)
  simulation_timestep: 0.0005                   # Physics timestep
  visualization_flag: True                      # Enable 3D visualization
  
  # Body parameters
  M_total: 60                                   # Total mass (kg)
  H_total: 1.59                                 # Total height (m)
  
  # Ankle joint parameters
  ankle_initial_position_radians: -0.03         # Initial ankle angle
  ankle_initial_velocity: 0.3                   # Initial ankle velocity
  ankle_position_setpoint_radians: 0.00         # Ankle setpoint
  mrtd_df: 148                                  # Max rate of torque development for dorsiflexion
  mrtd_pf: 389                                  # Max rate of torque development for plantarflexion
  
  # Hip joint parameters
  hip_initial_position_radians: 0.1             # Initial hip angle
  hip_initial_velocity: 0.0                     # Initial hip velocity
  hip_position_setpoint_radians: 0.0            # Hip setpoint
  hip_mrtd: 500                                 # Max rate of torque development for hip
  
  # Controller configuration
  controllers:
    human:
      type: "Precomputed"                       # Options: "LQR", "PD", "Precomputed"
      # Controller-specific parameters...
      
    exo:
      type: "PD"                                # Options: "PD", "None"
      # Controller-specific parameters...
      
    hip:
      type: "PD"                                # Options: "PD", "None"
      # Controller-specific parameters...
```

## How to Run a Simulation

Run the simulation with:

```bash
python simulation.py
```

This will:
1. Load the configuration from `config.yaml`
2. Initialize the model, controllers, and data handler
3. Run the simulation for the configured duration
4. Save results and plots to a timestamped directory in `data/`

## Understanding the Model

### Physical Model

The double-joint humanoid model consists of:

1. **Foot**: Base body that can slide and rotate relative to the ground
2. **Lower Leg**: Connected to the foot via an ankle hinge joint
3. **Upper Body**: Connected to the lower leg via a hip hinge joint

The model uses anthropometric data to set realistic segment lengths, masses, and center of mass positions based on the total height and weight specified in the configuration.

### Actuators and Controllers

The model has three actuators:
1. **Human Ankle Actuator**: Represents human-generated ankle torque
2. **Exo Ankle Actuator**: (Optional) Represents torque from an ankle exoskeleton
3. **Hip Actuator**: Represents human-generated hip torque

Each actuator is controlled by a corresponding controller that can be configured separately.

## Extending from Single-Joint to Double-Joint

This section explains the key modifications made to extend the single-joint model to support two joints.

### 1. XML Model Changes

The major changes to `initial_humanoid.xml` include:

1. **Added Upper Body Segment**:
```xml
<body name="upper_body">
    <joint type="hinge" name="hip_hinge" axis="0 1 0" damping="2.5"/>
    <geom name="upper_body_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.025" />
    <geom name="upper_body_com" type="sphere" size="0.05" pos="0 0 0.25"/>
</body>
```

2. **Added Hip Actuator**:
```xml
<actuator>
    <motor joint="ankle_hinge" name="human_ankle_actuator" gear="1" ctrlrange="-150 150" ctrllimited="true"/> 
    <motor joint="ankle_hinge" name="exo_ankle_actuator" gear="1" ctrlrange="-50 50" ctrllimited="true"/>
    <motor joint="hip_hinge" name="hip_actuator" gear="1" ctrlrange="-150 150" ctrllimited="true"/>
</actuator>
```

3. **Added Hip Sensors**:
```xml
<sensor>
    <jointpos joint='ankle_hinge'/>
    <touch site='front_foot_site'/>
    <touch site='back_foot_site'/>
    <jointpos joint='hip_hinge'/>
    <jointvel joint='hip_hinge'/>
    <actuatorfrc actuator='hip_actuator'/>
</sensor>
```

### 2. Controller Additions

In `controllers.py`, we added:

1. **Hip Controller Classes**:
   - `HipController`: Base class 
   - `HipPDController`: PD control implementation
   - `HipNoneController`: Disabled controller

2. **Factory Function for Hip Controllers**:
```python
def create_hip_controller(controller_type, model, data, params):
    """Create a hip controller instance based on type."""
    if controller_type == "PD":
        return HipPDController(
            model=model,
            data=data,
            kp=params.get('kp', 100),
            kd=params.get('kd', 10),
            mrtd=params.get('mrtd', None)
        )
    elif controller_type == "None":
        return HipNoneController(
            model=model,
            data=data
        )
    else:
        raise ValueError(f"Unknown hip controller type: {controller_type}")
```

### 3. Simulation Class Updates

The main simulation class (`AnkleHipExoSimulation`) was updated to:

1. **Initialize Hip Controller**:
```python
# Get hip controller configuration
hip_config = self.config['controllers']['hip']
hip_type = hip_config['type']

# Prepare hip parameters
hip_params = {
    'mrtd': self.config.get('hip_mrtd')
}

# Add controller-specific parameters
if hip_type == "PD":
    pd_params = hip_config['pd_params']
    hip_params.update({
        'kp': pd_params.get('kp', 700),
        'kd': pd_params.get('kd', 50)
    })

# Create hip controller
self.hip_controller = create_hip_controller(
    hip_type, model, data, hip_params
)
```

2. **Control Both Joints**:
```python
def controller(self, model, data):
    """Controller function for both ankle and hip joints."""
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
    
    # Compute and apply ankle control
    data.ctrl[0] = self.human_controller.compute_control(
        state=ankle_state,
        target=self.ankle_position_setpoint
    )
    data.ctrl[1] = self.exo_controller.compute_control(
        state=ankle_state,
        target=self.ankle_position_setpoint
    )
    
    # Compute and apply hip control
    data.ctrl[2] = self.hip_controller.compute_control(
        state=hip_state,
        target=self.hip_position_setpoint
    )
```

### 4. Data Handling Updates

The `data_handler.py` was updated to:

1. **Track Hip Data**:
```python
# Create additional datasets for hip
self.create_dataset("hip_position", 2)    # time, position
self.create_dataset("hip_velocity", 2)    # time, velocity
self.create_dataset("hip_torque", 2)      # time, torque
self.create_dataset("hip_gravity_torque", 2) # time, gravity torque
self.create_dataset("hip_rtd", 3)         # time, rtd_value, rtd_limit
```

2. **Generate Hip Plots**:
```python
def create_hip_plots(self):
    """Create a 3x1 plot with hip angle, velocity, and torque."""
    # ... plotting code for hip data ...
    plt.savefig(os.path.join(self.run_dir, "hip_plots.png"))
```

### 5. Configuration Updates

The `config.yaml` was extended with hip parameters:

```yaml
# Hip joint parameters
hip_initial_position_radians: 0.1
hip_initial_velocity: 0.0  
hip_position_setpoint_radians: 0.0
hip_mrtd: 500

# Hip controller configuration
controllers:
  hip:
    type: "PD"
    pd_params:
      kp: 700
      kd: 50
```

## Visualization and Data Analysis

### Visualization

The simulation provides a 3D visualization using MuJoCo's GLFW-based renderer. The visualization shows:
- The foot, lower leg, and upper body segments
- The ankle and hip joints
- The exoskeleton (if enabled)
- Optional visual elements like contact forces and center of mass

### Data Analysis

After each simulation run, the data handler generates:

1. **CSV Files**:
   - `ankle_state.csv`: Ankle angle and velocity
   - `hip_state.csv`: Hip angle and velocity
   - `ankle_torque.csv`: Human ankle, exo, and gravity torques
   - `hip_torque.csv`: Hip and gravity torques
   - `force.csv`: Contact and constraint forces

2. **Plot Files**:
   - `ankle_plots.png`: Time series of ankle angle, velocity, and torques
   - `hip_plots.png`: Time series of hip angle, velocity, and torque

3. **Video Recording**:
   - `simulation.mp4`: Video of the 3D visualization if recording is enabled

## Troubleshooting

### Common Issues

1. **Unstable Simulation**:
   - Try reducing the simulation timestep
   - Adjust controller gains to be less aggressive
   - Check for unrealistic initial conditions

2. **Controller Issues**:
   - Verify controller gains are appropriate for the system
   - Check that controller types are correctly specified
   - Ensure actuator limits are reasonable

3. **Visualization Problems**:
   - Make sure OpenGL dependencies are installed
   - Try adjusting camera settings in config.yaml
   - Verify that GLFW-related libraries are available

4. **Data Issues**:
   - Check for NaN values in CSV output (usually indicates simulation instability)
   - Ensure all required files are in the correct locations
   - Verify that file paths are correct for your operating system

If you encounter any additional issues or need assistance with extending the model further, please consult the MuJoCo documentation or reach out to the project maintainers.

---

By following this guide, you should be able to run the double-joint humanoid simulation and understand how it was developed from the single-joint version. The simulation provides a flexible platform for biomechanical analysis and controller development for both ankle and hip joints.