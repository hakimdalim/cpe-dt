# FFF Temperature Distribution Analysis

A comprehensive simulation framework for analyzing temperature distribution within printed walls during the Fused Filament Fabrication (FFF) process using PLA material.

## Project Overview

This project implements a complete digital twin simulation for FFF 3D printing temperature analysis, designed to complement thermal camera measurements. The simulation models the sequential deposition process, heat transfer physics, and provides realistic temperature distributions for validation against experimental data.

### Key Features

- **Sequential Element Activation**: Models layer-by-layer printing process following actual nozzle toolpath
- **Realistic Physics**: Includes heated bed effects, glass transition behavior, and convective cooling
- **Thermal Camera Integration**: Designed for validation against MLX90640 thermal camera measurements
- **Multiple Wall Configurations**: Supports 30mm and 70mm wall lengths with different print speeds
- **Professional Visualizations**: Creates animations and analysis plots for presentations

## Architecture Overview

The codebase is structured into modular components:

```
├── Core Simulation Components
│   ├── mesh.py           # Finite element mesh generation
│   ├── activation.py     # Sequential element activation (nozzle following)
│   ├── thermal.py        # Heat transfer calculations
│   └── simulator.py      # Main simulation controller
├── Analysis & Visualization
│   ├── analyze_results.py         # Basic results analysis
│   ├── analyze_realistic_results.py # Enhanced physics analysis
│   └── create_realistic_animations.py # Professional animations
├── Configuration & Execution
│   ├── config.yaml       # Simulation parameters
│   └── run_fff_simulation.py # Main execution script
└── Documentation
    ├── requirements.txt  # Python dependencies
    └── README.md        # This file
```

## Core Components Explained

### 1. Mesh Module (`mesh.py`)

**Purpose**: Creates the finite element discretization of the wall geometry.

**Key Concepts**:
- **Complete Geometry First**: Generates mesh for the entire final wall shape
- **Sequential Activation**: Elements start inactive and are turned "on" during printing
- **2D Approach**: Models wall in x-z plane (length × height), assumes uniform temperature across thickness

**Key Classes**:
- `Node`: Temperature calculation points
- `Element`: Rectangular mesh elements connecting 4 nodes
- `Mesh`: Main mesh management class

**Usage Example**:
```python
mesh = Mesh("config.yaml")  # Creates complete wall mesh
mesh.activate_element(0, temperature=175.0)  # Activates first element
active_elements = mesh.get_active_elements()  # Get currently active elements
```

### 2. Activation Module (`activation.py`)

**Purpose**: Simulates the nozzle movement and sequential element activation during printing.

**Key Concepts**:
- **Toolpath Following**: Nozzle moves left-to-right, layer by layer
- **Time-Based Activation**: Elements activate when nozzle reaches them
- **Print Speed Integration**: Activation timing based on actual print speeds

**Key Classes**:
- `NozzleState`: Current nozzle position and status
- `ActivationEvent`: Records when/where elements were activated
- `SequentialActivator`: Main activation controller

**Printing Pattern**:
```
Layer 0: (0,0) → (30,0) [left to right]
Layer 1: (0,0.2) → (30,0.2) [next layer up]
Layer 2: (0,0.4) → (30,0.4) [continue...]
```

### 3. Thermal Module (`thermal.py`)

**Purpose**: Handles heat transfer calculations and cooling physics.

**Key Features**:
- **Convective Cooling**: Heat loss to ambient air
- **Material Properties**: Uses actual PLA thermal properties
- **Temperature-Dependent Effects**: Supports glass transition modeling
- **Boundary Conditions**: Ambient temperature and heat transfer coefficients

**Physics Implemented**:
- Heat conduction within printed material
- Convective cooling to air
- Optional: Heated bed influence, glass transition effects

### 4. Simulator Module (`simulator.py`)

**Purpose**: Main simulation controller that orchestrates all components.

**Key Responsibilities**:
- **Time Stepping**: Manages simulation time advancement
- **Component Coordination**: Synchronizes mesh, activation, and thermal modules
- **Results Management**: Saves temperature histories and simulation data
- **Progress Monitoring**: Tracks printing progress and simulation status

**Main Loop**:
1. Update element activation (follow nozzle)
2. Calculate heat transfer
3. Update monitoring points
4. Save results at intervals
5. Check completion status
6. Advance time

## Configuration System

### Main Configuration (`config.yaml`)

The simulation is controlled through a YAML configuration file:

```yaml
simulation:
  wall_length: 30.0      # mm (30 or 70 for experiments)
  wall_thickness: 2.0    # mm (task requirement)
  layer_height: 0.2      # mm
  print_speed: 20.0      # mm/s (20 or 40 for experiments)
  total_layers: 50       # number of layers

material:
  name: "PLA"
  thermal_conductivity: 0.13  # W/mK (from task materials)
  density: 1250               # kg/m³
  specific_heat: 1800         # J/kgK
  deposition_temp: 175.0      # °C (from experiments)
  ambient_temp: 25.0          # °C

mesh:
  element_size_x: 1.0    # mm (spatial resolution)
  element_size_z: 0.2    # mm (matches layer height)

output:
  monitoring_points:     # Temperature measurement locations
    - [5.0, 1.0]        # [x, z] coordinates in mm
    - [15.0, 2.0]
    - [25.0, 5.0]
```

## Usage Guide

### 1. Basic Simulation Run

```bash
# Single simulation
python run_fff_simulation.py --wall_length 30 --print_speed 20

# Parameter study (all combinations)
python run_fff_simulation.py --parameter_study
```

### 2. Analysis Workflow

```bash
# Run simulation first
python run_fff_simulation.py --wall_length 30 --print_speed 20

# Analyze results
python analyze_results.py

# Create professional animations
python create_realistic_animations.py
```

### 3. Integration with Thermal Camera

The simulation is designed to complement thermal camera measurements:

1. **Setup**: Mount MLX90640 camera for side-view temperature measurement
2. **Calibration**: Calibrate camera using known temperature references
3. **Measurement**: Record thermal videos during actual printing
4. **Validation**: Compare simulation results with camera measurements

## Output Files Structure

```
data/output/wall_30mm_speed_20mms_[timestamp]/
├── simulation_config.yaml           # Configuration used
├── simulation_results.csv           # Main time-series results
├── simulation_metadata.json         # Complete simulation info
├── Point_1_temperature.csv          # Temperature at monitoring point 1
├── Point_2_temperature.csv          # Temperature at monitoring point 2
├── Point_3_temperature.csv          # Temperature at monitoring point 3
├── temperature_fields.npy           # Full temperature field snapshots
├── snapshot_times.npy               # Corresponding time points
└── simulation_summary.png           # Summary visualization
```

## Analysis Capabilities

### Temperature Evolution Analysis
- Temperature-time curves at specific monitoring points
- Cooling rate calculations and exponential fitting
- Peak temperature identification and timing

### Spatial Analysis
- Temperature distribution across wall height
- Layer-by-layer temperature profiles
- Heat gradient visualization

### Physics Validation
- Heated bed influence on bottom temperatures
- Glass transition effects at 58°C
- Convective cooling behavior

### Comparative Studies
- Different wall lengths (30mm vs 70mm)
- Print speed effects (20 mm/s vs 40 mm/s)
- Validation against thermal camera data

## Installation and Dependencies

### Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies**:
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Visualization
- `pandas>=1.3.0` - Data handling
- `pyyaml>=6.0` - Configuration management
- `scipy>=1.7.0` - Scientific computing

### Optional for Advanced Features
- `ffmpeg` - Video creation for animations
- `numba` - Performance optimization

## Task-Specific Implementation

This codebase directly addresses the experimental task requirements:

### Task 1: Camera Mount Design
- Simulation provides validation targets for camera measurements
- Monitoring points correspond to camera measurement locations

### Task 2: Temperature Measurement
- **Wall Dimensions**: 2mm thick walls, 30mm and 70mm lengths ✓
- **Layer Heights**: Multiple monitoring points at different heights ✓
- **Print Speeds**: Configurable speeds (20 mm/s, 40 mm/s) ✓
- **Temperature Curves**: Detailed temperature-time data ✓

### Task 3: Simulation Requirements
- **Heat Distribution Model**: Full finite element thermal simulation ✓
- **Moving Heat Source**: Nozzle following with realistic toolpath ✓
- **Material Properties**: Actual PLA properties from task specification ✓
- **Results Comparison**: Structured for comparison with measurements ✓

## Advanced Features

### Realistic Physics Extensions

For enhanced realism, the code supports:

- **Heated Bed Effects**: 60°C bed temperature influence
- **Glass Transition**: Temperature-dependent material properties at 58°C
- **Enhanced Heat Transfer**: Combined convection and radiation
- **Temperature Gradients**: Realistic spatial temperature variations

### Professional Animations

Creates publication-quality animations showing:
- 2D thermal evolution with nozzle movement
- Physics comparison (basic vs realistic models)
- Comprehensive dashboard with multiple views
- Temperature gradient visualization

## Integration with Experimental Setup

### Hardware Integration
- **Raspberry Pi**: Interfaces with thermal camera
- **MLX90640**: 32×24 thermal sensor array
- **3D Printer**: Any FFF printer with accessible hotend

### Data Flow
1. **Simulation**: Predicts temperature distributions
2. **Experimental**: Measures actual temperatures with thermal camera
3. **Validation**: Compare simulation vs measurement
4. **Refinement**: Adjust simulation parameters based on experimental data

## Troubleshooting

### Common Issues

**Simulation Too Slow**:
- Reduce `element_size_x` and `element_size_z` for coarser mesh
- Increase `time_step` (within stability limits)
- Use simplified thermal model

**Memory Issues**:
- Reduce `total_layers` for shorter walls
- Increase `save_interval` to save less data
- Disable `save_temperature_history` for large meshes

**Convergence Problems**:
- Check time step stability (thermal diffusion limit)
- Verify material properties are physically reasonable
- Ensure boundary conditions are appropriate

### Debug Mode

Enable detailed debugging:
```python
# In config.yaml
solver:
  debug_mode: true
  verbose_output: true
```

## Contributing

### Code Structure Guidelines
- Each module has a single clear responsibility
- Configuration-driven design for flexibility
- Comprehensive error handling and validation
- Professional documentation and comments

### Testing
```bash
# Test individual modules
python mesh.py
python activation.py
python thermal.py
python simulator.py

# Full integration test
python run_fff_simulation.py --wall_length 30 --print_speed 20
```

## References

- **Task Documentation**: Based on "Analysis of temperature distribution within a printed wall" requirements
- **Material Properties**: PLA thermal properties from task specification
- **Experimental Setup**: MLX90640 thermal camera integration
- **Physics**: Fused Filament Fabrication heat transfer modeling

## License

This code is developed for academic research purposes as part of the FFF temperature distribution analysis project.

---

**Contact**: For questions about implementation or experimental integration, refer to the project documentation or task supervisors.
