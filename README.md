# Task 02: Analysis of Temperature Distribution within a Printed Wall
## Protocol and Simulation Framework

This repository contains the complete experimental protocol and simulation framework for analyzing temperature distribution during PLA (Polylactic Acid) 3D printing processes. The project combines physical thermal measurements with digital twin simulation for comprehensive validation and analysis.

## Project Overview

### Experimental Component
Systematic thermal calibration and data collection using MLX90640 thermal camera to capture real-time temperature distributions during FFF (Fused Filament Fabrication) printing.

### Simulation Component
Complete digital twin simulation framework that models the sequential deposition process, heat transfer physics, and provides realistic temperature distributions for validation against experimental data.

## Repository Structure

```
├── docs/
│   ├── protocol/                    # LaTeX protocol documentation
│   ├── calibration/              # Thermocouple and camera calibration data
│   └── data/                     # Experimental thermal datasets
├── Simulation Framework/
│   ├── scripts/                  # Core simulation components
│   │   ├── mesh.py              # Finite element mesh generation
│   │   ├── activation.py        # Sequential element activation
│   │   ├── thermal.py           # Heat transfer calculations
│   │   └── simulator.py         # Main simulation controller
│   ├── analysis/                # Analysis and visualization tools
│   └── config/                  # Configuration files
└── Documentation/
    ├── README.md                # This file
    └── requirements.txt         # Dependencies
```

## Experimental Protocol

### Laboratory Information
- **Institution**: Lehrstuhl für Computational Physics in Engineering (CPE), RPTU Kaiserslautern
- **Supervisor**: Prof. Dr.-Ing. Kristin de Payrebrune
- **Students**: Yating Wei, Hakim Dalim, Yahya Badine
- **Lab Assistants**: Iram, Harshad

### Three-Phase Experimental Workflow

#### Phase 1: Thermocouple Calibration
- Reference calibration using controlled incubator
- Multiple temperature setpoints (25°C, 48°C, 60°C)
- Accuracy tolerance: ±0.5°C

#### Phase 2: Thermal Camera Calibration
- MLX90640 calibration against reference thermocouple
- Linear regression analysis for correction equation
- Format: `T_calibrated = a × T_camera + b`

#### Phase 3: Data Collection
- Real-time thermal monitoring during 3D printing
- ESP32-based data acquisition system
- Python visualization with automatic processing

### Experimental Dataset

| File | Print Settings | Total Frames | Active Session | Crop Range |
|------|----------------|--------------|----------------|------------|
| **30×1×50_Speed** | Speed optimized | 10,224 | 800-8,500 | **800-8500** |
| **30×1×50_Structural** | Structural quality | 7,617 | 400-7,200 | **400-7200** |
| **70×1×50_Speed** | Speed optimized | 16,711 | 1,000-14,500 | **1000-14500** |
| **70×1×50_Structural** | Structural quality | 15,402 | 1,200-13,000 | **1200-13000** |

## Simulation Framework

### Core Architecture

The simulation implements a complete digital twin of the FFF process:

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
└── Configuration & Execution
    ├── config.yaml       # Simulation parameters
    └── run_fff_simulation.py # Main execution script
```

### Key Simulation Features

- **Sequential Element Activation**: Models layer-by-layer printing following actual nozzle toolpath
- **Realistic Physics**: Includes heated bed effects, glass transition behavior, convective cooling
- **Thermal Camera Integration**: Designed for direct validation against MLX90640 measurements
- **Multiple Configurations**: Supports experimental wall geometries (30mm/70mm lengths)

### Simulation Configuration

```yaml
simulation:
  wall_length: 30.0      # mm (matches experimental setup)
  wall_thickness: 2.0    # mm (task requirement)
  layer_height: 0.2      # mm
  print_speed: 20.0      # mm/s (speed vs structural settings)
  total_layers: 50       # number of layers

material:
  name: "PLA"
  thermal_conductivity: 0.13  # W/mK
  density: 1250               # kg/m³
  specific_heat: 1800         # J/kgK
  deposition_temp: 175.0      # °C (from experiments)
  ambient_temp: 25.0          # °C
```

## Integrated Workflow: Experiment + Simulation

### 1. Hardware Setup
- **Thermal Camera**: MLX90640 (24×32 pixel array)
- **Interface**: ESP32 microcontroller with I2C communication
- **3D Printer**: Standard FDM printer with PLA filament
- **Data Acquisition**: USB serial at 921600 baud

### 2. Calibration Process
```bash
# 1. Thermocouple calibration (experimental)
# 2. Camera calibration (experimental)
# 3. Simulation parameter validation
python run_fff_simulation.py --calibration_mode
```

### 3. Data Collection
```bash
# Experimental data collection
python display_thermal_camera_esp32_8fps_v2.py

# Simulation data generation
python run_fff_simulation.py --wall_length 30 --print_speed 20
```


## Critical Implementation Details

### Experimental Considerations
- **MLX90640 Warm-up**: 180-second stabilization period before data collection
- **Camera Distance**: 6-7 cm from build plate for optimal field-of-view
- **Broken Pixel Handling**: Temperatures >180°C replaced with NaN values
- **Temperature Filtering**: Valid range 25-230°C for PLA printing

### Simulation Physics
- **Heat Conduction**: Within printed material using PLA thermal properties
- **Convective Cooling**: Heat loss to ambient air
- **Sequential Deposition**: Time-based element activation following nozzle movement
- **Boundary Conditions**: Heated bed (60°C) and ambient temperature effects

## Data Processing Pipeline

### Experimental Data
1. **HDF5 File Loading**: Raw thermal imaging sequences
2. **Quality Control**: Frame validation and outlier removal
3. **Temporal Cropping**: Active printing session isolation
4. **Calibration Application**: Camera correction equation
5. **Analysis**: Temperature distribution and evolution

### Simulation Data
1. **Mesh Generation**: Finite element discretization
2. **Time Integration**: Heat transfer calculations
3. **Results Export**: Temperature fields and monitoring points
4. **Visualization**: 2D thermal evolution animations

## Usage Instructions

### Complete Workflow Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run experimental protocol (hardware required)
python mlx_data_logger.py

# 3. Execute simulation
python run_fff_simulation.py --parameter_study

```

### Individual Components

```bash
# Experimental data analysis only
python analyze_experimental_data.py --data_path data/

# Simulation only
python run_fff_simulation.py --wall_length 30 --print_speed 20

# Visualization
python create_realistic_animations.py
```

## Output Structure

### Experimental Results
```
experimental_data/
├── calibration/
│   ├── thermocouple_calibration.csv
│   └── camera_calibration_coefficients.json
├── real_data/
│   ├── 30x1x50_Speed.h5
│   ├── 30x1x50_Structural.h5
│   ├── 70x1x50_Speed.h5
│   └── 70x1x50_Structural.h5
```

### Simulation Results
```
simulation_data/
├── wall_30mm_speed_20mms_[timestamp]/
│   ├── simulation_config.yaml
│   ├── simulation_results.csv
│   ├── temperature_fields.npy
│   └── monitoring_points/
└── validation/
    ├── experiment_vs_simulation.png
    └── validation_metrics.json
```

## Validation and Comparison

### Key Validation Metrics
- **Temperature Evolution**: Time-series comparison at monitoring points
- **Spatial Distribution**: 2D temperature field validation
- **Cooling Rates**: Exponential decay analysis
- **Peak Temperatures**: Maximum temperature validation

### Statistical Analysis
- Root Mean Square Error (RMSE) between experiment and simulation
- Correlation coefficients for temperature evolution
- Spatial temperature gradient comparison

## Installation and Dependencies

### Core Requirements
```txt
numpy>=1.21.0          # Numerical computations
matplotlib>=3.5.0      # Visualization
pandas>=1.3.0          # Data handling
h5py>=3.7.0           # HDF5 file handling
pyyaml>=6.0           # Configuration management
scipy>=1.7.0          # Scientific computing
opencv-python>=4.5.0  # Image processing
```

### Hardware Requirements
- MLX90640 thermal camera
- ESP32 development board
- 3D printer with accessible hotend
- USB connections for data acquisition


## Troubleshooting

### Experimental Issues
- **Camera Calibration**: Ensure 180s warm-up time
- **Data Quality**: Check for broken pixels and temperature outliers
- **Frame Synchronization**: Verify crop ranges for active printing

### Simulation Issues
- **Convergence**: Check time step stability limits
- **Memory Usage**: Reduce mesh resolution for large problems
- **Physics Validation**: Compare material properties with literature

## Contact

**Institution**: Lehrstuhl für Computational Physics in Engineering (CPE)  
**University**: RPTU Kaiserslautern  
**Address**: Gottlieb-Daimler-Straße, Gebäude 74, 67663 Kaiserslautern, Germany  
**Email**: dalim@rptu.de

### Contributors
- **Students**: Yating Wei, Hakim Dalim, Yahya Badine
- **Lab Assistants**: Iram, Harshad  
- **Supervisor**: Prof. Dr.-Ing. Kristin de Payrebrune

## License

This work is developed for academic research purposes as part of the comprehensive analysis of temperature distribution in 3D printed walls. Both experimental protocols and simulation framework are provided for educational and research use.

---

**Note**: This integrated framework provides both experimental validation and computational modeling capabilities for comprehensive thermal analysis in additive manufacturing processes.
