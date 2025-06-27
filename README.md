# FFF Thermal Distribution Analysis

A comprehensive system for analyzing temperature distribution in Fused Filament Fabrication (FFF) 3D printing through thermal imaging and computational simulation.

## Overview

This project compares thermal camera measurements from an MLX90640 sensor with computational finite element simulations to analyze temperature distribution in 3D printed walls. The system integrates hardware data acquisition with GPU-accelerated simulation models to provide insights into thermal behavior during the FFF process.

## Project Specifications

### Test Parameters

- **Wall Dimensions**: 30mm x 50mm x 1mm and 30mm x 70mm x 1mm
- **Print Speeds**: 20, 30, 50 mm/s
- **Material**: PLA with thermal properties:
    - Thermal conductivity: 0.13 W/mK
    - Density: 1250 kg/m³
    - Specific heat capacity: 1800 J/kgK

### Hardware Components

- Raspberry Pi with MLX90640 thermal camera
- 3D printed camera mount for side-view measurements
- FFF 3D printer for test specimen fabrication

## Repository Structure

```
fff-thermal-analysis/
├── src/                          # Source code
│   ├── simulation/               # Thermal simulation modules
│   ├── hardware/                 # Hardware interface modules
│   ├── analysis/                 # Data analysis and comparison
│   └── utils/                    # Utility functions
├── hardware/                     # Hardware design files
│   ├── camera_mount/             # CAD files for camera mount
│   └── raspberry_pi/             # Pi setup and configuration
├── data/                         # Data storage
│   ├── gcode/                    # G-code files
│   ├── measurements/             # Thermal camera data
│   └── simulation_results/       # Simulation outputs
├── simulation/                   # Simulation models and configs
├── analysis/                     # Analysis scripts and results
├── docs/                         # Documentation
├── tests/                        # Unit tests and validation
├── scripts/                      # Utility and automation scripts
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Installation

### Prerequisites

**System Requirements:**

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Raspberry Pi 4 with Raspbian OS (for hardware interface)

**Hardware Setup:**

- MLX90640 thermal camera connected to Raspberry Pi I2C
- 3D printer with accessible print area for camera mounting (Prusa MK4)

### Software Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/fff-thermal-analysis.git
cd fff-thermal-analysis
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

or conda
conda activate env
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Install package in development mode:**

```bash
pip install -e .
```

5. **GPU acceleration (optional):**

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### Raspberry Pi Setup

1. **Enable I2C interface:**

```bash
sudo raspi-config
# Navigate to Interface Options > I2C > Enable
```

2. **Install MLX90640 library:**

```bash
pip install adafruit-circuitpython-mlx90640
```

3. **Test camera connection:**

```bash
python scripts/test_camera.py
```

## Quick Start

### 1. Thermal Camera Data Acquisition

```python
from src.hardware.thermal_camera import MLX90640Interface

# Initialize camera
camera = MLX90640Interface()
camera.calibrate() #using thermocouple

# Start measurement session
session = camera.start_measurement_session(
    duration=300,  # 5 minutes
    output_dir="data/measurements/test_session_001"
)
```

### 2. G-code Mesh Generation

```python
from src.simulation.mesh_generator import GCodeMeshGenerator

# Generate finite element mesh from G-code
generator = GCodeMeshGenerator(resolution=0.1, use_gpu=True)
generator.read_gcode("data/gcode/wall_30x50_20mms.gcode")
mesh_data = generator.generate_mesh(wall_thickness=1.0)
```

### 3. Thermal Simulation

```python
from src.simulation.thermal_solver import ThermalSimulation

# Run thermal simulation
sim = ThermalSimulation(mesh_data)
sim.set_boundary_conditions(ambient_temp=25, nozzle_temp=210)
results = sim.run_simulation(duration=300, time_step=0.1)
```

### 4. Data Analysis and Comparison

```python
from src.analysis.comparison import ThermalComparison

# Compare simulation with measurements
comparison = ThermalComparison()
comparison.load_measurement_data("data/measurements/test_session_001")
comparison.load_simulation_results("data/simulation_results/test_sim_001")
comparison.generate_comparison_report("analysis/results/comparison_001.html")
```

## Usage

### Camera Mount Installation

1. Print camera mount using files in `hardware/camera_mount/`
2. Attach mount to 3D printer hotend assembly
3. Install MLX90640 camera in mount with clear view of print area
4. Connect camera to Raspberry Pi via I2C interface

### Running Complete Analysis

Use the automation script for full workflow:

```bash
python scripts/run_complete_analysis.py \
    --gcode data/gcode/wall_30x50_20mms.gcode \
    --measurement-session data/measurements/session_001 \
    --output-dir analysis/results/complete_001 \
    --gpu-acceleration
```

### Configuration

Edit `config/simulation_config.yaml` to adjust:

- Mesh resolution parameters
- Material properties
- Boundary conditions
- Output formats

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_simulation.py
python -m pytest tests/test_hardware.py -k "not integration"

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Validation Tests

Validate simulation accuracy against known benchmarks:

```bash
python scripts/validate_simulation.py
```

## API Reference

### Core Modules

**simulation.mesh_generator.GCodeMeshGenerator**

- `read_gcode(filepath)`: Parse G-code file
- `generate_mesh(wall_thickness)`: Create finite element mesh
- `export_simulation_data(output_dir)`: Export mesh for simulation

**hardware.thermal_camera.MLX90640Interface**

- `calibrate()`: Perform camera calibration
- `start_measurement_session()`: Begin data acquisition
- `get_temperature_matrix()`: Retrieve current temperature data

**simulation.thermal_solver.ThermalSimulation**

- `set_boundary_conditions()`: Define thermal boundaries
- `run_simulation()`: Execute time-stepping simulation
- `export_results()`: Save simulation output data

### Configuration Files

**config/simulation_config.yaml**: Simulation parameters **config/hardware_config.yaml**: Hardware interface settings **config/analysis_config.yaml**: Data analysis options

## Data Formats

### Measurement Data

- **Format**: hdf5 with timestamp, temperature matrix, metadata
- **Location**: `data/measurements/session_id/`
- **Filename**: `thermal_data_YYYYMMDD_HHMMSS.csv`

### Simulation Results

- **Format**: JSON with mesh data, temperature fields, activation sequence
- **Location**: `data/simulation_results/sim_id/`
- **Files**: `mesh.json`, `temperature_field.json`, `metadata.json`

### G-code Files

- **Format**: Standard G-code with layer information
- **Location**: `data/gcode/`
- **Naming**: `wall_WxHxT_SPEEDmms.gcode`

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-analysis-method`
3. Make changes and add tests
4. Run test suite: `python -m pytest`
5. Submit pull request with detailed description

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Document all public functions with docstrings
- Maintain test coverage above 80%

## Troubleshooting

### Common Issues

**GPU acceleration not working:**

- Verify CUDA installation: `nvidia-smi`
- Check CuPy installation: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`

**MLX90640 camera not detected:**

- Check I2C connection: `sudo i2cdetect -y 1`
- Verify camera address (should be 0x33)
- Ensure adequate power supply for Raspberry Pi

**Simulation convergence issues:**

- Reduce time step size in configuration
- Check mesh quality with validation script
- Verify boundary condition settings

**Memory issues with large meshes:**

- Reduce mesh resolution parameter
- Enable GPU acceleration for larger capacity
- Use batch processing for multiple simulations

### Performance Optimization

**For large-scale simulations:**

- Use GPU acceleration when available
- Optimize mesh resolution for balance of accuracy/speed
- Enable parallel processing for multiple test cases

**For real-time measurements:**

- Adjust camera sampling rate based on print speed
- Use buffered data acquisition to prevent loss
- Configure appropriate storage capacity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in academic research, please cite:

```bibtex
@software{fff_thermal_analysis,
  title={FFF Thermal Distribution Analysis},
  author={[Yating Wei], [Yahya Badine], [Hakim Dalim]},
  institution={RPTU Kaiserslautern-Landau},
  year={2025},
  url={https://github.com/hakimdalim/fff-thermal-analysis}
}
```

## Acknowledgments

- RPTU Kaiserslautern-Landau for research support
- MLX90640 thermal camera documentation and community
- Open-source finite element method libraries

## Contact

- **Author**: [Hakim Dalim]
- **Institution**: RPTU Kaiserslautern-Landau
- **Email**: [dalim@rptu.de]
- **Project Issues**: [GitHub Issues](https://github.com/your-username/fff-thermal-analysis/issues)

## Version History

- **v1.0.0** - Initial release with basic simulation and measurement capabilities
- **v0.9.0** - Beta release with GPU acceleration support
- **v0.8.0** - Alpha release with hardware interface implementation
