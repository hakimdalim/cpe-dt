# Measurement Data
This directory will contain measurement data collected during the project.
# PLA 3D Printing Thermal Dataset

## Overview
This dataset contains thermal imaging data collected during PLA (Polylactic Acid) 3D printing processes using an MLX90640 thermal camera. The data captures temperature distribution patterns on printed walls under different printing configurations.

## Dataset Structure

### Files Description
| File Name | Print Configuration | Description |
|-----------|-------------------|-------------|
| `30x1x50_Speed.h5` | Speed optimized | 30mm width × 1mm layer × 50mm height wall |
| `30x1x50_Structural.h5` | Structural quality | 30mm width × 1mm layer × 50mm height wall |
| `70x1x50_Speed.h5` | Speed optimized | 70mm width × 1mm layer × 50mm height wall |
| `70x1x50_Structural.h5` | Structural quality | 70mm width × 1mm layer × 50mm height wall |


## Data Production Process

### 1. Equipment Setup
- **Thermal Camera**: MLX90640 (24×32 pixel thermal sensor array)
- **Microcontroller**: ESP32 for sensor interfacing
- **3D Printer**: Standard FDM printer with PLA filament
- **Data Interface**: USB serial communication at 921600 baud

### 2. Calibration Process
1. **Thermocouple Calibration**: Reference sensor calibrated using incubator at multiple temperature points (25°C, 48°C, 60°C)
2. **Thermal Camera Calibration**: MLX90640 calibrated against reference thermocouple using linear regression analysis
3. **Correction Equation**: Applied as `T_calibrated = a × T_camera + b`

### 3. Data Collection Protocol

#### Pre-collection Setup
- Camera positioned 15-20 cm from build plate for optimal field-of-view
- 180-second warm-up period for MLX90640 sensor stabilization
- Python visualization script initialized for real-time monitoring

#### Collection Parameters
- **Frame Rate**: Continuous acquisition during printing
- **Temperature Range**: 25-230°C (filtered during preprocessing)
- **Spatial Resolution**: 24×32 thermal pixels
- **Data Format**: 32-bit floating-point temperature values

### 4. Print Configurations

#### Speed Optimized Settings
- Higher print speeds for rapid prototyping
- Reduced layer adhesion time
- Optimized for time efficiency

#### Structural Quality Settings  
- Lower print speeds for enhanced layer bonding
- Increased thermal stabilization between layers
- Optimized for mechanical properties

### 5. Data Preprocessing

#### Quality Control Measures
- **Broken Pixel Filtering**: Temperatures >180°C replaced with NaN
- **Frame Validation**: Structural integrity checks on HDF5 files
- **Temporal Cropping**: Active printing sessions isolated using recommended ranges
- **Outlier Removal**: Temperature values outside 25-230°C range filtered

#### Data Structure
- **Raw Format**: HDF5 files containing thermal image sequences
- **Dimensions**: Time × Height × Width (frames × 24 × 32 pixels)
- **Data Type**: Float32 temperature values in Celsius
- **Metadata**: Timestamps, sensor parameters, print settings

## Usage Guidelines

### Recommended Analysis Workflow
1. Load HDF5 file using appropriate library (h5py for Python)
2. Apply temporal cropping using recommended frame ranges
3. Filter temperature data (25-230°C range)
4. Handle NaN values from broken pixel compensation
5. Apply thermal camera calibration correction if needed

### Frame Selection
- **Start Frame**: Begin analysis after initial warm-up period
- **End Frame**: Conclude before print completion artifacts
- **Active Duration**: Use cropped ranges for consistent analysis
- **Total Frames**: Raw data includes pre/post printing periods

### Data Considerations
- **Thermal Drift**: Account for sensor warm-up in first 180 seconds
- **Ambient Effects**: Laboratory temperature variations may affect baseline
- **Spatial Resolution**: 24×32 pixels covers printing area with adequate detail
- **Temporal Resolution**: Frame rate optimized for thermal process timescales

## Technical Specifications

### Hardware Configuration
- **Sensor Type**: MLX90640 far-infrared thermal sensor
- **Interface**: I2C communication protocol
- **Processing**: ESP32 microcontroller with USB serial output
- **Software**: Python-based data acquisition and visualization

### File Format Details
- **Container**: HDF5 hierarchical data format
- **Compression**: Optional compression for storage efficiency
- **Metadata**: Embedded acquisition parameters and timestamps
- **Compatibility**: Cross-platform readable with standard HDF5 libraries

## Citation
When using this dataset, please reference the associated publication and acknowledge the CPE Laboratory at RPTU Kaiserslautern.

**Laboratory**: Lehrstuhl für Computational Physics in Engineering (CPE)  
**Institution**: RPTU Kaiserslautern  
**Contributors**: Yating Wei, Hakim Dalim, Yahya Badine  
**Supervision**: Prof. Dr.-Ing. Kristin de Payrebrune

## Contact
For questions regarding this dataset or access to additional data, contact:
- **Email**: cpe-sekretariat@mv.rptu.de
- **Address**: Gottlieb-Daimler-Straße, Gebäude 74, 67663 Kaiserslautern, Germany