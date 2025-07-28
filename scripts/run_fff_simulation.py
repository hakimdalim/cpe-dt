"""
Main FFF Temperature Simulation Runner

This script runs the complete FFF temperature simulation according to task requirements:
- 2mm thick walls of 30mm and 70mm length
- Different print speeds (20 mm/s and 40 mm/s)  
- Temperature measurement at different layer heights
- Results for comparison with thermal camera measurements

Usage:
    python run_fff_simulation.py --wall_length 30 --print_speed 20
    python run_fff_simulation.py --wall_length 70 --print_speed 40
"""

import argparse
import yaml
import os
import time
import json
from datetime import datetime

from simulator import FFFilamentFabricationSimulator  # Corrected import


def create_simulation_config(wall_length: float, print_speed: float, 
                           output_dir: str = None) -> str:
    """Create configuration file for specific simulation parameters"""
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/output/wall_{wall_length}mm_speed_{print_speed}mms_{timestamp}"
    
    # Base configuration
    config = {
        'simulation': {
            'wall_length': float(wall_length),
            'wall_thickness': 1.0,
            'layer_height': 0.2,
            'print_speed': float(print_speed),
            'total_layers': int(50)  # 10mm height / 0.2mm per layer
        },
        'material': {
            'name': "PLA",
            'thermal_conductivity': 0.13,
            'density': 1250,
            'specific_heat': 1800,
            'deposition_temp': 175.0,  # From experimental data
            'ambient_temp': 25.0
        },
        'mesh': {
            'element_size_x': 0.5,  # 0.5mm elements along length for better resolution
            'element_size_z': 0.2,  # Match layer height
            'dimension': "2D"
        },
        'solver': {
            'time_step': 0.001,  # Small time step for stability
            'max_time': 200.0,   # Allow enough time for complete print
            'save_interval': 0.1,
            'thermal_method': "explicit",  # More stable for finite element
            'theta': 0 #1.0  # Fully implicit
        },
        'output': {
            'results_dir': output_dir,
            'save_mesh': True,
            'save_temperature_history': True,
            'save_heat_flux': True,
            'monitoring_points': [
                [5.0, 1.0],   # Near start, low layer
                [wall_length/2, 2.0],  # Middle, medium layer
                [wall_length-5.0, 5.0],  # Near end, high layer
                [wall_length/4, 0.2],    # Quarter point, first layer
                [3*wall_length/4, 4.0]   # Three-quarter point, high layer
            ],
            'thermal_snapshots': {
                'interval': 1.0,
                'format': "numpy"
            }
        },
        'boundary_conditions': {
            'ambient_temperature': 25.0,
            'convection_coefficient': 10.0,
            'build_plate_temp': 60.0,
            'plate_contact_coeff': 100.0
        },
        'printing': {
            'nozzle_start_x': 0.0,
            'nozzle_end_x': float(wall_length),
            'print_direction': "alternating"  # Realistic printing pattern
        }
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration file
    config_file = os.path.join(output_dir, 'simulation_config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration created: {config_file}")
    return config_file


def run_simulation(wall_length: float, print_speed: float, 
                  output_dir: str = None, verbose: bool = True) -> dict:
    """
    Run complete FFF temperature simulation
    
    Args:
        wall_length: Wall length in mm (30 or 70)
        print_speed: Print speed in mm/s
        output_dir: Output directory (auto-generated if None)
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with simulation results and file paths
    """
    
    if verbose:
        print(f"\nFFF Temperature Simulation")
        print(f"=" * 50)
        print(f"Wall: {wall_length}mm × 2mm × 10mm")
        print(f"Print speed: {print_speed} mm/s")
        print(f"Material: PLA (deposition temp: 175°C)")
    
    start_time = time.time()
    
    # Create configuration
    config_file = create_simulation_config(wall_length, print_speed, output_dir)
    
    try:
        # Initialize and run simulation
        if verbose:
            print(f"\nInitializing simulation...")
        
        simulator = FFFilamentFabricationSimulator(config_file)
        
        if verbose:
            print(f"Simulation setup:")
            print(f"  Elements: {len(simulator.mesh.elements)}")
            print(f"  Nodes: {len(simulator.mesh.nodes)}")
            print(f"  Monitoring points: {len(simulator.monitoring_points)}")
            print(f"  Expected print time: {simulator.activator.toolpath[-1][0]:.1f}s")
        
        # Run simulation
        if verbose:
            print(f"\nRunning simulation...")
            
        results = simulator.run_simulation()
        
        # Calculate additional metrics
        end_time = time.time()
        wall_clock_time = end_time - start_time
        
        # Save simulation metadata
        metadata = {
            'simulation_parameters': {
                'wall_length_mm': wall_length,
                'wall_thickness_mm': 2.0,
                'wall_height_mm': 10.0,
                'print_speed_mm_per_s': print_speed,
                'deposition_temp_celsius': 175.0,
                'ambient_temp_celsius': 25.0
            },
            'mesh_info': {
                'total_elements': len(simulator.mesh.elements),
                'total_nodes': len(simulator.mesh.nodes),
                'element_size_x_mm': simulator.mesh.dx,
                'element_size_z_mm': simulator.mesh.dz
            },
            'simulation_results': results,
            'timing': {
                'wall_clock_time_seconds': wall_clock_time,
                'simulated_time_seconds': results['final_time'],
                'speed_ratio': results['final_time'] / wall_clock_time if wall_clock_time > 0 else 0
            },
            'files': {
                'config_file': config_file,
                'results_directory': results['results_directory']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_file = os.path.join(results['results_directory'], 'simulation_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            print(f"\n" + "=" * 50)
            print(f"✓ Simulation completed successfully!")
            print(f"  Simulated time: {results['final_time']:.1f}s")
            print(f"  Wall clock time: {wall_clock_time:.1f}s")
            print(f"  Speed ratio: {results['final_time']/wall_clock_time:.1f}x" if wall_clock_time > 0 else "")
            print(f"  Active elements: {results['active_elements']}/{results['total_elements']}")
            print(f"  Completion: {results['completion_percentage']:.1f}%")
            print(f"  Results saved in: {results['results_directory']}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_parameter_study():
    """Run parameter study for task requirements"""
    print(f"\nFFF Temperature Analysis - Parameter Study")
    print(f"=" * 60)
    print(f"According to task requirements:")
    print(f"- 2mm thick walls")
    print(f"- Wall lengths: 30mm and 70mm") 
    print(f"- Print speeds: 20 mm/s and 40 mm/s")
    print(f"- Temperature measurement at different layer heights")
    
    # Parameter combinations to test
    parameters = [
        (30, 45),  # 30mm wall, 45 mm/s
        (30, 170),  # 30mm wall, 170 mm/s  
        (70, 45),  # 70mm wall, 45 mm/s
        (70, 170),  # 70mm wall, 170 mm/s
    ]
    
    results = {}
    
    for i, (wall_length, print_speed) in enumerate(parameters):
        print(f"\n--- Simulation {i+1}/4: {wall_length}mm wall @ {print_speed} mm/s ---")
        
        # Create specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/parameter_study/wall_{wall_length}mm_speed_{print_speed}mms_{timestamp}"
        
        # Run simulation
        result = run_simulation(wall_length, print_speed, output_dir, verbose=True)
        
        if result:
            results[f"wall_{wall_length}mm_speed_{print_speed}mms"] = result
            print(f"✓ Simulation {i+1} completed")
        else:
            print(f"❌ Simulation {i+1} failed")
    
    # Save parameter study summary
    study_dir = "data/parameter_study"
    os.makedirs(study_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(study_dir, f"parameter_study_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"Parameter Study Summary")
    print(f"=" * 60)
    
    for name, result in results.items():
        if result:
            sim_results = result['simulation_results']
            timing = result['timing']
            print(f"{name:25s}: {sim_results['completion_percentage']:5.1f}% complete, "
                  f"{timing['wall_clock_time_seconds']:6.1f}s runtime")
    
    print(f"\nDetailed results saved in: {summary_file}")
    print(f"\nReady for comparison with thermal camera measurements!")
    
    return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='FFF Temperature Simulation')
    parser.add_argument('--wall_length', type=float, choices=[30, 70], 
                       help='Wall length in mm (30 or 70)')
    parser.add_argument('--print_speed', type=float, choices=[45, 170],
                       help='Print speed in mm/s (20 or 40)')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--parameter_study', action='store_true',
                       help='Run complete parameter study (all combinations)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if args.parameter_study:
        # Run complete parameter study
        results = run_parameter_study()
        
    elif args.wall_length and args.print_speed:
        # Run single simulation
        result = run_simulation(
            wall_length=args.wall_length,
            print_speed=args.print_speed, 
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        if result:
            print(f"\n✓ Simulation completed successfully!")
        else:
            print(f"\n❌ Simulation failed!")
            return 1
            
    else:
        # Show usage
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python run_fff_simulation.py --wall_length 30 --print_speed 45")
        print(f"  python run_fff_simulation.py --wall_length 70 --print_speed 170")  
        print(f"  python run_fff_simulation.py --parameter_study")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())