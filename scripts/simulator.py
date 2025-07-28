"""
Main FFF Temperature Simulation Controller

This module coordinates:
1. Time stepping loop
2. Sequential element activation 
3. Heat transfer calculations
4. Results saving and output
5. Overall simulation management
"""

import numpy as np
import yaml
import os
import time as time_module
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

from mesh import Mesh
from activation import SequentialActivator
from thermal import FiniteElementThermalSolver


@dataclass
class SimulationState:
    """Current state of the simulation"""
    current_time: float = 0.0           # seconds
    time_step: float = 0.01             # seconds
    iteration: int = 0                  # current iteration number
    active_elements: int = 0            # number of active elements
    total_elements: int = 0             # total elements in mesh
    is_complete: bool = False           # whether simulation is finished
    wall_start_time: float = 0.0        # when this wall started printing


@dataclass
class MonitoringPoint:
    """Point where we track temperature over time"""
    name: str
    x: float                            # mm
    z: float                            # mm  
    element_id: Optional[int] = None    # which element contains this point
    temperature_history: List[float] = None   # temperature vs time
    time_history: List[float] = None           # time points
    
    def __post_init__(self):
        if self.temperature_history is None:
            self.temperature_history = []
        if self.time_history is None:
            self.time_history = []


class FFFilamentFabricationSimulator:
    """
    Main FFF Temperature Simulation Controller
    
    This is the conductor that orchestrates all components:
    - Mesh: Manages elements and temperatures
    - Activator: Follows nozzle path and activates elements  
    - Thermal: Calculates heat transfer (to be implemented)
    - Results: Saves and visualizes data
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize simulation with configuration"""
        self.config_file = config_file
        self.load_config()
        
        # Initialize components
        print("Initializing simulation components...")
        self.mesh = Mesh(config_file)
        self.activator = SequentialActivator(self.mesh, config_file)
        
        # Initialize thermal solver with error handling
        try:
            self.thermal_solver = SimplifiedThermalSolver(self.mesh, config_file)
            self.thermal_enabled = True
            print("  ✓ Thermal solver initialized")
        except Exception as e:
            print(f"  ⚠ Thermal solver initialization failed: {e}")
            print("  → Using simplified thermal model")
            self.thermal_solver = None
            self.thermal_enabled = False
        
        # Simulation state
        self.state = SimulationState(
            time_step=self.time_step,
            total_elements=len(self.mesh.elements)
        )
        
        # Monitoring points for detailed temperature tracking
        self.monitoring_points = self._setup_monitoring_points()
        
        # Results storage
        self.results = {
            'time': [],
            'active_elements': [],
            'nozzle_x': [],
            'nozzle_z': [],
            'layer': [],
            'progress': []
        }
        
        # Temperature fields (saved at intervals)
        self.temperature_snapshots = []
        self.snapshot_times = []
        
        print(f"Simulation initialized:")
        print(f"  Wall: {self.mesh.wall_length}mm × {self.mesh.wall_height}mm")
        print(f"  Elements: {self.state.total_elements}")
        print(f"  Time step: {self.state.time_step}s")
        print(f"  Save interval: {self.save_interval}s")
        print(f"  Monitoring points: {len(self.monitoring_points)}")
        
    def load_config(self):
        """Load simulation configuration"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Time stepping
        solver_config = config.get('solver', {})
        self.time_step = solver_config.get('time_step', 0.01)
        self.max_time = solver_config.get('max_time', 100.0)  # Maximum simulation time
        
        # Output configuration
        output_config = config.get('output', {})
        self.save_interval = output_config.get('save_interval', 0.1)
        self.results_dir = output_config.get('results_dir', 'data/output')
        self.save_temperature_history = output_config.get('save_temperature_history', True)
        
        # Material properties
        material_config = config.get('material', {})
        self.ambient_temp = material_config.get('ambient_temp', 25.0)
        
        # Create output directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _setup_monitoring_points(self) -> List[MonitoringPoint]:
        """Setup monitoring points from configuration"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        monitoring_config = config.get('output', {}).get('monitoring_points', [])
        points = []
        
        for i, (x, z) in enumerate(monitoring_config):
            point = MonitoringPoint(
                name=f"Point_{i+1}",
                x=float(x),
                z=float(z)
            )
            
            # Find which element contains this point
            element_id = self.mesh.get_element_at_position(x, z)
            point.element_id = element_id
            
            if element_id is not None:
                print(f"Monitoring point {point.name} at ({x}, {z}) → Element {element_id}")
            else:
                print(f"Warning: Monitoring point {point.name} at ({x}, {z}) is outside mesh")
                
            points.append(point)
            
        return points
        
    def run_simulation(self) -> Dict:
        """
        Run the complete FFF temperature simulation
        
        Returns:
            Dictionary with simulation results and statistics
        """
        print(f"\nStarting FFF Temperature Simulation")
        print(f"{'='*60}")
        
        start_wall_time = time_module.time()
        self.state.wall_start_time = start_wall_time
        
        last_save_time = 0.0
        last_progress_time = 0.0
        progress_interval = 1.0  # seconds between progress updates
        
        try:
            # Main simulation loop
            while (self.state.current_time < self.max_time and 
                   not self.state.is_complete):
                
                # Step 1: Update element activation
                newly_activated = self.activator.update(self.state.current_time)
                
                # Step 2: Update thermal physics (simplified for now)
                self._update_thermal_physics()
                
                # Step 3: Update monitoring points
                self._update_monitoring_points()
                
                # Step 4: Save results at intervals
                if self.state.current_time - last_save_time >= self.save_interval:
                    self._save_timestep_results()
                    last_save_time = self.state.current_time
                    
                # Step 5: Check completion
                if self.activator.is_printing_complete(self.state.current_time):
                    self.state.is_complete = True
                    
                # Step 6: Progress reporting
                if self.state.current_time - last_progress_time >= progress_interval:
                    self._report_progress()
                    last_progress_time = self.state.current_time
                    
                # Step 7: Advance time
                self.state.current_time += self.state.time_step
                self.state.iteration += 1
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nSimulation error: {e}")
            raise
            
        # Finalize simulation
        end_wall_time = time_module.time()
        simulation_wall_time = end_wall_time - start_wall_time
        
        print(f"\nSimulation completed!")
        print(f"  Simulated time: {self.state.current_time:.2f} seconds")
        print(f"  Wall clock time: {simulation_wall_time:.2f} seconds")
        print(f"  Iterations: {self.state.iteration}")
        print(f"  Final progress: {self.activator.get_print_progress(self.state.current_time)*100:.1f}%")
        
        # Save final results
        results = self._finalize_results()
        
        return results
        
    def _update_thermal_physics(self):
        """
        Update heat transfer calculations
        """
        if self.thermal_enabled and self.thermal_solver:
            # Use the thermal solver for proper heat transfer physics
            self.thermal_solver.solve_timestep(self.state.time_step)
        else:
            # Fallback to simple cooling model
            self._simple_thermal_update()
            
    def _simple_thermal_update(self):
        """
        Simple thermal update as fallback
        """
        # Simple cooling model for demonstration
        cooling_rate = 0.1  # °C/s cooling rate
        dt = self.state.time_step
        
        for element in self.mesh.elements:
            if element.is_active:
                # Get all nodes in this element
                for node_id in element.node_ids:
                    node = self.mesh.nodes[node_id]
                    
                    # Simple cooling toward ambient temperature
                    temp_diff = node.temperature - self.ambient_temp
                    cooling = cooling_rate * temp_diff * dt
                    node.temperature = max(self.ambient_temp, 
                                         node.temperature - cooling)
                                         
    def _update_monitoring_points(self):
        """Update temperature history at monitoring points"""
        for point in self.monitoring_points:
            if point.element_id is not None and point.element_id < len(self.mesh.elements):
                element = self.mesh.elements[point.element_id]
                
                if element.is_active:
                    # Get average temperature of element nodes
                    temps = [self.mesh.nodes[nid].temperature for nid in element.node_ids]
                    avg_temp = np.mean(temps)
                else:
                    avg_temp = self.ambient_temp
                    
                point.temperature_history.append(avg_temp)
                point.time_history.append(self.state.current_time)
                
    def _save_timestep_results(self):
        """Save results for current timestep"""
        # Update state
        self.state.active_elements = len(self.mesh.get_active_elements())
        
        # Get nozzle position
        nozzle_x, nozzle_z = self.activator.get_nozzle_position(self.state.current_time)
        current_layer = int(nozzle_z / self.mesh.layer_height)
        progress = self.activator.get_print_progress(self.state.current_time)
        
        # Store results
        self.results['time'].append(self.state.current_time)
        self.results['active_elements'].append(self.state.active_elements)
        self.results['nozzle_x'].append(nozzle_x)
        self.results['nozzle_z'].append(nozzle_z)
        self.results['layer'].append(current_layer)
        self.results['progress'].append(progress)
        
        # Save temperature snapshot if requested
        if self.save_temperature_history:
            temp_field = [node.temperature for node in self.mesh.nodes]
            self.temperature_snapshots.append(temp_field)
            self.snapshot_times.append(self.state.current_time)
            
    def _report_progress(self):
        """Report simulation progress"""
        progress = self.activator.get_print_progress(self.state.current_time)
        active_elements = len(self.mesh.get_active_elements())
        nozzle_x, nozzle_z = self.activator.get_nozzle_position(self.state.current_time)
        
        print(f"t={self.state.current_time:6.2f}s | "
              f"Progress: {progress*100:5.1f}% | "
              f"Active: {active_elements:4d}/{self.state.total_elements} | "
              f"Nozzle: ({nozzle_x:5.1f}, {nozzle_z:4.1f})")
              
    def _finalize_results(self) -> Dict:
        """Finalize and save all simulation results"""
        print("\nSaving simulation results...")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Save main results
        results_file = os.path.join(self.results_dir, 'simulation_results.csv')
        df_results.to_csv(results_file, index=False)
        print(f"Main results saved to: {results_file}")
        
        # Save monitoring point data
        for point in self.monitoring_points:
            if point.temperature_history:
                df_point = pd.DataFrame({
                    'time': point.time_history,
                    'temperature': point.temperature_history
                })
                point_file = os.path.join(self.results_dir, f'{point.name}_temperature.csv')
                df_point.to_csv(point_file, index=False)
                print(f"Monitoring point {point.name} saved to: {point_file}")
                
        # Save temperature field snapshots
        if self.temperature_snapshots:
            temp_array = np.array(self.temperature_snapshots)
            temp_file = os.path.join(self.results_dir, 'temperature_fields.npy')
            np.save(temp_file, temp_array)
            
            times_file = os.path.join(self.results_dir, 'snapshot_times.npy')  
            np.save(times_file, np.array(self.snapshot_times))
            print(f"Temperature fields saved to: {temp_file}")
            
        # Create summary plots
        self._create_summary_plots()
        
        # Return summary statistics
        return {
            'final_time': self.state.current_time,
            'total_iterations': self.state.iteration,
            'active_elements': self.state.active_elements,
            'total_elements': self.state.total_elements,
            'completion_percentage': self.activator.get_print_progress(self.state.current_time) * 100,
            'results_directory': self.results_dir
        }
        
    def _create_summary_plots(self):
        """Create summary visualization plots"""
        print("Creating summary plots...")
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Activation progress over time
        ax1 = axes[0, 0]
        times = self.results['time']
        active_elements = self.results['active_elements']
        progress = [x * 100 for x in self.results['progress']]
        
        ax1.plot(times, progress, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Print Progress (%)')
        ax1.set_title('Printing Progress Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Nozzle path
        ax2 = axes[0, 1]
        nozzle_x = self.results['nozzle_x']
        nozzle_z = self.results['nozzle_z']
        
        ax2.plot(nozzle_x, nozzle_z, 'r-', linewidth=1, alpha=0.7)
        ax2.set_xlabel('X Position (mm)')
        ax2.set_ylabel('Z Position (mm)')
        ax2.set_title('Nozzle Toolpath')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Plot 3: Temperature histories at monitoring points
        ax3 = axes[1, 0]
        for point in self.monitoring_points:
            if point.temperature_history:
                ax3.plot(point.time_history, point.temperature_history, 
                        linewidth=2, label=f'{point.name} ({point.x}, {point.z})')
                        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Temperature at Monitoring Points')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Final mesh state
        ax4 = axes[1, 1]
        self.mesh.visualize_mesh()
        plt.sca(ax4)  # Switch to this axes for mesh plot
        
        plt.tight_layout()
        
        # Save figure
        plot_file = os.path.join(self.results_dir, 'simulation_summary.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Summary plots saved to: {plot_file}")
        
        plt.show()
        
    def visualize_temperature_evolution(self, time_points: Optional[List[float]] = None):
        """Create animation of temperature evolution"""
        if not self.temperature_snapshots:
            print("No temperature snapshots available for visualization")
            return
            
        if time_points is None:
            # Select evenly spaced time points
            n_frames = min(10, len(self.snapshot_times))
            indices = np.linspace(0, len(self.snapshot_times)-1, n_frames, dtype=int)
            time_points = [self.snapshot_times[i] for i in indices]
            
        # Create subplots for selected time points
        n_cols = min(4, len(time_points))
        n_rows = (len(time_points) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
            
        for i, t in enumerate(time_points):
            # Find closest snapshot
            time_idx = np.argmin(np.abs(np.array(self.snapshot_times) - t))
            temp_field = self.temperature_snapshots[time_idx]
            actual_time = self.snapshot_times[time_idx]
            
            # Reshape temperature field to mesh grid
            temps = np.array(temp_field).reshape(self.mesh.nz_nodes, self.mesh.nx_nodes)
            
            # Create contour plot
            ax = axes[i] if len(time_points) > 1 else axes
            
            x_coords = np.linspace(0, self.mesh.wall_length, self.mesh.nx_nodes)
            z_coords = np.linspace(0, self.mesh.wall_height, self.mesh.nz_nodes)
            X, Z = np.meshgrid(x_coords, z_coords)
            
            contour = ax.contourf(X, Z, temps, levels=20, cmap='hot')
            ax.set_xlabel('Length (mm)')
            ax.set_ylabel('Height (mm)')
            ax.set_title(f'T = {actual_time:.1f}s')
            ax.set_aspect('equal')
            
            # Add colorbar
            plt.colorbar(contour, ax=ax, label='Temperature (°C)')
            
        plt.tight_layout()
        
        # Save animation frames
        anim_file = os.path.join(self.results_dir, 'temperature_evolution.png')
        plt.savefig(anim_file, dpi=150, bbox_inches='tight')
        print(f"Temperature evolution saved to: {anim_file}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("FFF Temperature Simulation Test")
    print("=" * 50)
    
    try:
        # Create and run simulation
        simulator = FFFilamentFabricationSimulator("config.yaml")
        
        # Run simulation (short test)
        print("\nRunning simulation test...")
        results = simulator.run_simulation()
        
        # Display results
        print(f"\n=== Simulation Results ===")
        for key, value in results.items():
            print(f"{key}: {value}")
            
        # Create additional visualizations
        print(f"\n=== Creating Visualizations ===")
        simulator.visualize_temperature_evolution()
        
        print(f"\n✓ Simulation completed successfully!")
        print(f"Results saved in: {results['results_directory']}")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()