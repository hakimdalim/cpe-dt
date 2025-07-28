"""
Sequential Element Activation for FFF Temperature Simulation

This module handles:
1. Nozzle toolpath following (left-to-right, layer by layer)
2. Time-based element activation  
3. Application of deposition temperature (175°C)
4. Print speed and timing calculations
"""

import numpy as np
import yaml
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class NozzleState:
    """Current state of the printing nozzle"""
    x: float = 0.0          # mm, current x position
    z: float = 0.0          # mm, current z position (layer height)
    layer: int = 0          # current layer number
    is_printing: bool = False    # whether nozzle is actively depositing
    print_direction: int = 1     # 1 for left-to-right, -1 for right-to-left


@dataclass
class ActivationEvent:
    """Record of when an element was activated"""
    element_id: int
    time: float            # seconds when activated
    x_pos: float          # mm, nozzle x position when activated  
    z_pos: float          # mm, nozzle z position when activated
    temperature: float    # °C, temperature applied


class SequentialActivator:
    """
    Manages sequential element activation following nozzle toolpath
    
    The nozzle follows this pattern:
    Layer 0: (0,0) → (30,0) [left to right]
    Layer 1: (0,0.2) → (30,0.2) [left to right, next layer up]  
    Layer 2: (0,0.4) → (30,0.4) [continue...]
    
    Or alternating pattern:
    Layer 0: (0,0) → (30,0) [left to right]
    Layer 1: (30,0.2) → (0,0.2) [right to left]
    Layer 2: (0,0.4) → (30,0.4) [left to right]
    """
    
    def __init__(self, mesh, config_file: str = "config.yaml"):
        """Initialize with mesh and configuration"""
        self.mesh = mesh
        self.load_config(config_file)
        
        # Nozzle state
        self.nozzle = NozzleState()
        
        # Activation tracking
        self.activation_events: List[ActivationEvent] = []
        self.current_time = 0.0  # seconds
        self.elements_to_activate = []  # Queue of (time, element_id) pairs
        
        # Generate the complete toolpath
        self.generate_toolpath()
        
    def load_config(self, config_file: str):
        """Load configuration parameters"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Printing parameters
        self.print_speed = config['simulation']['print_speed']  # mm/s
        self.layer_height = config['simulation']['layer_height']  # mm
        self.total_layers = config['simulation']['total_layers']
        
        # Geometry
        self.wall_length = config['simulation']['wall_length']  # mm
        
        # Material
        self.deposition_temp = config['material']['deposition_temp']  # °C
        
        # Printing pattern
        printing_config = config.get('printing', {})
        self.print_direction = printing_config.get('print_direction', 'forward')
        self.nozzle_start_x = printing_config.get('nozzle_start_x', 0.0)
        self.nozzle_end_x = printing_config.get('nozzle_end_x', self.wall_length)
        
        # Time step calculation
        solver_config = config.get('solver', {})
        self.time_step = solver_config.get('time_step', 0.01)  # seconds
        
        print(f"Activation Config:")
        print(f"  Print speed: {self.print_speed} mm/s")
        print(f"  Deposition temperature: {self.deposition_temp}°C")
        print(f"  Print direction: {self.print_direction}")
        print(f"  Time step: {self.time_step}s")
        
    def generate_toolpath(self):
        """
        Generate complete nozzle toolpath for entire print
        
        Creates a sequence of (time, x, z) points representing where
        the nozzle should be at each moment in time
        """
        self.toolpath = []  # List of (time, x, z, element_id) tuples
        current_time = 0.0
        
        print(f"Generating toolpath for {self.total_layers} layers...")
        
        for layer in range(self.total_layers):
            z_pos = layer * self.layer_height
            
            # Determine print direction for this layer
            if self.print_direction == 'alternating':
                # Alternate direction each layer
                if layer % 2 == 0:
                    start_x, end_x, direction = self.nozzle_start_x, self.nozzle_end_x, 1
                else:
                    start_x, end_x, direction = self.nozzle_end_x, self.nozzle_start_x, -1
            else:
                # Always same direction
                start_x, end_x, direction = self.nozzle_start_x, self.nozzle_end_x, 1
            
            # Calculate time to traverse this layer
            layer_distance = abs(end_x - start_x)
            layer_time = layer_distance / self.print_speed
            
            # Get elements in this layer
            layer_elements = self.mesh.get_elements_in_layer(layer)
            
            if not layer_elements:
                continue
                
            # Distribute activation times across the layer
            for i, element_id in enumerate(layer_elements):
                element = self.mesh.elements[element_id]
                
                # Calculate when this element should be activated
                # Based on nozzle reaching the element center
                if direction > 0:
                    # Left to right
                    progress = (element.center_x - start_x) / layer_distance
                else:
                    # Right to left  
                    progress = (start_x - element.center_x) / layer_distance
                    
                progress = max(0, min(1, progress))  # Clamp to [0,1]
                activation_time = current_time + progress * layer_time
                
                # Add to toolpath
                self.toolpath.append((
                    activation_time,
                    element.center_x,
                    z_pos,
                    element_id
                ))
            
            # Update time for next layer
            current_time += layer_time
            
            # Add small delay between layers (travel time)
            current_time += 0.1  # 0.1 second layer transition
            
        # Sort toolpath by time
        self.toolpath.sort(key=lambda x: x[0])
        
        print(f"Toolpath generated: {len(self.toolpath)} activation points")
        print(f"Total print time: {self.toolpath[-1][0]:.2f} seconds")
        
    def update(self, current_time: float) -> List[int]:
        """
        Update activation state for given time
        
        Args:
            current_time: Current simulation time in seconds
            
        Returns:
            List of element IDs that were activated at this time step
        """
        self.current_time = current_time
        activated_elements = []
        
        # Find all elements that should be activated at this time
        for activation_time, x_pos, z_pos, element_id in self.toolpath:
            # Check if this element should be activated now
            if (activation_time <= current_time and 
                not self.mesh.elements[element_id].is_active):
                
                # Activate the element
                self.mesh.activate_element(element_id, self.deposition_temp)
                
                # Record activation event
                event = ActivationEvent(
                    element_id=element_id,
                    time=current_time,
                    x_pos=x_pos,
                    z_pos=z_pos,
                    temperature=self.deposition_temp
                )
                self.activation_events.append(event)
                activated_elements.append(element_id)
                
                # Update nozzle state
                self.nozzle.x = x_pos
                self.nozzle.z = z_pos
                self.nozzle.layer = int(z_pos / self.layer_height)
                self.nozzle.is_printing = True
                
        return activated_elements
        
    def get_nozzle_position(self, time: float) -> Tuple[float, float]:
        """
        Get nozzle position at given time
        
        Args:
            time: Time in seconds
            
        Returns:
            (x, z) position of nozzle in mm
        """
        if not self.toolpath:
            return (0.0, 0.0)
            
        # Find closest toolpath point
        for i, (t, x, z, _) in enumerate(self.toolpath):
            if t >= time:
                if i == 0:
                    return (x, z)
                else:
                    # Interpolate between previous and current point
                    t_prev, x_prev, z_prev, _ = self.toolpath[i-1]
                    if t == t_prev:
                        return (x_prev, z_prev)
                    
                    alpha = (time - t_prev) / (t - t_prev)
                    x_interp = x_prev + alpha * (x - x_prev)
                    z_interp = z_prev + alpha * (z - z_prev)
                    return (x_interp, z_interp)
                    
        # Time beyond end of print
        return self.toolpath[-1][1:3]
        
    def is_printing_complete(self, time: float) -> bool:
        """Check if printing is complete at given time"""
        if not self.toolpath:
            return True
        return time >= self.toolpath[-1][0]
        
    def get_print_progress(self, time: float) -> float:
        """
        Get printing progress as percentage
        
        Args:
            time: Current time in seconds
            
        Returns:
            Progress from 0.0 to 1.0
        """
        if not self.toolpath:
            return 1.0
            
        total_time = self.toolpath[-1][0]
        if total_time == 0:
            return 1.0
            
        return min(1.0, time / total_time)
        
    def get_activation_statistics(self) -> Dict:
        """Get statistics about activation process"""
        total_elements = len(self.mesh.elements)
        active_elements = len(self.mesh.get_active_elements())
        
        return {
            'total_elements': total_elements,
            'active_elements': active_elements,
            'activation_events': len(self.activation_events),
            'current_time': self.current_time,
            'nozzle_x': self.nozzle.x,
            'nozzle_z': self.nozzle.z,
            'current_layer': self.nozzle.layer,
            'is_printing': self.nozzle.is_printing,
            'progress_percent': (active_elements / total_elements) * 100
        }
        
    def get_elements_activated_in_timespan(self, start_time: float, end_time: float) -> List[int]:
        """Get elements activated within a time span"""
        activated = []
        for event in self.activation_events:
            if start_time <= event.time <= end_time:
                activated.append(event.element_id)
        return activated
        
    def get_layer_activation_times(self) -> Dict[int, Tuple[float, float]]:
        """
        Get start and end times for each layer
        
        Returns:
            Dict mapping layer_number -> (start_time, end_time)
        """
        layer_times = {}
        
        for activation_time, x_pos, z_pos, element_id in self.toolpath:
            layer = int(z_pos / self.layer_height)
            
            if layer not in layer_times:
                layer_times[layer] = [activation_time, activation_time]
            else:
                layer_times[layer][0] = min(layer_times[layer][0], activation_time)
                layer_times[layer][1] = max(layer_times[layer][1], activation_time)
                
        # Convert to tuples
        return {layer: tuple(times) for layer, times in layer_times.items()}
        
    def visualize_activation_sequence(self, max_time: Optional[float] = None):
        """Create visualization of activation sequence over time"""
        import matplotlib.pyplot as plt
        
        if max_time is None:
            max_time = self.toolpath[-1][0] if self.toolpath else 10.0
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Nozzle position over time
        times = np.linspace(0, max_time, 1000)
        x_positions = []
        z_positions = []
        
        for t in times:
            x, z = self.get_nozzle_position(t)
            x_positions.append(x)
            z_positions.append(z)
            
        ax1.plot(times, x_positions, 'b-', label='X position', linewidth=2)
        ax1.set_ylabel('Nozzle X Position (mm)')
        ax1.set_title('Nozzle Movement Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(times, z_positions, 'r-', label='Z position', linewidth=2)
        ax1_twin.set_ylabel('Nozzle Z Position (mm)')
        ax1_twin.legend(loc='upper right')
        
        # Bottom plot: Activation progress over time
        activation_times = [event.time for event in self.activation_events]
        cumulative_activations = list(range(1, len(activation_times) + 1))
        
        if activation_times:
            ax2.step(activation_times, cumulative_activations, 'g-', linewidth=2, where='post')
            
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Number of Active Elements')
        ax2.set_title('Element Activation Progress')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('activation_sequence.png', dpi=150, bbox_inches='tight')
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    from mesh import Mesh
    
    print("Testing Sequential Activation")
    print("=" * 50)
    
    # Create mesh and activator
    mesh = Mesh("config.yaml")
    activator = SequentialActivator(mesh, "config.yaml")
    
    # Test basic functionality
    print("\n=== Initial State ===")
    stats = activator.get_activation_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Simulate some time steps
    print("\n=== Simulating Activation ===")
    time_points = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for t in time_points:
        activated = activator.update(t)
        if activated:
            print(f"Time {t:4.1f}s: Activated {len(activated)} elements")
            nozzle_x, nozzle_z = activator.get_nozzle_position(t)
            print(f"           Nozzle at ({nozzle_x:.1f}, {nozzle_z:.1f})")
            
        progress = activator.get_print_progress(t)
        print(f"           Progress: {progress*100:.1f}%")
        
    print(f"\nPrint complete: {activator.is_printing_complete(10.0)}")
    
    # Show final statistics
    print("\n=== Final Statistics ===")
    final_stats = activator.get_activation_statistics()
    for key, value in final_stats.items():
        print(f"{key}: {value}")
        
    # Create visualization
    print("\n=== Creating Visualization ===")
    activator.visualize_activation_sequence()
    
    print("✓ Sequential activation testing complete!")