"""
Finite Element Mesh for FFF Temperature Simulation

This module handles:
1. Full geometry creation (complete wall shape)
2. Finite element discretization (breaking into small elements)  
3. Element activation/deactivation for sequential deposition simulation
"""

import numpy as np
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class Node:
    """A point in space where we solve for temperature"""
    id: int
    x: float  # mm, along wall length
    z: float  # mm, along wall height  
    temperature: float = 25.0  # °C, starts at ambient


@dataclass 
class Element:
    """A rectangular region connecting 4 nodes (in 2D)"""
    id: int
    node_ids: List[int]  # [bottom_left, bottom_right, top_right, top_left]
    is_active: bool = False  # Sequential activation: starts inactive
    center_x: float = 0.0
    center_z: float = 0.0


class Mesh:
    """
    2D Finite Element Mesh for Wall Geometry
    
    Coordinate System:
    - x: length direction (0 to wall_length mm)
    - z: height direction (0 to max_height mm) 
    - y: thickness (neglected in 2D, assume uniform temperature)
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize mesh from configuration file"""
        self.load_config(config_file)
        self.nodes: List[Node] = []
        self.elements: List[Element] = []
        self.node_grid = None  # 2D array for easy node lookup
        self.element_grid = None  # 2D array for easy element lookup
        
        # Create the complete geometry mesh
        self.create_mesh()
        
    def load_config(self, config_file: str):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Geometry parameters
        self.wall_length = config['simulation']['wall_length']  # mm
        self.layer_height = config['simulation']['layer_height']  # mm
        self.total_layers = config['simulation']['total_layers']
        self.wall_height = self.total_layers * self.layer_height  # mm
        
        # Mesh parameters
        self.dx = config['mesh']['element_size_x']  # mm
        self.dz = config['mesh']['element_size_z']  # mm
        
        # Material properties
        self.material = config['material']
        self.deposition_temp = self.material['deposition_temp']  # °C
        self.ambient_temp = self.material['ambient_temp']  # °C
        
        # Calculate mesh dimensions
        self.nx_elements = int(np.ceil(self.wall_length / self.dx))
        self.nz_elements = int(np.ceil(self.wall_height / self.dz))
        self.nx_nodes = self.nx_elements + 1
        self.nz_nodes = self.nz_elements + 1
        
        print(f"Mesh Info:")
        print(f"  Wall: {self.wall_length}mm × {self.wall_height}mm")
        print(f"  Elements: {self.nx_elements} × {self.nz_elements} = {self.nx_elements * self.nz_elements}")
        print(f"  Nodes: {self.nx_nodes} × {self.nz_nodes} = {self.nx_nodes * self.nz_nodes}")
        print(f"  Element size: {self.dx}mm × {self.dz}mm")
        
    def create_mesh(self):
        """
        Create complete wall geometry with finite elements
        
        Step 1: The Full Picture First
        - Generate all nodes for the complete final wall shape
        - Generate all elements covering the complete geometry
        
        Step 2: Breaking it Down  
        - Divide wall into regular rectangular elements
        - Each element connects 4 nodes in 2D
        
        Step 3: Turn Elements Off
        - All elements start as inactive (is_active = False)
        - Inactive elements have near-zero thermal properties
        """
        
        # Step 1 & 2: Create nodes for complete geometry
        self._create_nodes()
        
        # Step 2: Create elements connecting the nodes
        self._create_elements()
        
        # Step 3: All elements start inactive (deactivated)
        self._deactivate_all_elements()
        
        print(f"Mesh created: {len(self.nodes)} nodes, {len(self.elements)} elements")
        print(f"All elements initially INACTIVE (awaiting sequential activation)")
        
    def _create_nodes(self):
        """Create all nodes in a regular grid"""
        self.nodes = []
        self.node_grid = np.full((self.nz_nodes, self.nx_nodes), -1, dtype=int)
        
        node_id = 0
        for i in range(self.nz_nodes):  # height direction (z)
            for j in range(self.nx_nodes):  # length direction (x)
                x = j * self.dx
                z = i * self.dz
                
                node = Node(
                    id=node_id,
                    x=x, 
                    z=z,
                    temperature=self.ambient_temp
                )
                
                self.nodes.append(node)
                self.node_grid[i, j] = node_id
                node_id += 1
                
    def _create_elements(self):
        """Create rectangular elements connecting nodes"""
        self.elements = []
        self.element_grid = np.full((self.nz_elements, self.nx_elements), -1, dtype=int)
        
        element_id = 0
        for i in range(self.nz_elements):  # height direction
            for j in range(self.nx_elements):  # length direction
                
                # Get the 4 corner nodes for this element
                # Node ordering: [bottom_left, bottom_right, top_right, top_left]
                bottom_left = self.node_grid[i, j]
                bottom_right = self.node_grid[i, j + 1]  
                top_right = self.node_grid[i + 1, j + 1]
                top_left = self.node_grid[i + 1, j]
                
                # Element center coordinates
                center_x = (j + 0.5) * self.dx
                center_z = (i + 0.5) * self.dz
                
                element = Element(
                    id=element_id,
                    node_ids=[bottom_left, bottom_right, top_right, top_left],
                    is_active=False,  # Starts inactive!
                    center_x=center_x,
                    center_z=center_z
                )
                
                self.elements.append(element)
                self.element_grid[i, j] = element_id
                element_id += 1
                
    def _deactivate_all_elements(self):
        """Step 3: Turn all elements OFF initially"""
        for element in self.elements:
            element.is_active = False
            
    def activate_element(self, element_id: int, temperature: Optional[float] = None):
        """
        Activate an element (turn it ON for sequential deposition)
        
        Args:
            element_id: ID of element to activate
            temperature: Temperature to set (defaults to deposition_temp)
        """
        if 0 <= element_id < len(self.elements):
            self.elements[element_id].is_active = True
            
            # Set temperature of all nodes in this element to deposition temperature
            if temperature is None:
                temperature = self.deposition_temp
                
            for node_id in self.elements[element_id].node_ids:
                self.nodes[node_id].temperature = temperature
                
            print(f"Activated element {element_id} at T={temperature}°C")
        else:
            raise ValueError(f"Element ID {element_id} out of range")
            
    def deactivate_element(self, element_id: int):
        """Deactivate an element (for testing purposes)"""
        if 0 <= element_id < len(self.elements):
            self.elements[element_id].is_active = False
        else:
            raise ValueError(f"Element ID {element_id} out of range")
            
    def get_element_at_position(self, x: float, z: float) -> Optional[int]:
        """
        Find element ID at given physical coordinates
        
        Args:
            x: Position along wall length (mm)
            z: Position along wall height (mm)
            
        Returns:
            Element ID if position is within mesh, None otherwise
        """
        # Convert physical coordinates to element indices
        j = int(x / self.dx)
        i = int(z / self.dz)
        
        # Check bounds
        if 0 <= i < self.nz_elements and 0 <= j < self.nx_elements:
            return self.element_grid[i, j]
        else:
            return None
            
    def get_elements_in_layer(self, layer_number: int) -> List[int]:
        """
        Get all element IDs in a specific layer
        
        Args:
            layer_number: Layer index (0 = first layer)
            
        Returns:
            List of element IDs in that layer
        """
        if layer_number >= self.nz_elements:
            return []
            
        element_ids = []
        for j in range(self.nx_elements):
            element_id = self.element_grid[layer_number, j]
            element_ids.append(element_id)
            
        return element_ids
        
    def get_active_elements(self) -> List[int]:
        """Get list of currently active element IDs"""
        return [elem.id for elem in self.elements if elem.is_active]
        
    def get_thermal_properties(self, element_id: int) -> Dict[str, float]:
        """
        Get thermal properties for an element
        
        Active elements: Real material properties
        Inactive elements: Near-zero properties (ghost elements)
        """
        element = self.elements[element_id]
        
        if element.is_active:
            # Real PLA properties
            return {
                'thermal_conductivity': self.material['thermal_conductivity'],
                'density': self.material['density'], 
                'specific_heat': self.material['specific_heat']
            }
        else:
            # Near-zero properties for inactive elements
            return {
                'thermal_conductivity': 1e-6,  # Nearly zero
                'density': 1e-6,
                'specific_heat': 1e-6
            }
            
    def get_mesh_statistics(self) -> Dict:
        """Get mesh statistics for debugging"""
        active_count = len(self.get_active_elements())
        total_count = len(self.elements)
        
        return {
            'total_elements': total_count,
            'active_elements': active_count,
            'inactive_elements': total_count - active_count,
            'activation_percentage': (active_count / total_count) * 100,
            'total_nodes': len(self.nodes)
        }
        
    def visualize_mesh(self, show_elements: bool = True, show_activation: bool = True):
        """
        Visualize the mesh and element activation states
        
        Args:
            show_elements: Whether to show element boundaries
            show_activation: Whether to color elements by activation state
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot elements
        for element in self.elements:
            # Get element corner coordinates
            nodes = [self.nodes[nid] for nid in element.node_ids]
            
            # Element color based on activation state
            if show_activation:
                color = 'red' if element.is_active else 'lightgray'
                alpha = 0.8 if element.is_active else 0.3
            else:
                color = 'lightblue'
                alpha = 0.5
                
            # Draw rectangle
            bottom_left = nodes[0]
            rect = Rectangle(
                (bottom_left.x, bottom_left.z), 
                self.dx, self.dz,
                facecolor=color, 
                edgecolor='black' if show_elements else color,
                alpha=alpha,
                linewidth=0.5
            )
            ax.add_patch(rect)
            
        # Plot nodes
        x_coords = [node.x for node in self.nodes]
        z_coords = [node.z for node in self.nodes]
        ax.scatter(x_coords, z_coords, c='blue', s=1, alpha=0.6)
        
        # Formatting
        ax.set_xlim(-0.5, self.wall_length + 0.5)
        ax.set_ylim(-0.5, self.wall_height + 0.5)
        ax.set_xlabel('Length (mm)')
        ax.set_ylabel('Height (mm)')
        ax.set_title(f'FFF Wall Mesh - {len(self.get_active_elements())}/{len(self.elements)} Elements Active')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend
        if show_activation:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.8, label='Active Elements'),
                Patch(facecolor='lightgray', alpha=0.3, label='Inactive Elements')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create mesh
    mesh = Mesh("config.yaml")
    
    # Show initial state (all elements inactive)
    print("\n=== Initial Mesh State ===")
    stats = mesh.get_mesh_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Visualize initial mesh
    mesh.visualize_mesh()
    
    # Test element activation (simulate first few depositions)
    print("\n=== Testing Element Activation ===")
    
    # Activate first layer elements (simulating first layer printing)
    first_layer_elements = mesh.get_elements_in_layer(0)
    print(f"First layer has {len(first_layer_elements)} elements")
    
    # Activate first 5 elements to simulate partial printing
    for i in range(min(5, len(first_layer_elements))):
        element_id = first_layer_elements[i]
        mesh.activate_element(element_id)
        
    # Show updated statistics
    stats = mesh.get_mesh_statistics()
    print(f"\nAfter activating 5 elements:")
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Visualize with some activated elements
    mesh.visualize_mesh()
    
    # Test coordinate lookup
    print("\n=== Testing Coordinate Lookup ===")
    test_positions = [(5.0, 1.0), (15.0, 2.0), (25.0, 5.0)]
    for x, z in test_positions:
        element_id = mesh.get_element_at_position(x, z)
        if element_id is not None:
            element = mesh.elements[element_id]
            print(f"Position ({x}, {z}) mm -> Element {element_id} (active: {element.is_active})")
        else:
            print(f"Position ({x}, {z}) mm -> Outside mesh")