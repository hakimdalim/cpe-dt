"""
Improved Finite Element Thermal Solver for FFF Temperature Simulation

This module implements proper heat transfer physics while maintaining
compatibility with the existing simulation structure.
"""

import numpy as np
import yaml
from typing import Dict, Tuple, List
import warnings

# Try to import scipy for advanced features, fallback if not available
try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available, using dense matrices")
    SCIPY_AVAILABLE = False


class FiniteElementThermalSolver:
    """
    Proper finite element thermal solver for FFF temperature simulation
    
    Implements:
    1. Heat diffusion equation: ∂T/∂t = α∇²T + Q̇/(ρcp)
    2. Finite element spatial discretization
    3. Time integration (explicit/implicit)
    4. Proper boundary conditions
    """
    
    def __init__(self, mesh, config_file: str = "config.yaml"):
        """Initialize thermal solver with mesh and configuration"""
        self.mesh = mesh
        self.load_config(config_file)
        
        # Pre-allocate matrices
        self.n_nodes = len(self.mesh.nodes)
        self.setup_matrices()
        
        # Previous time step temperatures
        self.T_prev = np.array([node.temperature for node in self.mesh.nodes])
        
        print(f"FE Thermal solver initialized:")
        print(f"  Method: {self.solver_method}")
        print(f"  Nodes: {self.n_nodes}")
        print(f"  Thermal diffusivity: {self.alpha:.2e} m²/s")
        print(f"  Convection coefficient: {self.h_conv} W/m²K")
        print(f"  SciPy available: {SCIPY_AVAILABLE}")
        
    def load_config(self, config_file: str):
        """Load thermal solver configuration"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Material properties
        material = config['material']
        self.k_thermal = material['thermal_conductivity']  # W/mK
        self.rho = material['density']  # kg/m³
        self.cp = material['specific_heat']  # J/kgK
        self.T_ambient = material['ambient_temp']  # °C
        self.T_deposition = material['deposition_temp']  # °C
        
        # Boundary conditions
        bc_config = config.get('boundary_conditions', {})
        self.h_conv = bc_config.get('convection_coefficient', 10.0)  # W/m²K
        
        # Solver settings
        solver_config = config.get('solver', {})
        self.solver_method = solver_config.get('thermal_method', 'explicit')
        self.theta = solver_config.get('theta', 0)  # For theta-method 'TODO
        
        # Calculate thermal diffusivity
        self.alpha = self.k_thermal / (self.rho * self.cp)  # m²/s
        
        # Convert mesh dimensions to meters for calculations
        self.dx_m = self.mesh.dx / 1000.0  # mm to m
        self.dz_m = self.mesh.dz / 1000.0  # mm to m
        self.element_area = self.dx_m * self.dz_m  # m²
        
    def setup_matrices(self):
        """Set up global finite element matrices"""
        print("Assembling finite element matrices...")
        
        if SCIPY_AVAILABLE:
            # Use sparse matrices for efficiency
            self.K_global = lil_matrix((self.n_nodes, self.n_nodes))  # Conductivity matrix
            self.C_global = lil_matrix((self.n_nodes, self.n_nodes))  # Capacity matrix
            self.H_global = lil_matrix((self.n_nodes, self.n_nodes))  # Convection matrix
        else:
            # Fallback to dense matrices
            self.K_global = np.zeros((self.n_nodes, self.n_nodes))
            self.C_global = np.zeros((self.n_nodes, self.n_nodes))
            self.H_global = np.zeros((self.n_nodes, self.n_nodes))
        
        # Assemble element contributions
        for element in self.mesh.elements:
            self._assemble_element_matrices(element)
            
        if SCIPY_AVAILABLE:
            # Convert to CSR format for efficient operations
            self.K_global = self.K_global.tocsr()
            self.C_global = self.C_global.tocsr()
            self.H_global = self.H_global.tocsr()
        
        print(f"  Global matrices assembled: {self.n_nodes}×{self.n_nodes}")
        
    def _assemble_element_matrices(self, element):
        """Assemble matrices for a single element"""
        node_ids = element.node_ids
        
        # Get element properties (depends on activation state)
        props = self.mesh.get_thermal_properties(element.id)
        k_elem = props['thermal_conductivity']
        rho_elem = props['density'] 
        cp_elem = props['specific_heat']
        
        # Element conductivity matrix (4x4 for bilinear quad)
        K_elem = self._element_conductivity_matrix(k_elem)
        
        # Element capacity matrix (4x4)
        C_elem = self._element_capacity_matrix(rho_elem, cp_elem)
        
        # Element convection matrix (only for boundary elements)
        H_elem = self._element_convection_matrix(element)
        
        # Assemble into global matrices
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                self.K_global[node_i, node_j] += K_elem[i, j]
                self.C_global[node_i, node_j] += C_elem[i, j]
                self.H_global[node_i, node_j] += H_elem[i, j]
                
    def _element_conductivity_matrix(self, k_elem: float) -> np.ndarray:
        """
        Calculate element conductivity matrix for bilinear quad element
        
        For rectangular element with constant conductivity:
        K_elem = k * integral(∇N_i · ∇N_j) dA
        """
        dx, dz = self.dx_m, self.dz_m
        
        # Analytical conductivity matrix for rectangular element
        # Based on bilinear shape functions
        K_x = k_elem * dz / (3 * dx) * np.array([
            [ 1, -1, -1,  1],
            [-1,  1,  1, -1],
            [-1,  1,  1, -1],
            [ 1, -1, -1,  1]
        ])
        
        K_z = k_elem * dx / (3 * dz) * np.array([
            [ 1,  1, -1, -1],
            [ 1,  1, -1, -1],
            [-1, -1,  1,  1],
            [-1, -1,  1,  1]
        ])
        
        return K_x + K_z
        
    def _element_capacity_matrix(self, rho_elem: float, cp_elem: float) -> np.ndarray:
        """
        Calculate element capacity matrix
        
        C_elem = ρcp * integral(N_i * N_j) dA
        """
        area = self.element_area
        rho_cp = rho_elem * cp_elem
        
        # Consistent mass matrix for bilinear quad
        C_elem = (rho_cp * area / 36) * np.array([
            [4, 2, 1, 2],
            [2, 4, 2, 1], 
            [1, 2, 4, 2],
            [2, 1, 2, 4]
        ])
        
        return C_elem
        
    def _element_convection_matrix(self, element) -> np.ndarray:
        """
        Calculate convection matrix for boundary elements
        
        Only applies to elements on the boundary (exposed surfaces)
        """
        H_elem = np.zeros((4, 4))
        
        if self._is_boundary_element(element):
            # Get which edges are on boundary
            boundary_edges = self._get_boundary_edges(element)
            
            for edge in boundary_edges:
                # Add convection contribution for this edge
                edge_length = self._get_edge_length(edge)
                h_contrib = self.h_conv * edge_length / 6.0
                
                # Add to diagonal terms for nodes on this edge
                for node_idx in edge:
                    H_elem[node_idx, node_idx] += h_contrib
                    
        return H_elem
        
    def _is_boundary_element(self, element) -> bool:
        """Check if element is on the boundary"""
        # Get element position in grid
        elem_i = int(element.center_z / self.mesh.dz)
        elem_j = int(element.center_x / self.mesh.dx)
        
        # Check if on any boundary
        is_bottom = (elem_i == 0)
        is_top = (elem_i == self.mesh.nz_elements - 1)
        is_left = (elem_j == 0)
        is_right = (elem_j == self.mesh.nx_elements - 1)
        
        return is_bottom or is_top or is_left or is_right
        
    def _get_boundary_edges(self, element) -> List[List[int]]:
        """Get edges that are on the boundary"""
        elem_i = int(element.center_z / self.mesh.dz)
        elem_j = int(element.center_x / self.mesh.dx)
        
        edges = []
        
        # Check each edge (nodes are ordered: [bottom_left, bottom_right, top_right, top_left])
        if elem_i == 0:  # Bottom edge
            edges.append([0, 1])  # bottom edge
        if elem_j == self.mesh.nx_elements - 1:  # Right edge  
            edges.append([1, 2])  # right edge
        if elem_i == self.mesh.nz_elements - 1:  # Top edge
            edges.append([2, 3])  # top edge  
        if elem_j == 0:  # Left edge
            edges.append([3, 0])  # left edge
            
        return edges
        
    def _get_edge_length(self, edge: List[int]) -> float:
        """Get length of an edge"""
        # For rectangular elements, edges are either dx or dz
        if edge in [[0, 1], [2, 3]]:  # horizontal edges
            return self.dx_m
        else:  # vertical edges
            return self.dz_m
        
    def solve_timestep(self, dt: float):
        """
        Solve heat transfer for one time step
        
        Args:
            dt: Time step size in seconds
        """
        # Get current temperatures
        T_current = np.array([node.temperature for node in self.mesh.nodes])
        
        # Calculate heat source vector
        Q_vector = self._calculate_heat_sources()
        
        # Apply boundary conditions
        Q_bc = self._apply_boundary_conditions(T_current)
        Q_total = Q_vector + Q_bc
        
        # Solve based on method
        if self.solver_method == 'explicit':
            T_new = self._solve_explicit(T_current, dt, Q_total)
        elif self.solver_method == 'implicit':
            T_new = self._solve_implicit(T_current, dt, Q_total)
        else:  # theta-method or crank_nicolson
            T_new = self._solve_theta_method(T_current, dt, Q_total)
            
        # Update mesh temperatures
        for i, node in enumerate(self.mesh.nodes):
            node.temperature = float(T_new[i])
            
        # Store for next iteration
        self.T_prev = T_current.copy()
        
    def _calculate_heat_sources(self) -> np.ndarray:
        """Calculate heat source vector from newly deposited material"""
        Q = np.zeros(self.n_nodes)
        
        # Add heat sources for newly activated elements
        for element in self.mesh.elements:
            if element.is_active:
                # Heat source from deposition (could be more sophisticated)
                for node_id in element.node_ids:
                    # Simple heat source model - no internal generation for now
                    Q[node_id] += 0.0
                    
        return Q
        
    def _apply_boundary_conditions(self, T_current: np.ndarray) -> np.ndarray:
        """Apply convective boundary conditions"""
        # Convection: Q_conv = h*A*(T_ambient - T_surface)
        if SCIPY_AVAILABLE:
            Q_bc = self.H_global @ (self.T_ambient - T_current)
        else:
            Q_bc = self.H_global.dot(self.T_ambient - T_current)
        return Q_bc
        
    def _solve_explicit(self, T_current: np.ndarray, dt: float, Q: np.ndarray) -> np.ndarray:
        """Solve using explicit (forward Euler) method"""
        # Check stability
        dt_max = self.get_stability_limit()
        if dt > dt_max:
            warnings.warn(f"Time step {dt:.2e} exceeds stability limit {dt_max:.2e}")
            
        # Explicit: T_new = T + dt * C^(-1) * (Q - K*T)
        # Use lumped mass matrix for efficiency
        if SCIPY_AVAILABLE:
            C_lumped = np.array(self.C_global.sum(axis=1)).flatten()
            dT_dt = (Q - (self.K_global + self.H_global) @ T_current) / C_lumped
        else:
            C_lumped = np.sum(self.C_global, axis=1)
            dT_dt = (Q - (self.K_global + self.H_global).dot(T_current)) / C_lumped
            
        C_lumped[C_lumped == 0] = 1e-12  # Avoid division by zero
        T_new = T_current + dt * dT_dt
        
        return T_new
        
    def _solve_implicit(self, T_current: np.ndarray, dt: float, Q: np.ndarray) -> np.ndarray:
        """Solve using implicit (backward Euler) method"""
        # Implicit: (C + dt*(K+H)) * T_new = C * T + dt * Q
        LHS = self.C_global + dt * (self.K_global + self.H_global)
        
        if SCIPY_AVAILABLE:
            RHS = self.C_global @ T_current + dt * Q
            T_new = spsolve(LHS, RHS)
        else:
            RHS = self.C_global.dot(T_current) + dt * Q
            T_new = np.linalg.solve(LHS, RHS)
            
        return T_new
        
    def _solve_theta_method(self, T_current: np.ndarray, dt: float, Q: np.ndarray) -> np.ndarray:
        """Solve using theta-method (Crank-Nicolson when theta=0.5)"""
        theta = self.theta
        
        # (C + theta*dt*(K+H)) * T_new = (C - (1-theta)*dt*(K+H)) * T + dt * Q
        LHS = self.C_global + theta * dt * (self.K_global + self.H_global)
        RHS_matrix = self.C_global - (1-theta) * dt * (self.K_global + self.H_global)
        
        if SCIPY_AVAILABLE:
            RHS = RHS_matrix @ T_current + dt * Q
            T_new = spsolve(LHS, RHS)
        else:
            RHS = RHS_matrix.dot(T_current) + dt * Q
            T_new = np.linalg.solve(LHS, RHS)
            
        return T_new
        
    def get_stability_limit(self) -> float:
        """Calculate maximum stable time step for explicit method"""
        # For 2D explicit method: dt ≤ dx²/(4α)
        dx_min = min(self.dx_m, self.dz_m)
        dt_max = (dx_min**2) / (4 * self.alpha)
        
        # Safety factor
        dt_max *= 0.5
        
        return min(0.1, dt_max)  # Cap at 0.1 seconds
        
    def get_thermal_statistics(self) -> Dict:
        """Get thermal field statistics"""
        active_nodes = []
        all_temps = []
        
        for node in self.mesh.nodes:
            all_temps.append(node.temperature)
            if self._is_node_active(node.id):
                active_nodes.append(node.temperature)
                
        if not active_nodes:
            active_nodes = [self.T_ambient]
            
        return {
            'min_temp': min(all_temps),
            'max_temp': max(all_temps),
            'avg_temp': np.mean(all_temps),
            'std_temp': np.std(all_temps),
            'active_min_temp': min(active_nodes),
            'active_max_temp': max(active_nodes),
            'active_avg_temp': np.mean(active_nodes),
            'active_nodes': len([node for node in self.mesh.nodes if self._is_node_active(node.id)]),
            'total_nodes': len(self.mesh.nodes),
            'thermal_diffusivity': self.alpha
        }
        
    def _is_node_active(self, node_id: int) -> bool:
        """Check if node belongs to any active element"""
        for element in self.mesh.elements:
            if node_id in element.node_ids and element.is_active:
                return True
        return False
        
    def get_temperature_field(self) -> np.ndarray:
        """Get current temperature field as 2D array"""
        temps = np.array([node.temperature for node in self.mesh.nodes])
        return temps.reshape(self.mesh.nz_nodes, self.mesh.nx_nodes)
        
    def get_heat_flux(self, element_id: int) -> Tuple[float, float]:
        """Calculate heat flux in element (qx, qz)"""
        element = self.mesh.elements[element_id]
        
        # Get nodal temperatures
        T_nodes = np.array([self.mesh.nodes[nid].temperature for nid in element.node_ids])
        
        # Calculate temperature gradients (simplified)
        dT_dx = (T_nodes[1] + T_nodes[2] - T_nodes[0] - T_nodes[3]) / (2 * self.dx_m)
        dT_dz = (T_nodes[2] + T_nodes[3] - T_nodes[0] - T_nodes[1]) / (2 * self.dz_m)
        
        # Heat flux: q = -k * grad(T)
        props = self.mesh.get_thermal_properties(element_id)
        k = props['thermal_conductivity']
        
        qx = -k * dT_dx
        qz = -k * dT_dz
        
        return qx, qz


# Compatibility function to replace SimplifiedThermalSolver
def SimplifiedThermalSolver(mesh, config_file: str = "config.yaml"):
    """
    Compatibility wrapper - returns the improved solver
    
    This allows existing code to work without changes while getting
    the improved thermal physics.
    """
    return FiniteElementThermalSolver(mesh, config_file)


# Test the thermal solver
if __name__ == "__main__":
    print("Testing Improved Thermal Solver")
    print("=" * 50)
    
    try:
        from mesh import Mesh
        
        # Create mesh and solver
        mesh = Mesh("config.yaml")
        solver = FiniteElementThermalSolver(mesh, "config.yaml")
        
        # Test basic functionality
        print("\n=== Initial State ===")
        stats = solver.get_thermal_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        # Activate some elements and test
        print("\n=== Testing Heat Transfer ===")
        
        # Activate first layer
        first_layer = mesh.get_elements_in_layer(0)
        for elem_id in first_layer[:3]:
            mesh.activate_element(elem_id, 175.0)  # Deposition temperature
            
        # Run a few time steps
        dt = min(0.001, solver.get_stability_limit())  # Use small stable time step
        print(f"Using time step: {dt:.4f} seconds")
        
        for step in range(10):
            solver.solve_timestep(dt)
            if step % 5 == 0:
                stats = solver.get_thermal_statistics()
                print(f"Step {step:2d}: Max T = {stats['max_temp']:.1f}°C, Active avg = {stats['active_avg_temp']:.1f}°C")
        
        print("\n✓ Thermal solver test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()