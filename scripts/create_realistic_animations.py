"""
Professional Animation Creator for Realistic FFF Simulation - FIXED VERSION

Creates multiple animation types showing:
1. 2D side view with nozzle movement and temperature
2. 3D volume animation with temperature-based transparency  
3. Temperature evolution dashboard
4. Physics effects visualization (bed heating, glass transition)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
import warnings


def safe_contour_levels(temp_data, n_levels=30, min_range=1.0):
    """
    Generate safe contour levels that are guaranteed to be increasing
    
    Args:
        temp_data: 2D temperature array
        n_levels: Number of contour levels desired
        min_range: Minimum temperature range to ensure
        
    Returns:
        levels: Array of increasing contour levels
    """
    # Get temperature range
    temp_min = np.nanmin(temp_data)
    temp_max = np.nanmax(temp_data)
    
    # Handle edge cases
    if np.isnan(temp_min) or np.isnan(temp_max):
        # All NaN data
        return np.linspace(20, 180, n_levels)
    
    if temp_min == temp_max or (temp_max - temp_min) < 1e-6:
        # All temperatures are the same - create artificial range
        temp_center = temp_min if not np.isnan(temp_min) else 100.0
        temp_min = temp_center - min_range/2
        temp_max = temp_center + min_range/2
    
    # Ensure minimum range
    if (temp_max - temp_min) < min_range:
        temp_center = (temp_max + temp_min) / 2
        temp_min = temp_center - min_range/2
        temp_max = temp_center + min_range/2
    
    # Generate evenly spaced levels
    levels = np.linspace(temp_min, temp_max, n_levels)
    
    return levels


def safe_contourf(ax, X, Y, Z, **kwargs):
    """
    Safe contourf that handles edge cases and uniform temperature fields
    """
    try:
        # Check for valid data
        if np.all(np.isnan(Z)) or np.all(Z == 0):
            # All NaN or zero - create text placeholder
            ax.text(0.5, 0.5, 'No thermal data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            return None
        
        # Get temperature range
        temp_min, temp_max = np.nanmin(Z), np.nanmax(Z)
        
        # Generate safe levels
        if 'levels' not in kwargs:
            kwargs['levels'] = safe_contour_levels(Z, n_levels=kwargs.get('levels', 30))
        
        # Ensure we have valid levels
        levels = kwargs['levels']
        if isinstance(levels, int):
            kwargs['levels'] = safe_contour_levels(Z, n_levels=levels)
        elif len(levels) < 2:
            kwargs['levels'] = safe_contour_levels(Z, n_levels=20)
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, **kwargs)
        return contour
        
    except Exception as e:
        print(f"Contour plot failed: {e}")
        print(f"Temperature range: {np.nanmin(Z):.3f} to {np.nanmax(Z):.3f}")
        
        # Fallback to simple image plot
        try:
            vmin = kwargs.get('vmin', np.nanmin(Z))
            vmax = kwargs.get('vmax', np.nanmax(Z))
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
                
            im = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                          origin='lower', aspect='auto', 
                          cmap=kwargs.get('cmap', 'hot'),
                          vmin=vmin, vmax=vmax)
            return im
        except:
            # Final fallback - just show text
            ax.text(0.5, 0.5, f'Temperature data\nMin: {np.nanmin(Z):.1f}°C\nMax: {np.nanmax(Z):.1f}°C', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            return None


class RealisticAnimationCreator:
    """
    Creates professional animations showing realistic FFF physics
    """
    
    def __init__(self, animation_data_file: str = "output/animation_data_realistic.npz"):
        """Load animation data and setup"""
        
        print("Loading realistic simulation data for animation...")
        
        # Load data
        data = np.load(animation_data_file, allow_pickle=True)
        
        self.time_frames = data['time_frames']
        self.temperature_fields = data['temperature_fields'] 
        self.nozzle_trajectory = data['nozzle_trajectory']
        self.mesh_info = data['mesh_geometry'].item()
        self.config = data['physics_config'].item()
        
        # Extract key parameters
        self.wall_length = self.mesh_info['wall_length']
        self.wall_height = self.mesh_info['wall_height']
        self.dx = self.mesh_info['dx']
        self.dz = self.mesh_info['dz']
        self.nx_nodes = self.mesh_info['nx_elements'] + 1
        self.nz_nodes = self.mesh_info['nz_elements'] + 1
        
        # Physics parameters
        self.bed_temp = self.config['environment']['heated_bed']['temperature']
        self.glass_transition_temp = self.config['material']['glass_transition_temp']
        self.ambient_temp = self.config['environment']['ambient_temperature']
        
        # Create output directory
        Path("output/animations").mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Loaded {len(self.time_frames)} animation frames")
        print(f"  Wall: {self.wall_length}×{self.wall_height} mm")
        print(f"  Physics: Bed {self.bed_temp}°C, Tg {self.glass_transition_temp}°C")
        
    def create_2d_thermal_animation(self, fps: int = 10, duration_seconds: int = 30):
        """
        Create 2D side-view animation showing:
        - Temperature field as heatmap
        - Nozzle movement with trail
        - Element activation sequence
        - Key temperature thresholds
        """
        
        print("Creating 2D thermal animation...")
        
        # Setup figure
        fig, (ax_main, ax_temp) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate frame indices for desired duration
        total_frames = len(self.time_frames)
        frame_step = max(1, total_frames // (fps * duration_seconds))
        frame_indices = range(0, total_frames, frame_step)
        
        # Create coordinate grids
        x = np.linspace(0, self.wall_length, self.nx_nodes)
        z = np.linspace(0, self.wall_height, self.nz_nodes)
        X, Z = np.meshgrid(x, z)
        
        def animate_frame(frame_idx):
            """Animate single frame"""
            ax_main.clear()
            ax_temp.clear()
            
            # Get current data
            time_idx = frame_indices[frame_idx]
            current_time = self.time_frames[time_idx]
            temp_field = self.temperature_fields[time_idx]
            nozzle_pos = self.nozzle_trajectory[time_idx]
            
            # Reshape temperature field to grid
            temp_grid = temp_field.reshape(self.nz_nodes, self.nx_nodes)
            
            # === Main plot: Temperature field ===
            
            # Temperature heatmap using safe contour
            im = safe_contourf(ax_main, X, Z, temp_grid, levels=50, cmap='plasma', 
                              vmin=self.ambient_temp, vmax=175)
            
            # Add isotherms for key temperatures (only if we have valid contour)
            if im is not None:
                try:
                    cs1 = ax_main.contour(X, Z, temp_grid, levels=[self.bed_temp], 
                                         colors='red', linewidths=2, alpha=0.8)
                    cs2 = ax_main.contour(X, Z, temp_grid, levels=[self.glass_transition_temp], 
                                         colors='orange', linewidths=2, alpha=0.8)
                except:
                    pass  # Skip contour lines if they fail
            
            # Mark bed heating zone
            ax_main.axhspan(0, 0.5, alpha=0.3, color='red', label=f'Heated bed ({self.bed_temp}°C)')
            
            # Nozzle position and trail
            if len(nozzle_pos) >= 2:
                # Current nozzle position
                ax_main.scatter(nozzle_pos[0], nozzle_pos[1], c='white', s=200, 
                               marker='o', edgecolor='black', linewidth=2, 
                               label='Nozzle', zorder=10)
                
                # Nozzle trail (recent path)
                trail_length = min(20, time_idx)
                if trail_length > 1:
                    trail_x = [self.nozzle_trajectory[i][0] for i in range(max(0, time_idx-trail_length), time_idx)]
                    trail_z = [self.nozzle_trajectory[i][1] for i in range(max(0, time_idx-trail_length), time_idx)]
                    ax_main.plot(trail_x, trail_z, 'w-', linewidth=3, alpha=0.7, label='Nozzle trail')
            
            # Formatting
            ax_main.set_xlim(0, self.wall_length)
            ax_main.set_ylim(0, self.wall_height)
            ax_main.set_xlabel('Length (mm)')
            ax_main.set_ylabel('Height (mm)')
            ax_main.set_title(f'FFF Realistic Thermal Animation\nTime: {current_time:.1f}s')
            ax_main.set_aspect('equal')
            ax_main.legend(loc='upper right')
            
            # === Temperature evolution plot ===
            
            # Extract temperature at monitoring points
            monitoring_points = [
                (5.0, 1.0, "Bottom Point"),
                (15.0, 4.0, "Mid Point"), 
                (25.0, 8.0, "Top Point")
            ]
            
            colors = ['blue', 'green', 'red']
            
            for i, (x_mon, z_mon, label) in enumerate(monitoring_points):
                # Convert to indices with bounds checking
                j_idx = min(int(x_mon / self.dx), self.nx_nodes - 1)
                i_idx = min(int(z_mon / self.dz), self.nz_nodes - 1)
                
                if i_idx < self.nz_nodes and j_idx < self.nx_nodes:
                    node_id = i_idx * self.nx_nodes + j_idx
                    
                    # Extract temperature history up to current time
                    temp_history = [self.temperature_fields[t][node_id] 
                                   for t in range(min(time_idx + 1, len(self.temperature_fields)))]
                    time_history = self.time_frames[:len(temp_history)]
                    
                    ax_temp.plot(time_history, temp_history, color=colors[i], 
                                linewidth=2, label=label)
            
            # Add key temperature lines
            ax_temp.axhline(y=self.bed_temp, color='red', linestyle='--', 
                           alpha=0.7, label=f'Bed temp ({self.bed_temp}°C)')
            ax_temp.axhline(y=self.glass_transition_temp, color='orange', linestyle='--', 
                           alpha=0.7, label=f'Glass transition ({self.glass_transition_temp}°C)')
            ax_temp.axhline(y=self.ambient_temp, color='blue', linestyle='--', 
                           alpha=0.7, label=f'Ambient ({self.ambient_temp}°C)')
            
            # Current time marker
            ax_temp.axvline(x=current_time, color='black', linestyle='-', alpha=0.8)
            
            ax_temp.set_xlabel('Time (s)')
            ax_temp.set_ylabel('Temperature (°C)')
            ax_temp.set_title('Temperature Evolution at Key Points')
            ax_temp.legend()
            ax_temp.grid(True, alpha=0.3)
            ax_temp.set_ylim(20, 180)
            
            # Add colorbar to main plot (only on first frame)
            if frame_idx == 0 and im is not None and hasattr(im, 'levels'):
                try:
                    cbar = plt.colorbar(im, ax=ax_main)
                    cbar.set_label('Temperature (°C)')
                except:
                    pass
            
            plt.tight_layout()
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(frame_indices),
                                      interval=1000//fps, blit=False, repeat=True)
        
        # Save animation
        output_file = "output/animations/fff_realistic_2d_thermal.mp4"
        print(f"Saving 2D thermal animation to {output_file}...")
        
        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='FFF Simulation'),
                                           bitrate=1800)
            anim.save(output_file, writer=writer)
            print(f"✓ 2D thermal animation saved ({len(frame_indices)} frames)")
        except Exception as e:
            print(f"❌ Animation save failed: {e}")
            # Try saving as GIF instead
            gif_file = output_file.replace('.mp4', '.gif')
            anim.save(gif_file, writer='pillow', fps=max(1, fps//2))
            print(f"✓ Saved as GIF instead: {gif_file}")
        
        plt.close()
        return output_file
        
    def create_physics_comparison_animation(self, fps: int = 8):
        """
        Create animation comparing different physics effects:
        - Basic vs realistic boundary conditions
        - Glass transition effects
        - Bed heating influence
        """
        
        print("Creating physics comparison animation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate frame indices
        total_frames = len(self.time_frames)
        frame_step = max(1, total_frames // 100)  # ~100 frames max
        frame_indices = range(0, total_frames, frame_step)
        
        def animate_physics_frame(frame_idx):
            """Animate physics comparison frame"""
            
            for ax in axes.flat:
                ax.clear()
                
            time_idx = frame_indices[frame_idx]
            current_time = self.time_frames[time_idx]
            temp_field = self.temperature_fields[time_idx]
            
            # Reshape to grid
            temp_grid = temp_field.reshape(self.nz_nodes, self.nx_nodes)
            x = np.linspace(0, self.wall_length, self.nx_nodes)
            z = np.linspace(0, self.wall_height, self.nz_nodes)
            X, Z = np.meshgrid(x, z)
            
            # Debug temperature data
            temp_min, temp_max = np.nanmin(temp_grid), np.nanmax(temp_grid)
            print(f"Frame {frame_idx}: T range {temp_min:.1f} to {temp_max:.1f}°C")
            
            # 1. Overall temperature field
            ax = axes[0, 0]
            im1 = safe_contourf(ax, X, Z, temp_grid, levels=30, cmap='plasma', vmin=25, vmax=175)
            
            # Add contour lines safely
            try:
                if temp_max > temp_min + 1:  # Only if we have meaningful range
                    ax.contour(X, Z, temp_grid, levels=[self.bed_temp, self.glass_transition_temp], 
                              colors=['red', 'orange'], linewidths=2)
            except:
                pass
                
            ax.set_title(f'Temperature Field (t={current_time:.1f}s)')
            ax.set_xlabel('Length (mm)')
            ax.set_ylabel('Height (mm)')
            ax.set_aspect('equal')
            
            # 2. Bed heating effect (bottom vs top)
            ax = axes[0, 1]
            if temp_grid.shape[0] > 1:
                bottom_temps = temp_grid[0, :]  # Bottom row
                top_temps = temp_grid[-1, :]    # Top row
                
                ax.plot(x, bottom_temps, 'r-', linewidth=3, label=f'Bottom (near {self.bed_temp}°C bed)')
                ax.plot(x, top_temps, 'b-', linewidth=3, label='Top (exposed to air)')
                ax.axhline(y=self.bed_temp, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=self.ambient_temp, color='blue', linestyle='--', alpha=0.7)
                
                ax.set_xlabel('Length (mm)')
                ax.set_ylabel('Temperature (°C)')
                ax.set_title('Bed Heating Effect\n(Bottom vs Top Temperatures)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(20, 180)
            else:
                ax.text(0.5, 0.5, 'Single layer - no gradient', transform=ax.transAxes, ha='center')
            
            # 3. Glass transition zones
            ax = axes[1, 0]
            
            # Create glass transition indicator
            tg_indicator = np.where(temp_grid > self.glass_transition_temp, 1, 0)
            
            try:
                ax.contourf(X, Z, tg_indicator, levels=[0, 0.5, 1], 
                           colors=['blue', 'orange'], alpha=0.7)
                ax.contour(X, Z, temp_grid, levels=[self.glass_transition_temp], 
                          colors=['orange'], linewidths=3)
            except:
                # Fallback to simple visualization
                ax.imshow(tg_indicator, extent=[0, self.wall_length, 0, self.wall_height],
                         origin='lower', alpha=0.7, cmap='coolwarm')
            
            ax.set_title(f'Glass Transition Zones\n(Orange: T > {self.glass_transition_temp}°C)')
            ax.set_xlabel('Length (mm)')
            ax.set_ylabel('Height (mm)')
            ax.set_aspect('equal')
            
            # 4. Temperature gradients
            ax = axes[1, 1]
            
            # Calculate vertical temperature gradient
            if temp_grid.shape[0] > 1:
                try:
                    temp_gradient = np.gradient(temp_grid, axis=0) / self.dz  # °C/mm
                    
                    im4 = safe_contourf(ax, X, Z, temp_gradient, levels=20, cmap='RdBu_r')
                    ax.set_title('Vertical Temperature Gradient\n(°C/mm)')
                    ax.set_xlabel('Length (mm)')
                    ax.set_ylabel('Height (mm)')
                    ax.set_aspect('equal')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Gradient calculation failed\n{str(e)[:50]}...', 
                           transform=ax.transAxes, ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Single layer - no gradient', transform=ax.transAxes, ha='center')
            
            plt.tight_layout()
            
        # Create animation with error handling
        try:
            anim = animation.FuncAnimation(fig, animate_physics_frame, 
                                          frames=len(frame_indices),
                                          interval=1000//fps, blit=False, repeat=True)
            
            # Save
            output_file = "output/animations/fff_physics_comparison.mp4"
            print(f"Saving physics comparison animation to {output_file}...")
            
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='FFF Physics'),
                                           bitrate=1800)
            anim.save(output_file, writer=writer)
            
            plt.close()
            print(f"✓ Physics comparison animation saved")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Physics comparison animation failed: {e}")
            plt.close()
            return None


def main():
    """Create all realistic physics animations with error handling"""
    
    print("REALISTIC FFF ANIMATION CREATOR")
    print("="*50)
    
    # Check if animation data exists
    if not Path("output/animation_data_realistic.npz").exists():
        print("❌ Animation data not found!")
        print("Please run: python analyze_realistic_results.py first")
        return
        
    # Create animator
    try:
        animator = RealisticAnimationCreator()
    except Exception as e:
        print(f"❌ Failed to initialize animator: {e}")
        return
    
    # Create different animation types
    print("\nCreating professional animations...")
    
    created_files = []
    
    try:
        # 1. 2D thermal evolution
        print("Creating 2D thermal animation...")
        file1 = animator.create_2d_thermal_animation(fps=10, duration_seconds=20)
        if file1:
            created_files.append(file1)
        
        # 2. Physics comparison with better error handling
        print("Creating physics comparison animation...")
        file2 = animator.create_physics_comparison_animation(fps=8)
        if file2:
            created_files.append(file2)
        
        print("\n" + "="*50)
        print("🎉 ANIMATIONS COMPLETED!")
        print("="*50)
        
        if created_files:
            print(f"\nSuccessfully created animations:")
            for i, file in enumerate(created_files, 1):
                print(f"{i}. {file}")
        else:
            print("⚠ No animations were successfully created")
        
        print(f"\nNext steps:")
        print(f"• Compare animations with thermal camera videos")
        print(f"• Use for validation and presentation")
        
    except Exception as e:
        print(f"❌ Animation creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()