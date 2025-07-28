"""
Example showing how to use user-defined crop regions in GPU thermal preprocessing
"""

from gpu_thermal_preprocessor import GPUThermalPreprocessor, quick_view_with_gpu_preprocessing
from gpu_thermal_config import GPUThermalConfig
import numpy as np
import matplotlib.pyplot as plt

def example_user_defined_cropping():
    """Complete example of using user-defined crop regions"""
    
    # File path
    filepath = r"C:\Users\hakim\Documents\cpe_dt\dataset\real\thermal_capture_20250721_195758.h5"
    
    print("🎯 USER-DEFINED CROP REGION EXAMPLE")
    print("=" * 50)
    
    # Initialize GPU preprocessor
    config = GPUThermalConfig.get_config('rtx_4060_optimized')
    preprocessor = GPUThermalPreprocessor(config, use_gpu=True)
    
    # Load thermal data
    thermal_images, metadata = preprocessor.load_thermal_data(filepath)
    print(f"Loaded thermal data: {thermal_images.shape}")
    
    # Method 1: Define crop region directly in function call
    print("\n📍 Method 1: Direct crop region specification")
    user_crop_region = {
        'x_range': (14, 24),   # Crop X from pixel 14 to 24
        'y_range': (10, 25)    # Crop Y from pixel 10 to 25
    }
    
    print(f"User-defined crop region: X=[{user_crop_region['x_range'][0]}:{user_crop_region['x_range'][1]}], "
          f"Y=[{user_crop_region['y_range'][0]}:{user_crop_region['y_range'][1]}]")
    
    # Test single frame with user-defined crop
    processed_frame = preprocessor.test_single_frame(thermal_images, 
                                                   frame_idx=-1, 
                                                   user_crop_region=user_crop_region)
    
    # Method 2: Set crop region in preprocessor configuration
    print("\n⚙️ Method 2: Configure crop region in preprocessor")
    preprocessor.set_user_crop_region(x_range=(14, 24), y_range=(10, 25))
    
    # Check if crop region is set
    stored_crop = preprocessor.get_user_crop_region()
    if stored_crop:
        print(f"Stored crop region: {stored_crop}")
    
    # Process full dataset with user-defined crop
    print("\n🚀 Processing full dataset with user-defined crop region...")
    
    choice = input("Process full dataset with user-defined cropping? (y/n): ")
    
    if choice.lower() == 'y':
        results = preprocessor.process_full_dataset_gpu(thermal_images, 
                                                      user_crop_region=user_crop_region)
        
        # Save results
        output_path = filepath.replace('.h5', '_user_crop_processed.h5')
        preprocessor.save_processed_data(results, output_path)
        
        print(f"\n✅ User-defined crop processing complete!")
        print(f"Results saved to: {output_path}")
        
        # Show cropping statistics
        show_user_crop_statistics(results, user_crop_region)
        
        return results
    
    return None




import numpy as np
import matplotlib.pyplot as plt

def example_user_defined_cropping():
    """Complete example of using user-defined crop regions"""
    
    # File path
    filepath = r"C:\Users\hakim\Documents\cpe_dt\dataset\real\thermal_capture_20250721_195758.h5"
    
    print("🎯 USER-DEFINED CROP REGION EXAMPLE")
    print("=" * 50)
    
    # Initialize GPU preprocessor
    config = GPUThermalConfig.get_config('rtx_4060_optimized')
    preprocessor = GPUThermalPreprocessor(config, use_gpu=True)
    
    # Load thermal data
    thermal_images, metadata = preprocessor.load_thermal_data(filepath)
    print(f"Loaded thermal data: {thermal_images.shape}")
    
    # Method 1: Define crop region directly in function call
    print("\n📍 Method 1: Direct crop region specification")
    user_crop_region = {
        'x_range': (14, 24),   # Crop X from pixel 14 to 24
        'y_range': (10, 25)    # Crop Y from pixel 10 to 25
    }
    
    print(f"User-defined crop region: X=[{user_crop_region['x_range'][0]}:{user_crop_region['x_range'][1]}], "
          f"Y=[{user_crop_region['y_range'][0]}:{user_crop_region['y_range'][1]}]")
    
    # Test single frame with user-defined crop
    processed_frame = preprocessor.test_single_frame(thermal_images, 
                                                   frame_idx=-1, 
                                                   user_crop_region=user_crop_region)
    
    # Method 2: Set crop region in preprocessor configuration
    print("\n⚙️ Method 2: Configure crop region in preprocessor")
    preprocessor.set_user_crop_region(x_range=(14, 24), y_range=(10, 25))
    
    # Check if crop region is set
    stored_crop = preprocessor.get_user_crop_region()
    if stored_crop:
        print(f"Stored crop region: {stored_crop}")
    
    # Process full dataset with user-defined crop
    print("\n🚀 Processing full dataset with user-defined crop region...")
    
    choice = input("Process full dataset with user-defined cropping? (y/n): ")
    
    if choice.lower() == 'y':
        results = preprocessor.process_full_dataset_gpu(thermal_images, 
                                                      user_crop_region=user_crop_region)
        
        # Save results
        output_path = filepath.replace('.h5', '_user_crop_processed.h5')
        preprocessor.save_processed_data(results, output_path)
        
        print(f"\n✅ User-defined crop processing complete!")
        print(f"Results saved to: {output_path}")
        
        # Show cropping statistics
        show_user_crop_statistics(results, user_crop_region)
        
        return results
    
    return None


def enhanced_quick_view_with_user_crop(filepath: str, user_crop_region: dict = None):
    """Enhanced quick view function that supports user-defined crop regions"""
    
    print("🎯 GPU-Accelerated Thermal Preprocessing with User-Defined Cropping")
    print("=" * 70)
    
    # Initialize GPU preprocessor
    gpu_preprocessor = GPUThermalPreprocessor(use_gpu=True)
    
    # Load data
    thermal_images, metadata = gpu_preprocessor.load_thermal_data(filepath)
    
    print(f"Shape of thermal data: {thermal_images.shape}")
    print(f"Data type: {thermal_images.dtype}")
    
    if metadata:
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # Test preprocessing with user-defined crop region
    print("\n" + "="*70)
    print("TESTING GPU-ACCELERATED PREPROCESSING WITH USER-DEFINED CROP")
    print("="*70)
    
    if user_crop_region:
        print(f"Using user-defined crop region: {user_crop_region}")
        crop_info = f"X=[{user_crop_region['x_range'][0]}:{user_crop_region['x_range'][1]}], Y=[{user_crop_region['y_range'][0]}:{user_crop_region['y_range'][1]}]"
        print(f"Crop coordinates: {crop_info}")
    else:
        print("Using automatic target detection")
    
    # Test single frame with user-defined crop region
    processed_frame = gpu_preprocessor.test_single_frame(thermal_images, 
                                                       frame_idx=-1,
                                                       user_crop_region=user_crop_region)
    
    # Show statistics
    original_frame = thermal_images[-1]
    print(f"\n📊 Frame Statistics:")
    print(f"Original frame stats:")
    print(f"  Min: {original_frame.min():.2f}, Max: {original_frame.max():.2f}")
    print(f"  Mean: {original_frame.mean():.2f}, Std: {original_frame.std():.2f}")
    
    print(f"Processed frame stats:")
    print(f"  Min: {processed_frame.min():.2f}, Max: {processed_frame.max():.2f}")
    print(f"  Mean: {processed_frame.mean():.2f}, Std: {processed_frame.std():.2f}")
    
    # Show crop region statistics
    if user_crop_region:
        # Get cropped region for statistics
        cropped_region = gpu_preprocessor.crop_target_region(processed_frame, 
                                                           bbox=None, 
                                                           user_crop_region=user_crop_region)
        
        crop_width = user_crop_region['x_range'][1] - user_crop_region['x_range'][0]
        crop_height = user_crop_region['y_range'][1] - user_crop_region['y_range'][0]
        
        print(f"\n🎯 User-Defined Crop Statistics:")
        print(f"  Crop dimensions: {crop_width} x {crop_height} pixels")
        print(f"  Cropped region shape: {cropped_region.shape}")
        print(f"  Cropped area: {cropped_region.size} pixels")
        print(f"  Cropped stats: min={cropped_region.min():.3f}, max={cropped_region.max():.3f}, mean={cropped_region.mean():.3f}")
    
    # Benchmark if dataset is large enough
    if len(thermal_images) >= 10:
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK")
        print("="*50)
        gpu_preprocessor.benchmark_gpu_vs_cpu(thermal_images, num_test_frames=10)
    
    return gpu_preprocessor, thermal_images, metadata


def compare_auto_vs_user_cropping(filepath: str):
    """Compare automatic detection vs user-defined cropping"""
    
    print("\n🔄 COMPARISON: Auto Detection vs User-Defined Cropping")
    print("=" * 60)
    
    # Initialize preprocessor
    config = GPUThermalConfig.get_config('rtx_4060_optimized')
    preprocessor = GPUThermalPreprocessor(config, use_gpu=True)
    
    # Load data
    thermal_images, metadata = preprocessor.load_thermal_data(filepath)
    
    # Test frame
    test_frame = thermal_images[-1]
    processed_frame = preprocessor._preprocess_single_frame(test_frame)
    
    # 1. Automatic detection
    print("\n🤖 Automatic target detection:")
    auto_bbox = preprocessor.detect_target_region(processed_frame)
    auto_cropped = preprocessor.crop_target_region(processed_frame, auto_bbox)
    print(f"Auto-detected region: X=[{auto_bbox[0]}:{auto_bbox[2]}], Y=[{auto_bbox[1]}:{auto_bbox[3]}]")
    print(f"Auto-cropped size: {auto_cropped.shape}")
    
    # 2. User-defined cropping
    print("\n👤 User-defined cropping:")
    user_crop_region = {'x_range': (14, 24), 'y_range': (10, 25)}
    user_cropped = preprocessor.crop_target_region(processed_frame, bbox=None, 
                                                 user_crop_region=user_crop_region)
    print(f"User-defined region: X=[{user_crop_region['x_range'][0]}:{user_crop_region['x_range'][1]}], "
          f"Y=[{user_crop_region['y_range'][0]}:{user_crop_region['y_range'][1]}]")
    print(f"User-cropped size: {user_cropped.shape}")
    
    # Visualize comparison
    visualize_cropping_comparison(processed_frame, auto_bbox, auto_cropped, 
                                user_crop_region, user_cropped)
    
    # Statistics comparison
    print(f"\n📊 Comparison Statistics:")
    print(f"Auto-detected area: {(auto_bbox[2] - auto_bbox[0]) * (auto_bbox[3] - auto_bbox[1])} pixels")
    user_area = (user_crop_region['x_range'][1] - user_crop_region['x_range'][0]) * \
                (user_crop_region['y_range'][1] - user_crop_region['y_range'][0])
    print(f"User-defined area: {user_area} pixels")
    
    print(f"Auto-detected temperature stats: min={auto_cropped.min():.2f}, max={auto_cropped.max():.2f}, mean={auto_cropped.mean():.2f}")
    print(f"User-defined temperature stats: min={user_cropped.min():.2f}, max={user_cropped.max():.2f}, mean={user_cropped.mean():.2f}")


def visualize_cropping_comparison(frame, auto_bbox, auto_cropped, user_crop_region, user_cropped):
    """Visualize automatic vs user-defined cropping side by side"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original frame with both crop regions
    im1 = axes[0, 0].imshow(frame, cmap='inferno')
    axes[0, 0].set_title('Original Frame with Crop Regions')
    
    # Auto-detected region (green)
    auto_rect = plt.Rectangle((auto_bbox[0], auto_bbox[1]), 
                             auto_bbox[2] - auto_bbox[0], 
                             auto_bbox[3] - auto_bbox[1],
                             fill=False, color='green', linewidth=2, label='Auto-detected')
    axes[0, 0].add_patch(auto_rect)
    
    # User-defined region (blue)
    x_range = user_crop_region['x_range']
    y_range = user_crop_region['y_range']
    user_rect = plt.Rectangle((x_range[0], y_range[0]), 
                             x_range[1] - x_range[0], 
                             y_range[1] - y_range[0],
                             fill=False, color='blue', linewidth=2, label='User-defined')
    axes[0, 0].add_patch(user_rect)
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Auto-detected crop only
    axes[0, 1].imshow(frame, cmap='inferno')
    axes[0, 1].add_patch(plt.Rectangle((auto_bbox[0], auto_bbox[1]), 
                                      auto_bbox[2] - auto_bbox[0], 
                                      auto_bbox[3] - auto_bbox[1],
                                      fill=False, color='green', linewidth=3))
    axes[0, 1].set_title('Auto-Detected Region')
    
    # User-defined crop only  
    axes[0, 2].imshow(frame, cmap='inferno')
    axes[0, 2].add_patch(plt.Rectangle((x_range[0], y_range[0]), 
                                      x_range[1] - x_range[0], 
                                      y_range[1] - y_range[0],
                                      fill=False, color='blue', linewidth=3))
    axes[0, 2].set_title('User-Defined Region')
    
    # Auto-cropped result
    im4 = axes[1, 0].imshow(auto_cropped, cmap='inferno')
    axes[1, 0].set_title(f'Auto-Cropped ({auto_cropped.shape[1]}x{auto_cropped.shape[0]})')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # User-cropped result
    im5 = axes[1, 1].imshow(user_cropped, cmap='inferno')
    axes[1, 1].set_title(f'User-Cropped ({user_cropped.shape[1]}x{user_cropped.shape[0]})')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Difference visualization (if same size)
    if auto_cropped.shape == user_cropped.shape:
        diff = np.abs(auto_cropped - user_cropped)
        im6 = axes[1, 2].imshow(diff, cmap='coolwarm')
        axes[1, 2].set_title('Absolute Difference')
        plt.colorbar(im6, ax=axes[1, 2])
    else:
        axes[1, 2].text(0.5, 0.5, f'Different Sizes:\nAuto: {auto_cropped.shape}\nUser: {user_cropped.shape}', 
                       ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Size Comparison')
    
    plt.tight_layout()
    plt.show()


def show_user_crop_statistics(results, user_crop_region):
    """Show statistics specific to user-defined cropping"""
    
    print("\n" + "="*50)
    print("USER-DEFINED CROP STATISTICS")
    print("="*50)
    
    # Crop region info
    x_range = user_crop_region['x_range']
    y_range = user_crop_region['y_range']
    crop_width = x_range[1] - x_range[0]
    crop_height = y_range[1] - y_range[0]
    crop_area = crop_width * crop_height
    
    print(f"\n🎯 Crop Region Definition:")
    print(f"  X range: {x_range[0]} to {x_range[1]} (width: {crop_width} pixels)")
    print(f"  Y range: {y_range[0]} to {y_range[1]} (height: {crop_height} pixels)")
    print(f"  Total crop area: {crop_area} pixels")
    
    # Original vs cropped size comparison
    original_data = results['original']
    original_height, original_width = original_data.shape[1], original_data.shape[2]
    original_area = original_height * original_width
    
    print(f"\n📏 Size Comparison:")
    print(f"  Original frame size: {original_width} x {original_height} ({original_area} pixels)")
    print(f"  Cropped region size: {crop_width} x {crop_height} ({crop_area} pixels)")
    print(f"  Crop ratio: {crop_area/original_area:.1%} of original")
    
    # Cropped frames statistics
    cropped_frames = results['cropped_frames']
    if cropped_frames:
        # All cropped frames should have the same size with user-defined cropping
        first_crop_shape = cropped_frames[0].shape
        consistent_size = all(frame.shape == first_crop_shape for frame in cropped_frames)
        
        print(f"\n📊 Cropped Data Statistics:")
        print(f"  Number of cropped frames: {len(cropped_frames)}")
        print(f"  All frames same size: {'✓ Yes' if consistent_size else '✗ No'}")
        print(f"  Cropped frame dimensions: {first_crop_shape}")
        
        # Temperature statistics across all cropped frames
        all_cropped_values = np.concatenate([frame.flatten() for frame in cropped_frames])
        print(f"  Temperature range in cropped regions: {all_cropped_values.min():.3f} - {all_cropped_values.max():.3f}")
        print(f"  Mean temperature in cropped regions: {all_cropped_values.mean():.3f}")
        print(f"  Std deviation in cropped regions: {all_cropped_values.std():.3f}")
    
    # Memory efficiency
    if 'processed_frames' in results:
        processed_frames = results['processed_frames']
        total_cropped_size = sum(frame.nbytes for frame in cropped_frames)
        total_processed_size = processed_frames.nbytes
        
        print(f"\n💾 Memory Efficiency:")
        print(f"  Original processed data: {total_processed_size / 1024**2:.1f} MB")
        print(f"  Cropped data: {total_cropped_size / 1024**2:.1f} MB")
        print(f"  Memory reduction: {(1 - total_cropped_size/total_processed_size)*100:.1f}%")


# MAIN USAGE EXAMPLES
def main_usage_examples():
    """Main function showing different ways to use user-defined cropping"""
    
    # filepath = r"C:\Users\hakim\Documents\cpe_dt\dataset\real\30x1x50_Speed_thermal_capture_20250721_214207.h5"
    filepath = r"C:\Users\hakim\Documents\cpe_dt\dataset\real\30x1x50_Structural_thermal_capture_20250721_195758.h5"
    # filepath = r"dataset\real\70x1x50_Speed_thermal_capture_20250721_220957.h5"
    # filepath = r"dataset\real\70x1x50_Structural_thermal_capture_20250721_202100.h5"
    
    print("🚀 USER-DEFINED CROP REGION TUTORIAL")
    print("=" * 60)
    print("This tutorial shows different ways to use user-defined crop regions")
    print()
    
    # Example 1: Basic usage with enhanced quick view
    print("📍 EXAMPLE 1: Enhanced Quick View with User-Defined Crop")
    print("-" * 50)
    
    # Define your crop region
    my_crop_region = {
        'x_range': (14, 24),   # X from pixel 14 to 24 30x1x50_Structure
        'y_range': (10, 25)    # Y from pixel 10 to 25 30x1x50_Structure
        # 'x_range':  (9, 19),     # X from pixel 14 to 24 30x1x50_Speed
        # 'y_range':  (4, 20)    # Y from pixel 10 to 25 30x1x50_Speed
    #   'x_range': (8, 29),   # X from pixel 14 to 24  70x1x50_Structure
    #    'y_range': (5, 20)    # Y from pixel 10 to 25 70x1x50_Structure
        # 'x_range': (5, 26),   # X from pixel 14 to 24 70x1x50_Speed
        # 'y_range': (6, 21)    # Y from pixel 10 to 25 70x1x50_Speed
   }
    
    # Use enhanced quick view with user-defined crop
    gpu_preprocessor, thermal_images, metadata = enhanced_quick_view_with_user_crop(
        filepath, 
        user_crop_region=my_crop_region
    )
    
    print("\n" + "="*60)
    input("Press Enter to continue to Example 2...")
    
    # # Example 2: Compare auto vs user cropping
    # print("\n📍 EXAMPLE 2: Compare Automatic vs User-Defined Cropping")
    # print("-" * 50)
    
    #compare_auto_vs_user_cropping(filepath)
    
    print("\n" + "="*60)
    choice = input("Do you want to proceed with full dataset processing? (y/n): ")
    
    if choice.lower() == 'y':
        # Example 3: Full dataset processing
        print("\n📍 EXAMPLE 3: Full Dataset Processing with User-Defined Crop")
        print("-" * 50)
        
        results = gpu_preprocessor.process_full_dataset_gpu(
            thermal_images, 
            user_crop_region=my_crop_region
        )
        
        # Save results
        output_path = filepath.replace('.h5', '_user_defined_crop_processed.h5')
        gpu_preprocessor.save_processed_data(results, output_path)
        
        print(f"\n✅ Processing complete! Results saved to: {output_path}")
        
        # Show comprehensive statistics
        show_user_crop_statistics(results, my_crop_region)
        
        return results
    
    return None


if __name__ == "__main__":
    # Run the main tutorial
    results = main_usage_examples()
    
    print("\n🎉 Tutorial Complete!")
    print("You now know how to use user-defined crop regions in thermal preprocessing!")