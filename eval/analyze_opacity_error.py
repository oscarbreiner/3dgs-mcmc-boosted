import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import seaborn as sns
import os

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def read_ply_attributes(ply_path):
    """Read opacity and error_contrib from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    
    # Opacity is stored in inverse sigmoid (logit) space, convert to [0,1]
    opacity_raw = np.asarray(vertex_data['opacity'])
    opacity = sigmoid(opacity_raw)
    
    error_contrib = np.asarray(vertex_data['error_contrib'])
    
    return opacity, error_contrib

def compute_correlation_matrix(opacity, error_contrib):
    """Compute correlation matrix between opacity and error contribution."""
    # Stack into a 2D array where each column is a variable
    data = np.column_stack([opacity, error_contrib])
    
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(data.T)
    
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix, save_path=None):
    """Plot correlation matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    
    labels = ['Opacity', 'Error Contribution']
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='coolwarm', 
                xticklabels=labels, 
                yticklabels=labels,
                vmin=-1, 
                vmax=1,
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Correlation Matrix: Opacity vs Error Contribution')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {save_path}")
    
    plt.show()

def plot_histograms(opacity, error_contrib, save_path=None):
    """Plot histograms for opacity and error contribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Opacity histogram
    axes[0].hist(opacity, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Opacity', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Histogram of Opacity Values', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(opacity.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {opacity.mean():.4f}')
    axes[0].legend()
    
    # Error contribution histogram
    axes[1].hist(error_contrib, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Error Contribution', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Histogram of Error Contribution Values', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(error_contrib.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {error_contrib.mean():.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histograms to {save_path}")
    
    plt.show()

def main():
    # Configuration
    use_dead_gaussians = False  # Set to False to filter out dead Gaussians (opacity < 0.05)
    ignore_zero_values = True  # Set to False to include zero error contribution values
    
    # Define path to PLY file
    ply_path = "output/bicycle_random/point_cloud/iteration_3000/point_cloud.ply"
    
    # Check if file exists
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        print("Please update the path in the script to point to your output folder.")
        return
    
    print(f"Reading PLY file: {ply_path}")
    
    # Read data
    opacity, error_contrib = read_ply_attributes(ply_path)
    
    print(f"\nLoaded {len(opacity)} Gaussians")
    print(f"Opacity range: [{opacity.min():.4f}, {opacity.max():.4f}]")
    print(f"Error contribution range: [{error_contrib.min():.4f}, {error_contrib.max():.4f}]")
    
    # Filter out dead Gaussians if requested
    if not use_dead_gaussians:
        alive_mask = opacity >= 0.05
        n_dead = np.sum(~alive_mask)
        opacity = opacity[alive_mask]
        error_contrib = error_contrib[alive_mask]
        print(f"\nFiltered out {n_dead} dead Gaussians (opacity < 0.05)")
        print(f"Remaining {len(opacity)} alive Gaussians")
        print(f"Opacity range (alive): [{opacity.min():.4f}, {opacity.max():.4f}]")
        print(f"Error contribution range (alive): [{error_contrib.min():.4f}, {error_contrib.max():.4f}]")
    
    # Filter out zero error contribution values if requested
    if ignore_zero_values:
        nonzero_mask = error_contrib >= 0.05
        n_zero = np.sum(~nonzero_mask)
        opacity = opacity[nonzero_mask]
        error_contrib = error_contrib[nonzero_mask]
        print(f"\nFiltered out {n_zero} Gaussians with zero error contribution")
        print(f"Remaining {len(opacity)} Gaussians with non-zero error contribution")
        print(f"Opacity range (non-zero): [{opacity.min():.4f}, {opacity.max():.4f}]")
        print(f"Error contribution range (non-zero): [{error_contrib.min():.4f}, {error_contrib.max():.4f}]")
    
    # Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(opacity, error_contrib)
    print(f"\nCorrelation coefficient: {correlation_matrix[0, 1]:.4f}")
    
    # Plot correlation heatmap
    plot_correlation_heatmap(correlation_matrix)
    
    # Plot histograms
    plot_histograms(opacity, error_contrib)

if __name__ == "__main__":
    main()
