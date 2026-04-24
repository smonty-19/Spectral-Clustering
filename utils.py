"""
Utility functions for visualization, analysis, and reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


def visualize_spectral_clustering(X, y, labels, sc, feature_names, dataset_name=""):
    """
    Create comprehensive visualization of spectral clustering results.
    
    Plots:
    1. Original data with true labels (PCA 2D)
    2. Clustering results (PCA 2D)
    3. Original data (PCA 3D)
    4. Eigenvalue spectrum
    5. Eigenvalue gaps (pattern discovery)
    6. Projected data (eigenvector space)
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Spectral Clustering Results - {dataset_name} Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Project to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # =====================================================================
    # Plot 1: Original data with true labels (2D PCA)
    # =====================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                          edgecolors='black', s=80, alpha=0.7, linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Original Data\n(True Labels)', fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Class (0=Healthy, 1=Disease)')
    ax1.grid(True, alpha=0.3)
    
    # =====================================================================
    # Plot 2: Spectral clustering results (2D PCA)
    # =====================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma',
                          edgecolors='black', s=80, alpha=0.7, linewidth=0.5)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Spectral Clustering Results\n(Predicted Clusters)', fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Cluster')
    ax2.grid(True, alpha=0.3)
    
    # =====================================================================
    # Plot 3: Confusion visualization
    # =====================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Create confusion visualization
    colors_true = ['green' if label == 0 else 'red' for label in y]
    markers = ['o' if label == 0 else 's' for label in labels]
    
    for i in np.unique(labels):
        mask = labels == i
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c='blue' if i == 0 else 'orange',
                   marker='o' if i == 0 else 's',
                   s=80, alpha=0.6, edgecolors='black', linewidth=0.5,
                   label=f'Predicted Cluster {i}')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title('Cluster Visualization\n(Predicted)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # =====================================================================
    # Plot 4: Eigenvalue spectrum
    # =====================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    eigenvalues = sc.eigenvalues_
    ax4.plot(eigenvalues[:20], 'bo-', linewidth=2, markersize=7, label='Eigenvalues')
    ax4.axvline(x=sc.n_clusters, color='red', linestyle='--', linewidth=2,
               label=f'k={sc.n_clusters}')
    ax4.set_xlabel('Eigenvalue Index', fontweight='bold')
    ax4.set_ylabel('Eigenvalue (λ)', fontweight='bold')
    ax4.set_title('Eigenvalue Spectrum\n(First 20)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    
    # =====================================================================
    # Plot 5: Eigenvalue gaps (pattern discovery)
    # =====================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    gaps = np.diff(eigenvalues[:20])
    bars = ax5.bar(range(len(gaps)), gaps, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Highlight largest gap
    max_gap_idx = np.argmax(gaps)
    bars[max_gap_idx].set_color('red')
    
    ax5.set_xlabel('Gap Index', fontweight='bold')
    ax5.set_ylabel('Gap Size (λ_{i+1} - λ_i)', fontweight='bold')
    ax5.set_title('Eigenvalue Gaps\n(Pattern Discovery)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add text annotation
    ax5.text(max_gap_idx, gaps[max_gap_idx], f'Max\ngap={gaps[max_gap_idx]:.4f}',
            ha='center', va='bottom', fontweight='bold', color='red', fontsize=9)
    
    # =====================================================================
    # Plot 6: Projected data (eigenvector space)
    # =====================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    U = sc.projected_data_
    
    if U.shape[1] >= 2:
        scatter6 = ax6.scatter(U[:, 0], U[:, 1], c=labels, cmap='plasma',
                              edgecolors='black', s=80, alpha=0.7, linewidth=0.5)
        ax6.set_xlabel(f'Eigenvector 1 (λ_1={eigenvalues[0]:.4f})', fontweight='bold')
        ax6.set_ylabel(f'Eigenvector 2 (λ_2={eigenvalues[1]:.4f})', fontweight='bold')
        ax6.set_title('Projected Data Space\n(Eigenvector Basis)', fontweight='bold')
        cbar6 = plt.colorbar(scatter6, ax=ax6)
        cbar6.set_label('Cluster')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_clustering_heart_disease.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'spectral_clustering_heart_disease.png'")
    plt.show()

def print_medical_insights(X, y, labels, feature_names):
    """Print medical insights from clustering results."""
    print("\n" + "="*80)
    print("MEDICAL INSIGHTS FROM SPECTRAL CLUSTERING")
    print("="*80)
    
    print("\n" + "-"*80)
    print("CLUSTER-DISEASE RELATIONSHIP")
    print("-"*80)
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        disease_in_cluster = np.sum(y[cluster_mask])
        total_in_cluster = np.sum(cluster_mask)
        healthy_in_cluster = total_in_cluster - disease_in_cluster
        disease_ratio = disease_in_cluster / total_in_cluster * 100 if total_in_cluster > 0 else 0
        
        print(f"\nCluster {cluster_id}:")
        print(f"  - Total patients: {total_in_cluster}")
        print(f"  - Disease: {disease_in_cluster} ({disease_ratio:.1f}%)")
        print(f"  - Healthy: {healthy_in_cluster}")
        
        # Feature statistics for this cluster
        if feature_names and len(feature_names) > 0:
            print(f"  - Average measurements in this cluster:")
            for feat_idx, feat_name in enumerate(feature_names[:5]):  # First 5 features
                if feat_idx < X.shape[1]:
                    mean_value = np.mean(X[cluster_mask, feat_idx])
                    print(f"    • {feat_name}: {mean_value:.2f}")
    
    print("\n" + "-"*80)
    print("CLINICAL INTERPRETATION")
    print("-"*80)
    
    # Analyze if clusters align with disease status
    alignment = 0
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        disease_count = np.sum(y[cluster_mask])
        total_count = np.sum(cluster_mask)
        
        if disease_count > total_count / 2:
            alignment += total_count
    
    alignment_ratio = alignment / len(labels) * 100
    
    print(f"\nCluster-Disease Alignment: {alignment_ratio:.1f}%")
    if alignment_ratio > 70:
        print("  ✓ STRONG: Clusters well-separated by disease status")
    elif alignment_ratio > 50:
        print("  ⊘ MODERATE: Some separation between healthy and diseased patients")
    else:
        print("  ✗ WEAK: Clusters contain mixed healthy and diseased patients")
    
    print("\nInterpretation:")
    if alignment_ratio > 70:
        print("  • Spectral clustering successfully identifies disease-related patient subgroups")
        print("  • Patient physiological measurements naturally separate disease from healthy")
        print("  • Potential for clinical patient stratification")
    else:
        print("  • Disease is characterized by heterogeneous patient presentations")
        print("  • Multiple phenotypes within diseased and healthy populations")
        print("  • Clustering reveals subtypes not captured by binary disease status")
    
    print("\n" + "="*80)


def create_summary_report(X, y, labels, sc, feature_names, filename='report.txt'):
    """Create a text report of clustering results."""
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPECTRAL CLUSTERING - HEART DISEASE DATASET\n")
        f.write("Linear Algebra Mini Project Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Samples: {X.shape[0]} patients\n")
        f.write(f"Features: {X.shape[1]} medical measurements\n")
        f.write(f"Classes: 2 (Healthy/Disease)\n")
        f.write(f"Disease prevalence: {np.sum(y)/len(y)*100:.1f}%\n\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("-"*80 + "\n")
        for cluster_id in np.unique(labels):
            count = np.sum(labels == cluster_id)
            f.write(f"Cluster {cluster_id}: {count} patients\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Report saved as '{filename}'")