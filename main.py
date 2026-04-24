"""
Main Execution File - Spectral Clustering on Heart Disease Data

Complete mini-project pipeline following guidelines:
1. Load Real-World Data (Kaggle Heart Disease)
2. Apply Spectral Clustering (all intermediate steps)
3. End with Final Application Output (predictions, evaluation, insights)

This script ties together the complete linear algebra mini-project.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering as SklearnSpectralClustering

from data_loader import load_heart_disease_dataset
from spectral_clustering import SpectralClusteringHeartDisease, EvaluationMetrics
from utils import visualize_spectral_clustering, print_linear_algebra_concepts, print_medical_insights


def main():
    """Main execution function."""
    
    print("\n" + "#"*80)
    print("# SPECTRAL CLUSTERING - LINEAR ALGEBRA MINI PROJECT")
    print("# Heart Disease Patient Clustering")
    print("#"*80)
    
    # =========================================================================
    # STEP 1: LOAD REAL-WORLD DATA (Kaggle Heart Disease Dataset)
    # =========================================================================
    print("\n" + "="*80)
    print("LOADING KAGGLE HEART DISEASE DATASET")
    print("="*80)
    
    try:
        X, y, feature_names = load_heart_disease_dataset('heart.csv', verbose=True)
    except FileNotFoundError:
        print("\n" + "!"*80)
        print("DATASET NOT FOUND!")
        print("!"*80)
        return
    
    # =========================================================================
    # STEP 2: RUN SPECTRAL CLUSTERING PIPELINE
    # =========================================================================
    print("\n" + "="*80)
    print("APPLYING SPECTRAL CLUSTERING PIPELINE")
    print("="*80)
    
    # Initialize model
    sc = SpectralClusteringHeartDisease(
        n_clusters=2,
        sigma=1.0,
        use_normalized=False,
        verbose=True
    )
    
    # Fit model
    sc.fit(X, feature_names=feature_names)
    labels = sc.predict()
    
    # =========================================================================
    # STEP 3: FINAL APPLICATION OUTPUT - EVALUATION & INSIGHTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: FINAL APPLICATION OUTPUT - CLUSTERING EVALUATION")
    print("="*80)
    
    print("\n" + "-"*80)
    print("CLUSTERING QUALITY METRICS")
    print("-"*80)
    
    # Compute evaluation metrics
    silhouette = EvaluationMetrics.silhouette_score(X, labels)
    davies_bouldin = EvaluationMetrics.davies_bouldin_index(X, labels)
    ari = EvaluationMetrics.adjusted_rand_index(y, labels)
    purity = EvaluationMetrics.purity(y, labels)
    
    print(f"\nClustering Quality Metrics:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"    Range: [-1, 1] | Higher is better")
    print(f"    Interpretation: {'Good separation' if silhouette > 0.5 else 'Moderate separation' if silhouette > 0 else 'Poor separation'}")
    
    print(f"\n  - Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"    Range: [0, ∞] | Lower is better")
    print(f"    Interpretation: {'Good compactness' if davies_bouldin < 1.5 else 'Moderate compactness' if davies_bouldin < 2.5 else 'Poor compactness'}")
    
    print(f"\n  - Adjusted Rand Index (vs true labels): {ari:.4f}")
    print(f"    Range: [-1, 1] | 1 = perfect match with true labels")
    print(f"    Interpretation: {'High agreement' if ari > 0.5 else 'Moderate agreement' if ari > 0 else 'Low agreement'} with actual disease status")
    
    print(f"\n  - Purity: {purity:.4f}")
    print(f"    Range: [0, 1] | Higher is better")
    print(f"    Interpretation: {purity*100:.1f}% of patients clustered correctly")
    
    # Cluster distribution
    print(f"\n" + "-"*80)
    print("PATIENT CLUSTER DISTRIBUTION")
    print("-"*80)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(f"  - Cluster {cluster_id}: {count:3d} patients ({percentage:5.1f}%)")
        
        # Analyze disease distribution in cluster
        cluster_mask = labels == cluster_id
        disease_in_cluster = np.sum(y[cluster_mask])
        healthy_in_cluster = np.sum(~y[cluster_mask].astype(bool))
        print(f"    └─ Disease: {disease_in_cluster} | Healthy: {healthy_in_cluster}")
    
    # =========================================================================
    # COMPARISON WITH SKLEARN
    # =========================================================================
    print(f"\n" + "="*80)
    print("COMPARISON: Our Implementation vs Scikit-learn")
    print("="*80)
    
    sk_sc = SklearnSpectralClustering(n_clusters=2, affinity='rbf', random_state=42, n_init=10)
    sk_labels = sk_sc.fit_predict(X)
    
    sk_silhouette = EvaluationMetrics.silhouette_score(X, sk_labels)
    sk_davies_bouldin = EvaluationMetrics.davies_bouldin_index(X, sk_labels)
    sk_ari = EvaluationMetrics.adjusted_rand_index(y, sk_labels)
    sk_purity = EvaluationMetrics.purity(y, sk_labels)
    
    print(f"\nOur Implementation:")
    print(f"  - Silhouette: {silhouette:.4f}")
    print(f"  - Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"  - ARI: {ari:.4f}")
    print(f"  - Purity: {purity:.4f}")
    
    print(f"\nScikit-learn Implementation:")
    print(f"  - Silhouette: {sk_silhouette:.4f}")
    print(f"  - Davies-Bouldin: {sk_davies_bouldin:.4f}")
    print(f"  - ARI: {sk_ari:.4f}")
    print(f"  - Purity: {sk_purity:.4f}")
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print(f"\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualize_spectral_clustering(
        X=X,
        y=y,
        labels=labels,
        sc=sc,
        feature_names=feature_names,
        dataset_name="Heart Disease"
    )
    
    # =========================================================================
    # LINEAR ALGEBRA CONCEPTS
    # =========================================================================
    print_linear_algebra_concepts()
    
    # =========================================================================
    # MEDICAL INSIGHTS
    # =========================================================================
    print_medical_insights(X, y, labels, feature_names)
    
    # =========================================================================
    # PROJECT COMPLETION
    # =========================================================================
    print("\n" + "#"*80)
    print("# PROJECT COMPLETE")
    print("#"*80)

if __name__ == "__main__":
    main()