"""
Spectral Clustering Implementation for Heart Disease Dataset

This module implements complete spectral clustering with emphasis on
linear algebra concepts:

1. Matrix Representation (Raw Data)
2. Matrix Simplification (Gaussian Kernel - Affinity Matrix)
3. Structure of Space (Laplacian Matrix + Rank Analysis)
4. Remove Redundancy (Linear Dependencies)
5. Orthogonalization (Eigendecomposition - EIG problem)
6. Projection (Project to Eigenvector Subspace)
7. Prediction (K-means on Projected Data)
8. Pattern Discovery (Eigenvalue Analysis)

All steps emphasize mathematical foundation and medical interpretation.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


class SpectralClusteringHeartDisease:
    """
    Spectral Clustering specialized for heart disease patient clustering.
    
    Uses medical measurements to identify patient subgroups that may have
    different disease characteristics.
    """
    
    def __init__(self, n_clusters=2, sigma=1.0, use_normalized=False, verbose=True):
        """
        Initialize spectral clustering for heart disease data.
        
        Parameters:
        -----------
        n_clusters : int
            Number of patient clusters (default 2: disease vs no disease)
        sigma : float
            Bandwidth parameter for Gaussian kernel
        use_normalized : bool
            Use normalized Laplacian if True
        verbose : bool
            Print detailed step information
        """
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.use_normalized = use_normalized
        self.verbose = verbose
        
        # Results storage
        self.labels_ = None
        self.cluster_centers_ = None
        self.affinity_matrix_ = None
        self.laplacian_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.projected_data_ = None
        self.X_original_ = None
        self.feature_names_ = None
        
    def _print_step(self, step_num, title, description=""):
        """Helper to print steps with formatting."""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"STEP {step_num}: {title}")
            print(f"{'='*80}")
            if description:
                print(f"{description}")
    
    def _compute_affinity_matrix(self, X):
        """
        STEP 2: Matrix Simplification - Gaussian Kernel
        
        Transforms raw patient measurements into a similarity matrix.
        Similar patients (small measurement differences) have high affinity.
        
        Formula: W[i,j] = exp(-||X_i - X_j||^2 / (2*sigma^2))
        
        Linear Algebra Concepts:
        - Distance metric (Euclidean norm)
        - Exponential kernel (positive semi-definite matrix)
        - Graph representation of data
        """
        self._print_step(
            2, 
            "COMPUTE AFFINITY MATRIX (Gaussian Kernel)",
            "Create similarity matrix W: measures how similar each patient is to others"
        )
        
        # Compute pairwise Euclidean distances between all patients
        distances = cdist(X, X, metric='euclidean')
        
        if self.verbose:
            print(f"\nDistance Matrix Statistics:")
            print(f"  - Min distance: {np.min(distances[distances > 0]):.4f}")
            print(f"  - Max distance: {np.max(distances):.4f}")
            print(f"  - Mean distance: {np.mean(distances[distances > 0]):.4f}")
        
        # Apply Gaussian (RBF) kernel: higher similarity for closer patients
        self.affinity_matrix_ = np.exp(-distances**2 / (2 * self.sigma**2))
        
        if self.verbose:
            print(f"\nAffinity Matrix W = exp(-d²/2σ²) where σ={self.sigma}:")
            print(f"  - Shape: {self.affinity_matrix_.shape}")
            print(f"  - Diagonal (self-similarity): {np.diag(self.affinity_matrix_)[:5]} (should be ~1.0)")
            print(f"  - Min affinity: {np.min(self.affinity_matrix_[self.affinity_matrix_ < 1]):.6f}")
            print(f"  - Max affinity: {np.max(self.affinity_matrix_[self.affinity_matrix_ < 1]):.6f}")
            print(f"  - Mean affinity: {np.mean(self.affinity_matrix_[self.affinity_matrix_ < 1]):.6f}")
            print(f"  - Symmetric: {np.allclose(self.affinity_matrix_, self.affinity_matrix_.T)}")
            print(f"  - Positive Semi-Definite: Yes (Gaussian kernel property)")
        
        return self.affinity_matrix_
    
    def _compute_laplacian_matrix(self, W):
        """
        STEP 3: Structure of Space - Laplacian Matrix
        
        Creates the graph Laplacian from affinity matrix.
        The Laplacian encodes graph structure and reveals cluster boundaries.
        
        Unnormalized: L = D - W
        where D[i,i] = sum of row i of W (degree of node i)
        
        Linear Algebra Concepts:
        - Degree matrix (diagonal matrix)
        - Matrix subtraction
        - Graph theory (node degree)
        - Rank and null space
        """
        self._print_step(
            3,
            "COMPUTE LAPLACIAN MATRIX",
            "L = D - W: Encodes cluster structure through graph topology"
        )
        
        # Compute degree matrix
        degrees = np.sum(W, axis=1)
        D = np.diag(degrees)
        
        if self.verbose:
            print(f"\nDegree Matrix D (diagonal elements only):")
            print(f"  - Shape: {D.shape}")
            print(f"  - Degrees (first 10 patients): {degrees[:10]}")
            print(f"  - Min degree: {np.min(degrees):.4f}")
            print(f"  - Max degree: {np.max(degrees):.4f}")
            print(f"  - Mean degree: {np.mean(degrees):.4f}")
        
        if not self.use_normalized:
            # Unnormalized Laplacian
            self.laplacian_matrix_ = D - W
            lap_type = "UNNORMALIZED"
        else:
            # Normalized Laplacian: L_norm = I - D^(-1/2) * W * D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
            self.laplacian_matrix_ = np.eye(len(D)) - D_inv_sqrt @ W @ D_inv_sqrt
            lap_type = "NORMALIZED"
        
        if self.verbose:
            print(f"\nLaplacian Type: {lap_type}")
            print(f"  - Formula: L = D - W")
            print(f"  - Shape: {self.laplacian_matrix_.shape}")
            print(f"  - Symmetric: {np.allclose(self.laplacian_matrix_, self.laplacian_matrix_.T)}")
            
            # Rank analysis
            rank = np.linalg.matrix_rank(self.laplacian_matrix_)
            print(f"\nRank Analysis (Remove Redundancy):")
            print(f"  - Rank: {rank}")
            print(f"  - Max possible rank: {len(self.laplacian_matrix_)}")
            print(f"  - Rank deficiency: {len(self.laplacian_matrix_) - rank}")
            print(f"  - Interpretation: Rank deficiency indicates connected components")
            
            # Trace and diagonal properties
            print(f"\nLaplacian Properties:")
            print(f"  - Trace: {np.trace(self.laplacian_matrix_):.4f}")
            print(f"  - Diagonal elements: {np.diag(self.laplacian_matrix_)[:5]}")
        
        return self.laplacian_matrix_
    
    def _eigendecomposition(self, L):
        """
        STEP 4-5: Orthogonalization via Eigendecomposition
        
        Solves the symmetric eigenvalue problem: L*v = λ*v
        
        The eigenvectors form an orthonormal basis that reveals cluster structure.
        Small eigenvalues correspond to cluster structure.
        
        Linear Algebra Concepts:
        - Symmetric eigenvalue decomposition
        - Orthogonal eigenvectors (V^T*V = I)
        - Spectral theorem for symmetric matrices
        - Gram-Schmidt orthogonalization (implicit)
        - Basis transformation
        """
        self._print_step(
            4,
            "EIGENDECOMPOSITION (Orthogonalization)",
            "Solve L*v = λ*v: Find orthogonal directions revealing patient groupings"
        )
        
        # scipy.linalg.eigh: efficient symmetric eigendecomposition
        # Returns eigenvalues in ascending order with orthogonal eigenvectors
        eigenvalues, eigenvectors = eigh(L)
        
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        
        if self.verbose:
            print(f"\nEigenvalue Problem: L*v = λ*v")
            print(f"  - Number of eigenvalues: {len(eigenvalues)}")
            print(f"  - Smallest 10 eigenvalues: {eigenvalues[:10]}")
            print(f"  - Largest 5 eigenvalues: {eigenvalues[-5:]}")
            
            print(f"\nEigenvector Properties:")
            print(f"  - Shape: {eigenvectors.shape}")
            print(f"  - Each column is an eigenvector (patient direction)")
            
            # Verify orthogonality
            V = eigenvectors
            orthogonal_product = V.T @ V
            print(f"\nOrthogonality Verification (V^T * V ≈ I):")
            print(f"  - Diagonal sum: {np.trace(orthogonal_product):.4f} (should ≈ {len(V)})")
            off_diag_max = np.max(np.abs(orthogonal_product - np.eye(len(orthogonal_product))))
            print(f"  - Max off-diagonal: {off_diag_max:.2e} (should be ≈ 0)")
            print(f"  - Orthonormal: {'✓ YES' if off_diag_max < 1e-10 else '✗ NO'}")
            
            # Eigenvalue gap analysis (preview)
            gaps = np.diff(eigenvalues[:min(20, len(eigenvalues))])
            print(f"\nEigenvalue Gaps (λ_i+1 - λ_i):")
            print(f"  - First 10 gaps: {gaps[:10]}")
        
        return eigenvalues, eigenvectors
    
    def _find_optimal_clusters(self):
        """
        STEP 6: Pattern Discovery - Eigenvalue Gap Analysis
        
        The largest gap in eigenvalues indicates the optimal number of clusters.
        
        Medical Interpretation:
        - Large gap = clear separation between patient groups
        - Small gap = patient groups gradually blend
        
        Linear Algebra Concept: Spectral gap from graph Laplacian
        """
        self._print_step(
            5,
            "PATTERN DISCOVERY (Eigenvalue Gap Analysis)",
            "Identify natural patient groupings from eigenvalue spectrum"
        )
        
        gaps = np.diff(self.eigenvalues_[:min(30, len(self.eigenvalues_))])
        
        if self.verbose:
            print(f"\nSpectral Gap Analysis (Pattern Detection):")
            print(f"  - Eigenvalue gaps measure separation between clusters")
            print(f"  - Largest gap → optimal number of clusters")
            
            # Find top gaps
            top_gaps_idx = np.argsort(gaps)[-5:][::-1]
            print(f"\nTop 5 Eigenvalue Gaps:")
            for rank, idx in enumerate(top_gaps_idx, 1):
                gap_value = gaps[idx]
                lambda_k = self.eigenvalues_[idx]
                lambda_k1 = self.eigenvalues_[idx+1]
                print(f"  {rank}. Gap at k={idx+1}: {gap_value:.6f}")
                print(f"     λ_{idx} = {lambda_k:.6f} → λ_{idx+1} = {lambda_k1:.6f}")
            
            print(f"\nMedical Interpretation:")
            print(f"  - k=2: Two patient groups (likely disease vs no disease)")
            print(f"  - k>2: Patient subpopulations within disease categories")
    
    def _project_onto_eigenvectors(self):
        """
        STEP 5: Projection (Onto Eigenvector Subspace)
        
        Project data onto k-dimensional subspace spanned by k smallest eigenvectors.
        
        This dimensionality reduction makes clustering obvious while preserving
        the essential cluster structure information.
        
        Formula: U = V[:, :k]  where k = n_clusters
        
        Linear Algebra Concepts:
        - Orthogonal projection onto subspace
        - Dimensionality reduction
        - Basis and coordinate transformation
        - Subspace methods
        """
        self._print_step(
            6,
            "PROJECTION (Onto Eigenvector Subspace)",
            f"Project patients onto k={self.n_clusters} smallest eigenvectors"
        )
        
        # Select k smallest eigenvectors (corresponding to k smallest eigenvalues)
        U = self.eigenvectors_[:, :self.n_clusters]
        self.projected_data_ = U
        
        if self.verbose:
            print(f"\nProjection Matrix U = V[:, :k]:")
            print(f"  - Original space: {self.X_original_.shape[1]} dimensions ({len(self.feature_names_)} features)")
            print(f"  - Projected space: {self.n_clusters} dimensions")
            print(f"  - Projection matrix U shape: {U.shape}")
            
            print(f"\nSelected Eigenvalues (k smallest):")
            print(f"  - λ_1 to λ_{self.n_clusters}: {self.eigenvalues_[:self.n_clusters]}")
            print(f"  - Sum: {np.sum(self.eigenvalues_[:self.n_clusters]):.6f}")
            
            print(f"\nProjected Data Statistics:")
            print(f"  - Mean: {np.mean(U, axis=0)}")
            print(f"  - Std: {np.std(U, axis=0)}")
            print(f"  - First 5 projected patients:")
            for i in range(min(5, len(U))):
                print(f"    Patient {i}: {U[i]}")
    
    def _cluster_projected_data(self):
        """
        STEP 7: Prediction/Approximation using K-means
        
        Apply K-means clustering to the projected data.
        The projected space makes clustering straightforward because
        the cluster structure is now nearly linearly separable.
        
        Linear Algebra Concept: Linear separability in transformed space
        """
        self._print_step(
            7,
            "K-MEANS CLUSTERING (Prediction)",
            "Apply K-means to projected data to assign patients to clusters"
        )
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = kmeans.fit_predict(self.projected_data_)
        self.cluster_centers_ = kmeans.cluster_centers_
        
        if self.verbose:
            print(f"\nK-means Clustering Results:")
            print(f"  - Number of clusters: {self.n_clusters}")
            print(f"  - Algorithm: K-means (Lloyd's algorithm)")
            
            unique_labels, counts = np.unique(self.labels_, return_counts=True)
            print(f"\nCluster Assignments:")
            for label, count in zip(unique_labels, counts):
                percentage = count / len(self.labels_) * 100
                print(f"  - Cluster {label}: {count} patients ({percentage:.1f}%)")
            
            print(f"\nCluster Centers (in projected space):")
            print(self.cluster_centers_)
    
    def fit(self, X, feature_names=None):
        """
        Complete spectral clustering pipeline.
        
        Pipeline:
        1. Compute affinity matrix (Gaussian kernel) - Step 2
        2. Compute Laplacian matrix - Step 3
        3. Eigendecomposition - Step 4
        4. Find optimal clusters (pattern discovery) - Step 5
        5. Project onto eigenvectors - Step 6
        6. Apply K-means clustering - Step 7
        """
        self._print_step(0, "STARTING SPECTRAL CLUSTERING PIPELINE FOR HEART DISEASE DATA")
        
        # Store original data and feature names
        self.X_original_ = X.copy()
        self.feature_names_ = feature_names
        
        # Step 2: Compute affinity matrix
        self._compute_affinity_matrix(X)
        
        # Step 3: Compute Laplacian
        self._compute_laplacian_matrix(self.affinity_matrix_)
        
        # Step 4-5: Eigendecomposition
        self._eigendecomposition(self.laplacian_matrix_)
        
        # Step 6: Pattern discovery
        self._find_optimal_clusters()
        
        # Step 7: Project onto eigenvectors
        self._project_onto_eigenvectors()
        
        # Step 8: Cluster projected data
        self._cluster_projected_data()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("✓ SPECTRAL CLUSTERING PIPELINE COMPLETE")
            print(f"{'='*80}\n")
        
        return self
    
    def predict(self):
        """Return cluster labels for patients."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.labels_
    
    def get_eigenvalue_analysis(self):
        """Return eigenvalue analysis data for visualization."""
        return {
            'eigenvalues': self.eigenvalues_,
            'eigenvectors': self.eigenvectors_,
            'affinity_matrix': self.affinity_matrix_,
            'laplacian_matrix': self.laplacian_matrix_
        }


class EvaluationMetrics:
    """Evaluation metrics for clustering quality."""
    
    @staticmethod
    def silhouette_score(X, labels):
        """Silhouette score: -1 to 1, higher is better."""
        return silhouette_score(X, labels)
    
    @staticmethod
    def davies_bouldin_index(X, labels):
        """Davies-Bouldin index: lower is better."""
        return davies_bouldin_score(X, labels)
    
    @staticmethod
    def adjusted_rand_index(true_labels, pred_labels):
        """Adjusted Rand Index: -1 to 1, 1 is perfect match."""
        return adjusted_rand_score(true_labels, pred_labels)
    
    @staticmethod
    def purity(true_labels, pred_labels):
        """Clustering purity: 0 to 1, higher is better."""
        n = len(true_labels)
        w = np.zeros((len(np.unique(pred_labels)), len(np.unique(true_labels))))
        
        for i, pred in enumerate(np.unique(pred_labels)):
            for j, true in enumerate(np.unique(true_labels)):
                w[i, j] = np.sum((pred_labels == pred) & (true_labels == true))
        
        return np.sum(np.max(w, axis=1)) / n