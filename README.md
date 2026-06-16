# Spectral Clustering for Heart Disease Data

A project that implements spectral clustering from scratch (plus a scikit-learn comparison) on the Heart Disease dataset.

This repository focuses on the mathematical flow behind clustering:

- matrix representation of patient data
- Gaussian affinity graph construction
- Laplacian matrix analysis
- eigendecomposition and orthogonal basis projection
- clustering in spectral subspace
- quantitative evaluation against ground truth labels

## Why This Project

This project is designed to demonstrate how abstract Linear Algebra concepts map directly to a practical medical data problem. Instead of treating clustering as a black-box API call, the pipeline exposes each transformation and reports interpretable intermediate results.

## Repository Structure

- data_loader.py: loads heart.csv, handles preprocessing, and standardizes features
- spectral_clustering.py: custom spectral clustering implementation and clustering metrics
- utils.py: plotting and reporting utilities
- main.py: end-to-end execution pipeline
- heart.csv: input dataset
- requirements.txt: project dependencies

## Mathematical Pipeline

The custom implementation in spectral_clustering.py follows this sequence:

1. Build similarity graph using Gaussian kernel:
	$$W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$
2. Build Laplacian (unnormalized by default):
	$$L = D - W$$
3. Solve eigenvalue problem:
	$$L v = \lambda v$$
4. Select first k eigenvectors and project data to spectral space
5. Apply K-means in projected space for final cluster assignment

The code also reports eigenvalue gaps to support cluster-structure interpretation.

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:

- numpy
- scipy
- scikit-learn
- matplotlib
- pandas

## Run

From the repository root:

```bash
python main.py
```

What main.py does:

1. loads and preprocesses heart.csv
2. runs custom spectral clustering
3. computes metrics: Silhouette, Davies-Bouldin, ARI, Purity
4. compares results with scikit-learn SpectralClustering
5. generates visualizations and summary insights

## Output Artifacts

Running the project generates:

- console report with full step-by-step pipeline diagnostics
- spectral_clustering_heart_disease.png (saved visualization)

Optional utility output in utils.py:

- report.txt via create_summary_report(...)

## Interpreting Results

You should review:

- cluster balance (patients per cluster)
- agreement with disease labels (ARI and purity)
- separation quality (silhouette and Davies-Bouldin)
- eigenvalue spectrum and spectral gaps

Together, these indicate whether disease-related subgroups are well-separated in spectral space.

## Output Generated
<img width="5333" height="3544" alt="spectral_clustering_heart_disease" src="https://github.com/user-attachments/assets/1be36ca0-2c46-4616-a751-7b5e0f44644d" />


## Notes

- Default cluster count is k = 2 (healthy vs disease-oriented grouping).
- Data is standardized before affinity construction.
- The implementation supports normalized Laplacian via use_normalized=True.

## Future Improvements

- hyperparameter sweep for sigma and k
- normalized vs unnormalized Laplacian benchmark table
- multiple datasets and external validation
- reproducible experiment logging

## License

This project is intended for academic and educational use.
