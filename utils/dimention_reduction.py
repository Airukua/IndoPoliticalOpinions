import umap.umap_ as umap

def reduce_dimensions_umap(data, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=None):
    """
    Reduce the dimensionality of high-dimensional data using UMAP.

    Parameters:
        data (array-like): Input data (e.g., NumPy array or TF-IDF matrix).
        n_components (int): Target number of dimensions (default: 2).
        n_neighbors (int): Number of nearest neighbors to consider for local structure.
        min_dist (float): Minimum distance between points in the low-dimensional space.
        metric (str): Distance metric to use ('euclidean', 'cosine', etc.).
        random_state (int or None): Seed for reproducibility.

    Returns:
        ndarray: Low-dimensional representation of the input data.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, data.shape[0] - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    return reducer.fit_transform(data)
