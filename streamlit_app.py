import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import (
    make_classification,
    make_blobs,
    make_moons,
    make_circles
)
import numpy as np

# Custom dataset generators
def make_xor(n_samples=200, noise=0.1):
    X = np.random.rand(n_samples, 2)
    y = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)
    if noise > 0:
        X += noise * np.random.randn(*X.shape)
    return X, y.astype(int)

def make_spirals(n_samples=100, noise=0.2):
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    return X, y.astype(int)

def make_checkerboard(n_samples=200, noise=0.05):
    x = np.random.rand(n_samples) * 2 - 1
    y = np.random.rand(n_samples) * 2 - 1
    labels = ((x > 0) ^ (y > 0)).astype(int)
    X = np.stack([x, y], axis=1) + noise * np.random.randn(n_samples, 2)
    return X, labels

# Dataset generator map
dataset_generators = {
    "Linear": lambda: make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1),
    "XOR": lambda: make_xor(n_samples=200, noise=0.1),
    "Moons": lambda: make_moons(n_samples=200, noise=0.1),
    "Circles": lambda: make_circles(n_samples=200, noise=0.05, factor=0.5),
    "Spirals": lambda: make_spirals(n_samples=100, noise=0.5),
    "Blobs": lambda: make_blobs(n_samples=200, centers=3, cluster_std=1.0),
    "Varied Blobs": lambda: make_blobs(n_samples=200, centers=3, cluster_std=[1.0, 2.5, 0.5]),
    "Anisotropic": lambda: make_blobs(n_samples=200, centers=1, cluster_std=2.5),
    "Tight Circles": lambda: make_circles(n_samples=200, noise=0.02, factor=0.7),
    "Wide Circles": lambda: make_circles(n_samples=200, noise=0.1, factor=0.2),
    "Double Moons": lambda: make_moons(n_samples=200, noise=0.2),
    "Checkerboard": lambda: make_checkerboard(n_samples=200, noise=0.1),
    "Noisy Linear": lambda: make_classification(n_samples=200, n_features=2, n_informative=1, n_redundant=1, flip_y=0.2),
    "Overlapping Blobs": lambda: make_blobs(n_samples=200, centers=2, cluster_std=2.5),
    "Offset Moons": lambda: (make_moons(n_samples=200, noise=0.1)[0] + [1.5, 0.5], make_moons(n_samples=200, noise=0.1)[1]),
}

# Streamlit app
st.title("ML Toy Dataset Visualizer")
dataset_name = st.selectbox("Choose a dataset pattern", list(dataset_generators.keys()))

# Generate and plot dataset
X, y = dataset_generators[dataset_name]()
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)
ax.set_title(dataset_name)
ax.set_xticks([])
ax.set_yticks([])
st.pyplot(fig)

st.markdown("Built with ❤️ for ML pattern mastery")
