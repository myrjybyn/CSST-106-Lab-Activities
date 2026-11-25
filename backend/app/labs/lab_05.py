"""LAB 05: LFW Eigenfaces, whitened PCA, and SVM face recognition."""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def png_data(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor="#fffaf3")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_experiment(data_home):
    """Fetch/cache LFW, train the specified model, and return reviewable outputs."""
    from sklearn.datasets import fetch_lfw_people
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                           data_home=str(data_home), download_if_missing=True)
    images = lfw.images
    X = lfw.data
    y = lfw.target
    names = lfw.target_names
    n_samples, height, width = images.shape
    X_train, X_test, y_train, y_test, _, images_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42)
    n_components = min(150, X_train.shape[0] - 1, X_train.shape[1])
    if n_components < 1:
        raise ValueError("LFW did not provide enough training images for PCA.")

    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=42)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    classifier = SVC(kernel="rbf", class_weight="balanced")
    classifier.fit(X_train_pca, y_train)
    predictions = classifier.predict(X_test_pca)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    reached = np.flatnonzero(cumulative >= 0.9)
    components_90 = int(reached[0] + 1) if reached.size else None

    rng = np.random.default_rng(42)
    sample_indices = []
    for label in range(len(names)):
        candidates = np.flatnonzero(y == label)
        if candidates.size:
            sample_indices.append(int(rng.choice(candidates)))
    sample_indices = sample_indices[:12]
    columns = min(4, max(1, len(sample_indices)))
    rows = (len(sample_indices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.5 * rows), squeeze=False)
    for ax, index in zip(axes.flat, sample_indices):
        ax.imshow(images[index], cmap="gray")
        ax.set_title(str(names[y[index]]), fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(sample_indices):]:
        ax.axis("off")
    fig.suptitle("LFW sample faces (one per included identity)")
    sample_gallery = png_data(fig)

    eigenfaces = pca.components_.reshape((-1, height, width))
    chosen = list(range(min(12, n_components)))
    chosen += list(range(max(12, n_components - 3), n_components))
    chosen = list(dict.fromkeys(chosen))
    cols = 5
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.2 * rows), squeeze=False)
    for ax, index in zip(axes.flat, chosen):
        ax.imshow(eigenfaces[index], cmap="gray")
        ax.set_title(f"Eigenface {index + 1}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    fig.suptitle("First 12 and last 3 fitted eigenfaces")
    eigenface_gallery = png_data(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    component_numbers = np.arange(1, n_components + 1)
    ax.plot(component_numbers, cumulative, color="#76519d", linewidth=2)
    ax.axhline(0.9, color="#b34c86", linestyle="--", label="90% variance")
    if components_90 is not None:
        ax.axvline(components_90, color="#b34c86", linestyle=":", label=f"{components_90} components")
        ax.scatter([components_90], [cumulative[components_90 - 1]], color="#b34c86", zorder=3)
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
           xlim=(1, n_components), ylim=(0, 1.02), title="Variance retained by the first 150 components")
    ax.grid(alpha=0.2)
    ax.legend()
    variance_plot = png_data(fig)

    report = classification_report(y_test, predictions, labels=np.arange(len(names)),
                                   target_names=[str(name) for name in names],
                                   output_dict=True, zero_division=0)
    incorrect = np.flatnonzero(predictions != y_test)
    wrong_gallery = None
    wrong_items = []
    if incorrect.size:
        selected = incorrect[:12]
        columns = 4
        rows = (len(selected) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.7 * rows), squeeze=False)
        for ax, index in zip(axes.flat, selected):
            ax.imshow(images_test[index], cmap="gray")
            actual, predicted = str(names[y_test[index]]), str(names[predictions[index]])
            ax.set_title(f"True: {actual}\nPred: {predicted}", fontsize=7)
            ax.axis("off")
            wrong_items.append({"actual": actual, "predicted": predicted})
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        fig.suptitle("Misclassified test faces (up to 12)")
        wrong_gallery = png_data(fig)

    return {
        "dataset": {"samples": int(n_samples), "features": int(X.shape[1]),
                    "height": int(height), "width": int(width), "classes": int(len(names)),
                    "class_names": [str(name) for name in names],
                    "train_samples": int(len(y_train)), "test_samples": int(len(y_test))},
        "pca": {"components": int(n_components), "whiten": True,
                "explained_variance": [float(value) for value in cumulative],
                "variance_90_components": components_90,
                "variance_at_150": float(cumulative[-1])},
        "classifier": {"kernel": "rbf", "class_weight": "balanced",
                       "accuracy": float(accuracy_score(y_test, predictions)),
                       "misclassified": int(incorrect.size),
                       "report": report, "misclassified_faces": wrong_items},
        "plots": {"sample_faces": sample_gallery, "eigenfaces": eigenface_gallery,
                  "variance": variance_plot, "failures": wrong_gallery},
    }
