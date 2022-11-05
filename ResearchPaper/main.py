import typing as tp

import pandas as pd
from sklearn.base import ClusterMixin
import numpy as np
from sklearn.metrics import fowlkes_mallows_score, davies_bouldin_score
from sklearn.cluster import KMeans, SpectralClustering, BisectingKMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs, load_iris, load_wine, fetch_openml
from sklearn.model_selection import train_test_split
import os


def category_utility(X: np.ndarray, y_pred: np.ndarray, acuity: float = 1) -> float:
    # Page 40 of Gennari, Langley, Fisher's Models of Incremental Concept Formulation
    # recommends an acuity value of 1
    # https://www.sciencedirect.com/science/article/pii/0004370289900465?ref=cra_js_challenge&fr=RR-1
    num_clusters: int = np.unique(y_pred).size

    prob_clusters: np.ndarray = np.unique(y_pred, return_counts=True)[1] / y_pred.size
    overall_feature_std: np.ndarray = X.std(axis=0)

    overall_feature_std[np.logical_or(np.isnan(overall_feature_std),
                                      np.logical_or(np.isinf(overall_feature_std), overall_feature_std <= 0))] = acuity
    inv_overall_feature_std: np.ndarray = 1 / overall_feature_std
    inv_overall_feature_std[overall_feature_std <= acuity] = acuity

    norm: float = 1 / np.sqrt(np.pi)

    cluster_cu: np.ndarray = np.zeros((num_clusters,))

    for k in range(num_clusters):
        prob_cluster_k: float = prob_clusters[k]

        cluster_k: np.ndarray = X[y_pred == k, :]

        conditional_feature_std: np.ndarray = cluster_k.std(axis=0)

        conditional_feature_std[np.logical_or(np.isnan(conditional_feature_std),
                                              np.logical_or(np.isinf(conditional_feature_std),
                                                            conditional_feature_std <= 0))] = acuity

        inv_conditional_feature_std: np.ndarray = 1 / conditional_feature_std
        inv_conditional_feature_std[conditional_feature_std <= acuity] = acuity

        diff_std: np.ndarray = np.abs(inv_overall_feature_std - inv_conditional_feature_std)

        cluster_cu[k] = prob_cluster_k * norm * np.sum(diff_std)

    overall_category_utility: float = 1 / num_clusters * np.sum(cluster_cu)
    return overall_category_utility


def create_kmeans_model(num_clusters: int) -> KMeans:
    return KMeans(n_clusters=num_clusters, init='k-means++', random_state=13)


def create_spectral_model(num_clusters: int) -> SpectralClustering:
    return SpectralClustering(n_clusters=num_clusters, random_state=13, assign_labels='cluster_qr', eigen_solver='amg')


def create_agglomerative_model(num_clusters: int) -> AgglomerativeClustering:
    return AgglomerativeClustering(n_clusters=num_clusters)


def create_bisect_kmeans_model(num_clusters: int) -> BisectingKMeans:
    return BisectingKMeans(n_clusters=num_clusters, random_state=13, bisecting_strategy='largest_cluster')


def determine_best_num_clusters(cluster_gen_function: tp.Callable[[int], ClusterMixin], x: np.ndarray, y: np.ndarray,
                                min_clusters: int = 2, max_clusters: int = 20) -> tp.Dict[str, tp.Union[int, float]]:
    previous_category_utility: float = -1e10 + 1
    previous_fowlkes_mallows: float = -1e10 + 1
    previous_davies_bouldin: float = 1e10 - 1

    output: tp.Dict[str, tp.Union[int, float, bool]] = dict()

    for k in range(min_clusters, max_clusters):
        cluster_model: ClusterMixin = cluster_gen_function(k)

        y_pred: np.ndarray = cluster_model.fit_predict(x, y)

        current_category_utility: float = category_utility(x, y_pred)
        current_fowlkes_mallows: float = fowlkes_mallows_score(y, y_pred)
        current_davies_bouldin: float = davies_bouldin_score(x, y_pred)

        # Stop updating once you've found your first best one.
        if 'category_utility_score' not in output.keys() and previous_category_utility > current_category_utility:
            output['category_utility_recommendation'] = k
            output['category_utility_score'] = previous_category_utility
            output['category_utility_converged'] = True
        else:
            previous_category_utility = current_category_utility

        if 'fowlkes_mallows_score' not in output.keys() and previous_fowlkes_mallows > current_fowlkes_mallows:
            output['fowlkes_mallows_recommendation'] = k
            output['fowlkes_mallows_score'] = previous_fowlkes_mallows
            output['fowlkes_mallows_converged'] = True
        else:
            previous_fowlkes_mallows = current_fowlkes_mallows

        if 'davies_bouldin' not in output.keys() and previous_davies_bouldin < current_davies_bouldin:
            output['davies_bouldin_recommendation'] = k
            output['davies_bouldin_score'] = previous_davies_bouldin
            output['davies_bouldin_converged'] = True
        else:
            previous_davies_bouldin = current_davies_bouldin

        # Stop clustering when we've found our best ones.
        if len(output.keys()) == 9:
            break

    if 'category_utility_score' not in output.keys():
        output['category_utility_recommendation'] = max_clusters
        output['category_utility_score'] = previous_category_utility
        output['category_utility_converged'] = False

    if 'fowlkes_mallows_score' not in output.keys():
        output['fowlkes_mallows_recommendation'] = max_clusters
        output['fowlkes_mallows_score'] = previous_fowlkes_mallows
        output['fowlkes_mallows_converged'] = False

    if 'davies_bouldin_score' not in output.keys():
        output['davies_bouldin_recommendation'] = max_clusters
        output['davies_bouldin_score'] = previous_davies_bouldin
        output['davies_bouldin_converged'] = False

    return output


if __name__ == "__main__":
    wine_x, wine_y = load_wine(as_frame=False, return_X_y=True)
    iris_x, iris_y = load_iris(return_X_y=True)
    blob_x, blob_y = make_blobs(n_samples=1_000, n_features=2, centers=5, random_state=13, center_box=(-50, 50))
    mnist_x, mnist_y = fetch_openml(data_id=554, return_X_y=True)

    mnist_x = mnist_x.values / 255
    mnist_y = mnist_y.values

    mnist_x, _, mnist_y, _ = train_test_split(mnist_x, mnist_y, train_size=.25, random_state=13, shuffle=True)

    num_wine_classes: int = np.unique(wine_y).size
    num_iris_classes: int = np.unique(iris_y).size
    num_blob_classes: int = np.unique(blob_y).size
    num_mnist_classes: int = np.unique(mnist_y).size

    results_dl: tp.List[tp.Dict[str, tp.Union[str, bool, int, float]]] = []

    datasets: tp.List[tp.Tuple[str, int, np.ndarray, np.ndarray]] = [('Wine', num_wine_classes, wine_x, wine_y),
                                                                     ('Iris', num_iris_classes, iris_x, iris_y),
                                                                     ('Blobs', num_blob_classes, blob_x, blob_y),
                                                                     ('MNIST', num_mnist_classes, mnist_x, mnist_y)]

    models: tp.List[tp.Tuple[str, tp.Any]] = [('K-Means++', create_kmeans_model), ('Spectral', create_spectral_model),
                                              ('Agglomerative', create_agglomerative_model),
                                              ('Bisecting K-Means', create_bisect_kmeans_model)]

    models: tp.List[tp.Tuple[int, tp.Callable[[int], ClusterMixin]]] = [('K-Means++', create_kmeans_model),
                                                                        ('Spectral', create_spectral_model),
                                                                        ('Agglomerative', create_agglomerative_model),
                                                                        ('Bisecting K-Means',
                                                                         create_bisect_kmeans_model)]

    for dataset_name, num_classes, x, y in datasets:
        for model_name, model_gen_function in models:
            print(f'Trying out {model_name} model on the {dataset_name} dataset.')

            results: tp.Dict[str, tp.Union[int, bool, float]] = determine_best_num_clusters(model_gen_function, x, y)

            print(
                f"Did: Category Utility converge? {results['category_utility_converged']} | Fowlkes Mallows? {results['fowlkes_mallows_converged']} | Davies Bouldin {results['davies_bouldin_converged']}")

            results['dataset_name'] = dataset_name
            results['model_name'] = model_name
            results['true_num_classes'] = num_classes

            results_dl.append(results)

    save_path: str = os.path.join('.', 'alg_results.csv')

    print(f'Saving results to "{save_path}"...')

    pd.DataFrame.from_records(results_dl).to_csv(save_path, index=False)
