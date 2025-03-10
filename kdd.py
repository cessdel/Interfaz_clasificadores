import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def plot_decision_boundary(X, y, tree, metric):
    """Dibuja la región de decisión del clasificador KD-Tree"""
    h = 0.05  # Tamaño del paso en la malla
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    indices = tree.query(grid_points, k=1, return_distance=False)
    Z = np.array([y[idx][0] for idx in indices])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm", alpha=0.8)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"Región de Decisión - KDTree ({metric})")
    plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(y))
    st.pyplot(plt)


def run_kdtree(X, y, params):
    start_time = time.time()
    
    # Mapeo de nombres de métricas
    metric_map = {
        "euclidiana": "euclidean",
        "manhattan": "manhattan",
        "chebyshev": "chebyshev",
        "minkowski": "minkowski"
    }
    
    metric = metric_map.get(params.get("Métrica", "euclidiana").lower(), "euclidean")
    k_value = params.get("K", 3)
    tecnica_validacion = params.get("Técnica de validación", "Holdout")
    
    # Inicializar variables de resultados
    y_test, y_pred = None, None

    if tecnica_validacion == "Holdout":
        # División en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - params.get("Proporción", 0.8), random_state=42)
        tree = KDTree(X_train, metric=metric)
        indices = tree.query(X_test, k=k_value, return_distance=False)
        y_pred = np.array([np.bincount(y_train[idx]).argmax() for idx in indices])

    elif tecnica_validacion == "K-Fold":
        kf = KFold(n_splits=params.get("Folds", 10), shuffle=True, random_state=42)
        scores = []
        y_true_all, y_pred_all, X_test_all = [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test_fold = X[train_index], X[test_index]
            y_train, y_test_fold = y[train_index], y[test_index]
            
            tree = KDTree(X_train, metric=metric)
            indices = tree.query(X_test_fold, k=k_value, return_distance=False)
            y_pred_fold = np.array([np.bincount(y_train[idx]).argmax() for idx in indices])
            
            scores.append(accuracy_score(y_test_fold, y_pred_fold))
            y_true_all.extend(y_test_fold)
            y_pred_all.extend(y_pred_fold)
            X_test_all.extend(X_test_fold)
        
        accuracy = np.mean(scores)
        y_test, y_pred, X_test = np.array(y_true_all), np.array(y_pred_all), np.array(X_test_all)

    if y_test is None or y_pred is None:
        raise ValueError("Error: y_test o y_pred no están definidos correctamente.")

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    
    # Cálculo de especificidad
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  
    fp = cm.sum(axis=0) - np.diag(cm)  
    specificity = np.mean(tn / (tn + fp + 1e-7))  # Evita división por 0

    # Graficar matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión - KDTree ({tecnica_validacion})")
    st.pyplot(fig)

    # Graficar región de decisión si los datos son 2D
    if X.shape[1] == 2:
        plot_decision_boundary(X, y, tree, metric)

    return {
        "Clasificador": "KD-Tree",
        "Parámetros": params,
        "Exactitud": accuracy,
        "Precisión": precision,
        "Sensibilidad": recall,
        "Especificidad": specificity,
        "Tiempo de ejecución": time.time() - start_time
    }
