import time
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

def run_distancia_minima(X, y, params):
    start_time = time.time()
    
    # Mapeo de nombres de métricas
    metric_map = {
        "euclidiana": "euclidean",
        "manhattan": "manhattan",
        "chebyshev": "chebyshev"
    }
    
    metric = metric_map.get(params.get("Métrica", "euclidiana").lower(), "euclidean")
    tecnica_validacion = params.get("Técnica de validación", "Holdout")
    
    # Configuración de reducción de dimensionalidad
    if params.get("Reducción de dimensionalidad") == "PCA":
        pca = PCA(n_components=params.get("Componentes", 2))
        X = pca.fit_transform(X)
    elif params.get("Reducción de dimensionalidad") == "LDA":
        lda = LDA(n_components=params.get("Componentes", 1))
        X = lda.fit_transform(X, y)
    
    # Configuración de técnica de validación
    y_test, y_pred = None, None
    
    if tecnica_validacion == "Holdout":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - params.get("Proporción", 0.8), random_state=42)
        
        # Implementación de Distancia Mínima
        class_means = {label: X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)}
        y_pred = np.array([min(class_means, key=lambda c: np.linalg.norm(x - class_means[c])) for x in X_test])
    
    elif tecnica_validacion == "K-Fold":
        kf = KFold(n_splits=params.get("Folds", 10), shuffle=True, random_state=42)
        scores = []
        y_true_all, y_pred_all = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            class_means = {label: X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)}
            y_pred = np.array([min(class_means, key=lambda c: np.linalg.norm(x - class_means[c])) for x in X_test])
            
            scores.append(accuracy_score(y_test, y_pred))
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
        accuracy = np.mean(scores)
        y_test, y_pred = np.array(y_true_all), np.array(y_pred_all)
    
    if y_test is None or y_pred is None:
        raise ValueError("Error: y_test o y_pred no están definidos correctamente.")
    
    # Cálculo de métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Cálculo de especificidad
    cm = confusion_matrix(y_test, y_pred)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  # Verdaderos negativos
    fp = cm.sum(axis=0) - np.diag(cm)  # Falsos positivos
    specificity = np.mean(tn / (tn + fp + 1e-7))  # Evitar división por 0
    
    # Tiempo de ejecución
    execution_time = time.time() - start_time
    
    # Almacenar resultados en un JSON
    results = {
        "Clasificador": "Distancia Mínima",
        "Parámetros": params,
        "Exactitud": accuracy,
        "Precisión": precision,
        "Sensibilidad": recall,
        "Especificidad": specificity,
        "Tiempo de ejecución": execution_time
    }
    
    with open("results.json", "a") as f:
        json.dump(results, f, indent=4)
    
    return results