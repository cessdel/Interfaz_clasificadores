import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
import seaborn as sns


def run_knn(X, y, params):
    start_time = time.time()
    
    # Mapeo de nombres de métricas y ponderación
    metric_map = {
        "euclidiana": "euclidean",
        "manhattan": "manhattan",
        "chebyshev": "chebyshev"
    }
    
    weight_map = {
        "uniforme": "uniform",
        "distancia": "distance"
    }
    
    metric = metric_map.get(params.get("Métrica", "euclidiana").lower(), "euclidean")
    weights = weight_map.get(params.get("Ponderación", "uniforme").lower(), "uniform")
    tecnica_validacion = params.get("Tecnica de validación", "Holdout")
    
    # Configuración de reducción de dimensionalidad
    if params.get("Reducción de dimensionalidad") == "PCA":
        pca = PCA(n_components=params.get("Componentes", 2))
        X = pca.fit_transform(X)
    elif params.get("Reducción de dimensionalidad") == "LDA":
        lda = LDA(n_components=params.get("Componentes", 1))
        X = lda.fit_transform(X, y)

    # ✅ Optimización de K si está activada
    if params.get("Optimizar valor de K", False):
        k_range = range(1, 21)  # Rango de valores de K a probar
        scores = []
        
        best_score = 0
        best_k = 3  # Valor por defecto

        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
            score = cross_val_score(knn, X, y, cv=5, scoring="accuracy").mean()  # Validación cruzada
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        params["K"] = best_k  # Guarda el mejor K encontrado
        st.info(f"Mejor valor de K encontrado: {best_k}")
        
        # Graficar
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, scores, marker='o', linestyle='-', color='b', label="Exactitud")
        plt.axvline(best_k, linestyle="--", color="r", label=f"Mejor K = {best_k}")
        plt.xlabel("Valor de K")
        plt.ylabel("Exactitud de validación cruzada")
        plt.title("Búsqueda del mejor valor de K")
        plt.xticks(np.arange(1, 21, step=1))
        plt.grid()
        
        # Mostrar gráfico en Streamlit
        st.pyplot(plt)

    
    # ✅ Si no se optimiza, usa el valor manual de K
    k_value = params.get("K", 3)

    # Configuración de técnica de validación
    y_test, y_pred = None, None
    
    if tecnica_validacion == "Holdout":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - params.get("Proporción", 0.8), random_state=42)
        model = KNeighborsClassifier(n_neighbors=k_value, metric=metric, weights=weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    elif tecnica_validacion == "K-Fold":
        kf = KFold(n_splits=params.get("Folds", 10), shuffle=True, random_state=42)
        scores = []
        y_true_all, y_pred_all, X_test_all = [], [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test_fold = X[train_index], X[test_index]
            y_train, y_test_fold = y[train_index], y[test_index]
            model = KNeighborsClassifier(n_neighbors=k_value, metric=metric, weights=weights)
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test_fold)
            scores.append(accuracy_score(y_test_fold, y_pred_fold))
            y_true_all.extend(y_test_fold)
            y_pred_all.extend(y_pred_fold)
            X_test_all.extend(X_test_fold)
        accuracy = np.mean(scores)
        y_test, y_pred, X_test = np.array(y_true_all), np.array(y_pred_all), np.array(X_test_all)
    
    if y_test is None or y_pred is None or X_test is None:
        raise ValueError("Error: X_test, y_test o y_pred no están definidos correctamente.")
    
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
    
    # Graficar matriz de confusión
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)
    
    # Graficar las clases con los puntos del conjunto de prueba y sus áreas sombreadas
    if X.shape[1] == 2:
        fig, ax = plt.subplots()
        for i, label in enumerate(np.unique(y_test)):
            mask = y_test == label
            ax.scatter(X_test[mask, 0], X_test[mask, 1], label=f'Clase {label}', alpha=0.6, edgecolor='k')
            sns.kdeplot(x=X_test[mask, 0], y=X_test[mask, 1], ax=ax, levels=5, fill=True, alpha=0.3)
        
        ax.set_title("Distribución de Clases en el Conjunto de Prueba")
        ax.legend()
        st.pyplot(fig)
    
    # Almacenar resultados en un JSON
    results = {
        "Clasificador": "KNN",
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
