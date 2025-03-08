import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from distancia_minima import run_distancia_minima
from knn import run_knn

# Título del proyecto
st.title("Comparador de Clasificadores de Machine Learning")

# Descripción del proyecto
st.markdown(
    """
    Esta aplicación permite experimentar con distintos clasificadores de Machine Learning
    utilizando archivos CSV personalizados. Se pueden seleccionar múltiples clasificadores,
    ajustar sus parámetros y comparar métricas de desempeño como precisión, exactitud,
    sensibilidad, especificidad y tiempo de ejecución.
    
    ### Instrucciones: 
    1. Cargue un archivo CSV con datos.
    2. Elija los clasificadores a probar y ajuste sus parámetros.
    3. Compare los resultados obtenidos.
    """
)

st.divider()

# Carga de archivo CSV
st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo CSV", type=["csv"])


# Opción en el sidebar para mostrar estadísticas
display_stats = st.sidebar.checkbox("Ver detalles y estadística", value=True)

# Selección de clasificadores en el sidebar
st.sidebar.header("Seleccionar clasificadores a comparar")
classifiers = ["KNN", "Distancia Mínima", "SVM", "Random Forest", "Red Neuronal"]
selected_classifiers = st.sidebar.multiselect("Seleccione los clasificadores", classifiers)

# Estado global
if "classifier_index" not in st.session_state:
    st.session_state.classifier_index = 0
if "classifier_params" not in st.session_state:
    st.session_state.classifier_params = {}
if "experiment_ready" not in st.session_state:
    st.session_state.experiment_ready = False
if "results" not in st.session_state:
    st.session_state.results = []

# Función para reiniciar el proceso
def reset_experiment():
    st.session_state.classifier_index = 0
    st.session_state.classifier_params = {}
    st.session_state.experiment_ready = False
    st.session_state.results = []
    st.rerun()

# Configuración de clasificadores en secuencia
if selected_classifiers and not st.session_state.experiment_ready:
    # Número de clases y características
    num_classes = st.session_state["num_classes"]
    num_features = st.session_state["num_features"]

    # Máximo permitido para LDA
    max_components_lda = max(1, min(num_classes - 1, num_features))
    
    if st.session_state.classifier_index < len(selected_classifiers):
        current_classifier = selected_classifiers[st.session_state.classifier_index]
        st.write(f"### Configuración de {current_classifier}")
        
        params = {}
        if current_classifier == "KNN":
            params["Optimizar valor de K"] = st.checkbox("Optimizar valor de K", value=True)
            if not params["Optimizar valor de K"]:
                params["K"] = st.number_input("Número de vecinos (K)", min_value=1, value=3)
            params["Métrica"] = st.selectbox("Métrica de distancia", ["Euclidiana", "Manhattan", "Chebyshev"])
            params["Ponderación"] = st.selectbox("Ponderación de vecinos", ["Uniforme", "Distancia"])
            params["Tecnica de validación"] = st.selectbox("Técnica de validación", ["Holdout", "K-Fold"])
            if params["Tecnica de validación"] == "K-Fold":
                params["Folds"] = st.slider("Número de Folds", min_value=2, max_value=20, value=10)
            elif params["Tecnica de validación"] == "Holdout":
                params["Proporción"] = st.number_input("Proporción de datos de entrenamiento", min_value=0.1, max_value=0.9, value=0.8)

            # Reducción de dimensionalidad
            params["Reducción de dimensionalidad"] = st.selectbox("Reducción de dimensionalidad", ["Ninguna", "PCA", "LDA"])

            if params["Reducción de dimensionalidad"] == "PCA":
                params["Componentes"] = st.slider("Número de componentes para PCA", min_value=1, max_value=num_features, value=2)
                params["Varianza"] = st.number_input("Varianza mínima a explicar", min_value=0.0, max_value=1.0, value=0.95)

            elif params["Reducción de dimensionalidad"] == "LDA":
                params["Número de clases"] = num_classes  # Determinar automáticamente
                params["Componentes"] = max_components_lda  # Ajuste automático
                st.write(f"LDA se aplicará con {max_components_lda} componentes (clases - 1)")
                
        # Clasificador de Distancia Mínima       
        elif current_classifier == "Distancia Mínima":
            params["Métrica"] = st.selectbox("Métrica de distancia", ["Euclidiana", "Manhattan", "Chebyshev"])
            params["Tecnica de validación"] = st.selectbox("Técnica de validación", ["Holdout", "K-Fold"])
            if params["Tecnica de validación"] == "K-Fold":
                params["Folds"] = st.slider("Número de Folds", min_value=2, value=10)
            elif params["Tecnica de validación"] == "Holdout":
                params["Proporción"] = st.number_input("Proporción de datos de entrenamiento", min_value=0.1, max_value=0.9, value=0.8)
            params["Reducción de dimensionalidad"] = st.selectbox("Reducción de dimensionalidad", ["Ninguna", "PCA", "LDA"])
            if params["Reducción de dimensionalidad"] == "PCA":
                params["Componentes"] = st.number_input("Número de componentes para PCA", min_value=1, max_value=num_features, value=2)
                params["Varianza"] = st.number_input("Varianza mínima a explicar", min_value=0.0, max_value=1.0, value=0.95)
            elif params["Reducción de dimensionalidad"] == "LDA":
                params["Número de clases"] = num_classes  # Determinar automáticamente
                params["Componentes"] = max_components_lda  # Ajuste automático
                st.write(f"LDA se aplicará con {max_components_lda} componentes (clases - 1)")
            
            
        elif current_classifier == "SVM":
            params["Kernel"] = st.selectbox("Tipo de kernel", ["Lineal", "Polinomial", "RBF"])
            params["C"] = st.number_input("Valor de C", min_value=0.1, value=1.0)
        elif current_classifier == "Random Forest":
            params["Árboles"] = st.number_input("Número de árboles", min_value=1, value=100)
            params["Profundidad"] = st.number_input("Profundidad máxima", min_value=1, value=10)
        elif current_classifier == "Red Neuronal":
            params["Capas"] = st.number_input("Número de capas ocultas", min_value=1, value=3)
            params["Neuronas"] = st.number_input("Neuronas por capa", min_value=1, value=64)
        
        if st.button("Guardar configuración"):
            st.session_state.classifier_params[current_classifier] = params
            st.session_state.classifier_index += 1
            st.success(f"Configuración de {current_classifier} guardada correctamente.")
            st.rerun()
    
    else:
        st.session_state.experiment_ready = True
        st.rerun()
        
        
# Ejecución de experimentos
def run_experiments():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1].values  # Características
        y = df.iloc[:, -1].values   # Variable objetivo
        
        results = []
        for classifier, params in st.session_state.classifier_params.items():
            if classifier == "KNN":
                result = run_knn(X, y, params)
            elif classifier == "Distancia Mínima":
                result = run_distancia_minima(X, y, params)
            # Aquí se agregarán más clasificadores en el futuro
            results.append(result)
        
        # Guardar resultados en JSON
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        st.session_state.results = results
        st.success("Experimentos completados con éxito.")
        
# Mostrar configuraciones guardadas en tabla bonita y opciones finales
if st.session_state.experiment_ready:
    st.write("### Configuraciones Guardadas")
    st.info("Puede reiniciar la selección de clasificadores o empezar los experimentos.")
    st.warning("Revise bien las configuraciones antes de continuar.")
    df_params = pd.DataFrame.from_dict(st.session_state.classifier_params, orient="index")
    
    # Aplicar estilo bonito a la tabla
    st.markdown(
        df_params.style.set_table_styles(
            [
                {"selector": "thead", "props": "background-color: #4CAF50; color: white; font-weight: bold; text-align: center;"},
                {"selector": "td", "props": "text-align: center;"},
                {"selector": "th", "props": "text-align: center;"},
                {"selector": "tr:nth-child(even)", "props": "background-color: #f2f2f2;"},
                {"selector": "tr:hover", "props": "background-color: #ddd;"}
            ]
        ).to_html(), unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reiniciar selección"):
            reset_experiment()
    with col2:
        if st.button("Empezar experimentos"):
            st.write("Iniciando experimentos...")
            run_experiments()
            
# Mostrar resultados si ya se han ejecutado
if st.session_state.results:
    st.markdown("## 📊 Resultados de la Experimentación")
    
    # Convertir resultados en DataFrame
    df_results = pd.DataFrame(st.session_state.results)

    # Mostrar resultados con mejor formato
    st.dataframe(df_results, use_container_width=True, height=400)

    # Agregar opción para descargar los resultados
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar resultados en CSV",
        data=csv,
        file_name="resultados_experimentacion.csv",
        mime="text/csv",
    )





if uploaded_file is not None:
    
    if display_stats:
        # Carga de datos y visualización
        df = pd.read_csv(uploaded_file)
        
        st.write("### Vista previa de los datos:")
        st.write(df.head())
        # Normalización de datos (excepto la última columna que es la variable objetivo)
        scaler = MinMaxScaler()
        features = df.iloc[:, :-1]
        df.iloc[:, :-1] = scaler.fit_transform(features)
        # Detectar número de clases en la variable objetivo
        target_column = df.iloc[:, -1]  # Última columna como variable objetivo
        num_classes = len(np.unique(target_column))  # Contar clases únicas
        num_features = df.shape[1] - 1  # Número de características (sin contar la variable objetivo)
        # Guardar en `st.session_state`
        st.session_state["num_classes"] = num_classes
        st.session_state["num_features"] = num_features
            
        # Máximo permitido para LDA
        max_components_lda = max(1, min(num_classes - 1, num_features))  # Asegurar que no exceda las características o clases
        # Transformación de la variable objetivo a 0,1,2...
        label_encoder = LabelEncoder()
        df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])
        
        st.write("### Datos normalizados:")
        st.write(df.head())
        # Visualización interactiva de la distribución de datos por filas con animaciones
        st.write("### Distribución de los datos normalizados por filas")
        df_long = df.melt(id_vars=df.columns[-1], var_name="Característica", value_name="Valor")
        fig = px.histogram(df_long, x="Valor", color="Característica", nbins=30, 
                           animation_frame="Característica", title="Distribución de los Datos por Filas")
        st.plotly_chart(fig)
        
        # Cálculo de estadísticas descriptivas
        st.write("### Estadísticas Descriptivas")
        stats_df = pd.DataFrame({
            "Media": df.iloc[:, :-1].mean(),
            "Mediana": df.iloc[:, :-1].median(),
            "Moda": df.iloc[:, :-1].mode().iloc[0],
            "Desviación Estándar": df.iloc[:, :-1].std(),
            "Mínimo": df.iloc[:, :-1].min(),
            "Máximo": df.iloc[:, :-1].max()
        })
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={"index": "Característica"}, inplace=True)
        
        # Gráfico interactivo de estadísticas descriptivas con animaciones
        fig_stats = px.bar(stats_df.melt(id_vars=["Característica"], var_name="Métrica", value_name="Valor"), 
                            x="Característica", y="Valor", color="Métrica", 
                            animation_frame="Métrica", barmode="group", 
                            title="Estadísticas Descriptivas de los Datos")
        st.plotly_chart(fig_stats)