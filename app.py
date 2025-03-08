import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import streamlit.components.v1 as components
from time import sleep
from distancia_minima import run_distancia_minima
from knn import run_knn
st.header("Cesar Abraham Delgado Cardona")
# Título del proyecto
st.title("Comparador de Clasificadores")

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
else:
    st.error("Por favor, cargue un archivo CSV para continuar.")




# Inicialización de variables en session_state
if "num_classes" not in st.session_state:
    st.session_state.num_classes = 3  # Valor predeterminado
if "num_features" not in st.session_state:
    st.session_state.num_features = 5  # Valor predeterminado
if "classifier_list" not in st.session_state:
    st.session_state.classifier_list = ["KNN", "Distancia Mínima"]
if "classifier_configs" not in st.session_state or not isinstance(st.session_state.classifier_configs, list):
    st.session_state.classifier_configs = []  # Asegurar que sea una lista
if "experiment_ready" not in st.session_state:
    st.session_state.experiment_ready = False
if "results" not in st.session_state:
    st.session_state.results = []

# Función para reiniciar el proceso
def reset_experiment():
    st.session_state.classifier_configs = []
    st.session_state.experiment_ready = False
    st.session_state.results = []
    st.rerun()

# Selección de clasificadores
st.sidebar.header("Selección de Clasificadores")
selected_classifiers = st.sidebar.selectbox("Seleccione un clasificador para configurar", st.session_state.classifier_list)


# Configuración de clasificadores en secuencia
if selected_classifiers:
    # Número de clases y características
    num_classes = st.session_state["num_classes"]
    num_features = st.session_state["num_features"]

    # Máximo permitido para LDA
    max_components_lda = max(1, min(num_classes - 1, num_features))
    
        
    params = {}
    if selected_classifiers == "KNN":
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
    elif selected_classifiers == "Distancia Mínima":
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
        
        
    elif selected_classifiers == "SVM":
        params["Kernel"] = st.selectbox("Tipo de kernel", ["Lineal", "Polinomial", "RBF"])
        params["C"] = st.number_input("Valor de C", min_value=0.1, value=1.0)
    elif selected_classifiers == "Random Forest":
        params["Árboles"] = st.number_input("Número de árboles", min_value=1, value=100)
        params["Profundidad"] = st.number_input("Profundidad máxima", min_value=1, value=10)
    elif selected_classifiers == "Red Neuronal":
        params["Capas"] = st.number_input("Número de capas ocultas", min_value=1, value=3)
        params["Neuronas"] = st.number_input("Neuronas por capa", min_value=1, value=64)
    
    # Crear columnas para los botones con espacio
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("✅ Guardar configuración", use_container_width=True):
            st.session_state.classifier_configs.append({"Clasificador": selected_classifiers, "Parámetros": params})
            with st.spinner("Guardando configuración..."):
                sleep(1)
            st.success(f"Configuración de {selected_classifiers} guardada correctamente.")
            st.rerun()

    with col2:
        if st.button("🔄 Reiniciar selección", use_container_width=True):
            with st.spinner("Reiniciando..."):
                sleep(1)
            reset_experiment()

# Agregar animación visual
components.html(
    """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    .stButton>button {
        animation: fadeIn 0.5s ease-in-out;
        transition: all 0.3s;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
    """,
    height=0
)

    
# Mostrar configuraciones guardadas con mejor diseño
if st.session_state.classifier_configs:
    st.write("## 📌 Configuraciones Guardadas")
    st.info("Puede reiniciar la selección de clasificadores o empezar los experimentos.")
    st.warning("Revise bien las configuraciones antes de continuar.")
    # Convertir configuraciones a DataFrame con formato mejorado
    df_configs = pd.DataFrame(st.session_state.classifier_configs)
    
    # Usar expander para un diseño más limpio
    with st.expander("📋 Ver configuraciones guardadas", expanded=True):
        st.dataframe(df_configs, use_container_width=True, height=250)
    
    # Crear columnas para centrar el botón
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Empezar experimentos", use_container_width=True):
            with st.spinner("Ejecutando experimentos..."):
                sleep(1.5)
            st.session_state.experiment_ready = True
            st.rerun()
    
    # Animación visual
    components.html(
        """
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stButton>button {
            animation: fadeIn 0.5s ease-in-out;
            transition: all 0.3s;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #007BFF !important;
            color: white !important;
        }
        </style>
        """,
        height=0
    )

    
        
# Ejecución de experimentos
if st.session_state.experiment_ready:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1].values  # Características
        y = df.iloc[:, -1].values   # Variable objetivo
        
        results = []
        for config in st.session_state.classifier_configs:
            classifier = config["Clasificador"]
            params = config["Parámetros"]
            
            if classifier == "KNN":
                result = run_knn(X, y, params)
            elif classifier == "Distancia Mínima":
                result = run_distancia_minima(X, y, params)
            results.append(result)
        
        # Guardar resultados en JSON
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        st.session_state.results = results
        st.success("Experimentos completados con éxito.")
        
# Mostrar resultados de experimentación


if st.session_state.results:
    st.write("## 📊 Resultados de la Experimentación")
    
    df_results = pd.DataFrame(st.session_state.results)
    
    # Expandir el JSON de parámetros en columnas individuales
    df_expanded = df_results.copy()
    df_expanded = df_expanded.drop(columns=["Parámetros"]).join(df_results["Parámetros"].apply(pd.Series))
    
    # Redondear valores numéricos
    numeric_cols = ["Exactitud", "Precisión", "Sensibilidad", "Especificidad", "Tiempo de ejecución"]
    df_expanded[numeric_cols] = df_expanded[numeric_cols].apply(lambda x: x.round(4))
    
    # Usar un expander para mantener la interfaz organizada
    with st.expander("📋 Ver Resultados Detallados", expanded=True):
        st.dataframe(df_expanded, use_container_width=True, height=300)
    
    # 📊 Gráfica del tiempo de ejecución por experimento
    st.write("### ⏳ Comparación de Tiempo de Ejecución")
    fig_time = px.bar(
        df_expanded, 
        x=df_expanded.index, 
        y="Tiempo de ejecución", 
        color="Clasificador", 
        hover_data=["Exactitud", "Precisión", "Sensibilidad", "Especificidad"],
        labels={"index": "Experimento", "Tiempo de ejecución": "Tiempo (s)"},
        title="Tiempo de Ejecución por Experimento",
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Pastel  # Cambio de colores
    )
    fig_time.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig_time.update_layout(
        xaxis_title="Experimento", 
        yaxis_title="Tiempo de ejecución (s)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Mostrar el experimento con menor tiempo de ejecución
    min_time_exp = df_expanded.loc[df_expanded["Tiempo de ejecución"].idxmin()]
    st.info(f"⏳ **Menor tiempo de ejecución:** {min_time_exp['Tiempo de ejecución']}s con {min_time_exp['Clasificador']}. Parámetros: {min_time_exp.to_dict()}")
    
    # 📊 Gráfica de exactitud por experimento
    st.write("### 🎯 Comparación de Exactitud")
    fig_accuracy = px.bar(
        df_expanded, 
        x=df_expanded.index, 
        y="Exactitud", 
        color="Clasificador", 
        hover_data=["Precisión", "Sensibilidad", "Especificidad", "Tiempo de ejecución"],
        labels={"index": "Experimento", "Exactitud": "Exactitud"},
        title="Exactitud por Experimento",
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Bold  # Cambio de colores
    )
    fig_accuracy.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig_accuracy.update_layout(
        xaxis_title="Experimento", 
        yaxis_title="Exactitud",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Mostrar el experimento con mayor exactitud
    max_acc_exp = df_expanded.loc[df_expanded["Exactitud"].idxmax()]
    st.info(f"🎯 **Mayor exactitud:** {max_acc_exp['Exactitud']} con {max_acc_exp['Clasificador']}. Parámetros: {max_acc_exp.to_dict()}")
    
    # Centrar y mejorar el diseño del botón
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Reiniciar selección", key="reiniciar_btn", use_container_width=True):
            with st.spinner("Reiniciando..."):
                sleep(1)
            reset_experiment()
    
    # Agregar animaciones y diseño
    components.html(
        """
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stButton>button {
            animation: fadeIn 0.5s ease-in-out;
            transition: all 0.3s;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #FF5733 !important;
            color: white !important;
        }
        </style>
        """,
        height=0
    )
