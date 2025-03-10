import json
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import streamlit.components.v1 as components
from time import sleep
from distancia_minima import run_distancia_minima
from kdd import run_kdtree
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
# Función para reiniciar la aplicación
def reset_app():
    st.session_state.clear()
    st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)
st.divider()

# Carga de archivo CSV
st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo CSV", type=["csv"])
# Opción en el sidebar para mostrar estadísticas
display_stats = st.sidebar.checkbox("Ver detalles y estadística", value=False)

if uploaded_file is not None:
    # Leer el archivo solo si no ha sido guardado en session_state
    if "df" not in st.session_state:
        df = pd.read_csv(uploaded_file)
        st.session_state["df_original"] = df.copy()  # Guardar una copia original sin modificar
        st.session_state["df"] = df  # Guardar el DataFrame modificado
    else:
        df = st.session_state["df"]  # Usar el DataFrame almacenado
    
    # Detección automática de la variable objetivo (última columna por defecto, pero validada)
    target_column = df.columns[-1]  # Última columna por defecto
    if df[target_column].nunique() > df.shape[0] * 0.5:  # Si tiene demasiados valores únicos, no es categórica
        st.error("⚠️ No se detectó una variable objetivo categórica en la última columna. Seleccione manualmente.")
    else:
        if display_stats:
            
            st.write("### 📄 Vista previa de los datos:")
            st.write(df.head())

            # Identificar TODAS las columnas categóricas (incluyendo la variable objetivo si es categórica)
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            # Convertir columnas categóricas a numéricas usando LabelEncoder
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            # Verificar si aún quedan columnas no numéricas antes de continuar
            non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
            if non_numeric_cols:
                st.error(f"⚠️ Aún existen columnas no numéricas en el dataset: {non_numeric_cols}. Asegúrese de que todas fueron convertidas correctamente.")
                st.stop()

            # **Verificar si la variable objetivo tiene valores nulos después de la conversión**
            if df[target_column].isnull().any():
                st.error(f"⚠️ La variable objetivo `{target_column}` contiene {df[target_column].isnull().sum()} valores nulos. Filtrando datos...")
                df = df.dropna(subset=[target_column])  # Eliminar filas con etiquetas nulas

            # **Normalización de datos (excepto la variable objetivo)**
            scaler = MinMaxScaler()
            features = df.drop(columns=[target_column])
            df[features.columns] = scaler.fit_transform(features)

            # **Detección del número de clases**
            num_classes = df[target_column].nunique()
            num_features = df.shape[1] - 1  # Número de características sin contar la variable objetivo

            # **Guardar en `st.session_state`**
            st.session_state["num_classes"] = num_classes
            st.session_state["num_features"] = num_features
            st.session_state["df"] = df  # Guardar el DataFrame modificado para usar en experimentos

            # **Forzar conversión a valores numéricos antes de entrenar**
            st.session_state["df"] = df.apply(pd.to_numeric, errors='coerce')

            # **Verificar si después de la conversión hay valores nulos y eliminarlos**
            if st.session_state["df"].isnull().any().any():
                st.warning("⚠️ Se detectaron valores nulos después de la conversión. Eliminando filas con valores faltantes...")
                st.session_state["df"].dropna(inplace=True)

            # **Máximo permitido para LDA (asegurar que no exceda las características o clases)**
            max_components_lda = max(1, min(num_classes - 1, num_features))

            st.write("### 🔍 Datos normalizados:")
            st.write(st.session_state["df"].head())
            
            # Visualización interactiva de la distribución de datos
            st.write("### 📊 Distribución de los datos normalizados")
            df_long = df.melt(id_vars=[target_column], var_name="Característica", value_name="Valor")
            fig = px.histogram(df_long, x="Valor", color="Característica", nbins=30, 
                                animation_frame="Característica", title="Distribución de los Datos")
            st.plotly_chart(fig)
            
            # Cálculo de estadísticas descriptivas
            st.write("### 📈 Estadísticas Descriptivas")
            stats_df = pd.DataFrame({
                "Media": df.drop(columns=[target_column]).mean(),
                "Mediana": df.drop(columns=[target_column]).median(),
                "Moda": df.drop(columns=[target_column]).mode().iloc[0],
                "Desviación Estándar": df.drop(columns=[target_column]).std(),
                "Mínimo": df.drop(columns=[target_column]).min(),
                "Máximo": df.drop(columns=[target_column]).max()
            })
            stats_df.reset_index(inplace=True)
            stats_df.rename(columns={"index": "Característica"}, inplace=True)
            
            # Gráfico interactivo de estadísticas descriptivas
            fig_stats = px.bar(stats_df.melt(id_vars=["Característica"], var_name="Métrica", value_name="Valor"),
                                x="Característica", y="Valor", color="Métrica", 
                                animation_frame="Métrica", barmode="group", 
                                title="📊 Estadísticas Descriptivas de los Datos")
            st.plotly_chart(fig_stats)
else:
    st.error("⚠️ Por favor, cargue un archivo CSV para continuar.")

#Botón para reiniciar toda la aplicación
if st.sidebar.button("🔄 Reiniciar Todo"):
    reset_app()

# Inicialización de variables en session_state
if "num_classes" not in st.session_state:
    st.session_state.num_classes = 3  # Valor predeterminado
if "num_features" not in st.session_state:
    st.session_state.num_features = 5  # Valor predeterminado
if "classifier_list" not in st.session_state:
    st.session_state.classifier_list = ["KNN", "Distancia Mínima", "KD-Tree"]
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
            # Determinar el número máximo de componentes permitidos
            max_components = min(df.iloc[:, :-1].shape)

            # Botón para ejecutar PCA automáticamente
            if st.button("Ejecutar PCA automáticamente"):
                pca = PCA(n_components=max_components)  # Ajustar el número máximo permitido
                pca.fit(df.iloc[:, :-1])  # Aplicar PCA a las características
                
                explained_variance = np.cumsum(pca.explained_variance_ratio_)
                optimal_components = np.argmax(explained_variance >= params.get("Varianza", 0.95)) + 1
                optimal_components = min(optimal_components, max_components)  # Asegurar que no exceda el límite

                params["Componentes"] = optimal_components

                # Graficar la varianza explicada
                fig, ax = plt.subplots()
                sns.lineplot(x=range(1, max_components + 1), y=explained_variance, marker='o', ax=ax)
                ax.axhline(y=params.get("Varianza", 0.95), color='r', linestyle='--', label=f'{params.get("Varianza", 0.95) * 100}% Varianza Explicada')
                ax.axvline(x=optimal_components, color='g', linestyle='--', label=f'{optimal_components} Componentes')
                ax.set_xlabel("Número de Componentes")
                ax.set_ylabel("Varianza Explicada Acumulada")
                ax.set_title("Análisis de Componentes Principales (PCA)")
                ax.legend()
                st.pyplot(fig)

            # Configuración manual de parámetros con control del límite
            params["Componentes"] = st.slider(
                "Número de componentes para PCA", 
                min_value=1, 
                max_value=max_components, 
                value=params.get("Componentes", min(2, max_components))
            )
            params["Varianza"] = st.number_input("Varianza mínima a explicar", min_value=0.0, max_value=1.0, value=params.get("Varianza", 0.95))


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
    elif selected_classifiers == "KD-Tree":
        params["Métrica"] = st.selectbox("Métrica de distancia", ["Euclidiana", "Manhattan", "Chebyshev"])
        params["K"] = st.number_input("Número de vecinos (K)", min_value=1, value=3)
        params["Técnica de validación"] = st.selectbox("Técnica de validación", ["Holdout", "K-Fold"])
        if params["Técnica de validación"] == "K-Fold":
            params["Folds"] = st.slider("Número de Folds", min_value=2, max_value=20, value=5)
        elif params["Técnica de validación"] == "Holdout":
            params["Proporción"] = st.number_input("Proporción de datos de entrenamiento", min_value=0.1, max_value=0.9, value=0.8)
    
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
if "experiment_ready" in st.session_state and st.session_state.experiment_ready:
    if "df" in st.session_state:
        df = st.session_state["df"]  # Usar el DataFrame procesado

        # Asegurar que X sea numérico
        X = df.iloc[:, :-1].values.astype(float)

        # Verificar el tipo de la última columna
        y_raw = df.iloc[:, -1]

        # Manejo de valores faltantes
        if y_raw.isnull().any():
            st.error("⚠️ La columna de etiquetas contiene valores nulos. Verifica los datos.")
        else:
            # Intentar convertir a enteros
            try:
                y = y_raw.astype(int).values
            except ValueError:
                # Si falla, convertir etiquetas categóricas a números
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                y = encoder.fit_transform(y_raw)  # Convierte etiquetas categóricas a números
                st.warning("⚠️ Las etiquetas de salida eran categóricas y se han convertido a números automáticamente.")

            results = []
            for config in st.session_state.classifier_configs:
                classifier = config["Clasificador"]
                params = config["Parámetros"]

                if classifier == "KNN":
                    result = run_knn(X, y, params)
                elif classifier == "Distancia Mínima":
                    result = run_distancia_minima(X, y, params)
                elif classifier == "KD-Tree":
                    result = run_kdtree(X, y, params)
                results.append(result)

            # Guardar resultados en JSON
            with open("results.json", "w") as f:
                json.dump(results, f, indent=4)

            st.session_state.results = results
            st.success("✅ Experimentos completados con éxito.")

        
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
