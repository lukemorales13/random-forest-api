---
title: "API de Clasificación de Iris con Random Forest"
author: "lukemorales13"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    theme: united
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Nombre de equipo e integrantes

**Equipo:** GPT-4o mini
**Integrantes:**
- Alanís González Sebástian
- Arano Bejarano Melisa Asharet 
- Fonseca González Bruno 
- Morales Flores Luis Enrique

# API de Clasificación de Iris con Random Forest

Esta es una **API RESTful** construida en **Python** que sirve un modelo de *machine learning* (**Random Forest**) para predecir especies de flores del dataset *Iris*.

---

## Descripción

Este proyecto expone un modelo de **Random Forest entrenado**.  
Permite a los usuarios enviar las características de una flor (largo y ancho de sépalo y pétalo) a un endpoint de predicción y recibir la especie de *Iris* predicha como respuesta.

El modelo fue entrenado y experimentado en los notebooks que se encuentran en la carpeta `/notebooks`, y la API está construida con **FastAPI**.

---

## Características

- **API de Predicción:** Un endpoint `/predict` para obtener predicciones del modelo.  
- **Modelo Doble:** Incluye implementaciones tanto de *Scikit-learn* (`rf_sklearn.py`) como un *Random Forest personalizado* (`rf_custom.py`).  
- **Servicio Web:** Construido usando **FastAPI** y **Uvicorn**.  
- **Desplegable:** Configurado para un despliegue sencillo usando `render.yaml`.

---

## Estructura del Proyecto

```
.
├── app/
│   ├── main.py               # Lógica principal de la API (endpoints)
│   ├── srf_model.py          # Esquema de datos Pydantic para la entrada
│   └── requirements.txt      # Dependencias del proyecto
│
├── model/
│   ├── model.pkl             # Modelo serializado (ej. sklearn)
│   ├── srf_propio_model.pkl  # Modelo serializado (ej. custom)
│   ├── rf_custom.py          # Definición del modelo Random Forest custom
│   └── rf_sklearn.py         # Definición del modelo con Scikit-learn
│
├── notebooks/
│   ├── Lab_ML_P8.ipynb       # Notebooks de experimentación y entrenamiento
│   ├── ML.ipynb
│   ├── experiment.ipynb
│   ├── iris_train.csv        # Datos de entrenamiento
│   ├── iris_train_clean.csv  # Datos limpios
│   └── cleaning_report...md  # Reporte de limpieza
│
├── .gitignore
├── LICENSE
├── README.Rmd
└── render.yaml               # Configuración de despliegue para Render
```

---

## Tecnologías

Este proyecto utiliza las siguientes tecnologías principales:

- **Python 3.x**
- **FastAPI** – Creación de la API  
- **Uvicorn** – Servidor ASGI para FastAPI  
- **Scikit-learn** – Entrenamiento y serialización del modelo  
- **Pandas** – Manipulación de datos  
- **Pydantic** – Validación de datos de entrada en la API  

> **Nota:** Confirma que todas estas librerías estén listadas en `app/requirements.txt`.

---

## Instalación y Ejecución Local

### 1. Clona el repositorio:
```bash
git clone https://github.com/lukemorales13/random-forest-api.git
cd random-forest-api
```

### 2. Crea un entorno virtual e instálalo:
```bash
# (Opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```

### 3. Instala las dependencias:
> El archivo de requerimientos está dentro de la carpeta `app`.

```bash
pip install -r app/requirements.txt
```

### 4. Ejecuta la API:
> El comando de inicio (definido en `render.yaml`) usa **uvicorn**:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Una vez ejecutado, la API estará disponible en : https://random-forest-api-og0h.onrender.com/

---

## Endpoints de la API

### `/` (Root)
**Método:** GET  
**Descripción:** Endpoint de bienvenida que retorna un mensaje de saludo.

**Ejemplo de respuesta (JSON):**
```json
{
  "message": "Bienvenido a la API de Random Forest para Iris!"
}
```

---

### `/predict` (Predicción)
**Método:** POST  
**Descripción:** Recibe las 4 características de la flor *Iris* y retorna la especie predicha.

**Cuerpo de la petición (JSON):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Ejemplo de respuesta (JSON):**
```json
{
  "prediction": "Iris-setosa"
}
```

---

## Licencia

Este proyecto está bajo la **Licencia MIT**.  
Consulta el archivo `LICENSE` para más detalles.

