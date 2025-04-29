# 📈 Proyección del Empleo Formal e Informal en Colombia (2025–2040)

Este repositorio contiene el código para estimar y proyectar la Población Económicamente Activa (PEA), así como el empleo formal e informal en Colombia, con foco especial en los sectores de Agricultura e Industria Manufacturera.  
El modelo incorpora análisis de series de tiempo, inteligencia artificial y posibles impactos del IIoT y la IA en el mercado laboral futuro.

---

## 🧠 Estructura del Proyecto

- `etl/` – Extracción, limpieza y procesamiento de datos de la GEIH.
- `models/` – Modelos de predicción: ARIMAX, CatBoost, Prophet.
- `utils/` – Funciones auxiliares, calendarización, diagnósticos.
- `generate_analysis_table.py` – Pipeline de modelado y generación de resultados.
- `notebooks/` – Experimentos y pruebas interactivas.

---

## 📦 Instalación

### 📦 Opción 1:  Instalar con HTTPS (más fácil, no requiere SSH keys)

```
pip install git+https://github.com/squareroot-co/empleo-proyeccion.git
```

### 📦 Opción 2: Instalar con SSH (más seguro, pero necesitas configurar la clave SSH)

```
pip install git+ssh://git@github.com/squareroot-co/empleo-proyeccion.git

```

## ▶️ Ejemplos de uso

### 1. Obtener los datos crudos desde archivos Excel



```python
from epl_proyection.etl.preprocessing import generate_labor_data
df_before_preprocessing = generate_labor_data(
    path_national='data/anex-GEIH-feb2025.xlsx',
    sheet_name_national='Total nacional',
    path_sector='data/anex-GEIH-feb2025.xlsx',
    sheet_name_sector='Ocupados TN_T13_rama',
    path_formal_informal='data/anex-GEIHEISS-nov2024-ene2025.xlsx',
    sheet_name_formal_informal='Grandes dominios ',
    sector=['Agricultura, ganadería, caza, silvicultura y pesca', 'Industrias manufactureras']
)
```
### 2. Ejecutar el pipeline de procesamiento

```python
from epl_proyection.etl.preprocessing import run_preprocessing_pipeline
df_preprocessed = run_preprocessing_pipeline(
    path_national='data/anex-GEIH-feb2025.xlsx',
    sheet_name_national='Total nacional',
    path_sector='data/anex-GEIH-feb2025.xlsx',
    sheet_name_sector='Ocupados TN_T13_rama',
    path_formal_informal='data/anex-GEIHEISS-nov2024-ene2025.xlsx',
    sheet_name_formal_informal='Grandes dominios ',
    sector=['Agricultura, ganadería, caza, silvicultura y pesca', 'Industrias manufactureras'],
    cols_to_lag=[
        'Población en edad de trabajar (PET)', 'Población ocupada',
        'Agricultura, ganadería, caza, silvicultura y pesca',
        'Industrias manufactureras', 'Formal', 'Informal', 'Fuerza de trabajo  '
    ]
)
```
## 3. Generar análisis completo con modelos y predicciones

```python
from epl_proyection.generate_analysis_table import generate_analysis
df_with_predictions = generate_analysis.generate_analysis_table(
    path_national='data/anex-GEIH-feb2025.xlsx',
    sheet_name_national='Total nacional',
    path_sector='data/anex-GEIH-feb2025.xlsx',
    sheet_name_sector='Ocupados TN_T13_rama',
    path_formal_informal='data/anex-GEIHEISS-nov2024-ene2025.xlsx',
    sheet_name_formal_informal='Grandes dominios ',
    sector=['Agricultura, ganadería, caza, silvicultura y pesca', 'Industrias manufactureras'],
    cols_to_lag=[
        'Población en edad de trabajar (PET)', 'Población ocupada',
        'Agricultura, ganadería, caza, silvicultura y pesca',
        'Industrias manufactureras', 'Formal', 'Informal', 'Fuerza de trabajo  '
    ],
    path_activity_formal='data/anex-GEIHEISS-nov2024-ene2025.xlsx',
    sheet_name_activity_formal='Ramas de actividad CIIU 4 A.C'
)
```

## 🔗 Fuentes de Datos

[Datos Fuerza de Trabajo y empleos por sector](https://www.dane.gov.co/files/operaciones/GEIH/anex-GEIH-feb2025.xlsx) : anex-GEIH-feb2025.xlsx

[Datos Formales e Informales](https://www.dane.gov.co/files/operaciones/GEIH/anex-GEIHEISS-nov2024-ene2025.xlsx): anex-GEIHEISS-nov2024-ene2025.xlsx

