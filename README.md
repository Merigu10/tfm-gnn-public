# Predicción de Progresión de Alzheimer mediante Graph Neural Networks Multimodales

Este repositorio contiene el código fuente del Trabajo Fin de Máster (TFM) titulado **"Predicción de la Progresión del Alzheimer mediante Graph Neural Networks Multimodales"**, desarrollado en el contexto del Máster en Inteligencia Artificial de la Universidad Internacional de Valencia (VIU).

## 📋 Descripción

Este trabajo desarrolla un modelo basado en **Graph Neural Networks (GNNs) multimodales** para predecir la progresión temporal hacia Alzheimer, integrando datos de la cohorte ADNI (Alzheimer's Disease Neuroimaging Initiative) que incluyen:

- 🧠 **Biomarcadores de LCR**: Aβ42, tau total, tau fosforilada
- 🔬 **Neuroimagen**: Volúmenes de MRI (hipocampo, corteza entorrinal, ventrículos)
- 🧬 **Biomarcadores PET**: PET amiloide, PET-FDG
- 👤 **Variables demográficas**: Edad, género, nivel educativo, estado civil
- 📊 **Evaluaciones cognitivas**: MMSE, ADAS13, CDR-SB

### Resultados Principales

- **MAE**: 0.043 ± 0.040 años (~16 días) mediante 10-Fold Cross-Validation estratificado
- **Mejora del 48.2%** respecto al baseline demográfico
- **Análisis de ablación sistemático** revelando la jerarquía de modalidades:
  - CSF es crítico (+42.6% error sin él)
  - MRI aporta +27.5%
  - PET empeora el rendimiento (-10.8%), evidenciando redundancia con CSF
- **Análisis de fairness**: Sin sesgos detectables por género, edad o APOE4

## 🚀 Características Clave

- ✅ **Validación cruzada rigurosa**: 10-Fold CV estratificado por paciente (RID)
- ✅ **Early stopping**: Prevención de overfitting con validación interna
- ✅ **Manejo inteligente de datos faltantes**: Estrategia LEFT JOIN con indicadores de disponibilidad
- ✅ **Arquitecturas GNN**: GCN, GraphSAGE, GAT implementadas con PyTorch Geometric
- ✅ **Construcción de grafos**: Aristas k-NN (k=8) + aristas temporales longitudinales
- ✅ **Análisis exhaustivo**: Ablación por modalidad, fairness, visualizaciones (t-SNE, UMAP)

## 📁 Estructura del Proyecto

```
tfm-gnn-public/
├── notebooks/                          # Jupyter notebooks con análisis
│   ├── AllBiomarkers_KFold_CrossValidation.ipynb
│   ├── AllBiomarkers_LOPO_CrossValidation.ipynb
│   ├── PET_CSF_Analysis.ipynb
│   ├── Comprehensive_Analysis_Availability_Fairness.ipynb
│   └── ...
├── src/                                # Código fuente (si aplica)
├── results/                            # Resultados y figuras
├── latex/                              # Memoria del TFM en LaTeX
├── requirements.txt                    # Dependencias de Python
├── .gitignore
└── README.md
```

## 🔧 Requisitos

### Hardware Recomendado
- **GPU**: NVIDIA con soporte CUDA (probado en RTX 3080 10GB)
- **RAM**: 16GB+ (32GB recomendado)
- **Almacenamiento**: 10GB+ para datos y modelos

### Software
- Python 3.8+
- CUDA 11.7+ (para aceleración GPU)
- PyTorch 1.13.1+
- PyTorch Geometric 2.3.0+

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/Merigu10/tfm-gnn-public.git
cd tfm-gnn-public
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### Dependencias principales:
```
torch==1.13.1
torch-geometric==2.3.0
numpy==1.23.5
pandas==1.5.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
networkx==3.1
umap-learn==0.5.3
jupyter==1.0.0
```

## 📊 Obtención de Datos ADNI

**⚠️ IMPORTANTE**: Este repositorio **NO incluye** los datos de ADNI debido a restricciones de privacidad y acuerdos de uso de datos.

### Cómo obtener acceso a los datos:

1. **Solicitar acceso a ADNI**:
   - Visita: http://adni.loni.usc.edu/
   - Crear cuenta y solicitar acceso
   - Firmar el Data Use Agreement (DUA)
   - Esperar aprobación del comité de ADNI

2. **Descargar los datos necesarios**:
   - Accede al portal ADNI: http://adni.loni.usc.edu/data-samples/access-data/
   - Descarga las siguientes tablas:
     - `ADNIMERGE.csv` (datos demográficos y clínicos)
     - `UPENNBIOMK.csv` (biomarcadores CSF)
     - `UPENNBIOMK_ROCHE.csv` (biomarcadores CSF Roche Elecsys)
     - `UCBERKELEYAV45.csv` (PET amiloide)
     - `UCBERKELEYFDG.csv` (PET-FDG)
     - `UCSFFSX.csv` (volúmenes MRI FreeSurfer)

3. **Colocar los datos**:
   ```bash
   mkdir data
   # Coloca los archivos .csv descargados en la carpeta data/
   ```

## 🎯 Uso

### Ejecución de Notebooks

1. **Análisis con 10-Fold Cross-Validation**:
```bash
jupyter notebook AllBiomarkers_KFold_CrossValidation.ipynb
```

2. **Análisis LOPO (Leave-One-Patient-Out)**:
```bash
jupyter notebook AllBiomarkers_LOPO_CrossValidation.ipynb
```

3. **Análisis de ablación PET vs CSF**:
```bash
jupyter notebook PET_CSF_Analysis.ipynb
```

4. **Análisis de fairness**:
```bash
jupyter notebook Comprehensive_Analysis_Availability_Fairness.ipynb
```

### Ejemplo de uso del modelo:

```python
import torch
from torch_geometric.data import Data

# Construir grafo de pacientes
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Inicializar modelo GCN
model = GCNModel(
    input_dim=len(feature_columns),
    hidden_dim=128,
    output_dim=1,
    dropout=0.3
)

# Entrenamiento con early stopping
best_model, history = train_with_early_stopping(
    model,
    data,
    patience=20,
    max_epochs=100
)
```

## 📈 Resultados y Visualizaciones

Los notebooks generan múltiples visualizaciones:

- 📊 **Métricas de validación cruzada**: MAE/RMSE por fold con intervalos de confianza
- 🎨 **t-SNE/UMAP**: Proyecciones del espacio latente coloreadas por diagnóstico
- 📉 **Curvas de convergencia**: Early stopping y pérdida train/val por época
- 📦 **Box plots**: Comparación de rendimiento entre configuraciones
- 🔥 **Análisis de ablación**: Contribución de cada modalidad
- ⚖️ **Fairness**: Disponibilidad de biomarcadores por subgrupos demográficos

## 🔬 Metodología

### Arquitectura del Modelo

1. **Construcción del Grafo**:
   - Nodos: Visitas clínicas de pacientes (6,488 visitas)
   - Aristas k-NN: Conectan visitas similares (k=8 vecinos)
   - Aristas temporales: Conectan visitas del mismo paciente

2. **Graph Convolutional Network (GCN)**:
   - 3 capas de convolución gráfica
   - Dimensión oculta: 128
   - Dropout: 0.3
   - Activación: ReLU
   - Normalización: Batch Normalization

3. **Estrategia LEFT JOIN**:
   - Indicadores binarios: `HAS_CSF`, `HAS_PET`, `HAS_MRI`
   - El modelo aprende a modular confianza según disponibilidad
   - Superior a imputación o exclusión de datos faltantes

### Validación

- **10-Fold Cross-Validation** estratificado por paciente (RID)
- **Early stopping** con patience=20 épocas sobre validación interna
- **Intervalo de confianza 95%** para cuantificar incertidumbre
- **Leave-One-Patient-Out (LOPO)** como validación adicional

## 📖 Citar este Trabajo

Si utilizas este código o metodología en tu investigación, por favor cita:

```bibtex
@mastersthesis{tfm-gnn-alzheimer-2025,
  author  = {Tu Nombre},
  title   = {Predicción de la Progresión del Alzheimer mediante Graph Neural Networks Multimodales},
  school  = {Universidad Internacional de Valencia (VIU)},
  year    = {2025},
  type    = {Trabajo Fin de Máster},
  url     = {https://github.com/Merigu10/tfm-gnn-public}
}
```

**Datos ADNI**: Los datos utilizados provienen de ADNI (http://adni.loni.usc.edu/). Asegúrate de cumplir con las políticas de citación de ADNI:

> Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report.

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

**Nota sobre los datos**: Los datos de ADNI están sujetos a sus propios términos y condiciones. Este código es de libre uso, pero los datos deben obtenerse directamente desde ADNI bajo sus políticas.

## 🙏 Agradecimientos

- **ADNI**: Por proporcionar acceso a los datos longitudinales
- **Tutora**: Yaneth Moreno, por la supervisión del TFM
- **PyTorch Geometric**: Por la infraestructura de GNNs

## 📧 Contacto

Para preguntas o colaboraciones:
- GitHub Issues: [Crear issue](https://github.com/Merigu10/tfm-gnn-public/issues)

## 🔗 Enlaces Útiles

- 📄 [Memoria completa del TFM](latex/memoria.pdf) (cuando esté disponible)
- 🌐 [ADNI Official Website](http://adni.loni.usc.edu/)
- 📚 [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- 🎓 [Universidad Internacional de Valencia](https://www.universidadviu.com/)

---

**⚠️ Disclaimer**: Este trabajo es de naturaleza académica e investigacional. Los modelos predictivos desarrollados NO están validados para uso clínico y NO deben utilizarse para toma de decisiones médicas sin validación exhaustiva adicional y aprobación regulatoria.
