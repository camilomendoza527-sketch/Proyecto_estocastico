# Reconocedor y Clasificador de Caracteres en Tiempo Real (2D/3D) mediante HMM

Proyecto final de la asignatura **Procesos Estocásticos (Código 603504)** — Programa de Ingeniería de Sistemas, Universidad de los Llanos. Ver especificación oficial en [`docs/SPEC.pdf`](docs/SPEC.pdf) y desarrollo matemático para sustentación en [`docs/ecuaciones.md`](docs/ecuaciones.md).

Reconoce y proyecta en tiempo real las **27 letras minúsculas del alfabeto español** ($\mathcal{C} = \{a, b, \dots, n, \tilde{n}, o, \dots, z\}$) tanto en un lienzo cartesiano bidimensional (**Modo 2D**) como sobre la superficie de una esfera tridimensional (**Modo 3D**), utilizando Modelos Ocultos de Markov ($\lambda_c = (A_c, B_c, \pi_c)$) implementados de forma 100 % nativa en **NumPy**.

---

## 1. Instalación y Ejecución

```powershell
# 1. Crear entorno virtual e instalar dependencias (Python 3.12+)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Ejecutar la aplicación de escritorio (GUI: Ventanas A, B y C)
.\.venv\Scripts\python.exe main.py

# 3. Ejecutar la suite completa de pruebas unitarias e integración
.\.venv\Scripts\pytest.exe -q
```

### Scripts Offline de Entrenamiento, Evaluación y Captura
*(Independientes de la GUI principal, cumpliendo SPEC §4.1)*:

```powershell
# Entrenar los 27 modelos 2D y 27 modelos 3D (exporta a models/2d_*.json y models/3d_*.json)
.\.venv\Scripts\python.exe -m training.train --mode all

# Evaluar accuracy Top-1 / Top-3 en prefijos (30%, 50%, 70%, 100%) y comparar K=8 vs K=16
.\.venv\Scripts\python.exe -m training.evaluate

# Herramienta opcional de captura de muestras 2D a mano alzada
.\.venv\Scripts\python.exe -m training.capture_tool
```

---

## 2. Arquitectura del Proyecto (SPEC §4.2)

```text
config/config.toml         Única fuente de configuración: alfabeto (27 letras), K, estados N por letra
core/config.py             Cargador y validador de config/config.toml
core/HMMEngine.py          Motor matemático HMM puro en NumPy (Forward escalado, Backward, Viterbi log,
                           Baum-Welch multisequencia, proyección futura y clasificador Bayesiano)
core/topology.py           Inicialización Bakis izquierda-derecha y alineación segmentaria uniforme
core/FeatureExtractor.py   Geometría y discretización 2D y 3D sobre la esfera (sin lógica HMM)
main.py                    Punto de entrada de la GUI; solo carga models/*.json (nunca entrena)
gui/MainMenu.py            Ventana A: Menú Principal de Inicio institucional
gui/View2D.py              Ventana B: Reconocedor y Visualizador 2D con proyección punteada
gui/View3D.py              Ventana C: Reconocedor y Visualizador 3D sobre esfera interactiva OpenGL
gui/RankingPanel.py        Panel lateral derecho con ranking en tiempo real P(c | O_{1:t}) e iconos
training/                  Scripts offline: augment.py, train.py, evaluate.py, capture_tool.py
models/                    54 modelos entrenados: 2d_<letra>.json (27) y 3d_<letra>.json (27)
docs/ecuaciones.md         Guía paso a paso con todas las ecuaciones y demostraciones para el tablero
tests/                     Pruebas automatizadas (38 tests: extractor 2D/3D, motor HMM, pipeline y GUI)
```

---

## 3. Discretización de Trazos y Convención Obligatoria

| Símbolo | Significado |
|---|---|
| `0 … K-1` | Dirección del movimiento con la pluma abajo ($K=16$ sectores angulares; el sector `0` es el Este $\to$ y crecen en sentido antihorario). |
| `K+0 … K+3` | Salto con la pluma levantada entre sub-trazos: $\to, \uparrow, \leftarrow, \downarrow$. |

- **Modo 2D (`FeatureExtractor2D`)**: Normalización por caja envolvente (con piso `expected_letter_size` en tiempo real para conservar el prefijo de trazos parciales) $\to$ suavizado simétrico $\to$ remuestreo a paso de arco constante $D$ $\to$ cuantización angular $\operatorname{atan2}(\Delta y, \Delta x)$.
- **Modo 3D (`FeatureExtractor3D`)**: Rotación al marco ortonormal centrado en la cámara $R_{\text{cam}} \in SO(3)$ (ecuador frontal $(\theta=\pi/2, \phi=0)$ lejos de los polos) $\to$ conversión a esféricas $(\theta, \phi)$ con desenvolvimiento de $\Delta\phi$ en $\pm\pi$ $\to$ remuestreo geodésico constante mediante SLERP $\to$ cuantización del rumbo en el plano tangente local $\psi = \operatorname{atan2}(-d\theta,\, d\phi \cdot \sin\bar{\theta})$.

### Convención de trazo para letras multitrazo
1. **Primero el cuerpo de la letra; después el punto, la virgulilla o la barra** (`i`, `j`, `ñ`, `t`, `f`).
2. **La barra de la `t` se traza siempre de izquierda a derecha.**
3. **Los dos trazos de la `x` se trazan siempre de izquierda a derecha.**
