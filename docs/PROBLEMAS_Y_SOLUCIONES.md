# Registro de Problemas, Soluciones y Justificación Técnica

Este documento recopila los principales problemas detectados a lo largo del desarrollo del **Reconocedor de Escritura Manuscrita 2D/3D con Modelos Ocultos de Márkov (HMM)**, la solución implementada en cada caso y su justificación matemática, geométrica o de experiencia de usuario.

---

## 1. Motor Estocástico (HMM) y Calibración de Probabilidades

### 1.1 Confusión entre letras morfológicamente cercanas (`a`, `d`, `q`, `n`, `h`, `b`)
- **Problema:** Al evaluar únicamente la verosimilitud de emisión/transición $\log P(O_{1:T} \mid \lambda_c)$, letras con trayectorias locales muy parecidas (como bucles seguidos de astiles en `a`, `d`, `q`, o arcos iniciales en `n`, `h`, `b`) competían estrechamente y producían errores cuando variaba la velocidad o la proporción del trazo.
- **Solución:**
  - Se incorporó el 100% de las muestras disponibles por letra en el entrenamiento (`b0dae23`) y se mejoró la separación entre `a`, `d` y `q` (`8fe813b`).
  - En [`core/HMMEngine.py`](../core/HMMEngine.py), se combinó la log-verosimilitud secuencial con penalizaciones de duración (`HMM.length_deviation_penalty`) y de forma global (`shape_penalty`) en `HMMClassifier.classify()`.
- **Justificación:** La secuencia de direcciones discretizadas $O_{1:T}$ captura la dinámica local del trazo, pero es invariante a escala y longitud absoluta. Añadir la desviación respecto a la longitud media $\mu_L$ y la geometría global preserva el orden topológico del trazo sin perder invarianza espacial razonable, elevando la exactitud Top-1 sobre el conjunto de validación a **100.00%** tanto en 2D (`405/405`) como en 3D (`135/135`).

### 1.2 Saturación de probabilidades al `100.0%` en trazos extremos y rechazo OOD
- **Problema:**
  1. Como el puntaje `Score(c)` es extensivo en la longitud $T$, diferencias pequeñas por símbolo se amplifican linealmente con $T$; bajo un Softmax sin escala adaptativa, incluso garabatos largos o trazos atípicos saturaban al `100.0%`.
  2. Se evaluó temporalmente rechazar trazos fuera de distribución (OOD) con una hipótesis nula $H_0$, pero esto impedía visualizar el comportamiento estadístico continuo del modelo sobre cualquier trazo libre.
- **Solución:**
  - Se eliminó el rechazo OOD para que el clasificador siempre devuelva la distribución posterior normalizada sobre las 27 clases ($\sum_{c \in \mathcal{C}} P(c \mid O_{1:T}) = 100\%$).
  - En [`core/HMMEngine.py`](../core/HMMEngine.py):
    - Se huberizó `HMM.length_deviation_penalty()` mediante `_huber_quad(z)` con umbral `HMM.HUBER_Z_THRESHOLD = 4.0` (crecimiento cuadrático para $|z| \le 4$ y lineal para $|z| > 4$).
    - En `HMMClassifier.classify()` (`FINAL_FIT_REF_NATS = 1.8`), cuando `complete=True`, se escaló la temperatura efectiva con `fit_scale = max(1.0, (-max_log / float(n_obs)) / self.FINAL_FIT_REF_NATS)`.
- **Justificación:**
  - La función de Huber mantiene continuidad $C^1$ en $z = \pm 4$ y evita que trazos muy largos ($T \gg \mu_L$) exploten cuadráticamente por duración, lo que antes forzaba márgenes artificiales entre clases.
  - `fit_scale` actúa únicamente cuando la calidad media por símbolo del mejor modelo cae por debajo de la referencia de letras válidas (`-max_log / T > 1.8` nats/símbolo). De este modo, las letras bien trazadas conservan su nitidez y Top-1 intacto, mientras que los garabatos largos o ruidosos reparten su probabilidad entre las candidatas cercanas en lugar de colapsar a `100.0%`.

---

## 2. Geometría y Renderizado 3D sobre la Esfera

### 2.1 Eje vertical invertido, desfase cursor-esfera y proyección verde inoperante en 3D
- **Problema:** En versiones iniciales de la vista 3D:
  - El eje vertical estaba invertido al mapear la carta esférica.
  - El trazo sobre la esfera presentaba desfase respecto al puntero del ratón.
  - La proyección futura punteada (color verde) funcionaba en 2D, pero no se proyectaba de forma coherente sobre la superficie curva en 3D.
- **Solución:**
  - Se corrigió el signo del eje vertical (`f4029ef`) y se implementó ray-casting exacto 1:1 mediante intersección rayo-esfera con la matriz $(P \cdot V)^{-1}$ (`1220a47`).
  - Se adoptó una carta esférica intrínseca centrada en la orientación actual de la cámara (`active_camera_orientation`), reconstruyendo los arcos futuros mediante interpolación esférica SLERP (`0db2558`, `4cef933`).
  - Se desacopló la geometría esférica de la vista moviéndola a [`core/FeatureExtractor.py`](../core/FeatureExtractor.py) (`e1889c3`).
- **Justificación:** En la variedad $S^2$, usar diferencias planas de $(\theta, \phi)$ introduce distorsión cerca de los polos y rompe la invariancia frente a rotaciones de cámara. Proyectar en el marco tangente de la cámara e interpolar geodésicamente con SLERP mantiene el trazo fiel al cursor y permite que el predictor HMM dibuje la continuación esperada directamente sobre la curvatura de la esfera.

### 2.2 Visibilidad y tamaño del recuadro guía y trazo en Modo 3D
- **Problema:** El recuadro guía sobre la esfera resultaba poco visible o desalineado tras ajustar el tamaño y el color respecto al fondo OpenGL.
- **Solución:** En [`gui/View3D.py`](../gui/View3D.py) y [`core/FeatureExtractor.py`](../core/FeatureExtractor.py) (`build_guide_frame_curves`), se actualizó el recuadro guía sobre la esfera usando tonos naranja consistentes para el borde y la retícula (`(0.94, 0.45, 0.05, 0.88)` y `(0.94, 0.55, 0.18, 0.42)`), sincronizado con la orientación de cámara al rotar o centrar la esfera.
- **Justificación:** El contraste naranja delimita la zona frontal útil para escribir sin obstruir la esfera ni confundirse con la proyección predictiva verde.

---

## 3. Dataset, Guías Canónicas e Interfaz de Usuario

### 3.1 Desalineación entre la Guía de letras y el dataset manuscrito real (ej. `x`, `q`)
- **Problema:** Algunas trayectorias en la guía animada no coincidían con el orden de sub-trazos predominante en el dataset (por ejemplo, `x` tenía el segundo trazo invertido respecto a los datos, y letras de 3 sub-trazos como `q` no contaban con descripción explícita para el tercer trazo).
- **Solución:**
  - Se regeneró [`data/canonical_glyphs.json`](../data/canonical_glyphs.json) con los 27 medoides modales suavizados extraídos de `data/glifo-dataset-2026-11-12.json` (`361ef5c`, `f659b1c`), garantizando que `a`, `g`, `k`, `p` usen 2 sub-trazos, `q` use 3 sub-trazos y `x` trace primero arriba-izquierda $\rightarrow$ abajo-derecha y luego abajo-izquierda $\rightarrow$ arriba-derecha.
  - En [`gui/LetterGuide.py`](../gui/LetterGuide.py), se actualizó `LetterGuideView.select_letter()` para describir 1, 2 y 3+ sub-trazos (`1 (verde)`, `2 (naranja)`, `3 (morado)`), y se añadieron miniaturas manuscritas reales en los botones del selector (`9cb7dd0`).
- **Justificación:** Si la guía enseña un orden de trazo opuesto al entrenado en los HMM, los usuarios replican secuencias que el modelo penaliza. Alinear guía y dataset asegura consistencia didáctica y **100%** de reconocimiento sobre las 27 plantillas canónicas en 2D y 3D.

### 3.2 Botones duplicados y ausencia de telemetría inferior en Modo 2D
- **Problema:** En [`gui/View2D.py`](../gui/View2D.py), los botones `Reiniciar captura` y `Limpiar lienzo` ejecutaban exactamente la misma acción, y el pie del lienzo mostraba solo un texto estático en lugar de la telemetría en vivo presente en 3D.
- **Solución:**
  - Se implementó `RecognizerCanvas2D.undo_last_stroke()` (incluyendo reversión segura si el usuario presiona el botón en medio de un sub-trazo activo) y se reemplazó el botón redundante por `Deshacer trazo` junto a `Limpiar lienzo` (`f659b1c`).
  - Se incorporó la barra inferior `self.lbl_obs` con coordenadas `(x, y)`, número de sub-trazos, longitud discreta `T`, pasos de proyección de la letra líder y los últimos 5 símbolos discretos observados, manteniendo también el estado de Viterbi $q^*$ en el encabezado de [`gui/RankingPanel.py`](../gui/RankingPanel.py) (`d7a02a6`, `e1889c3`).
- **Justificación:** En letras multi-trazo (`i`, `ñ`, `t`, `x`, `q`), poder deshacer únicamente el último sub-trazo evita reiniciar toda la letra ante un error; además, exponer `T` y la secuencia de símbolos en tiempo real permite verificar visualmente la cadena de observaciones que consume el HMM.

---

## 4. Arquitectura, Pruebas y Documentación

### 4.1 Acoplamiento de vistas y cobertura de verificación
- **Problema:** Parte de la lógica geométrica 3D residía dentro de la capa GUI y los cambios en fórmulas o botones requerían verificación sistemática para evitar regresiones.
- **Solución:**
  - Se centralizó la geometría 3D en [`core/FeatureExtractor.py`](../core/FeatureExtractor.py) (`e1889c3`).
  - Se amplió la suite automatizada en [`tests/`](../tests) cubriendo extractor 2D/3D, motor HMM (Forward, Backward, Viterbi, Baum-Welch, penalización de Huber y calibración), pipeline de entrenamiento e interfaz GUI (`test_view2d_undo_button_and_live_telemetry_bar`, `test_letter_guide_modal_trajectories_and_q_3_strokes`).
  - Se consolidó la documentación en [`docs/`](../docs) con `AUDITORIA_ARQUITECTURA.pdf`, `SUSTENTACION_MATEMATICA.pdf`, `EVALUACION_PROYECTO.pdf` y el presente historial.
- **Justificación:** Separar el dominio matemático de la interfaz Qt/OpenGL permite ejecutar pruebas deterministas rápidas con `pytest` y garantiza que la documentación técnica refleje fielmente la implementación vigente.
