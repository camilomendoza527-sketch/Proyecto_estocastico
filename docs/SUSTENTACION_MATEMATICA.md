# Guía de Sustentación Matemática y Mapeo al Código Fuente

Este documento organiza **todo lo que el equipo debe explicar y demostrar en tablero durante la sustentación oral** (**Enunciado §5.2**), conectando cada modelo y ecuación matemática directamente con el archivo, clase y método exacto donde fue programado **100 % desde cero en NumPy**.

> Para el desarrollo extenso de las demostraciones algebraicas paso a paso, consultar también [`docs/ecuaciones.md`](ecuaciones.md).

---

## 1. Guion Estructurado para la Sustentación

1. **Bloque 1 — Demostración en Vivo (GUI 2D, 3D y Guía)**:
   - Mostrar Ventana A (`MainMenu`), luego **Modo 2D** dibujando letras continuas (`a`, `v`, `n`, `m`) y multitrazo (`d`, `i`, `ñ`, `t`, `x`), señalando cómo cambian en vivo las probabilidades $P(c \mid O_{1:t})$ y la proyección futura verde conforme evoluciona el prefijo.
   - Pasar a **Modo 3D** trazando dentro del recuadro azul de $\pm 48^\circ$ sobre la esfera, rotando la esfera para demostrar invarianza respecto al marco de cámara y mostrando el autocompletado geodésico.
   - Mostrar la **Guía de letras** explicando que la ordenación de sub-trazos coincide exactamente con la del dataset entrenado (ej. `d` en dos trazos: `c` + palo vertical).
2. **Bloque 2 — Arquitectura Limpia y Cero Librerías Prohibidas**:
   - Mostrar [`requirements.txt`](../requirements.txt) (solo `numpy`, `PySide6`, `pyqtgraph`, `PyOpenGL`, `pytest`) y destacar que en [`core/HMMEngine.py`](../core/HMMEngine.py) no existe ninguna importación de `hmmlearn`, `scikit-learn` ni `scipy.stats`.
   - Explicar la separación entre entrenamiento offline ([`training/train.py`](../training/train.py) $\to$ [`models/*.json`](../models/)) e inferencia en vivo ([`main.py`](../main.py) $\to$ [`gui/`](../gui/)).
3. **Bloque 3 — Defensa Matemática en Tablero (4 Pilares + Ejercicio Numérico)**:
   - Pilar A: Discretización angular 2D y geometría diferencial sobre la esfera $S^2$ (3D).
   - Pilar B: Definición de $\lambda_c = (A, B, \pi)$ con topología Bakis y Algoritmo **Forward Escalado**.
   - Pilar C: Clasificación Bayesiana Conjunta ($\text{Score}(c)$ + Log-Sum-Exp) y Decodificación **Viterbi en Log-Space**.
   - Pilar D: Aprendizaje **Baum-Welch** y Proyección Dinámica de Trayectoria Futura ($\mathbb{E}[d_j] = 1/(1-a_{jj})$ + mapa exponencial en $S^2$).

---

## 2. Mapeo Completo: Matemática $\leftrightarrow$ Código Fuente

### 2.1 Invariancia al Tiempo de Reloj, Escala y Discretización 2D (`FeatureExtractor2D`)
**Qué se explica en tablero:**
No usamos milisegundos del reloj porque una misma letra dibujada lento o rápido cambiaría su duración temporal. En su lugar parametrizamos por **longitud de arco geométrica** constante $D = 0.05$.

| Concepto / Ecuación Matemática | Implementación Exacta en el Código |
|---|---|
| **Inversión de eje pantalla y normalización con piso en tiempo real:**<br>$$y \leftarrow -y, \qquad s = \begin{cases} \max(w, h, 250) & \text{tiempo real} \\ \max(w, h, 10^{-6}) & \text{trazo completo} \end{cases}, \qquad p'_i = \frac{p_i - p_{\min}}{s}$$ | [`FeatureExtractor2D.normalize`](../core/FeatureExtractor.py#L153-L164) y [`View2D._evaluate_realtime`](../gui/View2D.py#L283-L288). |
| **Suavizado simétrico con bordes fijos ($w=3, r=1$):**<br>$$\tilde{p}_i = \frac{1}{2r_i + 1}\sum_{j=-r_i}^{r_i} p'_{i+j}, \qquad r_i = \min(r, i, n-1-i)$$ | [`FeatureExtractor._moving_average`](../core/FeatureExtractor.py#L138-L147) usando sumas acumuladas `np.cumsum` en $O(n)$. |
| **Remuestreo a longitud de arco constante ($D = 0.05$):**<br>$$s_i = \sum_{k=1}^{i}\|\tilde{p}_k - \tilde{p}_{k-1}\|_2, \qquad u_m = m \cdot D, \qquad q_m = \operatorname{interp}(u_m, s, \tilde{p})$$ | [`FeatureExtractor2D.resample`](../core/FeatureExtractor.py#L169-L180). |
| **Cuantización angular centrada ($K = 16$ sectores + $4$ saltos):**<br>$$o_m = \left\lfloor \frac{(\operatorname{atan2}(\Delta y_m, \Delta x_m) + \pi/K) \bmod 2\pi}{2\pi/K} \right\rfloor \bmod K$$<br>$$o_{\text{salto}} = K + \left(\left\lfloor \frac{(\theta_{\text{salto}} + \pi/4)\bmod 2\pi}{\pi/2} \right\rfloor \bmod 4\right)$$ | [`FeatureExtractor.quantize_direction`](../core/FeatureExtractor.py#L118-L122), [`quantize_jump`](../core/FeatureExtractor.py#L124-L127), [`directions`](../core/FeatureExtractor.py#L182-L186) y [`jump`](../core/FeatureExtractor.py#L188-L192). |
| **Descriptor morfológico global invariante $\mathbf{g} \in \mathbb{R}^7$:**<br>$$\mathbf{g} = [\log(w/h),\; x'_{\text{ini}},\; y'_{\text{ini}},\; x'_{\text{fin}},\; y'_{\text{fin}},\; \bar{x}',\; \bar{y}']^T$$ | [`compute_shape_features`](../core/FeatureExtractor.py#L27-L54). |

---

### 2.2 Geometría Diferencial Esférica en el Modo 3D (`FeatureExtractor3D`)
**Qué se explica en tablero:**
Al dibujar sobre una esfera $S^2$, en los polos ($\theta=0,\pi$) colapsa la métrica ($\sin\theta \to 0$) y en $\phi=\pm\pi$ hay discontinuidad de $2\pi$. Además, el usuario puede rotar la esfera a cualquier ángulo.

| Concepto / Ecuación Matemática | Implementación Exacta en el Código |
|---|---|
| **Intersección exacta rayo-esfera (trazo 3D con ratón):**<br>$$\| \mathbf{o} + t\mathbf{d} \|^2 = R^2 \implies \|\mathbf{d}\|^2 t^2 + 2(\mathbf{o}\cdot\mathbf{d})t + (\|\mathbf{o}\|^2 - R^2) = 0$$ | [`FeatureExtractor3D.intersect_ray_sphere`](../core/FeatureExtractor.py#L560-L582) invocada desde [`SphereCanvas3D._screen_to_sphere_point`](../gui/View3D.py#L192-L211). |
| **Marco ortonormal centrado en la vista de cámara ($R_{\text{cam}} \in SO(3)$):**<br>$$R_{\text{cam}} = \begin{bmatrix} \mathbf{e}_{\text{front}}^T \\ \mathbf{e}_{\text{right}}^T \\ \mathbf{e}_{\text{up}}^T \end{bmatrix}, \qquad \hat{p}_{\text{cam}} = R_{\text{cam}}\frac{p}{\|p\|}$$<br>Coloca el frente de la cámara en el ecuador $(\theta=\pi/2, \phi=0)$ donde $\sin\theta=1$. | [`FeatureExtractor3D.build_camera_matrix`](../core/FeatureExtractor.py#L292-L315) y [`rotate_to_camera_frame`](../core/FeatureExtractor.py#L317-L328). |
| **Esféricas y desenvolvimiento de $\phi$:**<br>$$\theta_i = \arccos(z_i), \qquad \phi_i = \operatorname{unwrap}(\operatorname{atan2}(y_i, x_i))$$ | [`FeatureExtractor3D.cartesian_to_spherical`](../core/FeatureExtractor.py#L331-L335). |
| **Remuestreo geodésico mediante SLERP sobre $S^2$:**<br>$$\Delta s_k = \arccos(\hat{p}_{k-1}\cdot\hat{p}_k), \qquad \hat{q}_m = \frac{\sin((1-t)\Omega)}{\sin\Omega}\hat{p}_k + \frac{\sin(t\Omega)}{\sin\Omega}\hat{p}_{k+1}$$ | [`FeatureExtractor3D.geodesic_distances`](../core/FeatureExtractor.py#L361-L366) y [`resample_geodesic`](../core/FeatureExtractor.py#L368-L392). |
| **Rumbo métrico en el plano tangente local:**<br>$$d\hat{p} = \mathbf{e}_\theta d\theta + (\sin\theta\,\mathbf{e}_\phi)d\phi \implies \psi_m = \operatorname{atan2}(-d\theta_m,\; d\phi_m\cdot\sin\bar{\theta}_m)$$ | [`FeatureExtractor3D.directions_spherical`](../core/FeatureExtractor.py#L394-L401). |

---

### 2.3 Modelo $\lambda_c = (A, B, \pi)$, Topología Bakis y Problema 1 (Forward Escalado)
**Qué se explica en tablero:**
Cada letra $c$ tiene su propio HMM con $N_c \in [4, 11]$ estados secuenciales. Usamos **topología Bakis (izquierda-derecha)** porque un trazo avanza en el tiempo sin retroceder:
$$\pi = [1, 0, \dots, 0]^T, \qquad a_{ij} = 0 \;\;\forall\, j \notin \{i, i+1, i+2\}$$
Si multiplicamos probabilidades crudas $\alpha_t(i)$ durante $T \approx 60$ pasos, $(1/20)^{60} \approx 10^{-78}$ genera **underflow**. Al reescalar con $c_t = \big(\sum_j \tilde\alpha_t(j)\big)^{-1}$, se demuestra por inducción que $\hat\alpha_T(i) = \big(\prod_{t=1}^T c_t\big)\alpha_T(i)$ y como $\sum_i \hat\alpha_T(i) = 1$, entonces:
$$\log P(O \mid \lambda) = -\sum_{t=1}^{T} \log(c_t)$$

| Concepto / Ecuación Matemática | Implementación Exacta en el Código |
|---|---|
| Restricción estocástica $\sum_j a_{ij}=1, \sum_k b_j(k)=1, \sum_i \pi_i=1$ | [`HMM._validate_shapes_and_stochastic`](../core/HMMEngine.py#L60-L74) |
| Inicialización Bakis y ruptura de simetría segmentaria | [`init_bakis_hmm`](../core/HMMEngine.py#L438-L456) y [`init_segmental_bakis_hmm`](../core/HMMEngine.py#L459-L489) |
| Algoritmo Forward Escalado (Rabiner, 1989) | [`HMM.forward`](../core/HMMEngine.py#L146-L166) |

---

### 2.4 Clasificación Bayesiana Conjunta y Log-Sum-Exp (`HMMClassifier.classify`)
**Qué se explica en tablero:**
Al evaluar una letra completa o parcial, letras distintas pueden compartir giros angulares (ej. `a` vs `o`, `n` vs `h`, `v` vs `x`, `i` vs `l`). Por ello el puntaje logarítmico integra secuencia + duración + forma + saltos:
$$\text{Score}(c) = \log P(O_{1:t} \mid \lambda_c) - P_{\text{dur}}(t) - P_{\text{shape}}(\mathbf{g}) - P_{\text{jump}}(J_{\text{obs}})$$
Con prior uniforme $P(c)=1/27$, normalizamos numéricamente sin overflow con la identidad **Log-Sum-Exp**:
$$P(c \mid O_{1:t}, \mathbf{g}) = \exp\!\left(\text{Score}(c) - m - \log\sum_{c'\in\mathcal{C}} e^{\text{Score}(c') - m}\right), \qquad m = \max_{c'} \text{Score}(c')$$

| Término en $\text{Score}(c)$ | Implementación Exacta en el Código |
|---|---|
| Penalización de duración $P_{\text{dur}}(T) = \frac{1}{2}\big(\frac{T - \mu_c}{\max(\sigma_c, 4)}\big)^2$ (en parciales solo si $T > \mu_c + 1.5\sigma_c$) | [`HMM.length_deviation_penalty`](../core/HMMEngine.py#L76-L95) |
| Penalización morfológica $P_{\text{shape}}(\mathbf{g}) = w\cdot\frac{1}{2}\sum_{d=1}^7 \big(\frac{g_d - \mu_{\text{shape},c,d}}{\sigma_{\text{shape},c,d}}\big)^2$ | [`HMM.shape_penalty`](../core/HMMEngine.py#L97-L124) |
| Penalización estructural de saltos entre sub-trazos $P_{\text{jump}}$ | [`HMM.jump_penalty`](../core/HMMEngine.py#L126-L141) |
| Clasificación multiclase y normalización Log-Sum-Exp | [`HMMClassifier.classify`](../core/HMMEngine.py#L504-L539) |

> **Nota clave de sustentación:** Como Bayes normaliza sobre un conjunto **cerrado** de 27 letras ($\sum_{c=1}^{27} P(c \mid O) = 100\%$), al finalizar cualquier trazo la clase ganadora se acerca al $100\%$ cuando supera al segundo lugar por $\Delta\text{Score} \gtrsim 15\text{–}20$. La calidad absoluta del parecido sigue estando medida por el $\text{Score}(c)$ logarítmico bruto (una letra bien trazada da $-25$ a $-55$; un garabato cae por debajo de $-170$).

---

### 2.5 Problema 2 — Decodificación: Viterbi en Log-Espacio (`HMM.viterbi`)
**Qué se explica en tablero:**
Forward suma todos los caminos posibles; **Viterbi** halla el **único camino óptimo de estados** $Q^* = \arg\max_Q P(Q, O \mid \lambda)$ para saber en qué fase $q_t^*$ de la letra va el usuario. Sumamos logaritmos en vez de multiplicar probabilidades:
$$\delta_t(j) = \max_{0 \le i < N} \big[\delta_{t-1}(i) + \log a_{ij}\big] + \log b_j(O_t), \qquad \psi_t(j) = \operatorname{argmax}_{0 \le i < N} \big[\delta_{t-1}(i) + \log a_{ij}\big]$$
Seguido de backtracking $q_t^* = \psi_{t+1}(q_{t+1}^*)$ en [`HMM.viterbi`](../core/HMMEngine.py#L190-L214).

---

### 2.6 Problema 3 — Aprendizaje: Baum-Welch (EM) Escalado (`HMM.backward` y `HMM.fit`)
**Qué se explica en tablero:**
Al calcular $\hat\beta_t(i)$ con los **mismos** factores $c_t$ de Forward, los productos de escala se cancelan exactamente en las variables posteriores del Paso E:
$$\hat\alpha_t(i)\hat\beta_t(i) = c_t \left(\prod_{\tau=1}^T c_\tau\right)\alpha_t(i)\beta_t(i) = c_t\,\gamma_t(i) \implies \gamma_t(i) = \frac{\hat\alpha_t(i)\hat\beta_t(i)}{c_t}$$
$$\xi_t(i, j) = \hat\alpha_t(i)\, a_{ij}\, b_j(O_{t+1})\, \hat\beta_{t+1}(j)$$
Y en el Paso M se acumulan conteos sobre todas las secuencias de entrenamiento $k = 1, \dots, K_{\text{seq}}$, aplicando al final suavizado circular de emisiones entre sectores vecinos ($0.78 \cdot b_j(k) + 0.11 \cdot b_j(k\pm 1 \bmod K)$) para tolerar variaciones angulares humanas de $\pm 22.5^\circ$:
- Backward: [`HMM.backward`](../core/HMMEngine.py#L171-L185)
- Paso E + Paso M + suavizado circular: [`HMM.fit`](../core/HMMEngine.py#L219-L323)
- Orquestación offline: [`_train_letter`](../training/train.py#L21-L54)

---

### 2.7 Proyección Dinámica de Trayectoria Futura (`project_future_symbols` y Mapa Exponencial)
**Qué se explica en tablero:**
1. Se elige la letra ganadora $\hat c$ y se decodifica su estado actual $q_t^*$ con Viterbi, contando cuántos pasos consecutivos $k_{\text{cons}}$ lleva en ese estado.
2. En una cadena de Markov, permanecer exactamente $d_j$ pasos en $S_j$ sigue una **distribución geométrica** con probabilidad de éxito $(1 - a_{jj})$, cuya esperanza matemática es:
   $$\mathbb{E}[d_j] = \frac{1}{1 - a_{jj}}$$
3. Desde el estado actual hasta $N-1$, se emiten los símbolos más probables $\hat o_j = \arg\max_k b_j(k)$ en [`HMM.project_future_symbols`](../core/HMMEngine.py#L328-L369).
4. Reconstrucción espacial desde la última posición del lápiz:
   - **En 2D** ([`FeatureExtractor2D.reconstruct_projected_strokes`](../core/FeatureExtractor.py#L265-L285)): $\hat p_k = \hat p_{k-1} + D\cdot(\cos\phi, \sin\phi)$ y desnormalización $p = s\cdot\hat p + p_{\min}$.
   - **En 3D** ([`FeatureExtractor3D.step_geodesic`](../core/FeatureExtractor.py#L519-L537) y [`reconstruct_projected_strokes`](../core/FeatureExtractor.py#L539-L557)): **mapa exponencial sobre la esfera $S^2$**:
     $$\hat p_k = \cos(\delta)\,\hat p_{k-1} + \sin(\delta)\,\hat t(\psi_{\hat o_k})$$

---

## 3. Preguntas Frecuentes en la Sustentación y Respuesta Exacta

| Pregunta del Docente | Respuesta Matemática y Técnica Respaldada en el Código |
|---|---|
| **¿Por qué su sistema no depende de si dibujo lento o rápido?** | Porque tanto `FeatureExtractor2D.resample` como `FeatureExtractor3D.resample_geodesic` parametrizan la curva por **longitud de arco** cada paso $D=0.05$ (o $D_{\text{geo}}$ con SLERP). Los timestamps `[t]` no entran en el extractor. |
| **¿Por qué eligieron $K=16$ sectores en lugar de $K=8$?** | Cada sector en $K=16$ mide $22.5^\circ$ (vs $45^\circ$ en $K=8$), permitiendo distinguir curvas finas (`a`, `c`, `e`, `g`, `s`). Además, [`training/evaluate.py`](../training/evaluate.py) compara experimentalmente ambos valores en prefijos $30\%, 50\%, 70\%, 100\%$. |
| **¿Cómo evitan la singularidad de los polos y el salto $\pm\pi$ en 3D?** | Primero rotamos la esfera al marco centrado en la cámara $R_{\text{cam}}$ para dibujar en el **ecuador frontal** ($\theta=\pi/2, \phi=0$), aplicamos `np.unwrap` al ángulo azimutal $\phi$ y medimos el rumbo en la base ortonormal local $(d\phi\sin\bar\theta, -d\theta)$. |
| **¿Cómo distinguen letras con trazos parecidos (`a` vs `o`, `v` vs `x`, `i` vs `l`, `n` vs `h`)?** | Combinando la secuencia HMM con la **longitud esperada** $P_{\text{dur}}$ (`h` vs `n`), el **descriptor morfológico 7D** $P_{\text{shape}}$ (`a` termina abajo-derecha vs `o` cierra arriba), la **penalización de saltos** $P_{\text{jump}}$ (1 trazo en `v`/`l` vs 2 trazos en `x`/`i`) y el **suavizado circular angular** en $B$. |
| **¿Por qué cuando termino un garabato marca 100% en alguna letra?** | Porque el clasificador aplica el **Teorema de Bayes** sobre un alfabeto cerrado de 27 clases sin clase de rechazo ($\sum_{c=1}^{27} P(c \mid O) = 1$). Aunque el ganador alcance $100\%$ relativo al superar al segundo por $\Delta\text{Score} \ge 20$, su $\text{Score}$ logarítmico bruto cae a $\approx -173$ (frente a $-30$ en una letra real). |
| **¿Cómo demuestran a mano Forward y Viterbi en tablero?** | Con el ejemplo completo $N=2, T=2$ ($\pi=[1,0]^T, A=[[0.6,0.4],[0,1]], B=[[0.8,0.2],[0.1,0.9]], O=(0,1)$) resuelto paso a paso en la Sección 8 de [`docs/ecuaciones.md`](ecuaciones.md#L208-L229). |
