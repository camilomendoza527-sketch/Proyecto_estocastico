# Guía de Defensa Matemática en Tablero — Proyecto Final HMM 2D/3D

Este documento contiene el desarrollo matemático completo y paso a paso de todas las ecuaciones programadas en [`core/HMMEngine.py`](../core/HMMEngine.py), [`core/topology.py`](../core/topology.py) y [`core/FeatureExtractor.py`](../core/FeatureExtractor.py), estructurado específicamente para la **sustentación oral en tablero** (SPEC §5.2).

---

## 1. Definición Formal del Modelo HMM $\lambda_c = (A, B, \pi)$ y Topología Bakis

Para cada una de las 27 letras minúsculas del alfabeto español $c \in \mathcal{C} = \{\text{a}, \text{b}, \dots, \text{ñ}, \dots, \text{z}\}$ se construye un Modelo Oculto de Markov discreto independiente $\lambda_c = (A_c, B_c, \pi_c)$:

1. **Espacio de estados ocultos ($S$)**:
   $$S = \{S_0, S_1, \dots, S_{N-1}\}, \qquad N \in [4, 11] \text{ según la complejidad de la letra}$$
   Cada estado representa un tramo secuencial del trazo de la letra.

2. **Alfabeto discreto de observaciones ($V$)**:
   $$V = \{0, 1, \dots, K-1\} \cup \{K, K+1, K+2, K+3\}, \qquad M = |V| = K + 4$$
   donde $0 \dots K-1$ son las $K=16$ (o $K=8$) direcciones con la pluma abajo y $K \dots K+3$ son los 4 rumbos cardinales $(\to, \uparrow, \leftarrow, \downarrow)$ de salto con la pluma levantada entre sub-trazos.

3. **Parámetros estocásticos ($\lambda = (A, B, \pi)$)**:
   - **Vector de distribución inicial** $\pi \in \mathbb{R}^N$:
     $$\pi_i = P(q_1 = S_i), \qquad \sum_{i=0}^{N-1} \pi_i = 1$$
   - **Matriz de transición de estados** $A = [a_{ij}] \in \mathbb{R}^{N \times N}$:
     $$a_{ij} = P(q_{t+1} = S_j \mid q_t = S_i), \qquad \sum_{j=0}^{N-1} a_{ij} = 1 \;\;\forall i$$
   - **Matriz de probabilidad de emisión** $B = [b_j(k)] \in \mathbb{R}^{N \times M}$:
     $$b_j(k) = P(O_t = v_k \mid q_t = S_j), \qquad \sum_{k=0}^{M-1} b_j(k) = 1 \;\;\forall j$$

### 1.1 Restricción de Topología Bakis (Izquierda-Derecha)
Dado que un carácter escrito evoluciona en el tiempo sin retroceder a fases iniciales, en [`core/topology.py`](../core/topology.py) se impone la topología **Bakis izquierda-derecha**:
$$\pi = [1, 0, 0, \dots, 0]^T, \qquad a_{ij} = 0 \quad \text{para todo } j < i \text{ o } j > i + 2$$
Es decir, desde el estado $S_i$ solo se permite permanecer en $S_i$ ($j=i$), avanzar al siguiente estado ($j=i+1$) o saltar un estado ($j=i+2$).

### 1.2 Inicialización por Alineación Segmentaria Uniforme
Si todas las filas de $B$ se inicializan uniformes ($b_j(k) = 1/M$), Baum-Welch parte de un punto silla simétrico. Para romper la simetría (`init_segmental_bakis_hmm`), cada secuencia de entrenamiento $O^{(r)}$ de longitud $T_r$ se divide en $N$ segmentos uniformes asignando al instante $t \in \{0, \dots, T_r - 1\}$ el estado:
$$q_t^{(r)} = \min\!\left(N - 1,\; \left\lfloor \frac{t \cdot N}{T_r} \right\rfloor\right)$$
y se estiman los conteos iniciales con suavizado de Laplace $\alpha = 0.1$:
$$b_j^{(0)}(k) = \frac{\sum_{r,t} \mathbf{1}(q_t^{(r)} = j,\, O_t^{(r)} = k) + \alpha}{\sum_{r,t} \mathbf{1}(q_t^{(r)} = j) + M\alpha}$$

---

## 2. Extracción Geométrica y Discretización 2D (`FeatureExtractor2D`)

Implementada en [`core/FeatureExtractor.py`](../core/FeatureExtractor.py). Recibe $m$ sub-trazos crudos $P^{(1)}, \dots, P^{(m)}$:

### Paso 1: Inversión de eje de pantalla y normalización de escala
En pantalla el eje $y$ crece hacia abajo; internamente trabajamos en coordenadas cartesianas estándar ($y$ hacia arriba):
$$y \leftarrow -y$$
Sea $p_{\min} = (x_{\min}, y_{\min})$ y $p_{\max} = (x_{\max}, y_{\max})$ la caja envolvente de **todos** los sub-trazos de la letra, con ancho $w = x_{\max} - x_{\min}$ y alto $h = y_{\max} - y_{\min}$:
$$s = \begin{cases} \max(w,\, h,\, 10^{-6}) & \text{modo completo (\texttt{realtime=False})} \\ \max(w,\, h,\, s_{\text{esp}}) & \text{modo tiempo real (\texttt{realtime=True})} \end{cases}, \qquad p'_i = \frac{p_i - p_{\min}}{s}$$
- **Justificación para el tablero**: En tiempo real ($s_{\text{esp}} = 250\text{ px}$), mientras el usuario apenas lleva el primer tramo del trazo ($w, h < s_{\text{esp}}$), dividir por $s_{\text{esp}}$ evita estirar ese pequeño segmento hasta tamaño $1$, logrando que los símbolos extraídos del trazo parcial coincidan exactamente con el prefijo de la letra completa.

### Paso 2: Suavizado por promedio móvil simétrico con bordes fijos
Para eliminar el temblor del pulso sin encoger los extremos del trazo (ventana impar $w = 3$, radio $r = (w-1)/2 = 1$):
$$\tilde{p}_i = \frac{1}{2r_i + 1} \sum_{j = -r_i}^{r_i} p'_{i+j}, \qquad r_i = \min(r,\; i,\; n - 1 - i)$$
En los extremos $i=0$ e $i=n-1$, $r_i = 0 \implies \tilde{p}_0 = p'_0$ y $\tilde{p}_{n-1} = p'_{n-1}$.

### Paso 3: Remuestreo a paso de longitud de arco constante $D$
Para lograr **invariancia a la velocidad de dibujo**, se parametriza la polilínea por su longitud de arco acumulada:
$$s_0 = 0, \qquad s_i = \sum_{k=1}^{i} \|\tilde{p}_k - \tilde{p}_{k-1}\|_2$$
Se muestrean posiciones equiespaciadas cada $D = 0.05$:
$$u_m = m \cdot D, \qquad m = 0, 1, \dots, \left\lfloor \frac{s_n}{D} \right\rfloor$$
obteniendo los puntos remuestreados $q_m = \tilde{p}(u_m)$ por interpolación lineal en el intervalo $[s_k, s_{k+1}]$ que contiene a $u_m$.

### Paso 4: Cuantización angular centrada ($K$ sectores + 4 saltos)
Para cada segmento remuestreado $\Delta_m = q_{m+1} - q_m = (\Delta x_m, \Delta y_m)$:
$$\theta_m = \operatorname{atan2}(\Delta y_m, \Delta x_m) \in (-\pi, \pi]$$
Se cuantiza en $K$ sectores de ancho $\Delta\theta = \frac{2\pi}{K}$ centrados (de modo que el sector $0$ cubre $[-\frac{\pi}{K}, +\frac{\pi}{K})$ alrededor del Este $\to$):
$$o_m = \left\lfloor \frac{(\theta_m + \pi/K) \bmod 2\pi}{2\pi / K} \right\rfloor \bmod K \;\;\in \{0, \dots, K-1\}$$
Entre el final del sub-trazo $k$ y el inicio del sub-trazo $k+1$, sea $\Delta_{\text{salto}} = p^{(k+1)}_{\text{inicio}} - p^{(k)}_{\text{fin}}$. Si $\|\Delta_{\text{salto}}\|_2 \ge D$, se emite uno de los 4 símbolos de pluma levantada:
$$o_{\text{salto}} = K + \left(\left\lfloor \frac{(\operatorname{atan2}(\Delta y, \Delta x) + \pi/4) \bmod 2\pi}{\pi / 2} \right\rfloor \bmod 4\right) \;\;\in \{K, K+1, K+2, K+3\}$$

---

## 3. Geometría Diferencial Esférica en el Modo 3D (`FeatureExtractor3D`)

Cuando el usuario dibuja una curva $p_i = (x_i, y_i, z_i) \in \mathbb{R}^3$ sobre la esfera de radio $R$, surgen dos problemas clásicos de las coordenadas esféricas estándar:
1. **Singularidad en los polos** ($\theta = 0, \pi \implies \sin\theta = 0$).
2. **Discontinuidad en el antimeridiano** ($\phi = \pm\pi$).

### Paso 1: Rotación al marco centrado en la vista de la cámara ($R_{\text{cam}} \in SO(3)$)
Dada la orientación de la cámara $(\text{azimut }\alpha, \text{elevación }\beta)$, construimos la base ortonormal derecha:
$$\mathbf{e}_{\text{front}} = (\cos\beta\cos\alpha,\; \cos\beta\sin\alpha,\; \sin\beta)$$
$$\mathbf{e}_{\text{right}} = (-\sin\alpha,\; \cos\alpha,\; 0)$$
$$\mathbf{e}_{\text{up}} = \mathbf{e}_{\text{front}} \times \mathbf{e}_{\text{right}} = (-\sin\beta\cos\alpha,\; -\sin\beta\sin\alpha,\; \cos\beta)$$
$$R_{\text{cam}} = \begin{bmatrix} \mathbf{e}_{\text{front}}^T \\ \mathbf{e}_{\text{right}}^T \\ \mathbf{e}_{\text{up}}^T \end{bmatrix} \in SO(3), \qquad \hat{p}_{\text{cam}, i} = R_{\text{cam}}\,\frac{p_i}{\|p_i\|}$$
En este marco, el centro de la pantalla frente al usuario es $\hat{p}_{\text{cam}} = (1, 0, 0)$, que corresponde a $(\theta = \pi/2, \phi = 0)$ (**pleno ecuador frontal**, donde $\sin\theta \approx 1$, lejos de los polos y lejos de $\phi = \pm\pi$).

### Paso 2: Conversión a coordenadas esféricas $(\theta, \phi)$ y desenvolvimiento
$$\theta_i = \arccos(\operatorname{clip}(z_i, -1, 1)) \in [0, \pi], \qquad \phi_i = \operatorname{atan2}(y_i, x_i) \in (-\pi, \pi]$$
$$\Delta\phi_i = \big((\phi_i - \phi_{i-1} + \pi) \bmod 2\pi\big) - \pi, \qquad \tilde{\phi}_i = \phi_0 + \sum_{k=1}^i \Delta\phi_k$$

### Paso 3: Remuestreo geodésico constante mediante SLERP
La distancia sobre la esfera unitaria $S^2$ entre dos puntos consecutivos es el ángulo de círculo máximo:
$$\Delta s_k = \arccos(\operatorname{clip}(\hat{p}_{k-1} \cdot \hat{p}_k, -1, 1)), \qquad s_i = \sum_{k=1}^i \Delta s_k$$
Con paso geodésico $D_{\text{geo}} = s_{\text{ang}} \cdot D$, cada punto remuestreado $\hat{q}_m$ a distancia $u_m = m \cdot D_{\text{geo}} \in [s_k, s_{k+1}]$ se interpola sobre la geodésica usando **SLERP** (con $\Omega = \Delta s_{k+1}$ y $t = (u_m - s_k)/\Omega$):
$$\hat{q}_m = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,\hat{p}_k + \frac{\sin(t\Omega)}{\sin\Omega}\,\hat{p}_{k+1}$$

### Paso 4: Rumbo en el plano tangente local $\psi = \operatorname{atan2}(-d\theta, d\phi\sin\theta)$
**Demostración para el tablero**:
Sobre la esfera unitaria $\hat{p}(\theta, \phi) = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$, un desplazamiento infinitesimal es:
$$d\hat{p} = \frac{\partial \hat{p}}{\partial \theta} d\theta + \frac{\partial \hat{p}}{\partial \phi} d\phi = \mathbf{e}_\theta\, d\theta + (\sin\theta\,\mathbf{e}_\phi)\, d\phi$$
donde $\mathbf{e}_\phi = (-\sin\phi, \cos\phi, 0)$ apunta al **Este** (derecha en pantalla) y $-\mathbf{e}_\theta = (-\cos\theta\cos\phi, -\cos\theta\sin\phi, \sin\theta)$ apunta al **Norte** (arriba en pantalla). Ambos son unitarios y ortogonales ($\|\mathbf{e}_\phi\| = \|-\mathbf{e}_\theta\| = 1$, $\mathbf{e}_\phi \cdot (-\mathbf{e}_\theta) = 0$).
Por tanto, las componentes métricas ortonormales del desplazamiento en el plano tangente local son:
$$(\text{Este}, \text{Norte}) = (d\phi \cdot \sin\bar{\theta},\; -d\theta)$$
y su ángulo respecto al Este es exactamente:
$$\boxed{\psi_m = \operatorname{atan2}(-d\theta_m,\; d\phi_m \cdot \sin\bar{\theta}_m)}$$
que se cuantiza en los mismos $K$ sectores + 4 saltos que el modo 2D.

---

## 4. Problema 1 — Evaluación: Algoritmo Forward con Escalado (`HMM.forward`)

### 4.1 Por qué se necesita el escalado
La variable forward clásica $\alpha_t(i) = P(O_1, \dots, O_t, q_t = S_i \mid \lambda)$ es un producto de $t$ probabilidades menores que $1$. Cuando $T \approx 40\text{–}80$, $\alpha_T(i) \sim (1/20)^{60} \approx 10^{-78}$, sufriendo **underflow numérico** en punto flotante si $T$ crece.

### 4.2 Ecuaciones del Algoritmo Forward Escalado (Rabiner, 1989)
1. **Inicialización ($t = 1$)**:
   $$\tilde{\alpha}_1(i) = \pi_i \, b_i(O_1), \qquad c_1 = \frac{1}{\sum_{i=0}^{N-1} \tilde{\alpha}_1(i)}, \qquad \hat{\alpha}_1(i) = c_1 \, \tilde{\alpha}_1(i)$$
2. **Inducción ($t = 1, \dots, T-1$)**:
   $$\tilde{\alpha}_{t+1}(j) = \left[\sum_{i=0}^{N-1} \hat{\alpha}_t(i)\, a_{ij}\right] b_j(O_{t+1})$$
   $$c_{t+1} = \frac{1}{\sum_{j=0}^{N-1} \tilde{\alpha}_{t+1}(j)}, \qquad \hat{\alpha}_{t+1}(j) = c_{t+1}\,\tilde{\alpha}_{t+1}(j)$$
3. **Demostración en tablero de la Log-Verosimilitud**:
   Por inducción sobre $t$, definiendo $C_t = \prod_{\tau=1}^t c_\tau$, se cumple que:
   $$\hat{\alpha}_t(i) = C_t \, \alpha_t(i) = \left(\prod_{\tau=1}^t c_\tau\right) \alpha_t(i)$$
   En $t = T$, como los coeficientes $c_T$ se eligen para que $\sum_{i=0}^{N-1} \hat{\alpha}_T(i) = 1$, tenemos:
   $$1 = \sum_{i=0}^{N-1} \hat{\alpha}_T(i) = \left(\prod_{t=1}^T c_t\right) \underbrace{\sum_{i=0}^{N-1} \alpha_T(i)}_{P(O \mid \lambda)} \implies P(O \mid \lambda) = \frac{1}{\prod_{t=1}^T c_t}$$
   Tomando logaritmo natural:
   $$\boxed{\log P(O \mid \lambda) = -\sum_{t=1}^{T} \log(c_t)}$$

### 4.3 Clasificación Bayesiana Multiclase con Log-Sum-Exp (`HMMClassifier.classify`)
Con prior uniforme $P(c) = 1/27$, la probabilidad a posteriori del carácter $c$ dada la secuencia parcial $O_{1:t}$ es:
$$P(c \mid O_{1:t}) = \frac{P(O_{1:t} \mid \lambda_c)}{\sum_{c' \in \mathcal{C}} P(O_{1:t} \mid \lambda_{c'})} = \frac{\exp(L_c)}{\sum_{c' \in \mathcal{C}} \exp(L_{c'})}, \qquad L_c = \log P(O_{1:t} \mid \lambda_c)$$
Para evitar underflow al exponenciar $L_c \ll 0$, se resta el máximo $m = \max_{c' \in \mathcal{C}} L_{c'}$ (**identidad Log-Sum-Exp**):
$$\log \sum_{c'} \exp(L_{c'}) = m + \log\!\left(\sum_{c' \in \mathcal{C}} \exp(L_{c'} - m)\right), \qquad \boxed{P(c \mid O_{1:t}) = \exp\!\left(L_c - m - \log\sum_{c'} e^{L_{c'}-m}\right)}$$

---

## 5. Problema 2 — Decodificación: Algoritmo de Viterbi en Log-Espacio (`HMM.viterbi`)

Para hallar la secuencia óptima de estados ocultos $Q^* = (q_1^*, \dots, q_T^*) = \arg\max_{Q} P(Q, O \mid \lambda)$ sin underflow, operamos con logaritmos definiendo $\delta_t(i) = \max_{q_1,\dots,q_{t-1}} \log P(q_1 \dots q_t = S_i, O_{1:t} \mid \lambda)$:

1. **Inicialización ($t = 1$)**:
   $$\delta_1(i) = \log \pi_i + \log b_i(O_1), \qquad \psi_1(i) = 0$$
2. **Recursión ($t = 2, \dots, T$)**:
   $$\delta_t(j) = \max_{0 \le i < N} \big[\delta_{t-1}(i) + \log a_{ij}\big] + \log b_j(O_t)$$
   $$\psi_t(j) = \operatorname{argmax}_{0 \le i < N} \big[\delta_{t-1}(i) + \log a_{ij}\big]$$
3. **Terminación**:
   $$q_T^* = \operatorname{argmax}_{0 \le i < N} \delta_T(i), \qquad \log P^* = \max_{0 \le i < N} \delta_T(i)$$
4. **Backtracking ($t = T-1, T-2, \dots, 1$)**:
   $$\boxed{q_t^* = \psi_{t+1}(q_{t+1}^*)}$$

---

## 6. Problema 3 — Aprendizaje: Algoritmo Baum-Welch Escalado (`HMM.backward`, `HMM.fit`)

### 6.1 Algoritmo Backward con los mismos factores $c_t$
1. **Inicialización ($t = T$)**:
   $$\hat{\beta}_T(i) = c_T \cdot 1$$
2. **Inducción hacia atrás ($t = T-1, \dots, 1$)**:
   $$\tilde{\beta}_t(i) = \sum_{j=0}^{N-1} a_{ij}\, b_j(O_{t+1})\, \hat{\beta}_{t+1}(j), \qquad \hat{\beta}_t(i) = c_t \, \tilde{\beta}_t(i)$$
Por inducción hacia atrás se verifica que $\hat{\beta}_t(i) = \left(\prod_{\tau=t}^T c_\tau\right) \beta_t(i)$.

### 6.2 Paso E: Variables Posteriores $\gamma_t(i)$ y $\xi_t(i, j)$ (Cancelación exacta de $c_t$)
**Demostración en tablero**:
Observemos qué ocurre al multiplicar $\hat{\alpha}_t(i)$ por $\hat{\beta}_t(i)$:
$$\hat{\alpha}_t(i)\,\hat{\beta}_t(i) = \left(\prod_{\tau=1}^t c_\tau\right)\alpha_t(i) \cdot \left(\prod_{\tau=t}^T c_\tau\right)\beta_t(i) = c_t \underbrace{\left(\prod_{\tau=1}^T c_\tau\right)}_{1 / P(O \mid \lambda)} \alpha_t(i)\beta_t(i) = c_t \, \frac{\alpha_t(i)\beta_t(i)}{P(O \mid \lambda)} = c_t \, \gamma_t(i)$$
Por lo tanto, dividiendo entre $c_t$ se obtiene directamente $\gamma_t(i) = P(q_t = S_i \mid O, \lambda)$:
$$\boxed{\gamma_t(i) = \frac{\hat{\alpha}_t(i)\,\hat{\beta}_t(i)}{c_t}}$$
De igual forma, para $\xi_t(i, j) = P(q_t = S_i, q_{t+1} = S_j \mid O, \lambda)$:
$$\hat{\alpha}_t(i)\, a_{ij}\, b_j(O_{t+1})\, \hat{\beta}_{t+1}(j) = \left(\prod_{\tau=1}^t c_\tau\right)\left(\prod_{\tau=t+1}^T c_\tau\right) \alpha_t(i) a_{ij} b_j(O_{t+1}) \beta_{t+1}(j) = \frac{\alpha_t(i) a_{ij} b_j(O_{t+1}) \beta_{t+1}(j)}{P(O \mid \lambda)}$$
El producto de escalas forma exactamente $\prod_{\tau=1}^T c_\tau = 1/P(O \mid \lambda)$ sin sobrar ningún factor:
$$\boxed{\xi_t(i, j) = \hat{\alpha}_t(i)\, a_{ij}\, b_j(O_{t+1})\, \hat{\beta}_{t+1}(j)}$$

### 6.3 Paso M: Reestimación Multisequencia ($k = 1, \dots, K_{\text{seq}}$)
Sumando los conteos esperados sobre todas las muestras de entrenamiento $O^{(k)}$:
$$\bar{\pi}_i = \frac{1}{K_{\text{seq}}} \sum_{k=1}^{K_{\text{seq}}} \gamma_1^{(k)}(i)$$
$$\bar{a}_{ij} = \frac{\sum_{k=1}^{K_{\text{seq}}} \sum_{t=1}^{T_k - 1} \xi_t^{(k)}(i, j)}{\sum_{k=1}^{K_{\text{seq}}} \sum_{t=1}^{T_k - 1} \gamma_t^{(k)}(i)}, \qquad \bar{b}_j(m) = \frac{\sum_{k=1}^{K_{\text{seq}}} \sum_{t:\, O_t^{(k)} = v_m} \gamma_t^{(k)}(j) + \varepsilon}{\sum_{k=1}^{K_{\text{seq}}} \sum_{t=1}^{T_k} \gamma_t^{(k)}(j) + M\varepsilon}$$

---

## 7. Proyección Dinámica de Trayectoria Futura (Autocompletado Espacial)

Implementada entre [`HMM.project_future_symbols`](../core/HMMEngine.py) y [`reconstruct_projected_strokes`](../core/FeatureExtractor.py) (SPEC §3.5):

1. **Identificación del modelo ganador y estado actual**:
   Dada la secuencia parcial $O_{1:t}$ trazada hasta el instante $t$:
   $$\hat{c} = \operatorname{argmax}_{c \in \mathcal{C}} P(c \mid O_{1:t}), \qquad Q^* = \operatorname{Viterbi}(O_{1:t},\, \lambda_{\hat{c}}), \qquad q_t^* = Q^*_t$$
   Se cuenta cuántos pasos consecutivos $k_{\text{cons}}$ lleva el camino $Q^*$ en el estado actual $q_t^*$.

2. **Deducción del Tiempo de Permanencia Esperado $\mathbb{E}[d_j]$**:
   En una cadena de Markov con probabilidad de auto-transición $a_{jj}$, la cantidad de pasos consecutivos $d_j \in \{1, 2, \dots\}$ que el proceso permanece en el estado $S_j$ sigue una **distribución geométrica**:
   $$P(d_j = k) = (a_{jj})^{k-1}(1 - a_{jj}), \qquad k = 1, 2, 3, \dots$$
   Su esperanza matemática es:
   $$\boxed{\mathbb{E}[d_j] = \sum_{k=1}^{\infty} k\,(a_{jj})^{k-1}(1 - a_{jj}) = \frac{1}{1 - a_{jj}}}$$
   En el estado actual $j = q_t^*$, restan $\max(0, \operatorname{round}(\mathbb{E}[d_j]) - k_{\text{cons}})$ pasos por emitir; en cada estado posterior $j > q_t^*$ hasta $N-1$, se emiten $\operatorname{round}(\mathbb{E}[d_j])$ veces el símbolo dominante:
   $$\hat{o}_j = \operatorname{argmax}_{0 \le k < M} b_j(k)$$
   avanzando al siguiente estado Bakis más probable $j_{\text{next}} = \operatorname{argmax}_{k > j} a_{jk}$.

3. **Reconstrucción Geométrica desde el Cursor Actual**:
   - **En 2D**: A partir del último punto normalizado $\hat{p}_0 = q_{\text{último}}$, cada símbolo proyectado $\hat{o}_k$ avanza:
     $$\hat{p}_k = \hat{p}_{k-1} + D \cdot (\cos\phi_{\hat{o}_k},\; \sin\phi_{\hat{o}_k}), \qquad p_{\text{lienzo}, k} = s \cdot \hat{p}_k + p_{\min}$$
   - **En 3D (Geodésica sobre $S^2$)**: A partir del último punto unitario $\hat{p}_0 \in S^2$, cada símbolo avanza un arco geodésico $\delta = D_{\text{geo}}$ en la dirección tangente unitaria $\hat{t}(\psi_{\hat{o}_k}) = \cos\psi\,\mathbf{e}_\phi + \sin\psi\,(-\mathbf{e}_\theta)$ mediante el **mapa exponencial en $S^2$**:
     $$\boxed{\hat{p}_k = \cos(\delta)\,\hat{p}_{k-1} + \sin(\delta)\,\hat{t}(\psi_{\hat{o}_k}), \qquad p_{\text{mundo}, k} = R \cdot \hat{p}_k \cdot R_{\text{cam}}}$$

---

## 8. Ejercicio Numérico Corto para Demostración en Tablero ($N=2, T=2$)

Si durante la sustentación se solicita resolver a mano un ejemplo numérico de Forward escalado y Viterbi, puede escribirse este modelo de $N=2$ estados y secuencia observada $O = (0, 1)$:

$$\pi = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \qquad A = \begin{bmatrix} 0.6 & 0.4 \\ 0.0 & 1.0 \end{bmatrix}, \qquad B = \begin{bmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \end{bmatrix}, \qquad O = (O_1 = 0,\; O_2 = 1)$$

### A. Forward Escalado a Mano
- **Paso $t = 1$ ($O_1 = 0$)**:
  $$\tilde{\alpha}_1(0) = \pi_0 b_0(0) = 1 \times 0.8 = 0.8, \qquad \tilde{\alpha}_1(1) = \pi_1 b_1(0) = 0 \times 0.1 = 0$$
  $$c_1 = \frac{1}{0.8 + 0} = 1.25 \implies \hat{\alpha}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$
- **Paso $t = 2$ ($O_2 = 1$)**:
  $$\tilde{\alpha}_2(0) = (\hat{\alpha}_1(0) a_{00} + \hat{\alpha}_1(1) a_{10}) b_0(1) = (1.0 \times 0.6) \times 0.2 = 0.12$$
  $$\tilde{\alpha}_2(1) = (\hat{\alpha}_1(0) a_{01} + \hat{\alpha}_1(1) a_{11}) b_1(1) = (1.0 \times 0.4) \times 0.9 = 0.36$$
  $$\sum_j \tilde{\alpha}_2(j) = 0.12 + 0.36 = 0.48 \implies c_2 = \frac{1}{0.48}, \qquad \hat{\alpha}_2 = \begin{bmatrix} 0.25 \\ 0.75 \end{bmatrix}$$
- **Verosimilitud Total**:
  $$P(O \mid \lambda) = \frac{1}{c_1 c_2} = 0.8 \times 0.48 = 0.384, \qquad \log P(O \mid \lambda) = -\log c_1 - \log c_2 = \log(0.384)$$

### B. Viterbi y Proyección a Mano
-En $t=2$, los dos caminos posibles desde $q_1 = S_0$ tienen probabilidad:
  - Camino $(S_0, S_0)$: $0.8 \times 0.6 \times 0.2 = 0.096$
  - Camino $(S_0, S_1)$: $0.8 \times 0.4 \times 0.9 = 0.288 \implies Q^* = (S_0, S_1)$, estado actual $q_2^* = S_1$.
- En $S_1$, el símbolo más probable es $\arg\max_k b_1(k) = 1$, proyectando la dirección $v_1$ para autocompletar el trazo.
