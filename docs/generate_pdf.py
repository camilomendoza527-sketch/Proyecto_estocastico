"""Generador del PDF explicativo e intuitivo del proceso HMM (docs/generate_pdf.py)."""

import sys
from pathlib import Path
from PySide6.QtCore import QMarginsF
from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PDF = PROJECT_ROOT / "docs" / "GUIA_INTUITIVA_PROCESO_HMM.pdf"

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #1e293b;
    line-height: 1.5;
    font-size: 11pt;
    margin: 0;
    padding: 0;
  }
  .header-box {
    background-color: #f0fdf4;
    border: 2px solid #16a34a;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    text-align: center;
  }
  .title {
    color: #166534;
    font-size: 20pt;
    font-weight: bold;
    margin: 0 0 6px 0;
  }
  .subtitle {
    color: #475569;
    font-size: 12pt;
    margin: 0 0 4px 0;
  }
  .meta {
    color: #64748b;
    font-size: 9pt;
    margin-top: 6px;
  }
  h2 {
    color: #0f766e;
    font-size: 13.5pt;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }
  h3 {
    color: #1e293b;
    font-size: 11.5pt;
    margin-top: 12px;
    margin-bottom: 5px;
    page-break-after: avoid;
  }
  p {
    margin: 0 0 8px 0;
    text-align: justify;
  }
  .card, .card-blue, .card-amber {
    page-break-inside: avoid;
  }
  .card {
    background-color: #f8fafc;
    border-left: 4px solid #16a34a;
    padding: 10px 14px;
    margin: 10px 0;
  }
  .card-blue {
    background-color: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 10px 14px;
    margin: 10px 0;
  }
  .card-amber {
    background-color: #fefce8;
    border-left: 4px solid #ca8a04;
    padding: 10px 14px;
    margin: 10px 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }
  th {
    background-color: #f1f5f9;
    color: #334155;
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
    font-weight: bold;
    text-align: left;
  }
  td {
    padding: 7px 10px;
    border: 1px solid #e2e8f0;
    vertical-align: top;
  }
  .badge {
    background-color: #dcfce7;
    color: #166534;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .step-num {
    background-color: #16a34a;
    color: #ffffff;
    font-weight: bold;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 10pt;
  }
  ul, ol {
    margin: 0 0 10px 20px;
    padding: 0;
  }
  li {
    margin-bottom: 5px;
  }
  code {
    font-family: 'Consolas', 'Courier New', monospace;
    background-color: #f1f5f9;
    color: #0f766e;
    padding: 1px 4px;
    font-size: 9.5pt;
  }
</style>
</head>
<body>

<div class="header-box">
  <div class="title">¿Cómo Funciona y Cómo se Entrenó el Reconocedor HMM?</div>
  <div class="subtitle">Guía Intuitiva y Paso a Paso para Entender el Proceso Estocástico</div>
  <div class="meta">Universidad de los Llanos &bull; Procesos Estocásticos &bull; Autor: Camilo Mendoza</div>
</div>

<h2>1. La Gran Pregunta y la Intuición Básica</h2>
<p>
¿Cómo puede una computadora saber qué letra estás dibujando en tiempo real si cada persona escribe diferente, con distinto tamaño y a distinta velocidad?
</p>
<div class="card">
  <strong>La analogía de la brújula:</strong> No importa si dibujas una letra gigante o diminuta, ni si tardas 1 segundo o 10 segundos. Si estás dibujando una <strong>letra 'a'</strong>, tu mano siempre hace el mismo viaje:
  <ol>
    <li>Empieza arriba a la derecha.</li>
    <li>Curva hacia la izquierda y hacia abajo haciendo un círculo.</li>
    <li>Sube a cerrar el círculo.</li>
    <li>Baja verticalmente para hacer el vástago final.</li>
  </ol>
  El secreto del proyecto es <strong>convertir ese viaje del lápiz en una serie de rumbos de brújula</strong> y enseñarle a un modelo estocástico de probabilidades a reconocer ese camino.
</div>

<h2>2. El Rol del Dataset: La Materia Prima (data/)</h2>
<p>
El archivo <code>data/glifo-dataset-2026-11-06.json</code> contiene más de <strong>5.500 trazos reales</strong> hechos por personas. Para trabajar ordenadamente, lo dividimos en 27 archivos individuales dentro de <code>data/letras/</code> (como <code>a.json</code>, <code>b.json</code>, etc.).
</p>
<p>
Cada archivo guarda las coordenadas de pantalla puras:
<br>
<code>[ [x0, y0], [x1, y1], [x2, y2], ... ]</code>
<br>
Estos datos crudos son el "libro de texto" con el que la inteligencia artificial aprende. Sin estos datos, la máquina no sabría cómo luce una letra humana real.
</p>

<h2>3. La Transformación Mágica: De Dibujo a Símbolos</h2>
<p>
Los Modelos Ocultos de Markov (HMM) no entienden coordenadas continuas (como <code>x=142.5, y=320.1</code>). El módulo <code>core/FeatureExtractor.py</code> transforma ese dibujo mediante 3 pasos clave:
</p>

<div class="card-blue">
  <strong>Paso 3.1: Invariancia al Tamaño (Normalización)</strong><br>
  Calculamos la caja que rodea a la letra y la escalamos para que siempre mida <code>1.0 x 1.0</code>. Si la dibujas en una esquina pequeña o en toda la pantalla, el modelo la ve exactamente del mismo tamaño.
</div>

<div class="card-blue">
  <strong>Paso 3.2: Invariancia a la Velocidad (Paso de Longitud de Arco)</strong><br>
  Si escribes muy rápido, el ratón captura pocos puntos; si escribes lento, captura miles. Para solucionarlo, medimos la longitud de la línea y colocamos puntos virtuales equiespaciados cada <code>0.05</code> unidades. ¡Ahora la velocidad de escritura ya no afecta en absoluto!
</div>

<div class="card-blue">
  <strong>Paso 3.3: La Brújula de 16 Sectores</strong><br>
  Miramos la dirección entre cada par de puntos consecutivos y la convertimos en un número del 0 al 15:
  <ul>
    <li><code>0</code> = hacia el Este (derecha &rarr;)</li>
    <li><code>4</code> = hacia el Norte (arriba &uarr;)</li>
    <li><code>8</code> = hacia el Oeste (izquierda &larr;)</li>
    <li><code>12</code> = hacia el Sur (abajo &darr;)</li>
    <li><code>16 al 19</code> = cuando levantas el lápiz para hacer un segundo trazo (ej. el punto de la <code>i</code>).</li>
  </ul>
</div>

<p>
<strong>Resultado:</strong> Tu dibujo de la <code>a</code> se convirtió en una secuencia de números como esta:
<br>
<code>Secuencia de Observaciones O = [7, 8, 9, 10, 11, 12, 0, 1, 4, 12, 12]</code>
</p>

<h2>4. ¿Qué es un HMM y la Topología Bakis?</h2>
<p>
Para cada letra del alfabeto construimos un <strong>Modelo Oculto de Markov</strong> &lambda; = (A, B, &pi;):
</p>
<ul>
  <li><strong>Estados Ocultos (S):</strong> Son las etapas invisibles del trazo. Por ejemplo, en la 'a':
    <br><em>Estado 0:</em> iniciar el círculo &bull; <em>Estado 1:</em> fondo de la curva &bull; <em>Estado 2:</em> subir &bull; <em>Estado 3:</em> bajar la pata.
  </li>
  <li><strong>Topología Bakis (Izquierda-Derecha):</strong> Al escribir, el tiempo solo va hacia adelante. No puedes dibujar la pata final antes de hacer el círculo. La regla Bakis obliga a que desde el estado <code>S_i</code> solo se pueda quedar en <code>S_i</code>, avanzar a <code>S_i+1</code> o saltar a <code>S_i+2</code>. <strong>Nunca se puede retroceder.</strong></li>
  <li><strong>Matriz A (Transición):</strong> Probabilidad de pasar de una etapa a la siguiente.</li>
  <li><strong>Matriz B (Emisión):</strong> Estando en cierta etapa, ¿qué dirección de brújula es más probable ver?</li>
</ul>

<h2>5. El Entrenamiento: Algoritmo de Baum-Welch (EM)</h2>
<p>
¿Cómo aprendió la computadora las matrices A y B para cada letra? Mediante el algoritmo de <strong>Baum-Welch</strong> (Esperanza-Maximización):
</p>
<div class="card-amber">
  <ol>
    <li><strong>Alineación Inicial:</strong> Divide las secuencias de dibujo en tramos iguales para dar una primera suposición de las probabilidades.</li>
    <li><strong>Paso E (Expectation):</strong> Pasa las secuencias de entrenamiento usando las funciones <em>Forward</em> (&alpha;) y <em>Backward</em> (&beta;) y calcula la probabilidad de qué estado produjo cada movimiento.</li>
    <li><strong>Paso M (Maximization):</strong> Ajusta las matrices A y B para aumentar la probabilidad de haber visto esos trazos reales.</li>
    <li><strong>Repetición:</strong> Se repite hasta 25 iteraciones hasta que las matrices se estabilizan y ya no cambian.</li>
  </ol>
</div>

<div class="card">
  <strong>¿Por qué esto resolvió el problema entre 'a' y 'q'?</strong><br>
  Anteriormente se usaban plantillas artificiales casi idénticas. Al entrenar con cientos de muestras humanas reales, Baum-Welch aprendió que la letra <strong>'q'</strong> emite una larga serie de símbolos hacia el sur profundo (<code>12, 12, 12</code>) con un vástago largo, mientras que la <strong>'a'</strong> hace una curva corta y se detiene en la base. ¡La confusión quedó resuelta al 100%!
</div>

<h2>6. ¿Qué hay en la carpeta models/? (El Cerebro Guardado)</h2>
<p>
El entrenamiento toma varios minutos. No podemos hacer esperar al usuario cada vez que abre el programa. Por eso, el resultado final de las matrices de probabilidad se guarda en los <strong>54 archivos JSON de models/</strong> (27 para 2D y 27 para 3D).
</p>
<p>
Cuando abres <code>main.py</code>, la aplicación carga esos JSON en <strong>menos de 1 segundo</strong> en memoria RAM. Están listos para predecir al instante.
</p>

<h2>7. En Tiempo Real: ¿Qué pasa cuando mueves el ratón?</h2>
<p>
Cada vez que mueves el ratón o rotas la esfera 3D:
</p>
<ol>
  <li>El extractor calcula los símbolos del trazo parcial que llevas hasta ese milisegundo.</li>
  <li>El algoritmo <strong>Forward incremental</strong> calcula en tiempo real qué tan probable es esa secuencia para cada uno de los 27 modelos:
    <br>
    <code>P(letra | trazo_actual)</code>
  </li>
  <li>El <strong>RankingPanel</strong> actualiza las barras de probabilidad en pantalla inmediatamente.</li>
  <li>El algoritmo de <strong>Viterbi</strong> averigua en qué estado oculto vas y <strong>proyecta con líneas punteadas el trazo futuro</strong> que te falta por dibujar.</li>
</ol>

<h2>8. Resumen Comparativo de Términos</h2>
<table>
  <tr>
    <th>Concepto</th>
    <th>¿Dónde está en el código?</th>
    <th>¿Qué significa en lenguaje sencillo?</th>
  </tr>
  <tr>
    <td><strong>Dataset Maestro</strong></td>
    <td><code>data/glifo-dataset-*.json</code></td>
    <td>Las 5.500 firmas y dibujos originales hechos por personas de carne y hueso.</td>
  </tr>
  <tr>
    <td><strong>Dataset por Letras</strong></td>
    <td><code>data/letras/*.json</code></td>
    <td>Las muestras humanas organizadas en 27 carpetas/archivos para entrenar cada letra.</td>
  </tr>
  <tr>
    <td><strong>Extractor de Características</strong></td>
    <td><code>core/FeatureExtractor.py</code></td>
    <td>El traductor que convierte trazos de ratón en direcciones de brújula (0..15).</td>
  </tr>
  <tr>
    <td><strong>Algoritmo Baum-Welch</strong></td>
    <td><code>core/HMMEngine.py</code></td>
    <td>El profesor que ajusta las probabilidades de cada letra analizando los ejemplos.</td>
  </tr>
  <tr>
    <td><strong>Archivos de Modelos</strong></td>
    <td><code>models/*.json</code></td>
    <td>Las matrices de probabilidad ya aprendidas y listas para usar.</td>
  </tr>
  <tr>
    <td><strong>Algoritmo Forward</strong></td>
    <td><code>core/HMMEngine.py</code></td>
    <td>El motor en tiempo real que calcula qué porcentaje de parecido tiene tu trazo con cada letra.</td>
  </tr>
</table>

<div class="card" style="text-align: center; margin-top: 20px;">
  <strong>Conclusión:</strong> El sistema es un puente elegante entre la geometría del trazo y la teoría estocástica de probabilidades, permitiendo clasificar y anticipar la intención humana en tiempo real.
</div>

</body>
</html>
"""

def generate_pdf():
    app = QApplication.instance() or QApplication(sys.argv)

    doc = QTextDocument()
    doc.setDefaultFont(QFont("Segoe UI", 10))
    doc.setHtml(HTML_CONTENT)

    writer = QPdfWriter(str(OUTPUT_PDF))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    
    # Márgenes de 18mm
    margins = QMarginsF(18, 18, 18, 18)
    layout = QPageLayout(
        QPageSize(QPageSize.PageSizeId.A4),
        QPageLayout.Orientation.Portrait,
        margins,
        QPageLayout.Unit.Millimeter,
    )
    writer.setPageLayout(layout)

    doc.print_(writer)
    print(f"PDF generado exitosamente en: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()
