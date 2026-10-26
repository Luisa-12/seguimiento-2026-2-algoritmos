# Seguimiento 2026-2 — Requerimiento 3

Ranking de términos con algoritmos de ordenamiento no comparativos
(Counting Sort, Radix Sort, Bucket Sort) sobre los abstracts de los 20 artículos.

Requisitos: Python 3.9+ y

    pip install pypdf matplotlib

Ejecución:

    python main.py      # mide tiempos, operaciones y escalabilidad -> resultados/

Archivos:
- extraccion.py         abstract, título y autores de cada PDF -> datos/corpus.json (ya incluido)
- preprocesamiento.py   tokenización, stopwords, frecuencias, orden alfabético (Radix de cadenas)
- ordenamiento.py       Counting Sort, Radix Sort y Bucket Sort
- analisis_empirico.py  cálculo de operaciones elementales y experimento de escalabilidad
- main.py               medición en ms, validación, CSV y gráficas
- datos/                corpus extraído de los 20 artículos
- resultados/           tiempos, operaciones, escalabilidad, ranking y gráficas de la ejecución entregada
