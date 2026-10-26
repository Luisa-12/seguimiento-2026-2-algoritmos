"""
analisis_empirico.py
--------------------
Complementos de la medición empírica del Requerimiento 3:

1. Cálculo de operaciones elementales: cuántas operaciones predice el
   análisis de cada implementación para la lista del corpus, y cómo se
   compara esa predicción con los tiempos medidos.

2. Experimento de escalabilidad: se ejecutan los tres algoritmos sobre
   listas cada vez más grandes (la lista del corpus replicada 1, 2, 4, 8,
   y 16 veces) para comprobar que el tiempo crece de forma LINEAL con n.

Cómo se cuentan las operaciones: cada recorrido completo de un arreglo de
tamaño n cuenta como n operaciones (una lectura/escritura por elemento).
"""

import gc
import math
import statistics
import time

FACTORES = [1, 2, 4, 8, 16]       # veces que se replica la lista del corpus
REPETICIONES_ESC = 100            # corridas medidas por tamaño
BASE_RADIX = 10


# ---------------------------------------------------------------------------
# 1. Operaciones elementales según cada implementación (ordenamiento.py)
# ---------------------------------------------------------------------------
def digitos(valor, base=BASE_RADIX):
    """Número de pasadas de Radix Sort: dígitos de la clave máxima."""
    d = 0
    while valor > 0:
        valor //= base
        d += 1
    return d


def operaciones(pares):
    """Devuelve, para cada algoritmo, la fórmula, sus parámetros y el total.

    Counting Sort:  max + min (2n) + iniciar conteo (k) + contar (n)
                    + suma acumulada (k) + iniciar salida (n) + colocar (n)
                    = 5n + 2k
    Radix Sort:     max + claves + max de claves (3n) y, por cada uno de los
                    d dígitos: iniciar conteo (b) + contar (n) + acumular (b)
                    + iniciar 2 salidas (2n) + colocar (n)
                    = 3n + d(4n + 2b)
    Bucket Sort:    max + min (2n) + crear m cubetas (m) + distribuir (n)
                    + insertion sort (n - c comparaciones, c = cubetas no
                    vacías) + concatenar (n)
                    = 4n + m + (n - c)
    """
    n = len(pares)
    frec = [f for _, f in pares]
    f_max, f_min = max(frec), min(frec)
    k = f_max - f_min + 1
    d = digitos(f_max - f_min)
    b = BASE_RADIX
    m = n
    c = len(set(frec))            # cubetas no vacías (una por frecuencia distinta)
    return {
        "parametros": {"n": n, "k": k, "d": d, "b": b, "m": m, "c": c},
        "Counting Sort": {"formula": "5n + 2k",
                          "reemplazo": f"5({n}) + 2({k})",
                          "total": 5 * n + 2 * k},
        "Radix Sort": {"formula": "3n + d(4n + 2b)",
                       "reemplazo": f"3({n}) + {d}(4({n}) + 2({b}))",
                       "total": 3 * n + d * (4 * n + 2 * b)},
        "Bucket Sort": {"formula": "4n + m + (n − c)",
                        "reemplazo": f"4({n}) + {m} + ({n} − {c})",
                        "total": 4 * n + m + (n - c)},
    }


# ---------------------------------------------------------------------------
# 2. Experimento de escalabilidad
# ---------------------------------------------------------------------------
def replicar(pares, factor):
    """Lista de tamaño factor·n con la misma distribución de frecuencias
    (cada término se copia con un sufijo), de modo que k no cambia."""
    if factor == 1:
        return list(pares)
    salida = []
    for r in range(factor):
        salida.extend((f"{t}_{r}", f) for t, f in pares)
    return salida


def _minimo_ms(algoritmo, entrada, repeticiones):
    """Tiempo mínimo de varias corridas: es el estimador más estable del
    costo real del algoritmo, porque el ruido del sistema solo SUMA tiempo
    (es el criterio que recomienda la documentación del módulo timeit)."""
    for _ in range(3):
        algoritmo(list(entrada))
    tiempos = []
    gc_activo = gc.isenabled()
    gc.disable()
    for _ in range(repeticiones):
        copia = list(entrada)
        t0 = time.perf_counter_ns()
        algoritmo(copia)
        tiempos.append((time.perf_counter_ns() - t0) / 1_000_000)
    if gc_activo:
        gc.enable()
    return min(tiempos)


def regresion_lineal(xs, ys):
    """Ajuste por mínimos cuadrados  t = a·n + b  y coeficiente R²."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


def escalabilidad(pares, algoritmos):
    n0 = len(pares)
    filas = []
    for factor in FACTORES:
        entrada = replicar(pares, factor)
        n = len(entrada)
        fila = {
            "factor": factor,
            "n": n,
            # Crecimiento que se esperaría para cada complejidad
            "esperado_lineal": n / n0,
            "esperado_nlogn": (n * math.log2(n)) / (n0 * math.log2(n0)),
            "esperado_cuadratico": (n / n0) ** 2,
        }
        for nombre, algoritmo in algoritmos.items():
            fila[nombre] = _minimo_ms(algoritmo, entrada, REPETICIONES_ESC)
        filas.append(fila)

    ajustes = {}
    xs = [f["n"] for f in filas]
    for nombre in algoritmos:
        ys = [f[nombre] for f in filas]
        a, b, r2 = regresion_lineal(xs, ys)
        ajustes[nombre] = {
            "pendiente_ms_por_elemento": a,
            "intercepto_ms": b,
            "r2": r2,
            "crecimiento_medido": ys[-1] / ys[0],
            "ns_por_elemento": [y * 1e6 / x for x, y in zip(xs, ys)],
        }
    return filas, ajustes


def graficar_escalabilidad(filas, nombres, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = ["#2a78d6", "#eb6834", "#1baf7a"]
    tinta, tinta2, rejilla = "#0b0b0b", "#52514e", "#e4e3df"
    xs = [f["n"] for f in filas]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for nombre, color in zip(nombres, colores):
        ys = [f[nombre] for f in filas]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                label=nombre)
        ax.annotate(nombre, (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=tinta)
    ax.set_xlabel("Tamaño de la lista n (términos)", color=tinta2)
    ax.set_ylabel("Tiempo mínimo (ms)", color=tinta2)
    ax.set_title("Escalabilidad: tiempo de ejecución frente a n\n"
                 "(lista del corpus replicada ×1 a ×16)",
                 color=tinta, fontsize=11, loc="left")
    ax.grid(color=rejilla, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(rejilla)
    ax.tick_params(colors=tinta2, length=0)
    ax.set_xlim(0, xs[-1] * 1.22)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
