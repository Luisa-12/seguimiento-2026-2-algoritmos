"""
main.py
-------
Requerimiento 3 - Ranking de términos mediante algoritmos de ordenamiento
no basados en comparaciones (Counting Sort, Radix Sort, Bucket Sort).

Flujo:
    1. Carga el corpus (datos/corpus.json). Si no existe, lo genera desde
       los PDF con extraccion.py.
    2. Preprocesa los 20 abstracts y construye la lista de frecuencias.
    3. Ejecuta los tres algoritmos sobre la lista completa, midiendo el
       tiempo real de ejecución en milisegundos (REPETICIONES corridas).
    4. Valida que los tres rankings sean idénticos entre sí y frente a un
       ordenamiento de referencia (sorted de Python, estable).
    5. Calcula las operaciones elementales que predice cada fórmula y las
       compara con los tiempos medidos.
    6. Experimento de escalabilidad: repite la medición con la lista
       replicada x1..x16 para comprobar que el tiempo crece linealmente.
    7. Exporta resultados (CSV, JSON) y gráficas (PNG) a resultados/.

Uso:
    python main.py [carpeta_con_los_pdf]
"""

import csv
import gc
import json
import os
import platform
import statistics
import sys
import time

from preprocesamiento import lista_de_frecuencias
from ordenamiento import ALGORITMOS
from analisis_empirico import (operaciones, escalabilidad, graficar_escalabilidad,
                               FACTORES, REPETICIONES_ESC)

REPETICIONES = 500      # corridas medidas por algoritmo
CALENTAMIENTO = 5       # corridas previas no medidas
TOP = 15
SALIDA = "resultados"


def cargar_corpus(carpeta_pdf):
    ruta = os.path.join("datos", "corpus.json")
    if not os.path.exists(ruta):
        from extraccion import construir_corpus
        corpus = construir_corpus(carpeta_pdf)
        os.makedirs("datos", exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def medir(algoritmo, pares):
    """Devuelve (resultado, lista de tiempos en ms)."""
    for _ in range(CALENTAMIENTO):
        algoritmo(list(pares))
    tiempos = []
    resultado = None
    # Se desactiva el recolector de basura durante la medición (igual que
    # hace el módulo timeit) para que sus pausas no contaminen los tiempos.
    gc_activo = gc.isenabled()
    gc.disable()
    for _ in range(REPETICIONES):
        entrada = list(pares)                 # copia fuera del cronómetro
        t0 = time.perf_counter_ns()
        resultado = algoritmo(entrada)
        t1 = time.perf_counter_ns()
        tiempos.append((t1 - t0) / 1_000_000)  # ns -> ms
    if gc_activo:
        gc.enable()
    return resultado, tiempos


def guardar_csv(nombre, encabezado, filas):
    with open(os.path.join(SALIDA, nombre), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(encabezado)
        w.writerows(filas)


def graficar(top, resumen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    azul, tinta, tinta2, rejilla = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    # --- Top 15 términos --------------------------------------------------
    terminos = [t for t, _ in top][::-1]
    frec = [f for _, f in top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=200)
    barras = ax.barh(terminos, frec, color=azul, height=0.68)
    for b, f in zip(barras, frec):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, str(f),
                va="center", fontsize=9, color=tinta2)
    ax.set_xlabel("Frecuencia absoluta en los 20 abstracts", color=tinta2)
    ax.set_title("15 términos más frecuentes del corpus\n"
                 "(ranking idéntico con Counting, Radix y Bucket Sort)",
                 color=tinta, fontsize=11, loc="left")
    ax.grid(axis="x", color=rejilla, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(rejilla)
    ax.tick_params(colors=tinta2, length=0)
    ax.set_xlim(0, max(frec) * 1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "top15_terminos.png"))
    plt.close(fig)

    # --- Tiempos -----------------------------------------------------------
    nombres = list(resumen.keys())
    medianas = [resumen[n]["mediana"] for n in nombres]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=200)
    barras = ax.bar(nombres, medianas, color=azul, width=0.5)
    for b, m in zip(barras, medianas):
        ax.text(b.get_x() + b.get_width() / 2, m + max(medianas) * 0.03,
                f"{m:.3f} ms", ha="center", fontsize=9, color=tinta)
    ax.set_ylabel("Tiempo (ms)", color=tinta2)
    ax.set_title(f"Tiempo de ejecución sobre la lista de frecuencias\n"
                 f"(mediana de {REPETICIONES} corridas)",
                 color=tinta, fontsize=11, loc="left")
    ax.grid(axis="y", color=rejilla, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(rejilla)
    ax.tick_params(colors=tinta2, length=0)
    ax.set_ylim(0, max(medianas) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "tiempos_algoritmos.png"))
    plt.close(fig)


def main():
    carpeta_pdf = sys.argv[1] if len(sys.argv) > 1 else "articulos"
    os.makedirs(SALIDA, exist_ok=True)

    corpus = cargar_corpus(carpeta_pdf)
    pares, total_crudos, total_validos = lista_de_frecuencias(corpus)
    n = len(pares)
    f_max = max(f for _, f in pares)
    f_min = min(f for _, f in pares)
    print(f"Artículos: {len(corpus)} | tokens: {total_crudos} | "
          f"términos válidos: {total_validos} | términos distintos (n): {n}")
    print(f"Frecuencia mínima: {f_min} | máxima: {f_max} | k = {f_max - f_min + 1}")

    guardar_csv("frecuencias_entrada.csv", ["termino", "frecuencia"], pares)

    # Ordenamiento de referencia para validar (no forma parte del análisis)
    referencia = sorted(pares, key=lambda p: -p[1])

    resultados, resumen = {}, {}
    for nombre, algoritmo in ALGORITMOS.items():
        resultado, tiempos = medir(algoritmo, pares)
        resultados[nombre] = resultado
        resumen[nombre] = {
            "media": statistics.mean(tiempos),
            "mediana": statistics.median(tiempos),
            "minimo": min(tiempos),
            "maximo": max(tiempos),
            "desv": statistics.stdev(tiempos),
            "correcto": resultado == referencia,
        }
        print(f"{nombre:<14} media {resumen[nombre]['media']:.4f} ms | "
              f"mediana {resumen[nombre]['mediana']:.4f} ms | "
              f"mín {resumen[nombre]['minimo']:.4f} ms | "
              f"correcto: {resumen[nombre]['correcto']}")

    nombres = list(ALGORITMOS)
    identicos = all(resultados[x] == resultados[nombres[0]] for x in nombres)
    print(f"Los tres rankings completos son idénticos: {identicos}")

    # --- Exportar ------------------------------------------------------------
    guardar_csv("tiempos.csv",
                ["algoritmo", "media_ms", "mediana_ms", "minimo_ms", "maximo_ms",
                 "desv_estandar_ms", "repeticiones", "coincide_con_referencia"],
                [[x, f"{r['media']:.4f}", f"{r['mediana']:.4f}", f"{r['minimo']:.4f}",
                  f"{r['maximo']:.4f}", f"{r['desv']:.4f}", REPETICIONES, r["correcto"]]
                 for x, r in resumen.items()])

    filas = []
    for i in range(TOP):
        fila = [i + 1]
        for x in nombres:
            fila += list(resultados[x][i])
        filas.append(fila)
    encabezado = ["posicion"]
    for x in nombres:
        encabezado += [f"{x} - termino", f"{x} - frecuencia"]
    guardar_csv("top15.csv", encabezado, filas)
    guardar_csv("ranking_completo.csv", ["posicion", "termino", "frecuencia"],
                [[i + 1, t, f] for i, (t, f) in enumerate(resultados[nombres[0]])])

    # --- Operaciones elementales vs. tiempo medido --------------------------
    ops = operaciones(pares)
    base = nombres[0]
    print("\nOperaciones elementales estimadas (fórmula -> total):")
    for x in nombres:
        ops[x]["razon_predicha"] = ops[x]["total"] / ops[base]["total"]
        ops[x]["razon_medida"] = resumen[x]["mediana"] / resumen[base]["mediana"]
        ops[x]["ns_por_operacion"] = resumen[x]["mediana"] * 1e6 / ops[x]["total"]
        print(f"  {x:<14} {ops[x]['formula']:<18} = {ops[x]['reemplazo']:<28} = "
              f"{ops[x]['total']:>6} | razón predicha {ops[x]['razon_predicha']:.2f} | "
              f"razón medida {ops[x]['razon_medida']:.2f}")
    guardar_csv("operaciones.csv",
                ["algoritmo", "formula", "reemplazo", "operaciones", "razon_predicha",
                 "razon_medida", "ns_por_operacion"],
                [[x, ops[x]["formula"], ops[x]["reemplazo"], ops[x]["total"],
                  f"{ops[x]['razon_predicha']:.2f}", f"{ops[x]['razon_medida']:.2f}",
                  f"{ops[x]['ns_por_operacion']:.1f}"] for x in nombres])

    # --- Experimento de escalabilidad --------------------------------------
    print(f"\nEscalabilidad (lista x{FACTORES[0]}..x{FACTORES[-1]}, "
          f"mínimo de {REPETICIONES_ESC} corridas por tamaño)...")
    filas_esc, ajustes = escalabilidad(pares, ALGORITMOS)
    for fila in filas_esc:
        print(f"  n = {fila['n']:>6} | " +
              " | ".join(f"{x}: {fila[x]:.3f} ms" for x in nombres))
    for x in nombres:
        print(f"  {x:<14} R² = {ajustes[x]['r2']:.4f} | crecimiento x{FACTORES[-1]} "
              f"en n -> x{ajustes[x]['crecimiento_medido']:.1f} en tiempo")
    guardar_csv("escalabilidad.csv",
                ["factor", "n"] + [f"{x}_ms" for x in nombres] +
                ["esperado_lineal", "esperado_nlogn", "esperado_cuadratico"],
                [[f["factor"], f["n"]] + [f"{f[x]:.4f}" for x in nombres] +
                 [f"{f['esperado_lineal']:.1f}", f"{f['esperado_nlogn']:.1f}",
                  f"{f['esperado_cuadratico']:.0f}"] for f in filas_esc])
    graficar_escalabilidad(filas_esc, nombres,
                           os.path.join(SALIDA, "escalabilidad.png"))

    with open(os.path.join(SALIDA, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump({
            "entorno": {"python": platform.python_version(),
                        "sistema": platform.platform(),
                        "procesador": platform.processor() or platform.machine()},
            "corpus": {"articulos": len(corpus), "tokens": total_crudos,
                       "terminos_validos": total_validos, "n": n,
                       "f_min": f_min, "f_max": f_max, "k": f_max - f_min + 1,
                       "terminos_frecuencia_1": sum(1 for _, f in pares if f == 1)},
            "tiempos_ms": resumen,
            "repeticiones": REPETICIONES,
            "rankings_identicos": identicos,
            "operaciones": ops,
            "escalabilidad": {"filas": filas_esc, "ajustes": ajustes,
                              "repeticiones": REPETICIONES_ESC},
            "top15": resultados[nombres[0]][:TOP],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nTop {TOP}:")
    for i, (t, f) in enumerate(resultados[nombres[0]][:TOP], 1):
        print(f"  {i:>2}. {t:<16} {f}")

    graficar(resultados[nombres[0]][:TOP], resumen)
    print(f"\nResultados guardados en {SALIDA}/")


if __name__ == "__main__":
    main()
