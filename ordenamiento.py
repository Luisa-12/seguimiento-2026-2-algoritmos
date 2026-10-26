"""
ordenamiento.py
---------------
Algoritmos de ordenamiento NO basados en comparaciones, aplicados a la
lista de frecuencias [(término, frecuencia), ...].

Los tres ordenan de MAYOR a MENOR frecuencia y son ESTABLES: si dos
términos tienen la misma frecuencia conservan el orden de entrada (que es
alfabético, ver preprocesamiento.py). Así los tres producen exactamente
el mismo ranking y se pueden validar entre sí.

Para obtener orden descendente sin comparar elementos se usa la clave
invertida  c = f_max - f  (un número entero >= 0) y se ordena de forma
ascendente por c. Calcularla cuesta O(n).

Notación:
    n = número de términos distintos (tamaño de la lista)
    k = rango de las claves = f_max - f_min + 1
    d = número de dígitos de f_max en la base b (Radix Sort)
"""


# ---------------------------------------------------------------------------
# 1. COUNTING SORT  ->  O(n + k)
# ---------------------------------------------------------------------------
def counting_sort(pares):
    n = len(pares)
    if n == 0:
        return []
    f_max = max(f for _, f in pares)                 # O(n)
    f_min = min(f for _, f in pares)                 # O(n)
    k = f_max - f_min + 1

    # (a) Contar cuántos términos tienen cada clave           O(n)
    conteo = [0] * k                                 # O(k)
    for _, f in pares:
        conteo[f_max - f] += 1

    # (b) Suma acumulada: conteo[c] = posición final siguiente
    #     al último elemento con clave c                        O(k)
    for c in range(1, k):
        conteo[c] += conteo[c - 1]

    # (c) Colocar cada elemento directamente en su posición,
    #     recorriendo la entrada al revés para ser estable      O(n)
    salida = [None] * n
    for i in range(n - 1, -1, -1):
        par = pares[i]
        c = f_max - par[1]
        conteo[c] -= 1
        salida[conteo[c]] = par
    return salida


# ---------------------------------------------------------------------------
# 2. RADIX SORT (LSD)  ->  O(d * (n + b))
# ---------------------------------------------------------------------------
def _counting_sort_por_digito(pares, claves, exp, base):
    """Counting Sort estable usando sólo el dígito (clave // exp) % base."""
    n = len(pares)
    conteo = [0] * base
    for c in claves:
        conteo[(c // exp) % base] += 1
    for i in range(1, base):
        conteo[i] += conteo[i - 1]
    salida_pares = [None] * n
    salida_claves = [0] * n
    for i in range(n - 1, -1, -1):
        dig = (claves[i] // exp) % base
        conteo[dig] -= 1
        salida_pares[conteo[dig]] = pares[i]
        salida_claves[conteo[dig]] = claves[i]
    return salida_pares, salida_claves


def radix_sort(pares, base=10):
    n = len(pares)
    if n == 0:
        return []
    f_max = max(f for _, f in pares)
    claves = [f_max - f for _, f in pares]           # claves >= 0
    clave_max = max(claves)

    salida = list(pares)
    exp = 1
    # Una pasada por dígito, del menos al más significativo: d pasadas
    while clave_max // exp > 0:
        salida, claves = _counting_sort_por_digito(salida, claves, exp, base)
        exp *= base
    return salida


# ---------------------------------------------------------------------------
# 3. BUCKET SORT  ->  O(n) esperado
# ---------------------------------------------------------------------------
def bucket_sort(pares, num_cubetas=None):
    n = len(pares)
    if n == 0:
        return []
    f_max = max(f for _, f in pares)
    f_min = min(f for _, f in pares)
    rango = f_max - f_min + 1
    m = num_cubetas or n                             # por defecto m = n cubetas

    # (a) Distribución: el índice de cubeta se CALCULA a partir del valor
    #     (no se compara con otros elementos)                   O(n + m)
    cubetas = [[] for _ in range(m)]
    for par in pares:
        c = f_max - par[1]                           # clave: 0 .. rango-1
        indice = (c * m) // rango                    # 0 .. m-1
        cubetas[indice].append(par)

    # (b) Ordenar cada cubeta con Insertion Sort estable por la clave
    #     c = f_max - f (clave mayor <=> frecuencia menor).
    #     Con m = n cubetas el tamaño esperado de cada una es O(1).
    for cubeta in cubetas:
        for i in range(1, len(cubeta)):
            actual = cubeta[i]
            j = i - 1
            while j >= 0 and cubeta[j][1] < actual[1]:
                cubeta[j + 1] = cubeta[j]
                j -= 1
            cubeta[j + 1] = actual

    # (c) Concatenar las cubetas en orden                      O(n + m)
    salida = []
    for cubeta in cubetas:
        salida.extend(cubeta)
    return salida


ALGORITMOS = {
    "Counting Sort": counting_sort,
    "Radix Sort": radix_sort,
    "Bucket Sort": bucket_sort,
}
