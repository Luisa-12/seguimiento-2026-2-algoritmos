"""
preprocesamiento.py
-------------------
Preprocesamiento de los abstracts y construcción de la lista de
frecuencias de términos del corpus completo.

Pasos:
    1. Normalización: minúsculas y Unicode NFKC.
    2. Tokenización: secuencias de letras a-z. Números, signos y guiones
       actúan como separadores ("AI-driven" -> "ai", "driven").
    3. Eliminación de palabras vacías (stopwords) del inglés y de tokens
       de una sola letra.
    4. Conteo de frecuencias con una tabla hash (dict): O(N) para N tokens.
    5. Orden alfabético previo de los términos con Radix Sort LSD sobre
       caracteres (no comparativo). Así, como los tres algoritmos de
       ranking son estables, los empates de frecuencia quedan en orden
       alfabético y los tres resultados son directamente comparables.
"""

import re
import unicodedata

# Palabras vacías del inglés (lista estándar tipo NLTK) + abreviaturas
# propias de los abstracts académicos (e.g., i.e., et al.) y numerales
# romanos usados para enumerar (i, ii, iii, iv).
STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn do
does doesn doing don down during each few for from further had hadn has hasn have
haven having he her here hers herself him himself his how i if in into is isn it
its itself just ll me more most mustn my myself needn no nor not now of off on
once only or other our ours ourselves out over own re same shan she should
shouldn so some such than that the their theirs them themselves then there these
they this those through to too under until up very was wasn we were weren what
when where which while who whom why will with won would wouldn you your yours
yourself yourselves also however thus whereof whilst upon via yet within among
may might must shall per eg ie al etc ii iii iv vi
""".split())

RE_TOKEN = re.compile(r"[a-z]+")


def tokenizar(texto):
    texto = unicodedata.normalize("NFKC", texto).lower()
    return RE_TOKEN.findall(texto)


def es_termino_valido(token):
    return len(token) > 1 and token not in STOPWORDS


def preprocesar_abstract(texto):
    """Devuelve la lista de términos válidos de un abstract."""
    return [t for t in tokenizar(texto) if es_termino_valido(t)]


def contar_frecuencias(corpus):
    """Frecuencia absoluta de cada término sobre los 20 abstracts.
    Retorna (frecuencias: dict término->frecuencia, total_tokens_crudos,
    total_terminos_validos)."""
    frecuencias = {}
    total_crudos = 0
    total_validos = 0
    for articulo in corpus:
        tokens = tokenizar(articulo["abstract"])
        total_crudos += len(tokens)
        for t in tokens:
            if es_termino_valido(t):
                total_validos += 1
                frecuencias[t] = frecuencias.get(t, 0) + 1
    return frecuencias, total_crudos, total_validos


def radix_sort_alfabetico(palabras):
    """Radix Sort LSD para cadenas de longitud variable.
    Cada cadena se rellena (conceptualmente) a la derecha con el símbolo
    0, que precede a 'a'..'z' (1..26). Se hace una pasada estable de
    Counting Sort por cada posición, de la última a la primera.
    Complejidad: O(L * (n + 27)), con L = longitud máxima de palabra."""
    if not palabras:
        return []
    L = max(len(p) for p in palabras)
    BASE = 27
    salida = list(palabras)
    for pos in range(L - 1, -1, -1):
        conteo = [0] * BASE
        for p in salida:
            c = ord(p[pos]) - 96 if pos < len(p) else 0
            conteo[c] += 1
        for i in range(1, BASE):
            conteo[i] += conteo[i - 1]
        temporal = [None] * len(salida)
        for p in reversed(salida):           # recorrido inverso => estable
            c = ord(p[pos]) - 96 if pos < len(p) else 0
            conteo[c] -= 1
            temporal[conteo[c]] = p
        salida = temporal
    return salida


def lista_de_frecuencias(corpus):
    """Lista [(término, frecuencia), ...] en orden alfabético: es la
    entrada común de los tres algoritmos de ordenamiento."""
    frecuencias, total_crudos, total_validos = contar_frecuencias(corpus)
    terminos = radix_sort_alfabetico(list(frecuencias.keys()))
    pares = [(t, frecuencias[t]) for t in terminos]
    return pares, total_crudos, total_validos
