"""
extraccion.py
-------------
Extracción del corpus documental (20 artículos en PDF entregados por el docente).

Para cada artículo se obtiene:
    - id       : número del artículo (prefijo del nombre del archivo)
    - titulo   : título del artículo
    - autores  : lista de autores
    - abstract : texto del resumen (insumo principal del procesamiento)

El resultado se guarda en  datos/corpus.json  para que el resto de la
aplicación trabaje siempre sobre el mismo corpus estático y controlado.

Dependencia: pypdf  (pip install pypdf)

Uso:
    python extraccion.py <carpeta_con_los_pdf>
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Metadatos (título y autores) verificados manualmente contra la primera
# página de cada PDF. Se dejan explícitos porque los metadatos internos de
# los PDF están incompletos o son incorrectos en varios artículos
# (p. ej. el art. 13 tiene como título "ET-04-2023-0117_proof 609..645" y
# varios sólo registran al primer autor).
# ---------------------------------------------------------------------------
METADATOS = {
    1: ("Artificial intelligence literacy in higher and adult education: A scoping literature review",
        ["Matthias Carl Laupichler", "Alexandra Aster", "Jana Schirch", "Tobias Raupach"]),
    2: ("Systematic literature review on opportunities, challenges, and future research recommendations of artificial intelligence in education",
        ["Thomas K.F. Chiu", "Qi Xia", "Xinyan Zhou", "Ching Sing Chai", "Miaoting Cheng"]),
    3: ("Myths, mis- and preconceptions of artificial intelligence: A review of the literature",
        ["Arne Bewersdorff", "Xiaoming Zhai", "Jessica Roberts", "Claudia Nerdel"]),
    4: ("Opportunities of artificial intelligence for supporting complex problem-solving: Findings from a scoping review",
        ["Srecko Joksimovic", "Dirk Ifenthaler", "Rebecca Marrone", "Maarten De Laat", "George Siemens"]),
    5: ("Application and theory gaps during the rise of Artificial Intelligence in Education",
        ["Xieling Chen", "Haoran Xie", "Di Zou", "Gwo-Jen Hwang"]),
    6: ("A multi-perspective study on Artificial Intelligence in Education: grants, conferences, journals, software tools, institutions, and researchers",
        ["Xieling Chen", "Haoran Xie", "Gwo-Jen Hwang"]),
    7: ("Human-centred learning analytics and AI in education: A systematic literature review",
        ["Riordan Alfredo", "Vanessa Echeverria", "Yueqiao Jin", "Lixiang Yan", "Zachari Swiecki",
         "Dragan Gašević", "Roberto Martinez-Maldonado"]),
    8: ("A systematic review of artificial intelligence techniques for collaborative learning over the past two decades",
        ["Seng Chee Tan", "Alwyn Vwen Yen Lee", "Min Lee"]),
    9: ("AI technologies for education: Recent research & future directions",
        ["Ke Zhang", "Ayse Begum Aslan"]),
    10: ("Educational applications of artificial intelligence in simulation-based learning: A systematic mapping review",
         ["Chih-Pu Dai", "Fengfeng Ke"]),
    11: ("A scoping review on how generative artificial intelligence transforms assessment in higher education",
         ["Qi Xia", "Xiaojing Weng", "Fan Ouyang", "Tzung Jin Lin", "Thomas K.F. Chiu"]),
    12: ("A meta systematic review of artificial intelligence in higher education: a call for increased ethics, collaboration, and rigour",
         ["Melissa Bond", "Hassan Khosravi", "Maarten De Laat", "Nina Bergdahl", "Violeta Negrea",
          "Emily Oxley", "Phuong Pham", "Sin Wang Chong", "George Siemens"]),
    13: ("Artificial intelligence for education and its emphasis on assessment and adversity quotient: a review",
         ["Jyoti Mudkanna Gavhane", "Reena Pagare"]),
    14: ("A comprehensive exploration of artificial intelligence competence frameworks for educators: A critical review",
         ["Tamar Mikeladze", "Paulien C. Meijer", "Roald P. Verhoeff"]),
    15: ("A Systematic Review of Generative AI for Teaching and Learning Practice",
         ["Bayode Ogunleye", "Kudirat Ibilola Zakariyyah", "Oluwaseun Ajao", "Olakunle Olayinka", "Hemlata Sharma"]),
    16: ("Generative Artificial Intelligence in Tertiary Education: Assessment Redesign Principles and Considerations",
         ["Che Yee Lye", "Lyndon Lim"]),
    17: ("A Systematic Review of AI-Driven Educational Assessment in STEM Education",
         ["Fan Ouyang", "Tuan Anh Dinh", "Weiqi Xu"]),
    18: ("Transforming Education: A Comprehensive Review of Generative Artificial Intelligence in Educational Settings through Bibliometric and Content Analysis",
         ["Zied Bahroun", "Chiraz Anane", "Vian Ahmed", "Andrew Zacca"]),
    19: ("What Is the Impact of ChatGPT on Education? A Rapid Review of the Literature",
         ["Chung Kwan Lo"]),
    20: ("Systematic review of research on artificial intelligence applications in higher education – where are the educators?",
         ["Olaf Zawacki-Richter", "Victoria I. Marín", "Melissa Bond", "Franziska Gouverneur"]),
}

# Marca de inicio del resumen: "ABSTRACT", "A B S T R A C T", "Abstract", "Abstract:"
RE_INICIO = re.compile(r"(?m)(A B S T R A C T|^\s*ABSTRACT|^\s*Abstract)\s*:?\s*")

# Marcas de fin del resumen (lo primero que aparece después del abstract)
RE_FIN = re.compile(
    r"(?m)^\s*(Keywords?\b|KEYWORDS\b|1\.\s*Introduction|Introduction\b|"
    r"Citation:|Paper type|\* Corresponding|\*\s*[A-Z][a-z]+ [A-Z][a-z]+\s*$)"
)

# Rótulos del resumen estructurado de Emerald (art. 13). Son encabezados,
# no contenido, y se retiran para no contaminar el conteo de términos.
RE_ROTULOS = re.compile(r"(Purpose|Design/methodology/approach|Findings|Originality/value)\s*[–-]\s*")

# Fin específico de algunos artículos en los que el texto que sigue al
# abstract no trae un encabezado reconocible (frase final del abstract).
FIN_ESPECIFICO = {
    9: "opment efforts.",
    14: "teaching practices.",
}


def normalizar_unicode(texto):
    """NFKC convierte ligaduras tipográficas (ﬁ, ﬂ, ﬀ...) en letras normales."""
    texto = texto.replace("­", "-")        # guion blando -> guion
    texto = texto.replace("‐", "-").replace("‑", "-")
    return unicodedata.normalize("NFKC", texto)


def vocabulario_referencia(textos):
    """Frecuencia de palabras en el texto completo de los 20 PDF.
    Se usa para decidir, con evidencia del propio corpus, cómo reparar
    palabras partidas por la extracción del PDF."""
    vocab = Counter()
    for t in textos:
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", t))
    return vocab


def reparar_ligaduras(texto, vocab):
    """Algunos PDF (arts. 5 y 6) separan la ligadura con un espacio:
    'Arti ﬁcial' -> 'Arti ficial'. Se unen los dos fragmentos sólo si la
    palabra resultante aparece en el texto completo del corpus
    ('in fluential' -> 'influential'; pero 'the field' se mantiene,
    porque 'thefield' no existe)."""
    patron = re.compile(r"\b([A-Za-z]+) (f[fil]{1,2}[a-z]*)")

    def unir(m):
        izq, der = m.group(1), m.group(2)
        unida = (izq + der).lower()
        if vocab[unida] > 0:
            return izq + der
        return m.group(0)

    return patron.sub(unir, texto)


def reparar_guiones(texto, vocab):
    """Une palabras divididas al final de línea ('edu-\\ncational').
    Si la palabra unida existe en el corpus se une sin guion; de lo
    contrario se conserva el guion (p. ej. 'AI-\\ndriven' -> 'AI-driven')."""
    patron = re.compile(r"([A-Za-z]+)\s?-\s*\n\s*([A-Za-z]+)")

    def unir(m):
        izq, der = m.group(1), m.group(2)
        if vocab[(izq + der).lower()] > 0:
            return izq + der
        return izq + "-" + der

    return patron.sub(unir, texto)


def extraer_abstract(texto_pagina, art_id):
    m = RE_INICIO.search(texto_pagina)
    if not m:
        raise ValueError(f"No se encontró el abstract en el artículo {art_id}")
    resto = texto_pagina[m.end():]

    # Formato Elsevier 2024 (art. 7): tras el rótulo aparece primero el bloque
    # de palabras clave (líneas cortas). Se omite hasta la primera línea larga.
    lineas = resto.split("\n")
    if lineas and lineas[0].strip().startswith("Keywords"):
        i = 1
        while i < len(lineas) and len(lineas[i].strip()) < 60:
            i += 1
        resto = "\n".join(lineas[i:])

    if art_id in FIN_ESPECIFICO:
        frase = FIN_ESPECIFICO[art_id]
        resto = resto[: resto.index(frase) + len(frase)]
    else:
        f = RE_FIN.search(resto)
        if f:
            resto = resto[: f.start()]
    return resto


def limpiar(texto):
    texto = RE_ROTULOS.sub("", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def construir_corpus(carpeta):
    archivos = {}
    for nombre in os.listdir(carpeta):
        m = re.match(r"^(\d+)\.", nombre)
        if m and nombre.lower().endswith(".pdf"):
            archivos[int(m.group(1))] = os.path.join(carpeta, nombre)

    # Texto completo (para el vocabulario de referencia) y primera página
    completos, primera = {}, {}
    for art_id, ruta in sorted(archivos.items()):
        lector = PdfReader(ruta)
        paginas = [normalizar_unicode(p.extract_text() or "") for p in lector.pages]
        completos[art_id] = "\n".join(paginas)
        primera[art_id] = paginas[0]
    vocab = vocabulario_referencia(completos.values())

    corpus = []
    for art_id in sorted(archivos):
        crudo = extraer_abstract(primera[art_id], art_id)
        crudo = reparar_guiones(crudo, vocab)
        crudo = reparar_ligaduras(crudo, vocab)
        titulo, autores = METADATOS[art_id]
        corpus.append({
            "id": art_id,
            "archivo": os.path.basename(archivos[art_id]),
            "titulo": titulo,
            "autores": autores,
            "abstract": limpiar(crudo),
        })
    return corpus


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "articulos"
    corpus = construir_corpus(carpeta)
    os.makedirs("datos", exist_ok=True)
    with open(os.path.join("datos", "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Corpus generado: {len(corpus)} artículos -> datos/corpus.json")
    for art in corpus:
        print(f"  [{art['id']:>2}] {len(art['abstract'].split()):>4} palabras | {art['titulo'][:70]}")


if __name__ == "__main__":
    main()
