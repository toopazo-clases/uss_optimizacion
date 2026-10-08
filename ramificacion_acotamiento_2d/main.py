"""
Ramificación y acotamiento paso a paso sobre el problema 2D de la figura
"PL vs. PLE" de Unidad_2_Clase (el mismo de fuerzabruta_LpProblem_entera):

    max z = x1 + x2
    R1: x2 <= 6
    R2: x1 <= 7
    R3: 2 x1 + 7 x2 <= 48
    R4: 5 x1 + 2 x2 <= 39
    x1, x2 >= 0 enteras

Genera el material del "Ejemplo 2" de la clase y de
Guias/Guia_6_Ramificación_y_acotamiento:

1. El árbol de referencia, con las reglas que se les dan a los estudiantes:
     - se ramifica la PRIMERA variable fraccionaria (x1 antes que x2), en
       x_j <= floor(v) (subproblema "-1") y x_j >= floor(v) + 1 (subproblema "-2");
     - los dos subproblemas se resuelven (acotamiento) y se prueban APENAS se
       crean, primero el "-1"; las pruebas se aplican en este orden:
         1) infactible                         -> podar (rojo)
         2) Z <= mejor solución entera conocida -> podar por cota (amarillo)
         3) solución entera                    -> podar (verde); pasa a ser
                                                  la mejor conocida
       si ninguna aplica, el nodo queda abierto. Cuando cambia la mejor
       conocida, se vuelve a aplicar la prueba 2 a los nodos abiertos;
     - se ramifica el nodo abierto de MEJOR COTA (mayor Z; si empatan, el
       creado primero);
     - la cota es el Z de la relajación, sin redondear.
2. La "tabla de subproblemas": TODAS las cajas de cotas a las que se puede
   llegar ramificando en cualquier variable fraccionaria y en cualquier
   orden (sin podar), cada una con su relajación resuelta con pulp/HiGHS.
   Los estudiantes no resuelven las relajaciones: plantean el subproblema
   (cotas acumuladas) y leen su solución en esta tabla. Las filas van
   ordenadas por cotas, no por el árbol, y la mayoría no se usa en el
   árbol de referencia (sirven, p. ej., para "¿y si se ramifica en x2?").

Uso:
    python main.py arbol p1   # árbol de referencia, paso a paso
    python main.py tabla p1   # tabla de subproblemas y recorrido (resultados/p1/)
"""

import csv
import math
import os
import random
import sys

import pulp

CARPETA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")

# --------------------------------------------------------------------------
# Banco de problemas (max c x, A x <= b, x >= 0 enteras). Cada uno es un
# Problema de Guias/Guia_6_Ramificación_y_acotamiento; p1 es el de la figura
# "PL vs. PLE" de la clase (Ejemplo 2).
# --------------------------------------------------------------------------
PROBLEMAS = {
    "p1": {
        "objetivo": (1, 1),
        "restricciones": [
            ((0, 1), 6, "R1"),
            ((1, 0), 7, "R2"),
            ((2, 7), 48, "R3"),
            ((5, 2), 39, "R4"),
        ],
    },
    # p2 a p5: encontrados con una búsqueda aleatoria filtrada (2 variables,
    # sin restricciones redundantes, óptimo entero único, los tres colores
    # de poda, árbol de 7 a 9 nodos y tabla que cabe en una página).
    "p2": {
        "objetivo": (5, 3),
        "restricciones": [((9, 1), 19, "R1"), ((2, 5), 26, "R2")],
    },
    "p3": {
        "objetivo": (6, 4),
        "restricciones": [((3, 9), 44, "R1"), ((5, 7), 41, "R2"), ((9, 1), 35, "R3")],
    },
    "p4": {
        "objetivo": (5, 4),
        "restricciones": [((5, 7), 48, "R1"), ((9, 1), 60, "R2")],
    },
    "p5": {
        "objetivo": (5, 2),
        "restricciones": [((1, 8), 36, "R1"), ((8, 2), 51, "R2")],
    },
    # Problema 3 de Evaluaciones/Solemne02 (Formas A y B): 11 nodos, 2 hojas
    # infactibles, 2 enteras y 2 por cota, óptimo único, tabla de una página;
    # árboles de forma distinta entre las dos formas.
    "solemne02_a": {
        "objetivo": (3, 5),
        "restricciones": [((1, 7), 31, "R1"), ((5, 6), 47, "R2")],
    },
    "solemne02_b": {
        "objetivo": (1, 4),
        "restricciones": [((1, 9), 63, "R1"), ((2, 5), 54, "R2")],
    },
}

# Problema activo (lo fija usar(); las funciones de abajo leen estos globales)
OBJETIVO = None
RESTRICCIONES = None
N = None
CARPETA_RESULTADOS = None


def usar(nombre):
    """Activa un problema del banco (y su carpeta resultados/<nombre>)."""
    global OBJETIVO, RESTRICCIONES, N, CARPETA_RESULTADOS
    problema = PROBLEMAS[nombre]
    OBJETIVO = tuple(problema["objetivo"])
    RESTRICCIONES = list(problema["restricciones"])
    N = len(OBJETIVO)
    CARPETA_RESULTADOS = os.path.join(CARPETA_BASE, nombre)


EPS = 1e-6
SEMILLA_TABLA = 2026  # orden (mezclado) de las filas de la tabla de subproblemas

ROJO, AMARILLO, VERDE = "infactible", "por cota", "entera"


# --------------------------------------------------------------------------
# Relajación de un subproblema (caja de cotas lo <= x <= hi)
# --------------------------------------------------------------------------


def relajacion(lo, hi):
    """Resuelve la relajación de PL con cotas lo[j] <= x_j <= hi[j]
    (hi[j] = None: sin cota superior). Devuelve None si es infactible, o
    (x, z) con valores redondeados a 6 decimales."""
    modelo = pulp.LpProblem("subproblema", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{j + 1}", lowBound=lo[j], upBound=hi[j]) for j in range(N)]
    modelo += pulp.lpSum(c * v for c, v in zip(OBJETIVO, x))
    for a, rhs, et in RESTRICCIONES:
        modelo += pulp.lpSum(aj * v for aj, v in zip(a, x)) <= rhs, et
    modelo.solve(pulp.HiGHS(msg=False))
    if pulp.LpStatus[modelo.status] != "Optimal":
        return None
    valores = tuple(round(v.value(), 6) + 0.0 for v in x)
    return valores, round(pulp.value(modelo.objective), 6)


def es_entero(v):
    return abs(v - round(v)) < EPS


def fraccionarias(x):
    return [j for j in range(N) if not es_entero(x[j])]


def hijos(lo, hi, j, v):
    """Cajas de los dos subproblemas al ramificar x_j = v: x_j <= floor(v) y
    x_j >= floor(v) + 1."""
    f = math.floor(v)
    hi1 = list(hi)
    hi1[j] = f
    lo2 = list(lo)
    lo2[j] = f + 1
    return (tuple(lo), tuple(hi1)), (tuple(lo2), tuple(hi))


# --------------------------------------------------------------------------
# Árbol de referencia
# --------------------------------------------------------------------------


def ramificacion_y_acotamiento():
    """Devuelve (nodos, pasos, incumbente). nodos: dict nombre -> datos;
    pasos: lista de textos en orden; incumbente: nombre del nodo."""
    nodos = {}
    pasos = []
    abiertos = []
    incumbente = None
    orden = [0]

    def crear(nombre, padre, restr, lo, hi):
        r = relajacion(lo, hi)
        nodo = {
            "nombre": nombre, "padre": padre, "restr": restr,
            "agregada": texto_restriccion(restr) if restr else None,
            "lo": lo, "hi": hi, "x": r[0] if r else None, "z": r[1] if r else None,
            "estado": None, "orden": orden[0],
        }
        orden[0] += 1
        nodos[nombre] = nodo
        return nodo

    def probar(nodo):
        nonlocal incumbente
        txt = f"{nodo['nombre']} ({describir_cotas(nodo['lo'], nodo['hi'])}): "
        if nodo["x"] is None:
            nodo["estado"] = ROJO
            pasos.append(txt + "infactible -> podar (rojo)")
            return
        txt += f"x = {fmt_x(nodo['x'])}, Z = {fmt(nodo['z'])}"
        if incumbente and nodo["z"] <= nodos[incumbente]["z"] + EPS:
            nodo["estado"] = AMARILLO
            pasos.append(txt + f" <= {fmt(nodos[incumbente]['z'])} -> podar por cota (amarillo)")
            return
        if not fraccionarias(nodo["x"]):
            nodo["estado"] = VERDE
            incumbente = nodo["nombre"]
            pasos.append(txt + " entera -> podar (verde); nueva mejor conocida")
            for otro in list(abiertos):
                if nodos[otro]["z"] <= nodo["z"] + EPS:
                    abiertos.remove(otro)
                    nodos[otro]["estado"] = AMARILLO
                    pasos.append(f"  {otro}: Z = {fmt(nodos[otro]['z'])} <= {fmt(nodo['z'])}"
                                 " -> podar por cota (amarillo)")
            return
        abiertos.append(nodo["nombre"])
        pasos.append(txt + " -> queda abierto")

    raiz = crear("P", None, None, (0,) * N, (None,) * N)
    probar(raiz)
    while abiertos:
        elegido = max(abiertos, key=lambda n: (nodos[n]["z"], -nodos[n]["orden"]))
        abiertos.remove(elegido)
        padre = nodos[elegido]
        j = fraccionarias(padre["x"])[0]
        v = padre["x"][j]
        padre["estado"] = f"ramificado en x{j + 1}"
        pasos.append(f"Ramificar {elegido} (mejor cota, Z = {fmt(padre['z'])}) en x{j + 1} = {fmt(v)}")
        base = "S" if elegido == "P" else elegido + "-"
        caja1, caja2 = hijos(padre["lo"], padre["hi"], j, v)
        f = math.floor(v)
        h1 = crear(f"{base}1", elegido, (j, "<=", f), *caja1)
        h2 = crear(f"{base}2", elegido, (j, ">=", f + 1), *caja2)
        probar(h1)
        probar(h2)
    return nodos, pasos, incumbente


# --------------------------------------------------------------------------
# Tabla de subproblemas como CAMINOS (restricciones acumuladas sin simplificar)
# --------------------------------------------------------------------------
# Un camino es la tupla de restricciones agregadas desde P, en orden:
# ((j, "<=" | ">=", valor), ...). Cada fila de la tabla es un camino, con las
# restricciones tal como se van agregando (p. ej. "x1 >= 6, x2 <= 4, x1 <= 6");
# simplificarlas (x1 = 6) queda a cargo del estudiante.


def caja_de(camino):
    lo, hi = [0] * N, [None] * N
    for j, sentido, v in camino:
        if sentido == "<=":
            hi[j] = v if hi[j] is None else min(hi[j], v)
        else:
            lo[j] = max(lo[j], v)
    return tuple(lo), tuple(hi)


def camino_de(nodos, nombre):
    """Restricciones agregadas desde P hasta el nodo, en orden."""
    camino = []
    while nodos[nombre]["padre"] is not None:
        camino.append(nodos[nombre]["restr"])
        nombre = nodos[nombre]["padre"]
    return tuple(reversed(camino))


def caminos_alcanzables():
    """Todos los caminos que se obtienen ramificando, sin podar, en
    CUALQUIER variable fraccionaria y en cualquier orden."""
    caminos = []

    def explorar(camino):
        caminos.append(camino)
        r = relajacion(*caja_de(camino))
        if r is None:
            return
        for j in fraccionarias(r[0]):
            f = math.floor(r[0][j])
            explorar(camino + ((j, "<=", f),))
            explorar(camino + ((j, ">=", f + 1),))

    explorar(())
    return caminos


def caminos_distractores(caminos):
    """Caminos que salen de errores típicos, para que la tabla no regale el
    árbol y un planteamiento equivocado lleve a una fila que existe:
      - redondear al revés: x_j <= ceil(v) o x_j >= floor(v);
      - olvidar una restricción heredada: quitar una restricción anterior
        a la última."""
    nuevos = set()
    for camino in caminos:
        r = relajacion(*caja_de(camino))
        if r is not None:
            for j in fraccionarias(r[0]):
                f = math.floor(r[0][j])
                nuevos.add(camino + ((j, "<=", f + 1),))
                nuevos.add(camino + ((j, ">=", f),))
        for k in range(len(camino) - 1):
            nuevos.add(camino[:k] + camino[k + 1:])
    return sorted(nuevos - set(caminos))


def clave_camino(camino):
    return (len(camino), tuple((j, 0 if s == "<=" else 1, v) for j, s, v in camino))


def tabla():
    caminos = caminos_alcanzables()
    caminos += caminos_distractores(caminos)
    # orden mezclado (semilla fija -> tabla reproducible): obliga a buscar
    # el camino, en vez de deducirlo del orden de las filas
    caminos = sorted(set(caminos), key=clave_camino)
    random.Random(SEMILLA_TABLA).shuffle(caminos)
    filas = []
    for i, camino in enumerate(caminos, start=1):
        r = relajacion(*caja_de(camino))
        filas.append({
            "id": f"F{i}",
            "camino": camino,
            "texto": texto_camino(camino),
            "x": r[0] if r else None,
            "z": r[1] if r else None,
        })
    return filas


# --------------------------------------------------------------------------
# Formato
# --------------------------------------------------------------------------


def fmt(v):
    if es_entero(v):
        return str(int(round(v)))
    return f"{v:.2f}"


def fmt_x(x):
    return "(" + ", ".join(fmt(v) for v in x) + ")"


def cota_variable(j, lo, hi):
    """Texto de las cotas de x_{j+1}: '—' si no tiene cotas adicionales."""
    nombre = f"x{j + 1}"
    if lo == 0 and hi is None:
        return "—"
    if hi is not None and lo == hi:
        return f"{nombre} = {lo}"
    if lo == 0:
        return f"{nombre} <= {hi}"
    if hi is None:
        return f"{nombre} >= {lo}"
    return f"{lo} <= {nombre} <= {hi}"


def describir_cotas(lo, hi):
    partes = [c for c in (cota_variable(j, lo[j], hi[j]) for j in range(N)) if c != "—"]
    return ", ".join(partes) if partes else "sin cotas adicionales"


def texto_restriccion(restr):
    j, sentido, v = restr
    return f"x{j + 1} {sentido} {v}"


def texto_camino(camino):
    return ", ".join(texto_restriccion(r) for r in camino) if camino else "—"


def celda_camino_latex(camino):
    """Celda de la tabla: una restricción por línea (apiladas), «---» si es P."""
    if not camino:
        return "---"
    lineas = r" \\ ".join(a_latex(texto_restriccion(r)).strip("$") for r in camino)
    # espaciado propio (más ajustado que el de la tabla) para las líneas apiladas
    return (r"{\setlength{\extrarowheight}{0pt}\renewcommand{\arraystretch}{1.0}"
            r"$\begin{array}{@{}l@{}}" + lineas + r"\end{array}$}")


def a_latex(texto):
    texto = texto.replace("<=", r"\le").replace(">=", r"\ge")
    texto = texto.replace("x1", "x_1").replace("x2", "x_2").replace(", ", r",\ ")
    return "$" + texto + "$" if texto != "—" else "---"


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------


def comando_arbol():
    nodos, pasos, incumbente = ramificacion_y_acotamiento()
    print("== Pasos ==")
    for p in pasos:
        print(p)
    print("\n== Nodos (en orden de creación) ==")
    print(f"{'nodo':10} {'agregada':10} {'cotas acumuladas':22} {'x':16} {'Z':7} estado")
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        x = fmt_x(n["x"]) if n["x"] else "—"
        z = fmt(n["z"]) if n["z"] is not None else "—"
        print(f"{n['nombre']:10} {n['agregada'] or '—':10} {describir_cotas(n['lo'], n['hi']):22} "
              f"{x:16} {z:7} {n['estado']}")
    inc = nodos[incumbente]
    print(f"\nÓptimo: {incumbente}, x = {fmt_x(inc['x'])}, Z = {fmt(inc['z'])}"
          f" ({len(nodos)} nodos)")

    # cada nodo del árbol debe estar en la tabla de subproblemas
    ids = {f["camino"]: f["id"] for f in tabla()}
    print("\n== Fila de la tabla de cada nodo ==")
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        camino = camino_de(nodos, n["nombre"])
        print(f"{n['nombre']:10} -> {ids[camino]}  ({texto_camino(camino)})")


def decision_latex(nodo, nodos, incumbentes):
    """Texto de la columna «Decisión» del recorrido (LaTeX)."""
    if nodo["estado"].startswith("ramificado"):
        return "ramificar en $x_" + nodo["estado"][-1] + "$"
    if nodo["estado"] == ROJO:
        return r"podar \ussfinmuestra{USSfinInfactible}"
    if nodo["estado"] == VERDE:
        return r"podar \ussfinmuestra{USSfinEntera}; mejor $=" + fmt(nodo["z"]) + "$"
    mejor = nodos[incumbentes[nodo["nombre"]]]["z"]
    return r"podar \ussfinmuestra{USSfinCota} ($" + fmt(nodo["z"]) + r"\le" + fmt(mejor) + "$)"


def expresion_latex(coefs):
    """5x_1 + 3x_2 (omite coeficientes 1 y términos 0)."""
    partes = []
    for j, c in enumerate(coefs):
        if c == 0:
            continue
        termino = ("" if c == 1 else str(c)) + f"x_{j + 1}"
        partes.append(termino)
    return " + ".join(partes)


def modelo_latex():
    """Modelo compacto: máx z = ... s.a. (restricciones en 2 columnas)."""
    filas = (len(RESTRICCIONES) + 1) // 2
    celdas = [rf"\text{{{et}:}} & {expresion_latex(a)} \le {b}" for a, b, et in RESTRICCIONES]
    lineas = []
    for i in range(filas):
        izq = celdas[i]
        der = celdas[i + filas] if i + filas < len(celdas) else "&"
        lineas.append(f"    {izq} \\qquad & {der} \\\\")
    variables = ",\\ ".join(f"x_{j + 1}" for j in range(N))
    return ("\\[\n  \\max\\ z = " + expresion_latex(OBJETIVO) +
            "\n  \\qquad \\text{s.a.} \\qquad\n  \\begin{array}{llll}\n" + "\n".join(lineas) +
            "\n  \\end{array}\n  \\qquad " + variables + " \\ge 0 \\text{ enteras}\n\\]\n")


def modelo_estandar_latex():
    """Modelo en forma estándar: una restricción por línea."""
    lineas = [f"    \\max & z & = & {expresion_latex(OBJETIVO)} \\\\[4pt]"]
    for a, b, et in RESTRICCIONES:
        lineas.append(f"    \\text{{{et}:}} & {expresion_latex(a)} & \\le & {b} \\\\")
    lineas[-1] += "[4pt]"
    variables = ",\\ ".join(f"x_{j + 1}" for j in range(N))
    lineas.append(f"    & \\multicolumn{{3}}{{l}}{{{variables} \\ge 0 \\text{{ enteras}}}}")
    return "\\[\n  \\begin{array}{lrcl}\n" + "\n".join(lineas) + "\n  \\end{array}\n\\]\n"


def arbol_latex(nodos):
    """Árbol solución en TikZ: hojas en orden de izquierda a derecha
    (subproblema «-1» a la izquierda), padres centrados sobre sus hijos."""
    hijos_de = {n: [] for n in nodos}
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        if n["padre"]:
            hijos_de[n["padre"]].append(n["nombre"])
    pos = {}
    hojas = [0]

    def ubicar(nombre, prof):
        if not hijos_de[nombre]:
            pos[nombre] = (hojas[0] * 3.0, -prof * 2.0)
            hojas[0] += 1
            return
        for h in hijos_de[nombre]:
            ubicar(h, prof + 1)
        xs = [pos[h][0] for h in hijos_de[nombre]]
        pos[nombre] = (sum(xs) / len(xs), -prof * 2.0)

    ubicar("P", 0)
    centro = pos["P"][0]
    tikz = lambda nombre: nombre.replace("-", "m")
    lineas = ["\\begin{tikzpicture}[x=1cm, y=1cm]"]
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        x, y = pos[n["nombre"]]
        texto = nombre_latex(n["nombre"])
        if n["x"] is not None:
            texto += f"\\\\$Z^*{{=}}{fmt(n['z'])}$\\\\$X^*{{=}}({fmt(n['x'][0])},\\,{fmt(n['x'][1])})$"
        lineas.append(f"  \\node[nodoba] ({tikz(n['nombre'])}) at ({x - centro:.2f}, {y:.2f}) {{{texto}}};")
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        if n["padre"]:
            lineas.append(f"  \\draw[flechaba] ({tikz(n['padre'])}) -- ({tikz(n['nombre'])})"
                          f" node[midway, etiqba] {{{a_latex(n['agregada'])}}};")
    colores = {ROJO: "USSfinInfactible", AMARILLO: "USSfinCota", VERDE: "USSfinEntera"}
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        if n["estado"] in colores:
            lineas.append(f"  \\ussfin{{{tikz(n['nombre'])}}}{{{colores[n['estado']]}}}")
    lineas.append("\\end{tikzpicture}")
    return "\n".join(lineas) + "\n"


def nombre_latex(nombre):
    if nombre == "P":
        return "$P$"
    partes = nombre[1:].split("-")
    return "$S_{" + r"\text{-}".join(partes) + "}$"


def comando_tabla():
    filas = tabla()
    nodos, _pasos, incumbente = ramificacion_y_acotamiento()
    usados = {camino_de(nodos, n): n for n in nodos}
    print(f"{'id':4} {'restricciones adicionales':34} {'Z':10} {'X':14} en árbol")
    for f in filas:
        z = fmt(f["z"]) if f["z"] is not None else "—"
        x = fmt_x(f["x"]) if f["x"] else "infactible"
        marca = usados.get(f["camino"], "")
        print(f"{f['id']:4} {f['texto']:34} {z:10} {x:14} {marca}")
    print(f"\n{len(filas)} caminos; {len(usados)} son del árbol de referencia")

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta_csv = os.path.join(CARPETA_RESULTADOS, "tabla_subproblemas.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "restricciones_adicionales", "z", "x"])
        for f in filas:
            if f["x"]:
                w.writerow([f["id"], f["texto"], fmt(f["z"]), fmt_x(f["x"])])
            else:
                w.writerow([f["id"], f["texto"], "", "infactible"])
    # la tabla va en 2 columnas: se parte en dos mitades de altura parecida
    # (altura de una fila = cantidad de restricciones apiladas, mínimo 1)
    alturas = [max(1, len(f["camino"])) for f in filas]
    total, acumulado, corte = sum(alturas), 0, len(filas)
    for k, h in enumerate(alturas):
        acumulado += h
        if acumulado >= total / 2:
            corte = k + 1
            break
    for parte, trozo in (("1", filas[:corte]), ("2", filas[corte:])):
        ruta_tex = os.path.join(CARPETA_RESULTADOS, f"tabla_subproblemas_{parte}.tex")
        with open(ruta_tex, "w", encoding="utf-8") as fh:
            fh.write("% Generado por uss_optimizacion/ramificacion_acotamiento_2d/main.py tabla\n")
            for k, f in enumerate(trozo):
                if k:  # línea delgada entre filas (el color lo fija la guía)
                    fh.write("\\specialrule{0.3pt}{1pt}{1pt}\n")
                if f["x"]:
                    resto = f"{fmt(f['z'])} & $({fmt(f['x'][0])},\\,{fmt(f['x'][1])})$"
                else:
                    resto = "--- & infactible"
                fh.write(f"{f['id']} & {celda_camino_latex(f['camino'])} & {resto} \\\\\n")
        print("Guardado:", ruta_tex)

    # recorrido de la solución (mismo orden de creación que el árbol)
    ids = {f["camino"]: f["id"] for f in filas}
    mejor_al_podar = {}
    mejor = None
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        if n["estado"] == VERDE:
            mejor = n["nombre"]
        mejor_al_podar[n["nombre"]] = mejor if n["estado"] != VERDE else n["nombre"]
    ruta_rec = os.path.join(CARPETA_RESULTADOS, "recorrido.tex")
    with open(ruta_rec, "w", encoding="utf-8") as fh:
        fh.write("% Generado por uss_optimizacion/ramificacion_acotamiento_2d/main.py tabla\n")
        for k, n in enumerate(sorted(nodos.values(), key=lambda n: n["orden"]), start=1):
            camino = camino_de(nodos, n["nombre"])
            agregada = a_latex(n["agregada"]) if n["agregada"] else "---"
            z = fmt(n["z"]) if n["z"] is not None else "---"
            x = f"$({fmt(n['x'][0])},\\,{fmt(n['x'][1])})$" if n["x"] else "infactible"
            fh.write(f"{k} & {nombre_latex(n['nombre'])} & {agregada} & {a_latex(texto_camino(camino))}"
                     f" & {ids[camino]} & {z} & {x} & {decision_latex(n, nodos, mejor_al_podar)} \\\\\n")
    cabecera = "% Generado por uss_optimizacion/ramificacion_acotamiento_2d/main.py tabla\n"
    inc = nodos[incumbente]
    extras = {
        "modelo.tex": modelo_latex(),
        "modelo_estandar.tex": modelo_estandar_latex(),
        "arbol.tex": arbol_latex(nodos),
        "optimo.tex": (f"$X^*=({fmt(inc['x'][0])},\\,{fmt(inc['x'][1])})$, $Z^*={fmt(inc['z'])}$"
                       f" (subproblema {nombre_latex(incumbente)}).\n"),
    }
    for archivo, texto in extras.items():
        with open(os.path.join(CARPETA_RESULTADOS, archivo), "w", encoding="utf-8") as fh:
            fh.write(cabecera + texto)
    print("Guardado:", ruta_csv)
    print("Guardado:", ruta_rec)
    print("Guardado: modelo.tex, modelo_estandar.tex, arbol.tex, optimo.tex")


def main():
    comandos = {"arbol": comando_arbol, "tabla": comando_tabla}
    if len(sys.argv) != 3 or sys.argv[1] not in comandos or sys.argv[2] not in PROBLEMAS:
        print(f"Uso: python main.py arbol|tabla <problema>   (problemas: {', '.join(PROBLEMAS)})")
        return
    usar(sys.argv[2])
    comandos[sys.argv[1]]()


if __name__ == "__main__":
    main()
