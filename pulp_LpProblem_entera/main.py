"""
Banco de problemas de programación entera (binaria, entera pura y entera
mixta): se resuelven con pulp/HiGHS y se reporta el óptimo entero, su
relajación de PL y, cuando TODAS las variables son binarias (PEB), también
la verificación por enumeración exhaustiva (numpy) y, si el problema lo pide
("arbol": True), el árbol de ramificación y acotamiento.

Uso:
    python main.py <nombre_problema>

Sin argumento, o con un nombre que no está en el banco, solo muestra la
lista de problemas disponibles.

Qué reporta (en este orden):
    1. El óptimo del modelo entero, tal como lo resuelve pulp (cat="Binary",
       "Integer" o "Continuous" según el tipo de cada variable).
    2. La relajación de PL (mismas restricciones, sin exigir integralidad:
       las binarias pasan a 0 <= x <= 1, las enteras a x >= 0 continuas).
       Su Z es una cota (superior si se maximiza, inferior si se minimiza)
       del óptimo entero -- ver Hillier & Lieberman, sección 11.5.
    3. Solo PEB: enumeración de las 2^n combinaciones (verificación
       independiente del solver, con numpy: sirve hasta n ~ 22). Reporta
       cuántas combinaciones son factibles, cuántas soluciones óptimas hay
       (si hay más de una, el "Resultado esperado" de la Guía sería
       ambiguo) y el segundo mejor valor distinto.
    4. Solo con "arbol": True: árbol de ramificación y acotamiento, con las
       reglas del algoritmo de la sección 11.6 de Hillier & Lieberman
       (variable de ramificación en orden natural x1, x2, ...; se ramifica
       el subproblema de creación más reciente, con empate por la cota más
       grande; cota = Z de la relajación de PL redondeado hacia abajo, válido
       porque los coeficientes de la función objetivo son enteros; las 3
       pruebas de sondeo). Cada nodo se resuelve con pulp, no "a mano".

A diferencia de pulp_LpProblem_3d/4d, acá no se reporta precio sombra ni costo
reducido: son propiedades de la relajación de PL, no de la región factible
entera (ver Hillier & Lieberman, cap. 11), así que no hay ranging que pedirle
a HiGHS; se usa solo como solver.
"""

import math
import sys

import numpy as np
import pulp

# --------------------------------------------------------------------------
# Banco de problemas
# --------------------------------------------------------------------------
# Cada problema es un diccionario con:
#   titulo        -> nombre para los reportes
#   sentido       -> "max" o "min" (el árbol de ramificación y acotamiento
#                    solo está implementado para "max", como en el libro)
#   tipos         -> (t1, ..., tn), con cada t_j en {"binaria", "entera",
#                    "continua"}. Las 3 llevan x_j >= 0; la binaria además
#                    x_j <= 1. No incluir esas cotas en "restricciones".
#   objetivo      -> (c1, ..., cn) tal que z = c1*x1 + ... + cn*xn
#   restricciones -> lista de (a1, ..., an, sentido, rhs) para
#                    a1*x1 + ... + an*xn {<=,>=,==} rhs, con "sentido"
#                    literal "<=", ">=" o "==". Las variables se llaman
#                    x1, ..., xn.
#
# Formato "esparcido" (para modelos con muchas variables, ver los builders
# _red_distribucion, etc.): en vez de "tipos"/"objetivo"/"restricciones" en
# forma de tuplas, el problema trae
#   variables     -> lista de nombres (todas binarias salvo que traiga
#                    "tipos" como dict {nombre: tipo})
#   objetivo      -> {nombre: coeficiente}
#   restricciones -> lista de ({nombre: coeficiente}, sentido, rhs, etiqueta)
# Los dos formatos se normalizan con _canonico().
#
# "arbol": True (opcional) -> también imprime el árbol de ramificación y
# acotamiento (solo tiene sentido con pocas variables, PEB y "max").
#
# Los problemas 1 y 2 son de Hillier & Lieberman (los datos se verificaron
# contra las imágenes del libro: la extracción de texto pierde los signos
# menos, p. ej. el -1 de la Tabla 11.3). Del 3 al 6 son inventados para
# práctica (inspirados en las aplicaciones de la sección 11.2 y las
# formulaciones de la sección 11.3), no de ningún libro.


def _supersuds(formulacion):
    """Ejemplo 2 de la sección 11.4 (Supersuds Corp., Tabla 11.3, pp.
    444-447): 5 comerciales de TV a 3 productos, máximo 3 por producto.
    ganancia[i][j] = ganancia (millones de dólares) del producto i con j
    comerciales (j = 0..3); NO es lineal en j (viola la proporcionalidad),
    por eso se usan binarias auxiliares y_ij. Dos formulaciones (la del
    libro, p. 445-447):
      "excluyentes":  y_ij = 1 si x_i = j, con j = 1, 2, 3. sum_j y_ij <= 1
                      (alternativas mutuamente excluyentes: todas en 0
                      significa x_i = 0), sum j*y_ij = 5. 9 variables.
                      Óptimo: y12 = y33 = 1 (x = (2, 0, 3)), Z = 7. Es la
                      que usa la Guía. (Equivalente: incluir y_i0 con
                      sum_j y_ij = 1 para j = 0..3, 12 variables; mismo Z,
                      mismas 12 combinaciones factibles, verificado.)
      "contingentes": y_ij = 1 si x_i >= j (j = 1, 2, 3).  y_i,j+1 <= y_ij
                      (decisiones contingentes), sum y_ij = 5; la ganancia
                      de y_ij es el incremento ganancia[i][j] -
                      ganancia[i][j-1]. Mismo óptimo, Z = 7 (libro: y11 =
                      y12 = y31 = y32 = y33 = 1). Hoy comentada en la Guía
                      (sección 2.4).
    Usado en Guias/Guia_5_Programacion_Entera_Binaria_Mixta (Problema 2)."""
    ganancia = {1: [0, 1, 3, 3], 2: [0, 0, 2, 3], 3: [0, -1, 2, 4]}
    js = (1, 2, 3)
    nombres = [f"y{i}{j}" for i in (1, 2, 3) for j in js]
    if formulacion == "excluyentes":
        objetivo = {f"y{i}{j}": ganancia[i][j] for i in (1, 2, 3) for j in js}
        restricciones = [
            ({f"y{i}{j}": 1 for j in js}, "<=", 1, f"Producto{i}_una_cantidad")
            for i in (1, 2, 3)
        ]
        restricciones.append(
            ({f"y{i}{j}": j for i in (1, 2, 3) for j in js}, "==", 5, "Cinco_comerciales")
        )
    else:
        objetivo = {
            f"y{i}{j}": ganancia[i][j] - ganancia[i][j - 1]
            for i in (1, 2, 3)
            for j in js
        }
        restricciones = [
            ({f"y{i}{j + 1}": 1, f"y{i}{j}": -1}, "<=", 0, f"y{i}{j + 1}_requiere_y{i}{j}")
            for i in (1, 2, 3)
            for j in (1, 2)
        ]
        restricciones.append(
            ({n: 1 for n in nombres}, "==", 5, "Cinco_comerciales")
        )
    return {
        "titulo": f"Supersuds (H&L, sección 11.4, ejemplo 2), formulación {formulacion}: PEB, {len(nombres)} variables",
        "sentido": "max",
        "variables": nombres,
        "objetivo": objetivo,
        "restricciones": restricciones,
    }


def _red_distribucion():
    """Problema 3 de la Guía: diseño de una red de distribución (sección
    11.2, "Diseño de una red de producción y distribución"). Se eligen
    centros de distribución (CD) a abrir entre 3 candidatos y a cada una de
    4 zonas de mercado se le asigna exactamente 1 CD (grupo de alternativas
    mutuamente excluyentes). Sub-decisiones contingentes al CD (el "padre"):
    una zona solo se atiende desde un CD abierto (atender <= abrir) y la
    cámara de frío solo se instala en un CD abierto (camara <= abrir); la
    zona Costa (mariscos) exige un CD con cámara de frío (atender_Costa_k <=
    camara_k). Se abren a lo más 2 CD. Minimiza el costo anual (MM$).
    Óptimo: abrir Antofagasta y Santiago, cámara en Santiago, Z = 80 (el
    segundo mejor es 84).
    En la Guía las variables se llaman x_ij (atender_<zona i>_<CD j>; zonas 1
    Norte, 2 Centro, 3 Sur, 4 Costa), y_j (abrir_<CD j>) y w_j (camara_<CD j>),
    con CD 1 Antofagasta, 2 Santiago, 3 Concepcion."""
    cds = ["Antofagasta", "Santiago", "Concepcion"]
    zonas = ["Norte", "Centro", "Sur", "Costa"]
    fijo = {"Antofagasta": 10, "Santiago": 18, "Concepcion": 14}  # abrir el CD
    frio = 6  # cámara de frío en un CD
    servir = {  # atender la zona desde el CD
        "Norte": {"Antofagasta": 6, "Santiago": 22, "Concepcion": 40},
        "Centro": {"Antofagasta": 24, "Santiago": 6, "Concepcion": 18},
        "Sur": {"Antofagasta": 44, "Santiago": 24, "Concepcion": 8},
        "Costa": {"Antofagasta": 30, "Santiago": 10, "Concepcion": 26},
    }
    nombres = (
        [f"abrir_{k}" for k in cds]
        + [f"camara_{k}" for k in cds]
        + [f"atender_{m}_{k}" for m in zonas for k in cds]
    )
    objetivo = {f"abrir_{k}": fijo[k] for k in cds}
    objetivo.update({f"camara_{k}": frio for k in cds})
    objetivo.update({f"atender_{m}_{k}": servir[m][k] for m in zonas for k in cds})
    restricciones = []
    for m in zonas:
        restricciones.append(
            ({f"atender_{m}_{k}": 1 for k in cds}, "==", 1, f"UnCD_{m}")
        )
    for m in zonas:
        for k in cds:
            restricciones.append(
                ({f"atender_{m}_{k}": 1, f"abrir_{k}": -1}, "<=", 0, f"{m}_{k}_requiere_CD_abierto")
            )
    for k in cds:
        restricciones.append(
            ({f"camara_{k}": 1, f"abrir_{k}": -1}, "<=", 0, f"camara_{k}_requiere_CD_abierto")
        )
        restricciones.append(
            ({f"atender_Costa_{k}": 1, f"camara_{k}": -1}, "<=", 0, f"Costa_{k}_requiere_camara")
        )
    restricciones.append(({f"abrir_{k}": 1 for k in cds}, "<=", 2, "MaximoDosCD"))
    return {
        "titulo": "Red de distribución (H&L, sección 11.2): PEB, 18 variables",
        "sentido": "min",
        "variables": nombres,
        "objetivo": objetivo,
        "restricciones": restricciones,
    }


def _programacion_actividades():
    """Problema 4 de la Guía: programación de actividades interrelacionadas
    (sección 11.2, "Programación de actividades interrelacionadas"). Cada
    actividad inicia en exactamente 1 de las 5 semanas (grupo de
    alternativas mutuamente excluyentes: la variable de cada semana es 1
    solo para una). Las precedencias son decisiones contingentes a la
    elección del predecesor: una actividad solo puede iniciar en la semana t
    si su predecesora inició en una semana anterior a t. Se puede iniciar a
    lo más 1 actividad por semana. Minimiza el costo total (MM$) de iniciar
    cada actividad en la semana elegida. Óptimo: Diseño 1, Sitio 2, Equipos
    3, Instalación 5, Z = 27 (con el tope 'Instalación a más tardar en la
    semana 4' sube a 28).
    En la Guía las variables se llaman x_it (inicia_<actividad i>_<semana t>;
    actividades 1 Diseno, 2 Equipos, 3 Sitio, 4 Instalacion)."""
    actividades = ["Diseno", "Equipos", "Sitio", "Instalacion"]
    semanas = [1, 2, 3, 4, 5]
    costo = {
        "Diseno": [6, 7, 8, 9, 10],
        "Equipos": [12, 10, 8, 9, 11],
        "Sitio": [7, 6, 6, 8, 9],
        "Instalacion": [15, 13, 11, 8, 7],
    }
    precedencias = [  # (sucesora, predecesora)
        ("Equipos", "Diseno"),
        ("Sitio", "Diseno"),
        ("Instalacion", "Equipos"),
        ("Instalacion", "Sitio"),
    ]
    nombres = [f"inicia_{a}_{t}" for a in actividades for t in semanas]
    objetivo = {f"inicia_{a}_{t}": costo[a][t - 1] for a in actividades for t in semanas}
    restricciones = []
    for a in actividades:
        restricciones.append(
            ({f"inicia_{a}_{t}": 1 for t in semanas}, "==", 1, f"{a}_una_semana")
        )
    for suc, pre in precedencias:
        for t in semanas:
            fila = {f"inicia_{suc}_{t}": 1}
            fila.update({f"inicia_{pre}_{s}": -1 for s in semanas if s < t})
            restricciones.append((fila, "<=", 0, f"{suc}_en_{t}_requiere_{pre}_antes"))
    for t in semanas:
        restricciones.append(
            ({f"inicia_{a}_{t}": 1 for a in actividades}, "<=", 1, f"UnInicio_semana{t}")
        )
    return {
        "titulo": "Programación de actividades (H&L, sección 11.2): PEB, 20 variables",
        "sentido": "min",
        "variables": nombres,
        "objetivo": objetivo,
        "restricciones": restricciones,
    }


def _asignacion_flota():
    """Problema 5 de la Guía: asignación de flota (sección 11.2, "Aplicaciones
    a líneas aéreas"). Cada una de 4 rutas se opera con exactamente 1 de 3
    tipos de avión (grupo de alternativas mutuamente excluyentes), con
    disponibilidad limitada de cada tipo. Sub-decisión contingente al tipo
    elegido: la clase ejecutiva de una ruta solo se puede habilitar si la
    ruta usa un avión B o C (premium_r <= asigna_r_B + asigna_r_C); a lo más
    2 rutas con clase ejecutiva. Maximiza la utilidad semanal (MM$); la
    clase ejecutiva suma su ingreso extra menos un costo de 5 por ruta.
    Óptimo: Lima B, BuenosAires B, PuertoMontt A, PuntaArenas C, clase
    ejecutiva en Lima y BuenosAires, Z = 170 (el segundo mejor es 168).
    En la Guía las variables se llaman x_ij (asigna_<ruta i>_<avion j>; rutas 1
    Lima, 2 BuenosAires, 3 PuertoMontt, 4 PuntaArenas; aviones A, B, C) e y_i
    (premium_<ruta i>)."""
    rutas = ["Lima", "BuenosAires", "PuertoMontt", "PuntaArenas"]
    tipos = ["A", "B", "C"]  # A pequeño, B mediano, C grande
    utilidad = {
        "Lima": {"A": 20, "B": 40, "C": 46},
        "BuenosAires": {"A": 26, "B": 48, "C": 44},
        "PuertoMontt": {"A": 30, "B": 36, "C": 33},
        "PuntaArenas": {"A": 18, "B": 30, "C": 42},
    }
    flota = {"A": 2, "B": 2, "C": 1}
    ingreso_premium = {"Lima": 9, "BuenosAires": 11, "PuertoMontt": 4, "PuntaArenas": 7}
    costo_premium = 5
    nombres = [f"asigna_{r}_{k}" for r in rutas for k in tipos] + [
        f"premium_{r}" for r in rutas
    ]
    objetivo = {f"asigna_{r}_{k}": utilidad[r][k] for r in rutas for k in tipos}
    objetivo.update({f"premium_{r}": ingreso_premium[r] - costo_premium for r in rutas})
    restricciones = []
    for r in rutas:
        restricciones.append(
            ({f"asigna_{r}_{k}": 1 for k in tipos}, "==", 1, f"{r}_un_avion")
        )
    for k in tipos:
        restricciones.append(
            ({f"asigna_{r}_{k}": 1 for r in rutas}, "<=", flota[k], f"Flota_{k}")
        )
    for r in rutas:
        restricciones.append(
            (
                {f"premium_{r}": 1, f"asigna_{r}_B": -1, f"asigna_{r}_C": -1},
                "<=",
                0,
                f"premium_{r}_requiere_B_o_C",
            )
        )
    restricciones.append(({f"premium_{r}": 1 for r in rutas}, "<=", 2, "MaximoDosPremium"))
    return {
        "titulo": "Asignación de flota (H&L, sección 11.2): PEB, 16 variables",
        "sentido": "max",
        "variables": nombres,
        "objetivo": objetivo,
        "restricciones": restricciones,
    }


def _minera_una_u_otra():
    """Problema 6 de la Guía: restricciones "una u otra" (sección 11.3).
    Grupos de alternativas mutuamente excluyentes: energía (a lo más 1 de
    Solar, Diésel, Red) y transporte (a lo más 1 de Camiones, Correa).
    Sub-decisiones contingentes al padre: Monitoreo solo con Solar,
    Mantención solo con Camiones. El plan se financia por UNA de dos vías:
    capital propio (inversión total <= 30 MM$) o contratista (horas de
    ingeniería <= 50 mil); solo la restricción de la vía elegida debe
    cumplirse. Con M grande y una binaria auxiliar y (y = 0: se exige el
    tope de capital; y = 1: se exige el de ingeniería):
        capital     <= 30 + M*y
        ingenieria  <= 50 + M*(1 - y)
    Óptimo: Solar + Camiones + Mantención, Z = 64, con y = 0 (capital 30,
    ingeniería 54: la restricción de ingeniería queda "eliminada"). Si se
    exigieran las dos a la vez, el óptimo baja a 56.
    En la Guía las variables se llaman x_1..x_7 (los proyectos en el orden de
    la lista: Solar, Diesel, Red, Camiones, Correa, Monitoreo, Mantencion) e y."""
    proyectos = ["Solar", "Diesel", "Red", "Camiones", "Correa", "Monitoreo", "Mantencion"]
    van = {"Solar": 30, "Diesel": 18, "Red": 22, "Camiones": 25, "Correa": 42, "Monitoreo": 12, "Mantencion": 9}
    capital = {"Solar": 16, "Diesel": 8, "Red": 11, "Camiones": 10, "Correa": 24, "Monitoreo": 5, "Mantencion": 4}
    horas = {"Solar": 30, "Diesel": 12, "Red": 10, "Camiones": 14, "Correa": 50, "Monitoreo": 20, "Mantencion": 10}
    tope_capital, tope_horas, M = 30, 50, 200
    restricciones = [
        ({"Solar": 1, "Diesel": 1, "Red": 1}, "<=", 1, "UnaEnergia"),
        ({"Camiones": 1, "Correa": 1}, "<=", 1, "UnTransporte"),
        ({"Monitoreo": 1, "Solar": -1}, "<=", 0, "Monitoreo_requiere_Solar"),
        ({"Mantencion": 1, "Camiones": -1}, "<=", 0, "Mantencion_requiere_Camiones"),
    ]
    fila_capital = {p: capital[p] for p in proyectos}
    fila_capital["y"] = -M
    restricciones.append((fila_capital, "<=", tope_capital, "Capital_si_y_0"))
    fila_horas = {p: horas[p] for p in proyectos}
    fila_horas["y"] = M
    restricciones.append((fila_horas, "<=", tope_horas + M, "Ingenieria_si_y_1"))
    return {
        "titulo": "Minera, restricciones una u otra (H&L, sección 11.3): PEB, 8 variables",
        "sentido": "max",
        "variables": proyectos + ["y"],
        "objetivo": {p: van[p] for p in proyectos},
        "restricciones": restricciones,
    }


PROBLEMAS = {
    "california_manufacturing": {
        # Ejemplo prototipo de la sección 11.1 de Hillier & Lieberman,
        # "Introducción a la Investigación de Operaciones" (cap. 11,
        # pp. 429-430; el mismo ejemplo se resuelve con ramificación y
        # acotamiento en la sección 11.6, pp. 454-461). Unidades: millones
        # de dólares. Decisiones sí/no: x1 fábrica en Los Ángeles, x2 fábrica
        # en San Francisco, x3 almacén en Los Ángeles, x4 almacén en San
        # Francisco. Óptimo del libro: (x1,x2,x3,x4) = (1,1,0,0), Z = 14;
        # relajación de PL: (5/6,1,0,1), Z = 16.5 (el libro la escribe
        # 16 1/2). Usado en Guias/Guia_5_Programacion_Entera_Binaria_Mixta
        # (Problema 1).
        "titulo": "California Manufacturing Co. (H&L, sección 11.1): PEB, 4 variables",
        "sentido": "max",
        "arbol": True,
        "objetivo": (9, 5, 6, 4),
        "tipos": ("binaria", "binaria", "binaria", "binaria"),
        "restricciones": [
            (6, 3, 5, 2, "<=", 10),  # Capital disponible (millones de dólares)
            (0, 0, 1, 1, "<=", 1),  # A lo sumo un almacén: x3 + x4 <= 1
            (-1, 0, 1, 0, "<=", 0),  # Almacén LA solo si hay fábrica LA: x3 <= x1
            (0, -1, 0, 1, "<=", 0),  # Almacén SF solo si hay fábrica SF: x4 <= x2
        ],
    },
    "supersuds_excluyentes": _supersuds("excluyentes"),
    "supersuds_contingentes": _supersuds("contingentes"),
    "red_distribucion": _red_distribucion(),
    "programacion_actividades": _programacion_actividades(),
    "asignacion_flota": _asignacion_flota(),
    "minera_una_u_otra": _minera_una_u_otra(),
}

EPS = 1e-6


def _canonico(problema):
    """Normaliza los dos formatos del banco a: nombres (lista), tipos
    (dict nombre -> tipo), objetivo (dict) y restricciones (lista de
    (dict, sentido, rhs, etiqueta))."""
    if "variables" in problema:
        nombres = list(problema["variables"])
        tipos = problema.get("tipos") or {}
        tipos = {n: tipos.get(n, "binaria") for n in nombres}
        objetivo = {n: problema["objetivo"].get(n, 0) for n in nombres}
        restricciones = [
            (dict(f), s, rhs, et) for f, s, rhs, et in problema["restricciones"]
        ]
    else:
        n = len(problema["objetivo"])
        nombres = [f"x{j}" for j in range(1, n + 1)]
        tipos = dict(zip(nombres, problema["tipos"]))
        objetivo = dict(zip(nombres, problema["objetivo"]))
        restricciones = []
        for k, fila in enumerate(problema["restricciones"], start=1):
            *coefs, sentido, rhs = fila
            restricciones.append(
                ({nm: a for nm, a in zip(nombres, coefs) if a != 0}, sentido, rhs, f"R{k}")
            )
    return {
        "titulo": problema["titulo"],
        "sentido": problema["sentido"],
        "nombres": nombres,
        "tipos": tipos,
        "objetivo": objetivo,
        "restricciones": restricciones,
    }


def _es_entero(v):
    return abs(v - round(v)) < EPS


def resolver(problema, relajar=False, fijos=None):
    """Resuelve el modelo con pulp/HiGHS. Con relajar=True elimina la
    exigencia de integralidad (relajación de PL); `fijos` es un dict
    {nombre: valor} que fija esas variables en ese valor (solo se usa en la
    relajación, para los nodos del árbol de ramificación y acotamiento).
    Devuelve {"estado": ...} si no es óptimo, y además "x" (dict nombre ->
    valor) y "z" si lo es."""
    p = _canonico(problema)
    fijos = fijos or {}
    sentido = pulp.LpMaximize if p["sentido"] == "max" else pulp.LpMinimize

    modelo = pulp.LpProblem("modelo", sentido)
    variables = {}
    for nombre in p["nombres"]:
        tipo = p["tipos"][nombre]
        if nombre in fijos:
            v = pulp.LpVariable(nombre, lowBound=fijos[nombre], upBound=fijos[nombre])
        elif tipo == "binaria":
            if relajar:
                v = pulp.LpVariable(nombre, lowBound=0, upBound=1)
            else:
                v = pulp.LpVariable(nombre, cat="Binary")
        elif tipo == "entera":
            v = pulp.LpVariable(
                nombre, lowBound=0, cat="Continuous" if relajar else "Integer"
            )
        elif tipo == "continua":
            v = pulp.LpVariable(nombre, lowBound=0)
        else:
            raise ValueError(f"tipo de variable no soportado: {tipo!r}")
        variables[nombre] = v

    modelo += pulp.lpSum(c * variables[n] for n, c in p["objetivo"].items()), "z"
    for fila, sentido_restr, rhs, _etiqueta in p["restricciones"]:
        lhs = pulp.lpSum(a * variables[n] for n, a in fila.items())
        if sentido_restr == "<=":
            modelo += lhs <= rhs
        elif sentido_restr == ">=":
            modelo += lhs >= rhs
        elif sentido_restr == "==":
            modelo += lhs == rhs
        else:
            raise ValueError(f"sentido de restricción no soportado: {sentido_restr!r}")

    modelo.solve(pulp.HiGHS(msg=False))
    estado = pulp.LpStatus[modelo.status]
    if estado != "Optimal":
        return {"estado": estado}
    return {
        "estado": estado,
        "x": {n: variables[n].value() for n in p["nombres"]},
        "z": pulp.value(modelo.objective),
    }


def es_peb(problema):
    return all(t == "binaria" for t in _canonico(problema)["tipos"].values())


def enumerar(problema):
    """Enumera las 2^n combinaciones binarias (numpy) y devuelve
    (n_factibles, z_optimo, optimos, segundo_z) con `optimos` = lista de
    dicts nombre -> 0/1 y segundo_z el segundo mejor valor distinto (None si
    no hay). Verificación independiente del solver (solo PEB)."""
    p = _canonico(problema)
    nombres = p["nombres"]
    n = len(nombres)
    if n > 22:
        raise ValueError("enumeración limitada a n <= 22 variables")
    indices = np.arange(2**n, dtype=np.int64)
    X = ((indices[:, None] >> np.arange(n)) & 1).astype(np.int64)  # 2^n x n
    factible = np.ones(2**n, dtype=bool)
    for fila, sentido, rhs, _etiqueta in p["restricciones"]:
        a = np.array([fila.get(nm, 0) for nm in nombres], dtype=np.int64)
        lhs = X @ a
        if sentido == "<=":
            factible &= lhs <= rhs
        elif sentido == ">=":
            factible &= lhs >= rhs
        else:
            factible &= lhs == rhs
    c = np.array([p["objetivo"][nm] for nm in nombres], dtype=np.int64)
    z = X @ c
    z_fact = z[factible]
    if z_fact.size == 0:
        return 0, None, [], None
    mejor = z_fact.max() if p["sentido"] == "max" else z_fact.min()
    distintos = sorted(set(z_fact.tolist()), reverse=(p["sentido"] == "max"))
    segundo = distintos[1] if len(distintos) > 1 else None
    optimos = [
        {nm: int(v) for nm, v in zip(nombres, X[i])}
        for i in np.flatnonzero(factible & (z == mejor))
    ]
    return int(z_fact.size), int(mejor), optimos, segundo


def ramificar_y_acotar(problema):
    """Ramificación y acotamiento para PEB (Hillier & Lieberman, sección
    11.6). Devuelve la lista de nodos en orden de creación, cada uno un dict
    con nombre, fijos, x, z, cota, sondeo (texto) y z_incumbente (Z* justo
    después de evaluarlo), más la solución incumbente final (z, x)."""
    p = _canonico(problema)
    if p["sentido"] != "max":
        raise ValueError("el árbol solo está implementado para maximización")
    nombres = p["nombres"]
    coef_enteros = all(float(c).is_integer() for c in p["objetivo"].values())

    def cota_de(z):
        # Z entero redondeado hacia abajo: válido solo con coeficientes enteros.
        return math.floor(z + EPS) if coef_enteros else z

    incumbente = {"z": -math.inf, "x": None}
    nodos = []
    abiertos = []

    def evaluar(fijos, iteracion):
        nodo = {
            "nombre": "Todo" if not nodos else f"Sub {len(nodos)}",
            "fijos": dict(fijos),
            "iter": iteracion,
            "x": None,
            "z": None,
            "cota": None,
            "sondeo": None,
        }
        r = resolver(problema, relajar=True, fijos=fijos)
        if r["estado"] != "Optimal":
            nodo["sondeo"] = "prueba 2 (relajación infactible)"
        else:
            x = [r["x"][nm] for nm in nombres]
            nodo["x"] = tuple(round(v, 4) + 0.0 for v in x)  # +0.0: evita "-0.0"
            nodo["z"] = round(r["z"], 4)
            nodo["cota"] = cota_de(r["z"])
            if nodo["cota"] <= incumbente["z"]:
                nodo["sondeo"] = f"prueba 1 (cota {nodo['cota']} <= Z* = {incumbente['z']})"
            elif all(_es_entero(v) for v in x):
                nodo["sondeo"] = "prueba 3 (solución entera)"
                if r["z"] > incumbente["z"]:
                    incumbente["z"] = round(r["z"])
                    incumbente["x"] = tuple(round(v) for v in x)
        nodo["z_incumbente"] = incumbente["z"]
        nodos.append(nodo)
        if nodo["sondeo"] is None:
            abiertos.append(nodo)
        return nodo

    evaluar({}, 0)
    iteracion = 0
    while abiertos:
        iteracion += 1
        # Creación más reciente; empate -> cota más grande.
        padre = max(abiertos, key=lambda nd: (nd["iter"], nd["cota"]))
        abiertos.remove(padre)
        j = len(padre["fijos"])  # siguiente variable, en orden natural
        for valor in (0, 1):
            evaluar({**padre["fijos"], nombres[j]: valor}, iteracion)
        # Con el Z* nuevo, se vuelve a aplicar la prueba 1 a los pendientes.
        for nd in list(abiertos):
            if nd["cota"] <= incumbente["z"]:
                nd["sondeo"] = f"prueba 1 (cota {nd['cota']} <= Z* = {incumbente['z']})"
                abiertos.remove(nd)
    return nodos, (incumbente["z"], incumbente["x"])


def _fmt_fijos(fijos, nombres):
    return "{" + ", ".join(f"{nombres.index(n) + 1}: {v}" for n, v in fijos.items()) + "}"


def _unos(x):
    return [n for n, v in x.items() if round(v) == 1]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROBLEMAS:
        print("Uso: python main.py <nombre_problema>")
        print(f"Problemas disponibles: {', '.join(PROBLEMAS)}")
        return

    problema = PROBLEMAS[sys.argv[1]]
    p = _canonico(problema)
    print(p["titulo"])
    print(f"{p['sentido']}, {len(p['nombres'])} variables, {len(p['restricciones'])} restricciones")
    print("Tipos:", ", ".join(sorted(set(p["tipos"].values()))))

    print("\n== Modelo entero (pulp/HiGHS) ==")
    r = resolver(problema)
    print("Estado:", r["estado"])
    if r["estado"] == "Optimal":
        print("Z =", round(r["z"], 4))
        print("Variables en 1:", _unos(r["x"]))

    print("\n== Relajación de PL ==")
    rel = resolver(problema, relajar=True)
    print("Estado:", rel["estado"])
    if rel["estado"] == "Optimal":
        print("Z =", round(rel["z"], 4))
        fracc = {n: round(v, 4) for n, v in rel["x"].items() if not _es_entero(v)}
        print("Variables fraccionarias:", fracc if fracc else "ninguna (la relajación ya es entera)")

    if not es_peb(problema):
        print("\n(Enumeración y árbol de ramificación: solo para PEB.)")
        return

    n = len(p["nombres"])
    n_fact, z_opt, optimos, segundo = enumerar(problema)
    print("\n== Enumeración ==")
    print("Combinaciones posibles:", 2**n)
    print("Combinaciones factibles:", n_fact)
    print("Z óptimo:", z_opt, "| soluciones óptimas:", len(optimos), "| segundo mejor Z distinto:", segundo)
    for x in optimos[:3]:
        print("  óptima:", _unos(x))
    if r["estado"] == "Optimal":
        coincide = abs(r["z"] - z_opt) < EPS
        print("Coincide con el solver:", "sí" if coincide else "NO  <-- revisar")

    if problema.get("arbol"):
        print("\n== Ramificación y acotamiento (H&L, sección 11.6) ==")
        nodos, (z_inc, x_inc) = ramificar_y_acotar(problema)
        for nd in nodos:
            base = f"{nd['nombre']:6}{_fmt_fijos(nd['fijos'], p['nombres']):26}"
            if nd["x"] is None:
                print(f"{base}-> infactible  [{nd['sondeo']}]")
            else:
                sondeo = f"  [{nd['sondeo']}]" if nd["sondeo"] else ""
                print(f"{base}-> x={nd['x']} Z={nd['z']} cota={nd['cota']}{sondeo}")
        print("Solución óptima (incumbente final):", x_inc, " Z* =", z_inc)


if __name__ == "__main__":
    main()
