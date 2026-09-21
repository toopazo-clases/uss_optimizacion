"""
Banco de problemas de programación lineal en 2 variables: se resuelven con
pulp y se grafican (región factible + curvas de nivel + óptimo).

Uso:
    python main.py <nombre_problema>

Solo procesa si <nombre_problema> existe en PROBLEMAS. Sin argumento, o con
un nombre que no está en el banco, no resuelve nada — solo muestra la lista
de problemas disponibles.

Flujo:
    1. pulp resuelve el LP y entrega el óptimo (x1*, x2*, z*). Es el único
       método usado para encontrar la solución — no scipy, no un método
       gráfico ad-hoc.
    2. pulp no entrega el resto del polígono de la región factible (solo el
       vértice óptimo), y eso hace falta para poder dibujar la región. Por
       eso, y solo para eso, se calculan "a mano" (intersección de pares de
       restricciones activas) los demás vértices del polígono.
    3. Con el óptimo de pulp y los vértices calculados se arma el gráfico.
"""

import itertools
import os
import shutil
import subprocess
import sys

import highspy
import matplotlib.pyplot as plt
import numpy as np
import pulp

# --------------------------------------------------------------------------
# Banco de problemas
# --------------------------------------------------------------------------
# Cada problema es un diccionario con:
#   titulo        -> nombre para el gráfico
#   sentido       -> "max" o "min"
#   objetivo      -> (C1, C2) tal que z = C1*x1 + C2*x2
#   restricciones -> lista de (a, b, rhs) para a*x1 + b*x2 <= rhs.
#                    No incluir x1>=0 ni x2>=0: pulp ya las aplica vía
#                    lowBound=0 (ver resolver_pulp) y la visualización las
#                    agrega sola (ver NO_NEGATIVIDAD) — ponerlas acá las
#                    duplicaría en el modelo de pulp.
#
# "ejercicio_*" son problemas inventados para práctica, no de ningún libro.

PROBLEMAS = {
    "reddy_mikks": {
        "titulo": "Reddy Mikks (Taha, ejemplo 2.2-1)",
        "sentido": "max",
        "objetivo": (5, 4),
        "restricciones": [
            (6, 4, 24),
            (1, 2, 6),
            (-1, 1, 1),
            (0, 1, 2),
        ],
    },
    "ozark_farms": {
        "titulo": "Ozark Farms (Taha, ejemplo 2.2-2)",
        "sentido": "min",
        "objetivo": (0.30, 0.90),
        "restricciones": [
            # (-(0.30 - 0.09), +(0.30 - 0.60), 0),
            # (+(0.02 - 0.05), -(0.06 - 0.05), 0),
            (+0.21, -0.30, 0),
            (-0.03, +0.01, 0),
            (-1, -1, -800),
        ],
    },
    "fabrica_secuencial_10hr": {
        "titulo": "Fabrica secuencial (Taha, problemas 2.2A-4)",
        "sentido": "max",
        "objetivo": (2, 3),
        "restricciones": [
            ((10 + 6 + 8), (5 + 20 + 10), 600),
        ],
    },
    "fabrica_paralela_10hr": {
        "titulo": "Fabrica paralela (Taha, problemas 2.2A-4)",
        "sentido": "max",
        "objetivo": (2, 3),
        "restricciones": [
            ((10), (5), 600),
            ((6), (20), 600),
            ((8), (10), 600),
        ],
    },
    "guia1_p1": {
        "titulo": "Guia 1, P1",
        "sentido": "max",
        "objetivo": (4000, 5000),
        "restricciones": [
            (125, 200, 50000),
            (150, 100, 30000),
            (72, 27, 10000),
        ],
    },
    "guia1_p3": {
        "titulo": "Guia 1, P3",
        "sentido": "max",
        "objetivo": (8500, 8100),
        "restricciones": [
            (2, 3, 550),
            (3, 1, 480),
            (-1, -1, -200),
        ],
    },
    "guia1_p6": {
        "titulo": "Guia 1, P6",
        "sentido": "min",
        "objetivo": (20 / 1000, 30 / 1000),
        "restricciones": [
            (5 / 100, 2.5 / 100, 40),
            (1 / 100, 3 / 100, 20),
            (-1, -1, -1000),
        ],
    },
    "guia1_p11": {
        "titulo": "Guia 1, P11",
        "sentido": "min",
        "objetivo": (50, 40),
        "restricciones": [
            (-4, -2, -20),
            (-2, -5, -20),
        ],
    },
    "guia1_p13": {
        "titulo": "Guia 1, P13",
        "sentido": "max",
        "objetivo": (80, 70),
        "restricciones": [
            (40, 20, 2000),
            (20, 30, 1500),
            (-2, 1, 0),
        ],
    },
    "simplex_estandar": {
        "titulo": "Simplex estándar",
        "sentido": "max",
        "objetivo": (1, 2),
        "restricciones": [
            (1, 4, 15),
            (3, 1, 16),
        ],
    },
    "simplex_combinado": {
        "titulo": "Simplex combinado",
        "sentido": "max",
        "objetivo": (1, 2),
        "restricciones": [(1, 4, 15), (3, 1, 18), (-2, -2, -10)],
    },
    "simplex_combinado2": {
        "titulo": "Simplex combinado",
        "sentido": "max",
        "objetivo": (1, 5),
        "restricciones": [(1, 4, 15), (3, 1, 18), (-2, -2, -10)],
    },
    "taha_fig_3.12": {
        "titulo": "Ejemplo Fig 3.12 (Taha)",
        "sentido": "max",
        "objetivo": (30, 20),
        "restricciones": [(2, 1, 8), (1, 3, 8)],
    },
    "taha_fig_3.12_dual": {
        # Dual de taha_fig_3.12: min w = 8y1 + 8y2 s.a. 2y1+y2>=30,
        # y1+3y2>=20, y1,y2>=0. Restricciones reescritas como <= (mismo
        # truco que guia1_p11/guia1_p6: multiplicar por -1) para calzar
        # con el formato (a, b, rhs) <= que usa el resto del banco.
        "titulo": "Ejemplo Fig 3.12 (Taha) -- Dual",
        "sentido": "min",
        "objetivo": (8, 8),
        "restricciones": [(-2, -1, -30), (-1, -3, -20)],
    },
    "florista_ambulante": {
        # Usado solo para graficar la región factible en (x1,x2) -- el
        # problema real (con un tercer producto x3) vive en
        # pulp_LpProblem_3d/main.py, problema "florista_ambulante_3d".
        # Esta entrada es exactamente el corte x3=0 de ese problema (mismos
        # x1*,x2*,z*,y1*,y2*,y3*), no una versión vieja/desactualizada:
        # x3 no aporta nada a R1/R2/R3 en x3=0, así que la región y el
        # vértice óptimo coinciden -- se reusa acá solo para graficar (con
        # 3 variables no se puede graficar la región 2D con este banco,
        # que es 2D).
        "titulo": "Florista ambulante",
        "sentido": "max",
        "objetivo": (2000, 1000),
        "restricciones": [
            (3, 1, 300),
            (1, 1, 140),
            (1, 3, 300),
        ],
    },
    "ejercicio_a": {
        "titulo": "Ejercicio A (práctica)",
        "sentido": "max",
        "objetivo": (3, 5),
        "restricciones": [
            (1, 0, 4),
            (0, 2, 12),
            (3, 2, 18),
        ],
    },
    "ejercicio_b": {
        "titulo": "Ejercicio B (práctica, minimización)",
        "sentido": "min",
        "objetivo": (2, 3),
        "restricciones": [
            (-1, -1, -2),
            (1, 0, 5),
            (0, 1, 5),
        ],
    },
    "ejercicio_c": {
        "titulo": "Ejercicio C (práctica)",
        "sentido": "max",
        "objetivo": (1, 2),
        "restricciones": [
            (1, 3, 15),
            (4, 1, 16),
        ],
    },
    "ejercicio_d": {
        "titulo": "Ejercicio D (práctica, triángulo)",
        "sentido": "max",
        "objetivo": (2, 3),
        "restricciones": [
            (1, 1, 6),
        ],
    },
}


# --------------------------------------------------------------------------
# Parte 1: resolver con pulp
# --------------------------------------------------------------------------


def resolver_pulp(problema):
    """Resuelve el LP con pulp. Devuelve (x1, x2, z, estado)."""
    c1, c2 = problema["objetivo"]
    sentido = (
        pulp.LpMaximize if problema["sentido"] == "max" else pulp.LpMinimize
    )

    modelo = pulp.LpProblem("modelo", sentido)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)

    modelo += c1 * x1 + c2 * x2, "z"
    for a, b, rhs in problema["restricciones"]:
        modelo += a * x1 + b * x2 <= rhs

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    estado = pulp.LpStatus[modelo.status]
    return pulp.value(x1), pulp.value(x2), pulp.value(modelo.objective), estado


# --------------------------------------------------------------------------
# Parte 1.5: sensibilidad -- precio sombra y su rango de validez (RHS)
# --------------------------------------------------------------------------
# HiGHS (el solver, no pulp) sí trae ranging de sensibilidad incorporado:
# Highs.getRanging() devuelve, en una sola llamada tras UN solve, el rango
# del lado derecho de cada restricción donde la base óptima (y por lo tanto
# el precio sombra) no cambia -- ver la fórmula B^{-1}b y su condición de
# factibilidad x_B(b)>=0 en la teoría; getRanging() es exactamente eso, ya
# resuelto por el solver. No hace falta re-resolver el LP en bucle (como en
# un intento anterior con bisección) ni tampoco salir de pulp: el wrapper
# pulp.HiGHS ya arma internamente un highspy.Highs() y lo deja accesible en
# modelo.solverModel, así que basta con resolver con ese backend en vez de
# CBC y leer modelo.solverModel.getRanging().
#
# Ojo con el signo de los duales: con CBC (resolver_pulp), restriccion.pi
# de una restricción activa de un problema Max sale positivo (+14); con
# HiGHS, restriccion.pi (y row_dual) sale con el signo contrario (-14) --
# es una convención de signo distinta entre backends de pulp, no un error.
# Por eso acá se normaliza con abs().


def analizar_sensibilidad(problema):
    """Para cada restricción real: su precio sombra y el rango de su lado
    derecho (b_i) donde ese precio sombra es válido -- ambos calculados por
    HiGHS en un único solve (Highs.getRanging()), vía pulp.HiGHS. Devuelve
    una lista de dicts (uno por restricción); [] si el problema no es
    óptimo."""
    c1, c2 = problema["objetivo"]
    sentido = (
        pulp.LpMaximize if problema["sentido"] == "max" else pulp.LpMinimize
    )

    modelo = pulp.LpProblem("modelo", sentido)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)

    modelo += c1 * x1 + c2 * x2, "z"
    restricciones_pulp = []
    for a, b, rhs in problema["restricciones"]:
        restriccion = a * x1 + b * x2 <= rhs
        modelo += restriccion
        restricciones_pulp.append(restriccion)

    modelo.solve(pulp.HiGHS(msg=False))
    if pulp.LpStatus[modelo.status] != "Optimal":
        return []

    estado_ranging, ranging = modelo.solverModel.getRanging()
    if estado_ranging != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS getRanging() falló: {estado_ranging}")

    filas = []
    for idx, ((a, b, rhs), restriccion) in enumerate(
        zip(problema["restricciones"], restricciones_pulp), start=1
    ):
        filas.append(
            {
                "idx": idx,
                "rhs": rhs,
                "pi": abs(restriccion.pi),
                "b_inf": ranging.row_bound_dn.value_[idx - 1],
                "b_sup": ranging.row_bound_up.value_[idx - 1],
            }
        )
    return filas


# --------------------------------------------------------------------------
# Parte 2: visualización
# --------------------------------------------------------------------------
# pulp ya entregó el óptimo. Lo que sigue es geometría para poder dibujar la
# región factible completa (pulp no la entrega, solo el vértice óptimo).
#
# pulp aplica x1>=0, x2>=0 vía lowBound=0, no como restricciones del modelo,
# así que esta parte no las hereda de ahí. Para poder cerrar el polígono
# (intersección de pares de restricciones) hacen falta como desigualdades
# explícitas, y se agregan acá una sola vez en vez de pedirle a cada
# problema del banco que las repita en su lista de restricciones.

TOL = 1e-9
SLACK = 1e-6
NO_NEGATIVIDAD = [(-1, 0, 0), (0, -1, 0)]


def _interseccion(c_i, c_j):
    a1, b1, r1 = c_i
    a2, b2, r2 = c_j
    det = a1 * b2 - a2 * b1
    if abs(det) < TOL:
        return None
    x1 = (r1 * b2 - r2 * b1) / det
    x2 = (a1 * r2 - a2 * r1) / det
    return x1, x2


def _es_factible(x1, x2, restricciones):
    return all(a * x1 + b * x2 <= rhs + SLACK for a, b, rhs in restricciones)


# --------------------------------------------------------------------------
# Parte 2.5: enunciado y método algebraico de vértices
# --------------------------------------------------------------------------
# pulp entrega el óptimo directo, sin pasar por esto. Esta parte reconstruye
# "a mano" el método de los vértices que normalmente se enseña antes de usar
# un solver: igualar de a pares las restricciones reales (regla de Cramer),
# evaluar z en el cruce, y chequear factibilidad evaluando numéricamente las
# restricciones restantes. Solo se usa para el PDF de álgebra
# (construir_tex_algebra, más abajo) — no reemplaza a pulp como método de
# solución. Es independiente del renderizador (no sabe nada de LaTeX ni de
# matplotlib): solo devuelve números y texto de fórmulas en bruto.


def _terminos_lineales(a, b, v1="x_1", v2="x_2"):
    """Texto mathtext de a*v1 + b*v2, con signos correctos y sin términos
    nulos (p. ej. a=-1,b=1 -> "-x_1+x_2"; a=0,b=1 -> "x_2")."""
    partes = []
    for coef, var in ((a, v1), (b, v2)):
        if coef == 0:
            continue
        signo = "-" if coef < 0 else ("+" if partes else "")
        valor = abs(coef)
        coef_str = "" if valor == 1 else _fmt(valor)
        partes.append(f"{signo}{coef_str}{var}")
    return "".join(partes) if partes else "0"


def _cero(v):
    """Normaliza -0.0 a 0.0 (si no, se formatea feo: "-0.0000")."""
    return 0.0 if v == 0 else v


def _fmt(v):
    """Formatea un número sin notación científica y sin perder precisión.
    A diferencia de ":g" (que para números grandes, p. ej. el z* de
    p1_guia1 ~1.3 millones, pasa a notación científica y lo aproxima a
    "1.3e+06"), esto siempre usa punto fijo: enteros tal cual, y
    fraccionarios con hasta 6 decimales sin ceros de más."""
    v = _cero(float(v))
    entero = round(v)
    if abs(v - entero) < 1e-6:
        return f"{entero}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _restantes_reales(x1, x2, restricciones, excluir):
    """Evalúa en (x1, x2) las restricciones reales cuyo índice (1-based) no
    está en `excluir`. Reutilizado por los ejes (excluye 1) y los pares
    (excluye 2)."""
    restantes = []
    for k, (a, b, rhs) in enumerate(restricciones, start=1):
        if k in excluir:
            continue
        valor = _cero(a * x1 + b * x2)
        restantes.append(
            {
                "tipo": "real",
                "etiqueta": f"R{k}",
                "a": a,
                "b": b,
                "rhs": rhs,
                "valor": valor,
                "cumple": valor <= rhs + SLACK,
            }
        )
    return restantes


def _restante_noneg(var, valor):
    valor = _cero(valor)
    return {
        "tipo": "noneg",
        "var": var,
        "valor": valor,
        "cumple": valor >= -SLACK,
    }


def _intersecciones_ejes(problema):
    """ "Esquinas fáciles" del método de los vértices: la intersección de
    cada restricción real R_i con cada eje de no-negatividad (x1=0, x2=0).
    A diferencia de _intersecciones_reales no hace falta resolver un
    sistema 2x2 — alcanza con despejar la otra variable."""
    restricciones = problema["restricciones"]
    c1, c2 = problema["objetivo"]
    puntos = []
    for i, (a, b, rhs) in enumerate(restricciones, start=1):
        # eje=1 fija x1=0, así que la variable que queda libre es x2 (su
        # coeficiente es b, no a) -- y viceversa para eje=2.
        for eje, coef_usado in ((1, b), (2, a)):
            if abs(coef_usado) < TOL:
                puntos.append(
                    {
                        "i": i,
                        "eje": eje,
                        "a": a,
                        "b": b,
                        "rhs": rhs,
                        "paralela": True,
                    }
                )
                continue

            otro_val = _cero(rhs / coef_usado)
            x1 = 0.0 if eje == 1 else otro_val
            x2 = otro_val if eje == 1 else 0.0
            z = _cero(c1 * x1 + c2 * x2)

            otra_var = "x_2" if eje == 1 else "x_1"
            restantes = _restantes_reales(
                x1, x2, restricciones, excluir={i}
            ) + [_restante_noneg(otra_var, otro_val)]
            factible = all(r["cumple"] for r in restantes)

            puntos.append(
                {
                    "i": i,
                    "eje": eje,
                    "a": a,
                    "b": b,
                    "rhs": rhs,
                    "paralela": False,
                    "coef_usado": coef_usado,
                    "otro_val": otro_val,
                    "x1": x1,
                    "x2": x2,
                    "z": z,
                    "restantes": restantes,
                    "factible": factible,
                }
            )
    return puntos


def _intersecciones_reales(problema):
    """Para cada par de restricciones reales, resuelve el sistema 2x2 (regla
    de Cramer), evalúa z y chequea factibilidad contra las restricciones
    restantes (las demás reales + no-negatividad). No empareja x1>=0/x2>=0
    entre sí ni contra una restricción real: eso ya lo cubre
    vertices_region_factible para el gráfico (y, para el caso R_i con un
    eje, _intersecciones_ejes)."""
    restricciones = problema["restricciones"]
    c1, c2 = problema["objetivo"]
    pares = []
    for (i, ci), (j, cj) in itertools.combinations(
        enumerate(restricciones, start=1), 2
    ):
        a1, b1, r1 = ci
        a2, b2, r2 = cj
        det = a1 * b2 - a2 * b1
        if abs(det) < TOL:
            pares.append(
                {"i": i, "j": j, "ci": ci, "cj": cj, "paralelas": True}
            )
            continue

        num_x1 = _cero(r1 * b2 - r2 * b1)
        num_x2 = _cero(a1 * r2 - a2 * r1)
        x1 = _cero(num_x1 / det)
        x2 = _cero(num_x2 / det)
        z = _cero(c1 * x1 + c2 * x2)

        restantes = _restantes_reales(x1, x2, restricciones, excluir={i, j}) + [
            _restante_noneg("x_1", x1),
            _restante_noneg("x_2", x2),
        ]
        factible = all(r["cumple"] for r in restantes)

        pares.append(
            {
                "i": i,
                "j": j,
                "ci": ci,
                "cj": cj,
                "paralelas": False,
                "det": det,
                "num_x1": num_x1,
                "num_x2": num_x2,
                "x1": x1,
                "x2": x2,
                "z": z,
                "restantes": restantes,
                "factible": factible,
            }
        )
    return pares


def _segmento_visible(a, b, rhs, x1_max, x2_max):
    """Recorta la recta a*x1 + b*x2 = rhs contra la caja [0,x1_max] x
    [0,x2_max] y devuelve sus dos extremos visibles, o None si la recta no
    cruza la caja (p. ej. queda fuera del rango que se está graficando)."""
    candidatos = []
    if abs(b) > TOL:
        for x1 in (0.0, x1_max):
            x2 = (rhs - a * x1) / b
            if -TOL <= x2 <= x2_max + TOL:
                candidatos.append((x1, x2))
    if abs(a) > TOL:
        for x2 in (0.0, x2_max):
            x1 = (rhs - b * x2) / a
            if -TOL <= x1 <= x1_max + TOL:
                candidatos.append((x1, x2))

    candidatos = list(set(candidatos))
    if len(candidatos) < 2:
        return None
    # La recta puede cruzar hasta 4 bordes de la caja (p. ej. justo por una
    # esquina, que se cuenta dos veces); nos quedamos con el par de puntos
    # más separados entre sí como extremos del segmento a dibujar.
    return max(
        itertools.combinations(candidatos, 2),
        key=lambda par: (par[0][0] - par[1][0]) ** 2
        + (par[0][1] - par[1][1]) ** 2,
    )


def vertices_region_factible(restricciones):
    """Vértices del polígono factible (a mano), ordenados en sentido
    antihorario — solo para poder graficar la región, pulp no los entrega."""
    puntos = []
    for c_i, c_j in itertools.combinations(restricciones, 2):
        p = _interseccion(c_i, c_j)
        if p is None:
            continue
        x1, x2 = p
        if _es_factible(x1, x2, restricciones):
            puntos.append((round(x1, 6) or 0.0, round(x2, 6) or 0.0))

    puntos = list(set(puntos))
    cx = sum(p[0] for p in puntos) / len(puntos)
    cy = sum(p[1] for p in puntos) / len(puntos)
    puntos.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return puntos


def graficar_region_factible(ax, vertices):
    x1v = [v[0] for v in vertices] + [vertices[0][0]]
    x2v = [v[1] for v in vertices] + [vertices[0][1]]
    ax.fill(x1v, x2v, color="lightblue", alpha=0.5, label="Región factible")
    ax.plot(x1v, x2v, color="steelblue", linewidth=2)


def graficar_restricciones(ax, restricciones, vertices, x1_max, x2_max):
    """Dibuja cada restricción real (sin no-negatividad, esas ya son los
    ejes) como recta punteada en todo el rango visible — incluidas las
    redundantes, que graficar_region_factible nunca dibuja porque no
    aportan ningún vértice al polígono factible (ver
    vertices_region_factible)."""
    for idx, (a, b, rhs) in enumerate(restricciones, start=1):
        segmento = _segmento_visible(a, b, rhs, x1_max, x2_max)
        if segmento is None:
            continue
        (x1_a, x2_a), (x1_b, x2_b) = segmento

        activa = any(abs(a * vx + b * vy - rhs) < 1e-4 for vx, vy in vertices)
        if activa:
            color, estilo, etiqueta = "darkorange", "--", f"R{idx}"
        else:
            color, estilo = "magenta", ":"
            etiqueta = f"R{idx} (redundante)"

        ax.plot(
            [x1_a, x1_b],
            [x2_a, x2_b],
            linestyle=estilo,
            linewidth=1,
            color=color,
            alpha=0.8,
            zorder=1,
        )
        ax.annotate(
            etiqueta,
            (x1_b, x2_b),
            fontsize=7,
            color=color,
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
        )


def graficar_curvas_nivel(ax, c1, c2, x1_max, x2_max, z_vertices):
    x1_grid = np.linspace(0, x1_max, 400)
    x2_grid = np.linspace(0, x2_max, 400)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = c1 * X1 + c2 * X2

    niveles = np.linspace(min(z_vertices), max(z_vertices), 8)
    contornos = ax.contour(
        X1, X2, Z, levels=niveles, colors="gray", linewidths=0.8
    )
    ax.clabel(contornos, inline=True, fontsize=8, fmt="%.1f")


def graficar_vertices(ax, vertices, c1, c2):
    for x1, x2 in vertices:
        z = c1 * x1 + c2 * x2
        ax.plot(x1, x2, "o", color="steelblue", markersize=4)
        ax.annotate(
            f"({_fmt(x1)},{_fmt(x2)})\nz={_fmt(z)}",
            (x1, x2),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )


def graficar_gradiente(ax, c1, c2, x1_max):
    """Vector gradiente ∇z = (c1, c2) desde el origen: dirección de máximo
    crecimiento de z."""
    grad = np.array([c1, c2], dtype=float)
    grad_dir = grad / np.linalg.norm(grad)
    longitud = 0.2 * x1_max
    punta = grad_dir * longitud
    ax.annotate(
        "",
        xy=punta,
        xytext=(0, 0),
        arrowprops=dict(
            facecolor="darkred", edgecolor="darkred", width=1.5, headwidth=8
        ),
    )
    ax.text(punta[0], punta[1], " ∇z", color="darkred", fontsize=10)


def graficar_optimo(ax, x1, x2, z):
    """Óptimo: viene de pulp, no del cálculo geométrico."""
    ax.plot(
        x1,
        x2,
        "D",
        color="red",
        markersize=10,
        label=f"Óptimo pulp ({_fmt(x1)}, {_fmt(x2)}), z={_fmt(z)}",
    )


def construir_grafico_solucion(problema, x1, x2, z, estado):
    """PNG simple con el resumen textual de la solución, en formato
    "footer" (proporción 2:10, alto:ancho) para pegar debajo del enunciado
    en una presentación. Equivalente a:
        print("Estado:", LpStatus[modelo.status])
        print("x1 =", value(x1))
        print("x2 =", value(x2))
        print("Z =", value(modelo.objective))
    """
    lineas = [
        problema["titulo"],
        f"Estado: {estado}",
        f"x1 = {x1:.4f}    x2 = {x2:.4f}    Z = {z:.4f}",
    ]
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        "\n".join(lineas),
        fontsize=24,
        family="monospace",
        va="center",
        transform=ax.transAxes,
    )
    fig.tight_layout(pad=0.15)
    return fig


def construir_grafico_sensibilidad(problema, filas):
    """PNG compacto (mismo estilo "footer" que construir_grafico_solucion)
    con el precio sombra y el rango de validez del lado derecho (b_i) de
    cada restricción real -- para pegar en una lámina, análogo a como
    construir_grafico_solucion resume el óptimo."""
    lineas = [
        f"{problema['titulo']} -- Sensibilidad (precio sombra y su rango)"
    ]
    for f in filas:
        if f["b_inf"] is None or f["b_sup"] is None:
            rango = "no acotado"
        else:
            rango = f"[{_fmt(f['b_inf'])}, {_fmt(f['b_sup'])}]"
        lineas.append(
            f"R{f['idx']}: b={_fmt(f['rhs'])}   "
            f"precio sombra = {_fmt(f['pi'])}   "
            f"válido para b en {rango}"
        )
    fig, ax = plt.subplots(figsize=(10, 0.7 + 0.45 * len(lineas)))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        "\n".join(lineas),
        fontsize=22,
        family="monospace",
        va="center",
        transform=ax.transAxes,
    )
    fig.tight_layout(pad=0.15)
    return fig


def _tex_escape(s):
    """Escapa caracteres especiales de LaTeX en texto plano (no en modo
    matemático). Defensivo: hoy ningún texto del banco los usa, pero un
    ".tex" generado con texto sin escapar rompe la compilación en vez de
    solo verse feo (como pasaría con matplotlib)."""
    reemplazos = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(reemplazos[ch] if ch in reemplazos else ch for ch in s)


def _tex_marca(cumple):
    """✓/✗ en macros LaTeX estándar (amssymb/pifont) en vez de los
    caracteres Unicode ✓/✗ que usaba la versión matplotlib — no dependen de
    que la fuente tenga esos glyphs, compilan con pdflatex sin más."""
    return r"\checkmark\ cumple" if cumple else r"\ding{55}\ NO cumple"


def _math_display(contenido):
    """Ecuación en modo display que se autoajusta al ancho de columna
    (paquete adjustbox). Sin esto, las fórmulas de Cramer con números
    grandes (p. ej. p1_guia1, con rhs de hasta 50000) son más anchas que
    una columna angosta de multicol y terminan saliéndose/superponiéndose
    con la columna vecina (overfull \\hbox)."""
    return (
        "\\[\n"
        f"\\adjustbox{{max width=\\linewidth}}{{$\\displaystyle {contenido}$}}\n"
        "\\]"
    )


def _enunciado_tex(problema):
    """Bloques \\[...\\] en LaTeX real del problema tal como está escrito:
    objetivo, restricciones reales (mismos índices R1..Rn que
    graficar_restricciones) y no-negatividad."""
    c1, c2 = problema["objetivo"]
    palabra = r"\max" if problema["sentido"] == "max" else r"\min"
    partes = [
        _math_display(f"{palabra}\\ z = {_terminos_lineales(c1, c2)}"),
        "s.a.:",
    ]
    for idx, (a, b, rhs) in enumerate(problema["restricciones"], start=1):
        partes.append(
            _math_display(
                f"R_{{{idx}}}:\\ {_terminos_lineales(a, b)} \\leq {_fmt(rhs)}"
            )
        )
    partes.append(_math_display(r"x_1,\ x_2 \geq 0"))
    return "\n".join(partes)


def _resultado_tex(c1, c2, x1, x2, z, restantes, factible):
    """Bloque LaTeX de z*, la evaluación de cada restricción restante (como
    lista) y el veredicto de factibilidad — común a las esquinas de los
    ejes y a los pares de restricciones."""
    partes = [
        _math_display(
            f"z^{{*}}={_fmt(c1)}({x1:.4f})+{_fmt(c2)}({x2:.4f})={z:.4f}"
        ),
        "Restricciones restantes:",
        r"\begin{itemize}",
    ]
    for r in restantes:
        marca = _tex_marca(r["cumple"])
        if r["tipo"] == "real":
            partes.append(
                f"\\item ${r['etiqueta']}:\\ "
                f"{_terminos_lineales(r['a'], r['b'])}"
                f"={r['valor']:.4f} \\leq {_fmt(r['rhs'])}$ \\quad {marca}"
            )
        else:
            partes.append(
                f"\\item ${r['var']} \\geq 0$: "
                f"${r['var']}={r['valor']:.4f}$ \\quad {marca}"
            )
    partes.append(r"\end{itemize}")
    veredicto = "FACTIBLE" if factible else "NO FACTIBLE"
    partes.append(f"$\\Rightarrow$ Punto \\textbf{{{veredicto}}}")
    return "\n".join(partes)


def _paso1_tex(problema):
    """Bloque LaTeX del paso 1: cada restricción real cruzada con cada eje
    (esquina fácil, sin sistema 2x2)."""
    c1, c2 = problema["objetivo"]
    partes = []
    for p in _intersecciones_ejes(problema):
        i, eje = p["i"], p["eje"]
        a, b, rhs = p["a"], p["b"], p["rhs"]
        eje_var = "x_1" if eje == 1 else "x_2"
        otra_var = "x_2" if eje == 1 else "x_1"
        partes.append(f"\\paragraph{{$R_{{{i}}}\\cap {eje_var}=0$}}")
        partes.append(
            _math_display(
                f"{_terminos_lineales(a, b)} = {_fmt(rhs)} "
                f"\\quad \\text{{con }} {eje_var}=0"
            )
        )
        if p["paralela"]:
            partes.append(
                f"${otra_var}$ no aparece en $R_{{{i}}}$ (recta paralela "
                f"al eje ${eje_var}=0$): sin intersección única."
            )
            continue

        coef_usado, otro_val = p["coef_usado"], p["otro_val"]
        x1, x2, z = p["x1"], p["x2"], p["z"]
        partes.append(
            _math_display(
                f"{otra_var}^{{*}}=\\frac{{{_fmt(rhs)}}}{{{_fmt(coef_usado)}}}"
                f"={otro_val:.4f}"
            )
        )
        partes.append(
            _resultado_tex(c1, c2, x1, x2, z, p["restantes"], p["factible"])
        )
    return "\n".join(partes)


def _paso2_tex(problema):
    """Bloque LaTeX del paso 2: cada par de restricciones reales cruzadas
    entre sí (regla de Cramer)."""
    c1, c2 = problema["objetivo"]
    partes = [
        r"Para cada par de restricciones activas $R_i,R_j$: "
        r"$a_ix_1+b_ix_2=r_i,\ \ a_jx_1+b_jx_2=r_j$",
        _math_display(
            r"\Rightarrow\ x_1^{*}=\frac{r_ib_j-r_jb_i}{a_ib_j-a_jb_i}"
            r"\ ,\quad x_2^{*}=\frac{a_ir_j-a_jr_i}{a_ib_j-a_jb_i}"
        ),
    ]

    if len(problema["restricciones"]) < 2:
        partes.append(
            "Este problema tiene una sola restricción real: no hay pares "
            "para intersectar (ya se cubrió todo en el paso 1)."
        )
        return "\n".join(partes)

    for p in _intersecciones_reales(problema):
        i, j = p["i"], p["j"]
        a1, b1, r1 = p["ci"]
        a2, b2, r2 = p["cj"]
        partes.append(f"\\paragraph{{$R_{{{i}}}\\cap R_{{{j}}}$}}")
        partes.append(
            _math_display(
                f"{_terminos_lineales(a1, b1)} = {_fmt(r1)} \\quad "
                f"{_terminos_lineales(a2, b2)} = {_fmt(r2)}"
            )
        )
        if p["paralelas"]:
            partes.append("Rectas paralelas: sin punto de intersección.")
            continue

        det, num_x1, num_x2 = p["det"], p["num_x1"], p["num_x2"]
        x1, x2, z = p["x1"], p["x2"], p["z"]
        partes.append(
            _math_display(
                f"x_1^{{*}}=\\frac{{({_fmt(r1)})({_fmt(b2)})-({_fmt(r2)})({_fmt(b1)})}}"
                f"{{({_fmt(a1)})({_fmt(b2)})-({_fmt(a2)})({_fmt(b1)})}}"
                f"=\\frac{{{_fmt(num_x1)}}}{{{_fmt(det)}}}={x1:.4f}"
            )
        )
        partes.append(
            _math_display(
                f"x_2^{{*}}=\\frac{{({_fmt(a1)})({_fmt(r2)})-({_fmt(a2)})({_fmt(r1)})}}"
                f"{{({_fmt(a1)})({_fmt(b2)})-({_fmt(a2)})({_fmt(b1)})}}"
                f"=\\frac{{{_fmt(num_x2)}}}{{{_fmt(det)}}}={x2:.4f}"
            )
        )
        partes.append(
            _resultado_tex(c1, c2, x1, x2, z, p["restantes"], p["factible"])
        )
    return "\n".join(partes)


_TEX_PREAMBULO = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage{amsmath,amssymb,pifont,parskip,adjustbox,graphicx}
\usepackage[margin=2cm]{geometry}
\usepackage{multicol}
\setlength{\columnsep}{0.8cm}
% multicol intenta balancear el alto de las dos columnas en la última
% página del entorno, y ese cálculo se equivoca con contenido de alto
% variable (itemize + ecuaciones en \[...\]): el texto termina
% superponiéndose en vez de cortar prolijo a la columna siguiente.
% \raggedcolumns lo desactiva -- las columnas quedan de alto natural, sin
% forzar el balanceo.
\raggedcolumns
\pagestyle{plain}
% Registros para medir el alto real del bloque título+enunciado en la
% página 1 y así saber cuánto le queda disponible al gráfico (ver
% construir_tex_algebra) -- \pagegoal/\pagetotal (el otro truco típico
% para esto) no sirve tan temprano en la página, antes de que el "page
% builder" de TeX los inicialice.
\newsavebox{\encabezadobox}
\newlength{\alturarestante}
% babel-spanish convierte "." en coma decimal en modo matemático por
% default; se revierte para que los números se vean igual que en el resto
% del proyecto (prints de Python, los otros dos PNG).
\decimalpoint"""


def construir_tex_algebra(problema, x1_opt, x2_opt, z_opt, ruta_grafico):
    """Fuente .tex completa: el enunciado del problema, el gráfico de la
    región factible (ruta_grafico, a ancho completo) y el método de los
    vértices en dos columnas al estilo paper (paquete multicol) — un único
    flujo continuo (paso 1, luego paso 2, luego el óptimo) que llena la
    columna izquierda y sigue en la derecha, paginando solo. A diferencia
    de paracol (columnas paralelas independientes, paso 1 siempre a la
    izquierda), acá "Óptimo (pulp)" queda donde corresponde: pegado al
    final del paso 2, no forzado a esperar que ambas columnas terminen. A
    diferencia de la versión matplotlib/mathtext (PNG) original, acá LaTeX
    se encarga de envolver el texto y de paginar; no hace falta medir nada
    a mano."""
    titulo = _tex_escape(problema["titulo"])
    # Ruta absoluta: evita cualquier ambigüedad de directorio de búsqueda
    # de \includegraphics respecto al cwd desde el que se invoque pdflatex.
    ruta_grafico_abs = os.path.abspath(ruta_grafico).replace("\\", "/")
    cuerpo = [
        _TEX_PREAMBULO,
        r"\begin{document}",
        # Título + enunciado van adentro de un \sbox (no se tipean todavía
        # en la página): así se puede medir su alto real (\ht+\dp) ANTES
        # de decidir cuánto le queda disponible al gráfico, sin depender
        # del "page builder" de TeX (que recién en la próxima línea, con
        # \usebox, los pone efectivamente en la página).
        r"\sbox{\encabezadobox}{%",
        r"\begin{minipage}[t]{\linewidth}",
        f"{{\\Large\\bfseries {titulo}}}\\par\\medskip",
        _enunciado_tex(problema),
        r"\end{minipage}%",
        r"}",
        r"\usebox{\encabezadobox}",
        "",
        # \textheight menos lo que acaba de medirse, con un colchón para
        # el espaciado entre el encabezado y la imagen (parskip + los
        # márgenes verticales de \begin{center}). width=\linewidth de tope
        # superior de ancho (por si sobra tanta altura que el límite pasa
        # a ser el ancho); \begin{center} por si el límite de alto termina
        # angostando la imagen.
        # OJO: \setlength{\x}{A-B-C} en LaTeX plano NO resta encadenado
        # -- toma solo el primer término (A) y descarta el resto sin
        # avisar. Hace falta \dimexpr...\relax (de e-TeX, disponible en
        # pdflatex) para que la resta se calcule de verdad.
        r"\setlength{\alturarestante}{\dimexpr"
        r"\textheight-\ht\encabezadobox-\dp\encabezadobox-50pt\relax}",
        r"\begin{center}",
        f"\\includegraphics[width=\\linewidth,height=\\alturarestante,"
        f"keepaspectratio]{{{ruta_grafico_abs}}}",
        r"\end{center}",
        # Fuerza que lo que sigue arranque en la página 2 -- con el gráfico
        # ya dimensionado para el sobrante real de la página 1, esto ya no
        # debería dejar una página 2 vacía antes del texto.
        r"\newpage",
        "",
        "Método de los vértices: el óptimo de un LP siempre está en un "
        "vértice de la región factible. Para encontrarlos todos no hace "
        "falta graficar --- primero se cruza cada restricción con los "
        "ejes (fácil), y luego se combinan de a pares las restricciones "
        "reales.",
        "",
        r"\begin{multicols}{2}",
        r"\section*{Paso 1 --- intersección con los ejes}",
        _paso1_tex(problema),
        r"\section*{Paso 2 --- intersección de a pares (regla de Cramer)}",
        _paso2_tex(problema),
        "",
        r"\section*{Paso 3 --- Seleccionar el óptimo}",
        f"Óptimo (pulp): $(x_1^{{*}},x_2^{{*}})=({_fmt(x1_opt)},"
        f"{_fmt(x2_opt)})$, $z^{{*}}={_fmt(z_opt)}$.",
        r"\end{multicols}",
        r"\end{document}",
    ]
    return "\n".join(cuerpo)


def compilar_pdf(tex_fuente, ruta_tex, ruta_pdf):
    """Escribe `tex_fuente` en `ruta_tex` y lo compila a `ruta_pdf` con
    pdflatex (una sola pasada — no hay referencias cruzadas ni índice).

    pdflatex (TeX Live) es una dependencia de *sistema*, no de pip —
    requirements.txt no la puede declarar. En Debian/Ubuntu:
    "sudo apt install texlive-latex-extra"."""
    if shutil.which("pdflatex") is None:
        raise RuntimeError(
            "No se encontró 'pdflatex' en el PATH. Instalar TeX Live "
            "(p. ej. 'sudo apt install texlive-latex-extra') para poder "
            "generar el PDF de álgebra."
        )

    with open(ruta_tex, "w", encoding="utf-8") as f:
        f.write(tex_fuente)

    carpeta = os.path.dirname(ruta_tex) or "."
    resultado = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={carpeta}",
            ruta_tex,
        ],
        capture_output=True,
        text=True,
    )

    base = os.path.splitext(ruta_tex)[0]
    pdf_generado = base + ".pdf"
    log = base + ".log"

    if resultado.returncode != 0 or not os.path.exists(pdf_generado):
        cola_log = ""
        if os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as f:
                cola_log = "".join(f.readlines()[-40:])
        raise RuntimeError(f"pdflatex falló compilando {ruta_tex}:\n{cola_log}")

    if os.path.abspath(pdf_generado) != os.path.abspath(ruta_pdf):
        os.replace(pdf_generado, ruta_pdf)

    for ext in (".aux", ".log", ".out"):
        aux = base + ext
        if os.path.exists(aux):
            os.remove(aux)


def construir_grafico(problema, x1_opt, x2_opt, z_opt):
    c1, c2 = problema["objetivo"]
    restricciones = problema["restricciones"] + NO_NEGATIVIDAD
    vertices = vertices_region_factible(restricciones)
    z_vertices = [c1 * x1 + c2 * x2 for x1, x2 in vertices]

    x1v = [v[0] for v in vertices] + [x1_opt]
    x2v = [v[1] for v in vertices] + [x2_opt]
    x1_max = max(x1v) * 1.3 or 1.0
    x2_max = max(x2v) * 1.3 or 1.0

    fig, ax = plt.subplots(figsize=(8, 7))

    graficar_region_factible(ax, vertices)
    graficar_restricciones(
        ax, problema["restricciones"], vertices, x1_max, x2_max
    )
    graficar_curvas_nivel(ax, c1, c2, x1_max, x2_max, z_vertices)
    graficar_vertices(ax, vertices, c1, c2)
    graficar_gradiente(ax, c1, c2, x1_max)
    graficar_optimo(ax, x1_opt, x2_opt, z_opt)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(
        f"{problema['titulo']}\n{problema['sentido']} z = {c1}x1 + {c2}x2"
    )
    ax.set_xlim(0, x1_max)
    ax.set_ylim(0, x2_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROBLEMAS:
        print("Uso: python main.py <nombre_problema>")
        print(f"Problemas disponibles: {', '.join(PROBLEMAS)}")
        return

    nombre = sys.argv[1]
    problema = PROBLEMAS[nombre]

    x1, x2, z, estado = resolver_pulp(problema)
    c1, c2 = problema["objetivo"]
    print(problema["titulo"])
    print(f"{problema['sentido']} z = {c1}x1 + {c2}x2")
    print("Estado:", estado)
    print("x1 =", x1)
    print("x2 =", x2)
    print("Z =", z)

    carpeta_resultados = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resultados"
    )
    os.makedirs(carpeta_resultados, exist_ok=True)

    fig = construir_grafico(problema, x1, x2, z)
    ruta = os.path.join(carpeta_resultados, f"{nombre}.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"Gráfico guardado en {ruta}")

    fig_solucion = construir_grafico_solucion(problema, x1, x2, z, estado)
    ruta_solucion = os.path.join(carpeta_resultados, f"{nombre}_solucion.png")
    fig_solucion.savefig(ruta_solucion, dpi=150)
    print(f"Resumen guardado en {ruta_solucion}")

    filas_sensibilidad = analizar_sensibilidad(problema)
    if filas_sensibilidad:
        print("Sensibilidad (precio sombra y rango de validez del RHS):")
        for f in filas_sensibilidad:
            rango = (
                f"[{f['b_inf']:.4f}, {f['b_sup']:.4f}]"
                if f["b_inf"] is not None and f["b_sup"] is not None
                else "no acotado"
            )
            print(
                f"  R{f['idx']}: b={f['rhs']}  precio sombra={f['pi']}  "
                f"válido para b en {rango}"
            )
        fig_sensibilidad = construir_grafico_sensibilidad(
            problema, filas_sensibilidad
        )
        ruta_sensibilidad = os.path.join(
            carpeta_resultados, f"{nombre}_sensibilidad.png"
        )
        fig_sensibilidad.savefig(
            ruta_sensibilidad, dpi=150, bbox_inches="tight"
        )
        print(f"Sensibilidad guardada en {ruta_sensibilidad}")

    tex_algebra = construir_tex_algebra(problema, x1, x2, z, ruta)
    ruta_tex = os.path.join(carpeta_resultados, f"{nombre}_algebra.tex")
    ruta_pdf = os.path.join(carpeta_resultados, f"{nombre}_algebra.pdf")
    compilar_pdf(tex_algebra, ruta_tex, ruta_pdf)
    print(f"Álgebra (LaTeX) guardada en {ruta_tex} y {ruta_pdf}")


if __name__ == "__main__":
    main()
