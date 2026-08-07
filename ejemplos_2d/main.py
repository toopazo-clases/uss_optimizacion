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
import sys

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
#   restricciones -> lista de (a, b, rhs) para a*x1 + b*x2 <= rhs
#                    (incluye x1>=0, x2>=0 como -x1<=0, -x2<=0)
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
            (-1, 0, 0),
            (0, -1, 0),
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
            (-1, 0, 0),
            (0, -1, 0),
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
            (-1, 0, 0),
            (0, -1, 0),
        ],
    },
    "ejercicio_c": {
        "titulo": "Ejercicio C (práctica)",
        "sentido": "max",
        "objetivo": (1, 2),
        "restricciones": [
            (1, 3, 15),
            (4, 1, 16),
            (-1, 0, 0),
            (0, -1, 0),
        ],
    },
    "ejercicio_d": {
        "titulo": "Ejercicio D (práctica, triángulo)",
        "sentido": "max",
        "objetivo": (2, 3),
        "restricciones": [
            (1, 1, 6),
            (-1, 0, 0),
            (0, -1, 0),
        ],
    },
}


# --------------------------------------------------------------------------
# Parte 1: resolver con pulp
# --------------------------------------------------------------------------

def resolver_pulp(problema):
    """Resuelve el LP con pulp. Devuelve (x1, x2, z)."""
    c1, c2 = problema["objetivo"]
    sentido = pulp.LpMaximize if problema["sentido"] == "max" else pulp.LpMinimize

    modelo = pulp.LpProblem("modelo", sentido)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)

    modelo += c1 * x1 + c2 * x2, "z"
    for a, b, rhs in problema["restricciones"]:
        modelo += a * x1 + b * x2 <= rhs

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    return pulp.value(x1), pulp.value(x2), pulp.value(modelo.objective)


# --------------------------------------------------------------------------
# Parte 2: visualización
# --------------------------------------------------------------------------
# pulp ya entregó el óptimo. Lo que sigue es geometría para poder dibujar la
# región factible completa (pulp no la entrega, solo el vértice óptimo).

TOL = 1e-9
SLACK = 1e-6


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


def graficar_curvas_nivel(ax, c1, c2, x1_max, x2_max, z_vertices):
    x1_grid = np.linspace(0, x1_max, 400)
    x2_grid = np.linspace(0, x2_max, 400)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = c1 * X1 + c2 * X2

    niveles = np.linspace(min(z_vertices), max(z_vertices), 8)
    contornos = ax.contour(X1, X2, Z, levels=niveles, colors="gray", linewidths=0.8)
    ax.clabel(contornos, inline=True, fontsize=8, fmt="%.1f")


def graficar_vertices(ax, vertices, c1, c2):
    for x1, x2 in vertices:
        z = c1 * x1 + c2 * x2
        ax.plot(x1, x2, "o", color="steelblue", markersize=4)
        ax.annotate(f"({x1:g},{x2:g})\nz={z:g}", (x1, x2),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)


def graficar_gradiente(ax, c1, c2, x1_max):
    """Vector gradiente ∇z = (c1, c2) desde el origen: dirección de máximo
    crecimiento de z."""
    grad = np.array([c1, c2], dtype=float)
    grad_dir = grad / np.linalg.norm(grad)
    longitud = 0.2 * x1_max
    punta = grad_dir * longitud
    ax.annotate("", xy=punta, xytext=(0, 0),
                arrowprops=dict(facecolor="darkred", edgecolor="darkred",
                                 width=1.5, headwidth=8))
    ax.text(punta[0], punta[1], " ∇z", color="darkred", fontsize=10)


def graficar_optimo(ax, x1, x2, z):
    """Óptimo: viene de pulp, no del cálculo geométrico."""
    ax.plot(x1, x2, "D", color="red", markersize=10,
            label=f"Óptimo pulp ({x1:g}, {x2:g}), z={z:g}")


def construir_grafico(problema, x1_opt, x2_opt, z_opt):
    c1, c2 = problema["objetivo"]
    vertices = vertices_region_factible(problema["restricciones"])
    z_vertices = [c1 * x1 + c2 * x2 for x1, x2 in vertices]

    x1v = [v[0] for v in vertices] + [x1_opt]
    x2v = [v[1] for v in vertices] + [x2_opt]
    x1_max = max(x1v) * 1.3 or 1.0
    x2_max = max(x2v) * 1.3 or 1.0

    fig, ax = plt.subplots(figsize=(8, 7))

    graficar_region_factible(ax, vertices)
    graficar_curvas_nivel(ax, c1, c2, x1_max, x2_max, z_vertices)
    graficar_vertices(ax, vertices, c1, c2)
    graficar_gradiente(ax, c1, c2, x1_max)
    graficar_optimo(ax, x1_opt, x2_opt, z_opt)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{problema['titulo']}\n{problema['sentido']} z = {c1}x1 + {c2}x2")
    ax.set_xlim(0, x1_max)
    ax.set_ylim(0, x2_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
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

    x1, x2, z = resolver_pulp(problema)
    c1, c2 = problema["objetivo"]
    print(problema["titulo"])
    print(f"{problema['sentido']} z = {c1}x1 + {c2}x2")
    print(f"x1 = {x1:.4f}  x2 = {x2:.4f}  z = {z:.4f}")

    fig = construir_grafico(problema, x1, x2, z)
    ruta = f"{nombre}.png"
    fig.savefig(ruta, dpi=150)
    print(f"Gráfico guardado en {ruta}")


if __name__ == "__main__":
    main()
