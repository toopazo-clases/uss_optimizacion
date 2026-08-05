"""
Banco de problemas de programación lineal en 2 variables, para practicar y
verificar visualmente. Cada problema se resuelve por el método gráfico
(vértices de la región factible, evaluando z en cada uno) y se dibuja con
su región factible y curvas de nivel.

Uso:
    python banco_problemas.py
        -> usa la selección por defecto (ver SELECCION más abajo)
    python banco_problemas.py reddy_mikks ejercicio_a ejercicio_b ejercicio_d
        -> elige explícitamente qué 4 problemas graficar

Genera banco_problemas.png con una grilla 2x2 (una celda por problema) y
además imprime por consola la solución óptima de cada uno, para poder
comparar el número contra lo que se ve en el gráfico.
"""

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

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

# Qué problemas se incluyen en la grilla 2x2 al correr este archivo sin
# argumentos. Cambia esta lista (máximo 4 nombres) para elegir otra
# combinación, o pásalos por línea de comandos.
SELECCION = ["reddy_mikks", "ejercicio_a", "ejercicio_b", "ejercicio_c"]


# --------------------------------------------------------------------------
# Geometría genérica: vértices de la región factible + evaluación de z
# --------------------------------------------------------------------------

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
    """Vértices del polígono factible, ordenados en sentido antihorario."""
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


def resolver_geometricamente(problema):
    """Encuentra el óptimo evaluando z en cada vértice (método gráfico)."""
    c1, c2 = problema["objetivo"]
    vertices = vertices_region_factible(problema["restricciones"])
    z_vertices = [c1 * x1 + c2 * x2 for x1, x2 in vertices]

    if problema["sentido"] == "max":
        idx_opt = int(np.argmax(z_vertices))
    else:
        idx_opt = int(np.argmin(z_vertices))

    return vertices, z_vertices, idx_opt


# --------------------------------------------------------------------------
# Graficado de un problema en un eje (Axes) dado
# --------------------------------------------------------------------------

def graficar_problema(ax, problema, vertices, z_vertices, idx_opt):
    """Dibuja región factible + curvas de nivel + óptimo de un problema."""
    c1, c2 = problema["objetivo"]
    x1_opt, x2_opt = vertices[idx_opt]
    z_opt = z_vertices[idx_opt]

    x1v = [v[0] for v in vertices]
    x2v = [v[1] for v in vertices]
    x1_max = max(x1v) * 1.3 or 1.0
    x2_max = max(x2v) * 1.3 or 1.0

    # Región factible.
    ax.fill(x1v + [x1v[0]], x2v + [x2v[0]], color="lightblue", alpha=0.5)
    ax.plot(x1v + [x1v[0]], x2v + [x2v[0]], color="steelblue", linewidth=2)

    # Curvas de nivel.
    x1_grid = np.linspace(0, x1_max, 300)
    x2_grid = np.linspace(0, x2_max, 300)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = c1 * X1 + c2 * X2
    niveles = np.linspace(min(z_vertices), max(z_vertices), 6)
    contornos = ax.contour(X1, X2, Z, levels=niveles, colors="gray", linewidths=0.7)
    ax.clabel(contornos, inline=True, fontsize=6, fmt="%.1f")

    # Vértices y óptimo.
    for x1, x2 in vertices:
        ax.plot(x1, x2, "o", color="steelblue", markersize=3)
    ax.plot(x1_opt, x2_opt, "D", color="red", markersize=8)

    signo = "max" if problema["sentido"] == "max" else "min"
    ax.set_title(
        f"{problema['titulo']}\n"
        f"{signo} z={c1}x1+{c2}x2  →  ({x1_opt:g}, {x2_opt:g}), z={z_opt:g}",
        fontsize=9,
    )
    ax.set_xlabel("x1", fontsize=8)
    ax.set_ylabel("x2", fontsize=8)
    ax.set_xlim(0, x1_max)
    ax.set_ylim(0, x2_max)
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def obtener_seleccion():
    """Lee la selección de problemas desde los argumentos de línea de
    comandos si se pasaron; si no, usa SELECCION."""
    argv = sys.argv[1:]
    if not argv:
        return SELECCION

    desconocidos = [n for n in argv if n not in PROBLEMAS]
    if desconocidos:
        disponibles = ", ".join(PROBLEMAS)
        raise SystemExit(
            f"Problema(s) desconocido(s): {', '.join(desconocidos)}.\n"
            f"Disponibles: {disponibles}"
        )
    if len(argv) > 4:
        print(f"Aviso: se pasaron {len(argv)} problemas, solo se grafican los primeros 4.")
    return argv[:4]


def main():
    seleccion = obtener_seleccion()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    print(f"Problemas seleccionados: {', '.join(seleccion)}\n")

    for nombre, ax in zip(seleccion, axes.flat):
        problema = PROBLEMAS[nombre]
        vertices, z_vertices, idx_opt = resolver_geometricamente(problema)
        x1_opt, x2_opt = vertices[idx_opt]
        z_opt = z_vertices[idx_opt]

        graficar_problema(ax, problema, vertices, z_vertices, idx_opt)

        c1, c2 = problema["objetivo"]
        print(
            f"{nombre}: {problema['sentido']} z = {c1}x1 + {c2}x2  →  "
            f"x1={x1_opt:g}  x2={x2_opt:g}  z={z_opt:g}"
        )

    # Si se seleccionaron menos de 4 problemas, apaga las celdas sobrantes.
    for ax in axes.flat[len(seleccion):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("banco_problemas.png", dpi=150)
    print("\nGráfico guardado en banco_problemas.png")


if __name__ == "__main__":
    main()
