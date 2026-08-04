"""
Modelo del ejemplo 2.2-1 (Reddy Mikks) - Taha, Operations Research: An Introduction.

max z = 5*x1 + 4*x2
sujeto a:
    6*x1 + 4*x2 <= 24
      x1 + 2*x2 <= 6
     -x1 +   x2 <= 1
                x2 <= 2
    x1, x2 >= 0

Solución óptima esperada: x1 = 3, x2 = 1.5, z = 21.

Este módulo es la única fuente de verdad del modelo: lo usan tanto los
solvers (scipy, pulp) como la visualización 3D, para no duplicar los datos
del problema en cada archivo.
"""

import itertools

import numpy as np

# Coeficientes de la función objetivo: z = C1*x1 + C2*x2
C1, C2 = 5, 4

# Restricciones en la forma a*x1 + b*x2 <= rhs (incluye x1>=0 y x2>=0
# expresadas como -x1<=0 y -x2<=0, para que sea el único lugar donde se
# define el modelo completo).
CONSTRAINTS = [
    (6, 4, 24),   # materia prima M1
    (1, 2, 6),    # materia prima M2
    (-1, 1, 1),   # límite de mercado
    (0, 1, 2),    # demanda máxima de pintura interior
    (-1, 0, 0),   # x1 >= 0
    (0, -1, 0),   # x2 >= 0
]

TOL = 1e-9
SLACK = 1e-6


def objetivo(x1, x2):
    return C1 * x1 + C2 * x2


def _interseccion(c_i, c_j):
    a1, b1, r1 = c_i
    a2, b2, r2 = c_j
    det = a1 * b2 - a2 * b1
    if abs(det) < TOL:
        return None
    x1 = (r1 * b2 - r2 * b1) / det
    x2 = (a1 * r2 - a2 * r1) / det
    return x1, x2


def _es_factible(x1, x2):
    return all(a * x1 + b * x2 <= rhs + SLACK for a, b, rhs in CONSTRAINTS)


def vertices_region_factible():
    """Vértices del polígono factible, ordenados en sentido antihorario."""
    puntos = []
    for c_i, c_j in itertools.combinations(CONSTRAINTS, 2):
        p = _interseccion(c_i, c_j)
        if p is None:
            continue
        x1, x2 = p
        if _es_factible(x1, x2):
            puntos.append((round(x1, 6), round(x2, 6)))

    puntos = list(set(puntos))

    cx = sum(p[0] for p in puntos) / len(puntos)
    cy = sum(p[1] for p in puntos) / len(puntos)
    puntos.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return puntos
