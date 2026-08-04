"""
Visualización 3D del ejemplo 2.2-1 (Reddy Mikks) - Taha, Operations Research.

Muestra el plano de la función objetivo z = 5*x1 + 4*x2 recortado a la forma
de la región factible (un "techo" apoyado sobre el polígono de restricciones),
con los vértices etiquetados y el óptimo resaltado. Genera un archivo HTML
interactivo (rotar / zoom / hover).
"""

import itertools

import numpy as np
import plotly.graph_objects as go

# Restricciones en la forma a*x1 + b*x2 <= rhs (incluye x1>=0 y x2>=0).
CONSTRAINTS = [
    (6, 4, 24),   # 6x1 + 4x2 <= 24
    (1, 2, 6),    # x1 + 2x2 <= 6
    (-1, 1, 1),   # -x1 + x2 <= 1
    (0, 1, 2),    # x2 <= 2
    (-1, 0, 0),   # x1 >= 0
    (0, -1, 0),   # x2 >= 0
]

C1, C2 = 5, 4  # z = C1*x1 + C2*x2

TOL = 1e-9
SLACK = 1e-6


def objetivo(x1, x2):
    return C1 * x1 + C2 * x2


def interseccion(c_i, c_j):
    a1, b1, r1 = c_i
    a2, b2, r2 = c_j
    det = a1 * b2 - a2 * b1
    if abs(det) < TOL:
        return None
    x1 = (r1 * b2 - r2 * b1) / det
    x2 = (a1 * r2 - a2 * r1) / det
    return x1, x2


def es_factible(x1, x2):
    return all(a * x1 + b * x2 <= rhs + SLACK for a, b, rhs in CONSTRAINTS)


def vertices_region_factible():
    """Vértices del polígono factible, ordenados en sentido antihorario."""
    puntos = []
    for c_i, c_j in itertools.combinations(CONSTRAINTS, 2):
        p = interseccion(c_i, c_j)
        if p is None:
            continue
        x1, x2 = p
        if es_factible(x1, x2):
            puntos.append((round(x1, 6), round(x2, 6)))

    puntos = list(set(puntos))

    cx = sum(p[0] for p in puntos) / len(puntos)
    cy = sum(p[1] for p in puntos) / len(puntos)
    puntos.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return puntos


def construir_figura():
    vertices = vertices_region_factible()
    z_vertices = [objetivo(x1, x2) for x1, x2 in vertices]

    x1s = [v[0] for v in vertices]
    x2s = [v[1] for v in vertices]

    idx_opt = int(np.argmax(z_vertices))
    x1_opt, x2_opt = vertices[idx_opt]
    z_opt = z_vertices[idx_opt]

    fig = go.Figure()

    # Base: contorno de la región factible en z=0.
    fig.add_trace(go.Scatter3d(
        x=x1s + [x1s[0]], y=x2s + [x2s[0]], z=[0] * (len(vertices) + 1),
        mode="lines", line=dict(color="gray", width=4),
        name="Región factible (base)",
    ))

    # Techo: plano del objetivo recortado a la región factible (triangulación
    # en abanico, válida porque el polígono es convexo).
    n = len(vertices)
    fig.add_trace(go.Mesh3d(
        x=x1s, y=x2s, z=z_vertices,
        i=[0] * (n - 2), j=list(range(1, n - 1)), k=list(range(2, n)),
        color="lightblue", opacity=0.6, showscale=False,
        name="z = 5x1 + 4x2",
    ))

    # Contorno del techo.
    fig.add_trace(go.Scatter3d(
        x=x1s + [x1s[0]], y=x2s + [x2s[0]], z=z_vertices + [z_vertices[0]],
        mode="lines", line=dict(color="steelblue", width=4),
        showlegend=False,
    ))

    # Postes verticales desde cada vértice de la base hasta el techo.
    for (x1, x2), z in zip(vertices, z_vertices):
        fig.add_trace(go.Scatter3d(
            x=[x1, x1], y=[x2, x2], z=[0, z],
            mode="lines", line=dict(color="lightgray", width=2, dash="dot"),
            showlegend=False,
        ))

    # Vértices etiquetados con su valor de z.
    fig.add_trace(go.Scatter3d(
        x=x1s, y=x2s, z=z_vertices,
        mode="markers+text",
        marker=dict(size=4, color="steelblue"),
        text=[f"({x1:g},{x2:g})  z={z:g}" for (x1, x2), z in zip(vertices, z_vertices)],
        textposition="top center",
        name="Vértices",
    ))

    # Óptimo resaltado.
    fig.add_trace(go.Scatter3d(
        x=[x1_opt], y=[x2_opt], z=[z_opt],
        mode="markers",
        marker=dict(size=8, color="red", symbol="diamond"),
        name=f"Óptimo ({x1_opt:g}, {x2_opt:g})  z={z_opt:g}",
    ))

    fig.update_layout(
        title="Ejemplo 2.2-1 (Reddy Mikks) — max z = 5x1 + 4x2",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="z"),
        legend=dict(itemsizing="constant"),
    )
    return fig


def main():
    fig = construir_figura()
    salida = "reddy_mikks_3d.html"
    fig.write_html(salida)
    print(f"Gráfico guardado en {salida}")


if __name__ == "__main__":
    main()
