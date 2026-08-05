"""
Ejemplo 2.2-1 (Reddy Mikks) - Taha, Operations Research: An Introduction.

Versión simple, todo en un solo archivo: resuelve el modelo con pulp y
grafica la región factible, las curvas de nivel de z y el vector gradiente.
Pensado como introducción antes de pasar a la versión modular (problema.py,
solver_pulp.py, solver_scipy.py, visualizacion.py, main.py).

max z = 5*x1 + 4*x2
sujeto a:
    6*x1 + 4*x2 <= 24
      x1 + 2*x2 <= 6
     -x1 +   x2 <= 1
                x2 <= 2
    x1, x2 >= 0

Solución óptima: x1 = 3, x2 = 1.5, z = 21.
"""

import matplotlib.pyplot as plt
import numpy as np
import pulp

# Coeficientes de la función objetivo: z = C1*x1 + C2*x2
C1, C2 = 5, 4


# --------------------------------------------------------------------------
# Parte 1: resolver el modelo con pulp
# --------------------------------------------------------------------------

def resolver_pulp():
    """Arma y resuelve el modelo de Reddy Mikks con pulp. Devuelve (x1, x2, z)."""
    modelo = pulp.LpProblem("Reddy_Mikks", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)

    modelo += C1 * x1 + C2 * x2, "z"

    modelo += 6 * x1 + 4 * x2 <= 24
    modelo += x1 + 2 * x2 <= 6
    modelo += -x1 + x2 <= 1
    modelo += x2 <= 2

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    return pulp.value(x1), pulp.value(x2), pulp.value(modelo.objective)


# --------------------------------------------------------------------------
# Parte 2: visualización (región factible + curvas de nivel + gradiente)
# --------------------------------------------------------------------------
#
# Esto es geometría para dibujar el gráfico, no parte de la optimización:
# pulp ya encontró el óptimo en resolver_pulp(). Lo de acá abajo solo sirve
# para poder verlo.

def vertices_region_factible():
    """
    Vértices del polígono factible, en orden, hallados a mano intersectando
    pares de restricciones activas — el método gráfico clásico.
    """
    return [
        (0, 0),    # x1=0           ∩  x2=0
        (4, 0),    # 6x1+4x2=24     ∩  x2=0
        (3, 1.5),  # 6x1+4x2=24     ∩  x1+2x2=6
        (2, 2),    # x1+2x2=6       ∩  x2=2
        (1, 2),    # x2=2           ∩  -x1+x2=1
        (0, 1),    # -x1+x2=1       ∩  x1=0
    ]


def graficar_region_factible(ax, vertices):
    """Dibuja el polígono de la región factible, relleno y con su contorno."""
    x1v = [v[0] for v in vertices] + [vertices[0][0]]
    x2v = [v[1] for v in vertices] + [vertices[0][1]]
    ax.fill(x1v, x2v, color="lightblue", alpha=0.5, label="Región factible")
    ax.plot(x1v, x2v, color="steelblue", linewidth=2)


def graficar_curvas_nivel(ax, x1_max, x2_max):
    """Dibuja curvas de nivel de z = C1*x1 + C2*x2, tipo mapa topográfico."""
    x1_grid = np.linspace(0, x1_max, 400)
    x2_grid = np.linspace(0, x2_max, 400)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = C1 * X1 + C2 * X2

    niveles = np.linspace(0, C1 * x1_max + C2 * x2_max, 8)
    contornos = ax.contour(X1, X2, Z, levels=niveles, colors="gray", linewidths=0.8)
    ax.clabel(contornos, inline=True, fontsize=8, fmt="%.1f")


def graficar_vertices(ax, vertices):
    """Marca cada vértice de la región factible y le anota su valor de z."""
    for x1, x2 in vertices:
        z = C1 * x1 + C2 * x2
        ax.plot(x1, x2, "o", color="steelblue", markersize=4)
        ax.annotate(f"({x1:g},{x2:g})\nz={z:g}", (x1, x2),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)


def graficar_gradiente(ax, x1_max):
    """Dibuja el vector gradiente ∇z = (C1, C2) desde el origen: indica la
    dirección de máximo crecimiento de z."""
    grad = np.array([C1, C2], dtype=float)
    grad_dir = grad / np.linalg.norm(grad)
    longitud = 0.2 * x1_max
    punta = grad_dir * longitud
    ax.annotate("", xy=punta, xytext=(0, 0),
                arrowprops=dict(facecolor="darkred", edgecolor="darkred",
                                 width=1.5, headwidth=8))
    ax.text(punta[0], punta[1], " ∇z", color="darkred", fontsize=10)


def graficar_optimo(ax, x1, x2, z):
    """Resalta el punto óptimo encontrado por el solver."""
    ax.plot(x1, x2, "D", color="red", markersize=10,
            label=f"Óptimo ({x1:g}, {x2:g}), z={z:g}")


def construir_grafico(x1_opt, x2_opt, z_opt):
    """Arma la figura completa combinando las funciones de graficado."""
    vertices = vertices_region_factible()
    x1_max = max(v[0] for v in vertices) * 1.3
    x2_max = max(v[1] for v in vertices) * 1.3

    fig, ax = plt.subplots(figsize=(8, 7))

    graficar_region_factible(ax, vertices)
    graficar_curvas_nivel(ax, x1_max, x2_max)
    graficar_vertices(ax, vertices)
    graficar_gradiente(ax, x1_max)
    graficar_optimo(ax, x1_opt, x2_opt, z_opt)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Ejemplo 2.2-1 (Reddy Mikks) — z = {C1}x1 + {C2}x2")
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
    x1, x2, z = resolver_pulp()
    print("Ejemplo 2.2-1 (Reddy Mikks)")
    print(f"max z = {C1}x1 + {C2}x2")
    print(f"x1 = {x1:.4f}  x2 = {x2:.4f}  z = {z:.4f}")

    fig = construir_grafico(x1, x2, z)
    fig.savefig("reddy_mikks_simple.png", dpi=150)
    print("Gráfico guardado en reddy_mikks_simple.png")
    plt.show()


if __name__ == "__main__":
    main()
