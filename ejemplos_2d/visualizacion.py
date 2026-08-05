"""
Visualización 2D del modelo definido en problema.py: curvas de nivel de la
función objetivo (como un mapa topográfico) sobre la región factible, con
el vector gradiente (dirección de máximo crecimiento de z) y el óptimo
resaltados.
"""

import matplotlib.pyplot as plt
import numpy as np

import problema


def construir_figura():
    vertices = problema.vertices_region_factible()
    x1v = [v[0] for v in vertices]
    x2v = [v[1] for v in vertices]
    z_vertices = [problema.objetivo(x1, x2) for x1, x2 in vertices]

    idx_opt = int(np.argmax(z_vertices))
    x1_opt, x2_opt = vertices[idx_opt]
    z_opt = z_vertices[idx_opt]

    margen = 0.3
    x1_max = max(x1v) * (1 + margen)
    x2_max = max(x2v) * (1 + margen)

    x1_grid = np.linspace(0, x1_max, 400)
    x2_grid = np.linspace(0, x2_max, 400)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = problema.objetivo(X1, X2)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Región factible sombreada.
    ax.fill(x1v + [x1v[0]], x2v + [x2v[0]], color="lightblue", alpha=0.5,
             label="Región factible")
    ax.plot(x1v + [x1v[0]], x2v + [x2v[0]], color="steelblue", linewidth=2)

    # Curvas de nivel de z, tipo mapa topográfico (se extienden más allá de
    # la región factible para dar contexto, como en un mapa real).
    niveles = np.linspace(min(z_vertices), max(z_vertices), 8)
    contornos = ax.contour(X1, X2, Z, levels=niveles, colors="gray",
                             linewidths=0.8)
    ax.clabel(contornos, inline=True, fontsize=8, fmt="%.1f")

    # Vértices del polígono, etiquetados con su z.
    for (x1, x2), z in zip(vertices, z_vertices):
        ax.plot(x1, x2, "o", color="steelblue", markersize=4)
        ax.annotate(f"({x1:g},{x2:g})\nz={z:g}", (x1, x2),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Óptimo resaltado.
    ax.plot(x1_opt, x2_opt, "D", color="red", markersize=10,
             label=f"Óptimo ({x1_opt:g}, {x2_opt:g}), z={z_opt:g}")

    # Vector gradiente ∇z = (C1, C2): dirección de máximo crecimiento,
    # dibujado desde el origen.
    origen = (0.0, 0.0)
    grad = np.array([problema.C1, problema.C2], dtype=float)
    grad_dir = grad / np.linalg.norm(grad)
    longitud = 0.2 * x1_max
    punta = (origen[0] + grad_dir[0] * longitud, origen[1] + grad_dir[1] * longitud)
    ax.annotate(
        "", xy=punta, xytext=origen,
        arrowprops=dict(facecolor="darkred", edgecolor="darkred", width=1.5,
                         headwidth=8),
    )
    ax.text(punta[0], punta[1], " ∇z", color="darkred", fontsize=10)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Ejemplo 2.2-1 (Reddy Mikks) — z = {problema.C1}x1 + {problema.C2}x2")
    ax.set_xlim(0, x1_max)
    ax.set_ylim(0, x2_max)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def guardar_png(fig, ruta="reddy_mikks_curvas_nivel.png"):
    fig.savefig(ruta, dpi=150)
    return ruta


def main():
    fig = construir_figura()
    ruta = guardar_png(fig)
    print(f"Gráfico guardado en {ruta}")
    plt.show()


if __name__ == "__main__":
    main()
