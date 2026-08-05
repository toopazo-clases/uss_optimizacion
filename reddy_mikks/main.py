"""
Ejemplo 2.2-1 (Reddy Mikks) - Taha, Operations Research: An Introduction.

Punto de entrada único del proyecto: resuelve el modelo con scipy y con
pulp, y luego abre la visualización 2D (curvas de nivel + gradiente).

Estructura del proyecto:
    problema.py       -> definición del modelo (única fuente de verdad)
    solver_scipy.py    -> resolución con scipy.optimize.linprog
    solver_pulp.py      -> resolución con pulp
    visualizacion.py     -> curvas de nivel de z + vector gradiente (matplotlib)
    main.py               -> orquesta todo lo anterior
"""

import matplotlib.pyplot as plt

import problema
import solver_pulp
import solver_scipy
import visualizacion


def main():
    print("Ejemplo 2.2-1 (Reddy Mikks)")
    print(f"max z = {problema.C1}x1 + {problema.C2}x2")
    print()

    x1, x2, z = solver_scipy.resolver()
    print("--- scipy.optimize.linprog ---")
    print(f"x1 = {x1:.4f}  x2 = {x2:.4f}  z = {z:.4f}")
    print()

    x1, x2, z, estado = solver_pulp.resolver()
    print("--- pulp ---")
    print(f"Estado: {estado}")
    print(f"x1 = {x1:.4f}  x2 = {x2:.4f}  z = {z:.4f}")
    print()

    print("--- Visualización (curvas de nivel + gradiente) ---")
    fig = visualizacion.construir_figura()
    ruta = visualizacion.guardar_png(fig)
    print(f"Gráfico guardado en {ruta}")
    plt.show()


if __name__ == "__main__":
    main()
