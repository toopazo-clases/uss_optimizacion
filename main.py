"""
Ejemplo 2.2-1 (Reddy Mikks) - Taha, Operations Research: An Introduction.

max z = 5*x1 + 4*x2
sujeto a:
    6*x1 + 4*x2 <= 24
      x1 + 2*x2 <= 6
     -x1 +   x2 <= 1
                x2 <= 2
    x1, x2 >= 0

Solución óptima esperada: x1 = 3, x2 = 1.5, z = 21.
"""

from scipy.optimize import linprog
import pulp


def resolver_scipy():
    # linprog minimiza, así que se maximiza negando la función objetivo.
    c = [-5, -4]
    A_ub = [
        [6, 4],
        [1, 2],
        [-1, 1],
        [0, 1],
    ]
    b_ub = [24, 6, 1, 2]
    bounds = [(0, None), (0, None)]

    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    x1, x2 = resultado.x
    z = -resultado.fun
    print("=== scipy.optimize.linprog ===")
    print(f"x1 = {x1:.4f}")
    print(f"x2 = {x2:.4f}")
    print(f"z  = {z:.4f}")
    print()


def resolver_pulp():
    modelo = pulp.LpProblem("Reddy_Mikks", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)

    modelo += 5 * x1 + 4 * x2, "z"

    modelo += 6 * x1 + 4 * x2 <= 24
    modelo += x1 + 2 * x2 <= 6
    modelo += -x1 + x2 <= 1
    modelo += x2 <= 2

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    print("=== pulp ===")
    print(f"Estado: {pulp.LpStatus[modelo.status]}")
    print(f"x1 = {pulp.value(x1):.4f}")
    print(f"x2 = {pulp.value(x2):.4f}")
    print(f"z  = {pulp.value(modelo.objective):.4f}")


if __name__ == "__main__":
    resolver_scipy()
    resolver_pulp()
