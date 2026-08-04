"""Resolución del modelo (problema.py) con pulp."""

import pulp

import problema


def resolver():
    modelo = pulp.LpProblem("Reddy_Mikks", pulp.LpMaximize)

    # Variables libres: la no-negatividad ya está en problema.CONSTRAINTS.
    x1 = pulp.LpVariable("x1", lowBound=None)
    x2 = pulp.LpVariable("x2", lowBound=None)

    modelo += problema.C1 * x1 + problema.C2 * x2, "z"

    for a, b, rhs in problema.CONSTRAINTS:
        modelo += a * x1 + b * x2 <= rhs

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    estado = pulp.LpStatus[modelo.status]
    return pulp.value(x1), pulp.value(x2), pulp.value(modelo.objective), estado


def main():
    x1, x2, z, estado = resolver()
    print("--- pulp ---")
    print(f"Estado: {estado}")
    print(f"x1 = {x1:.4f}  x2 = {x2:.4f}  z = {z:.4f}")


if __name__ == "__main__":
    main()
