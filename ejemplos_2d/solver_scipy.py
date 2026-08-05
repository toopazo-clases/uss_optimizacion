"""Resolución del modelo (problema.py) con scipy.optimize.linprog."""

from scipy.optimize import linprog

import problema


def resolver():
    # linprog minimiza, así que se maximiza negando la función objetivo.
    # La no-negatividad de x1, x2 ya está incluida en problema.CONSTRAINTS,
    # por eso las cotas (bounds) quedan libres.
    c = [-problema.C1, -problema.C2]
    A_ub = [[a, b] for a, b, _ in problema.CONSTRAINTS]
    b_ub = [rhs for _, _, rhs in problema.CONSTRAINTS]

    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method="highs")

    x1, x2 = resultado.x
    z = -resultado.fun
    return x1, x2, z


def main():
    x1, x2, z = resolver()
    print("--- scipy.optimize.linprog ---")
    print(f"x1 = {x1:.4f}  x2 = {x2:.4f}  z = {z:.4f}")


if __name__ == "__main__":
    main()
