"""
Banco de problemas de programación lineal en 3 variables: se resuelven con
pulp y se reporta el óptimo, el precio sombra (y su rango de validez del
RHS) y el costo reducido de cada variable.

Uso:
    python main.py <nombre_problema>

A diferencia de pulp_LpProblem_2d (pensado para 2 variables, con
graficación de la región factible incluida), este banco no grafica nada:
con 3 variables la región factible es un poliedro en 3D, y una imagen
estática de eso suele ser más confusa que útil para el curso. Este script
solo resuelve y reporta números -- solución, duales, costo reducido y
rango de validez -- todo con pulp/HiGHS, en un único solve por problema
(ver pulp_LpProblem_2d/main.py para el porqué de usar HiGHS y no CBC para
la sensibilidad).
"""

import sys

import highspy
import pulp

# --------------------------------------------------------------------------
# Banco de problemas
# --------------------------------------------------------------------------
# Cada problema es un diccionario con:
#   titulo        -> nombre para los reportes
#   sentido       -> "max" o "min"
#   objetivo      -> (c1, c2, c3) tal que z = c1*x1 + c2*x2 + c3*x3
#   restricciones -> lista de (a, b, c, rhs) para a*x1 + b*x2 + c*x3 <= rhs.
#                    No incluir x1>=0, x2>=0, x3>=0: pulp ya las aplica vía
#                    lowBound=0.

PROBLEMAS = {
    "florista_ambulante_3d": {
        # Extiende florista_ambulante (pulp_LpProblem_2d) agregando un
        # tercer producto x3. x3 se eligió para que su restricción dual se
        # cumpla con desigualdad estricta usando las MISMAS duales del
        # problema de 2 variables (y1*=500, y2*=500, y3*=0): por holgura
        # complementaria eso garantiza que el óptimo no cambia (x1*, x2*,
        # z* y las duales quedan iguales) -- y aparece, de forma natural,
        # el primer costo reducido > 0 del curso para una variable de
        # decisión (x3, no básica). Verificado abajo, no solo supuesto.
        "titulo": "Florista ambulante (3 variables)",
        "sentido": "max",
        "objetivo": (2000, 1000, 1800),
        "restricciones": [
            (3, 1, 2, 300),
            (1, 1, 2, 140),
            (1, 3, 2, 300),
        ],
    },
}


def resolver_pulp(problema):
    """Resuelve el LP con pulp/HiGHS. Devuelve un dict con el estado, el
    óptimo, el precio sombra + rango de validez de cada restricción, y el
    costo reducido de cada variable -- o None si no es óptimo."""
    coeficientes = problema["objetivo"]
    n = len(coeficientes)
    nombres = [f"x{i + 1}" for i in range(n)]
    sentido = (
        pulp.LpMaximize if problema["sentido"] == "max" else pulp.LpMinimize
    )

    modelo = pulp.LpProblem("modelo", sentido)
    variables = [pulp.LpVariable(nombre, lowBound=0) for nombre in nombres]
    modelo += pulp.lpSum(c * v for c, v in zip(coeficientes, variables)), "z"

    restricciones_pulp = []
    for fila in problema["restricciones"]:
        *coefs, rhs = fila
        restriccion = pulp.lpSum(a * v for a, v in zip(coefs, variables)) <= rhs
        modelo += restriccion
        restricciones_pulp.append(restriccion)

    modelo.solve(pulp.HiGHS(msg=False))
    estado = pulp.LpStatus[modelo.status]
    if estado != "Optimal":
        return {"estado": estado}

    # Mismo truco que en pulp_LpProblem_2d: HiGHS trae ranging de
    # sensibilidad incorporado (Highs.getRanging()), no hace falta
    # resolver el LP en bucle. Ojo con el signo: restriccion.pi de HiGHS
    # sale con signo contrario al de CBC -- se normaliza con abs().
    estado_ranging, ranging = modelo.solverModel.getRanging()
    if estado_ranging != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS getRanging() falló: {estado_ranging}")

    duales = [abs(r.pi) for r in restricciones_pulp]

    sensibilidad = []
    for idx, fila in enumerate(problema["restricciones"]):
        *_, rhs = fila
        sensibilidad.append(
            {
                "idx": idx + 1,
                "rhs": rhs,
                "pi": duales[idx],
                "b_inf": ranging.row_bound_dn.value_[idx],
                "b_sup": ranging.row_bound_up.value_[idx],
            }
        )

    # Costo reducido de cada variable j, con la convención del curso:
    #   costo reducido = costo de los recursos consumidos por unidad
    #                     - ingreso por unidad
    #                   = sum_i (a_ij * y_i*)  -  c_j
    # Para una variable básica (>0 en el óptimo) esto da 0; para una no
    # básica, da el excedente/pérdida neta de forzar su producción.
    costos_reducidos = []
    for j, (nombre, cj) in enumerate(zip(nombres, coeficientes)):
        costo_recursos = sum(
            problema["restricciones"][i][j] * duales[i]
            for i in range(len(problema["restricciones"]))
        )
        costos_reducidos.append(
            {
                "variable": nombre,
                "valor": variables[j].value(),
                "costo_recursos": costo_recursos,
                "cj": cj,
                "costo_reducido": costo_recursos - cj,
            }
        )

    return {
        "estado": estado,
        "z": pulp.value(modelo.objective),
        "valores": {n: v.value() for n, v in zip(nombres, variables)},
        "sensibilidad": sensibilidad,
        "costos_reducidos": costos_reducidos,
    }


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROBLEMAS:
        print("Uso: python main.py <nombre_problema>")
        print(f"Problemas disponibles: {', '.join(PROBLEMAS)}")
        return

    nombre = sys.argv[1]
    problema = PROBLEMAS[nombre]
    resultado = resolver_pulp(problema)

    print(problema["titulo"])
    terminos = " + ".join(
        f"{c}x{i + 1}" for i, c in enumerate(problema["objetivo"])
    )
    print(f"{problema['sentido']} z = {terminos}")
    print("Estado:", resultado["estado"])
    if resultado["estado"] != "Optimal":
        return

    for var, valor in resultado["valores"].items():
        print(f"{var} = {valor}")
    print("Z =", resultado["z"])

    print("\nSensibilidad (precio sombra y rango de validez del RHS):")
    for f in resultado["sensibilidad"]:
        rango = f"[{f['b_inf']:.4f}, {f['b_sup']:.4f}]"
        print(
            f"  R{f['idx']}: b={f['rhs']}  precio sombra={f['pi']}  "
            f"válido para b en {rango}"
        )

    print("\nCosto reducido de cada variable:")
    for c in resultado["costos_reducidos"]:
        print(
            f"  {c['variable']} (valor={c['valor']}): "
            f"costo_recursos={c['costo_recursos']}  c_j={c['cj']}  "
            f"costo_reducido={c['costo_reducido']}"
        )


if __name__ == "__main__":
    main()
