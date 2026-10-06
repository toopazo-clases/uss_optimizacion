"""
Banco de problemas de programación lineal en 4 variables: se resuelven con
pulp y se reporta el óptimo, el precio sombra (y su rango de validez del
RHS) y el costo reducido de cada variable.

Uso:
    python main.py <nombre_problema>

Igual que pulp_LpProblem_3d (código genérico, sirve para cualquier n), pero
acá el banco trae problemas de 4 variables / 4-5 restricciones. Soporta
restricciones mixtas (<= y >=, ver "sentido" más abajo) -- necesario para
control_2_a/control_2_b, que agregan una restricción de venta cruzada
(x3 >= 0.2*(x3+x4)) y una de producción mínima (suma de todas >= N), ambas
de tipo >=, además de los recursos habituales en <=. No se grafica nada
(con 4 variables la región factible es un politopo en 4D, ver
pulp_LpProblem_3d para el mismo argumento en 3D).
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
#   objetivo      -> (c1, c2, c3, c4) tal que z = c1*x1 + c2*x2 + c3*x3 + c4*x4
#   restricciones -> lista de (a, b, c, d, sentido, rhs) para
#                    a*x1 + b*x2 + c*x3 + d*x4 {<=,>=} rhs, con "sentido"
#                    literal "<=" o ">=" (no se soporta "==" por ahora,
#                    ninguno de los problemas del banco lo necesita).
#                    No incluir x1>=0, ..., x4>=0: pulp ya las aplica vía
#                    lowBound=0 (ver resolver_pulp).

PROBLEMAS = {
    "guia_3": {
        # Fábrica de muebles con 4 productos (Sillas x1, Mesas x2,
        # Escritorios x3, Repisas x4) y 4 recursos: 2 máquinas (Corte,
        # Ensamblaje), 1 materia prima (Madera) y 1 de personal (HH).
        # Los coeficientes de x1, x2 en Corte/Madera/Ensamblaje y las
        # utilidades c1, c2, c3 son los mismos que florista_ambulante_3d
        # (pulp_LpProblem_3d) -- por eso el óptimo de x1, x2, x3 y las
        # duales de Corte/Madera/Ensamblaje salen idénticos. Se agregaron
        # Personal (recurso nuevo, ocioso) y x4 (producto nuevo, peor que
        # x3: mismo margen, $1800, pero consume más materia prima por
        # unidad) para que el óptimo tenga, a la vez, 2 restricciones
        # activas (Corte, Madera) y 2 inactivas (Ensamblaje, Personal), y 2
        # variables básicas (x1, x2) y 2 no básicas (x3, x4) -- pensado
        # para ilustrar holgura complementaria con ambos casos presentes.
        # Verificado abajo, no solo supuesto. Utilidades en pesos chilenos
        # (CLP) -- un cero más que el ejemplo de la florista (2000 ->
        # 20000, etc.) para que las cifras se vean como precios reales.
        # Usado en Guias/Guia_3_Analisis_Sensibilidad -- de ahí el nombre
        # (antes "control_1", renombrado porque este SÍ es el problema que
        # se resuelve en esa Guía, no un control/evaluación).
        "titulo": "Guía 3: fábrica de muebles (4 variables)",
        "sentido": "max",
        "objetivo": (20000, 10000, 18000, 18000),
        "restricciones": [
            (3, 1, 2, 2, "<=", 300),  # Máquina de Corte (horas)
            (1, 1, 2, 3, "<=", 140),  # Materia prima: Madera (kg)
            (1, 3, 2, 2, "<=", 300),  # Máquina de Ensamblaje (horas)
            (2, 2, 3, 3, "<=", 400),  # Personal (horas-hombre)
        ],
    },
    "control_2_a": {
        # EP01 (Evaluaciones/EP01), Forma A. Taller de bicicletas "Andes"
        # con 4 modelos: Urbana x1, Montaña x2, Ruta x3, Eléctrica x4.
        # SOLO 3 restricciones (<=): Máquina, Materiales, Personal -- se
        # quitaron Venta cruzada y Producción mínima (ver conversación
        # donde se pidió; esas 2 quedan solo en control_2_c, reservado
        # para el próximo semestre). Vértice elegido a mano (x1=30,x2=40,
        # x3=0,x4=0) y coeficientes de objetivo derivados para que las
        # duales den exactamente lo buscado, luego verificado con pulp --
        # no al revés. Con solo 3 restricciones y 4 variables, un vértice
        # no degenerado necesita 2 variables en 0 (no solo x4) si se
        # quiere dejar 1 restricción con holgura -- Ruta (x3) también
        # queda no básica (costo reducido > 0), no solo Eléctrica.
        # A pedido explícito, Máquina y PERSONAL quedan activas (positivas)
        # y MATERIALES con holgura (versión anterior, ya reemplazada, tenía
        # Máquina+Materiales activas y Personal con holgura). Estos números
        # deben coincidir siempre con
        # Evaluaciones/EP01/datos/EP01-valores-A.tex.
        "titulo": "Control 2 (EP01), Forma A: taller de bicicletas (4 variables, 3 restricciones)",
        "sentido": "max",
        "objetivo": (9000, 12000, 10000, 18000),
        "restricciones": [
            (2, 3, 4, 6, "<=", 180),  # Horas de máquina
            (5, 6, 8, 9, "<=", 450),  # Materiales (kg)
            (2, 2, 3, 3, "<=", 140),  # Horas de personal
        ],
    },
    "control_2_b": {
        # EP01, Forma B. Misma historia y misma matriz de tecnología que
        # control_2_a -- solo cambian utilidades y disponibilidad de
        # recursos. Mismo patrón cualitativo verificado: x3=x4=0 (costo
        # reducido > 0 ambas), Máquina y Personal activas, Materiales con
        # holgura. Estos números deben coincidir siempre con
        # Evaluaciones/EP01/datos/EP01-valores-B.tex.
        "titulo": "Control 2 (EP01), Forma B: taller de bicicletas (4 variables, 3 restricciones)",
        "sentido": "max",
        "objetivo": (7400, 9900, 8000, 15000),
        "restricciones": [
            (2, 3, 4, 6, "<=", 155),  # Horas de máquina
            (5, 6, 8, 9, "<=", 400),  # Materiales (kg)
            (2, 2, 3, 3, "<=", 120),  # Horas de personal
        ],
    },
    "control_2_c": {
        # EP01, Forma C. IDÉNTICA a control_2_a (Forma A) -- agregada a
        # pedido explícito ("crea una forma C que sea igual a la forma
        # A"), antes de que A y B pierdan las restricciones de venta
        # cruzada y producción mínima. Ver esa conversación.
        "titulo": "Control 2 (EP01), Forma C: taller de bicicletas (4 variables, 5 restricciones)",
        "sentido": "max",
        "objetivo": (12700, 17200, 23200, 25000),
        "restricciones": [
            (2, 3, 4, 6, "<=", 285),  # Horas de máquina
            (5, 6, 8, 9, "<=", 610),  # Materiales (kg)
            (2, 2, 3, 3, "<=", 260),  # Horas de personal
            (0, 0, 4, -1, ">=", 0),  # Venta cruzada: 4*x3 - x4 >= 0
            (1, 1, 1, 1, ">=", 100),  # Producción mínima
        ],
    },
}


def resolver_pulp(problema):
    """Resuelve el LP con pulp/HiGHS. Devuelve un dict con el estado, el
    óptimo, el precio sombra + rango de validez de cada restricción, y el
    costo reducido de cada variable -- o None si no es óptimo.

    Precio sombra: HiGHS reporta .pi con el signo exactamente invertido del
    que se espera en la convención económica estándar (positivo = un
    recurso <= más disponible ayuda; negativo = un piso >= más exigente
    cuesta) -- y esa inversión es GLOBAL (afecta por igual a restricciones
    <= y >=, no solo a una), así que se corrige negando .pi directo (NO con
    abs(), que borraría el signo -- economicamente real -- de una
    restricción >= respecto de una <=). Verificado con un caso de prueba
    antes de aplicarlo acá (ver conversación donde se agregaron
    control_2_a/b).
    """
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
        *coefs, sentido_restr, rhs = fila
        lhs = pulp.lpSum(a * v for a, v in zip(coefs, variables))
        if sentido_restr == "<=":
            restriccion = lhs <= rhs
        elif sentido_restr == ">=":
            restriccion = lhs >= rhs
        else:
            raise ValueError(f"sentido de restricción no soportado: {sentido_restr!r}")
        modelo += restriccion
        restricciones_pulp.append(restriccion)

    modelo.solve(pulp.HiGHS(msg=False))
    estado = pulp.LpStatus[modelo.status]
    if estado != "Optimal":
        return {"estado": estado}

    # HiGHS trae ranging de sensibilidad incorporado (Highs.getRanging()),
    # no hace falta resolver el LP en bucle (igual que pulp_LpProblem_3d).
    estado_ranging, ranging = modelo.solverModel.getRanging()
    if estado_ranging != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS getRanging() falló: {estado_ranging}")

    # Ver docstring: se niega .pi (no abs()) para recuperar el signo
    # económico real, válido tanto para restricciones <= como >=.
    duales = [-r.pi for r in restricciones_pulp]

    sensibilidad = []
    for idx, fila in enumerate(problema["restricciones"]):
        *_, sentido_restr, rhs = fila
        sensibilidad.append(
            {
                "idx": idx + 1,
                "sentido": sentido_restr,
                "rhs": rhs,
                "pi": duales[idx],
                "b_inf": ranging.row_bound_dn.value_[idx],
                "b_sup": ranging.row_bound_up.value_[idx],
            }
        )

    # Costo reducido de cada variable j: se lee directo de .dj (nativo de
    # pulp/HiGHS, con abs() por la misma razón de siempre -- ver Guía 3).
    # "costo_recursos" es solo una verificación manual (costo de los
    # recursos consumidos a precio sombra, con las duales YA con signo
    # económico correcto): costo_recursos - c_j debería coincidir con
    # costo_reducido salvo por el signo/orden de magnitud cuando hay
    # restricciones >= de por medio -- se reporta igual para que el
    # estudiante pueda cotejar a mano, pero el valor de referencia es
    # abs(dj).
    costos_reducidos = []
    for j, (nombre, cj, var) in enumerate(zip(nombres, coeficientes, variables)):
        costo_recursos = sum(
            problema["restricciones"][i][j] * duales[i]
            for i in range(len(problema["restricciones"]))
        )
        costos_reducidos.append(
            {
                "variable": nombre,
                "valor": var.value(),
                "costo_recursos": costo_recursos,
                "cj": cj,
                "costo_reducido": abs(var.dj),
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
            f"  R{f['idx']} ({f['sentido']}): b={f['rhs']}  precio sombra={f['pi']}  "
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
