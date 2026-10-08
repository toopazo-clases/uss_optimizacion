"""
Solemne 02, Problema 1 - Qué productos fabricar: problema de costo fijo.

Técnica "problema de costo fijo" (Hillier & Lieberman, sección 11.3; mismo
patrón que el Problema 4 de Guias/Guia_5_Programacion_Entera_Binaria_Mixta,
pero maximizando). Una empresa puede fabricar 3 productos; cada uno deja una
utilidad p_j por unidad, consume horas de máquina y horas-hombre, tiene una
demanda máxima d_j y, SI se fabrica, paga un costo fijo de preparación k_j.

    x_j >= 0 (continua): unidades del producto j
    y_j en {0, 1}: 1 si se fabrica el producto j

    max  z = sum_j p_j x_j - sum_j k_j y_j
    s.a. sum_j hm_j x_j <= HORAS_MAQUINA
         sum_j hh_j x_j <= HORAS_HOMBRE
         x_j <= d_j y_j                 (demanda y vínculo de costo fijo: M = d_j)

Los datos se eligieron (búsqueda aleatoria) para que el óptimo fabrique 2 de
los 3 productos, para que sin costos fijos se fabriquen los 3, y para que las
Formas A y B, que difieren SOLO en las horas disponibles, dejen fuera un
producto distinto.

Uso:
    python main.py
"""

import pulp

PRODUCTOS = ("A", "B", "C")
UTILIDAD = {"A": 8, "B": 7, "C": 12}  # miles de $ por unidad
COSTO_FIJO = {"A": 400, "B": 200, "C": 750}  # miles de $
HORAS_MAQUINA_POR_UNIDAD = {"A": 1, "B": 4, "C": 1}
HORAS_HOMBRE_POR_UNIDAD = {"A": 2, "B": 1, "C": 4}
DEMANDA = {"A": 100, "B": 100, "C": 150}  # unidades

FORMAS = {
    "A": {"horas_maquina": 900, "horas_hombre": 600},
    "B": {"horas_maquina": 800, "horas_hombre": 400},
}


def resolver(horas_maquina, horas_hombre, con_costo_fijo=True):
    modelo = pulp.LpProblem("solemne02_p1", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in PRODUCTOS}
    y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in PRODUCTOS}
    fijo = pulp.lpSum(COSTO_FIJO[j] * y[j] for j in PRODUCTOS) if con_costo_fijo else 0
    modelo += pulp.lpSum(UTILIDAD[j] * x[j] for j in PRODUCTOS) - fijo, "utilidad_neta"
    modelo += pulp.lpSum(HORAS_MAQUINA_POR_UNIDAD[j] * x[j] for j in PRODUCTOS) <= horas_maquina, "maquina"
    modelo += pulp.lpSum(HORAS_HOMBRE_POR_UNIDAD[j] * x[j] for j in PRODUCTOS) <= horas_hombre, "hombre"
    for j in PRODUCTOS:
        modelo += x[j] <= DEMANDA[j] * y[j], f"demanda_y_costo_fijo_{j}"
    modelo.solve(pulp.HiGHS(msg=False))
    return {
        "estado": pulp.LpStatus[modelo.status],
        "z": pulp.value(modelo.objective),
        "x": {j: x[j].value() for j in PRODUCTOS},
        "y": {j: round(y[j].value()) for j in PRODUCTOS},
    }


def imprimir(titulo, res):
    print(titulo, f"(estado: {res['estado']})")
    for j in PRODUCTOS:
        print(f"  {j}: x = {res['x'][j]:8.2f}   y = {res['y'][j]}")
    print(f"  z = {res['z']:.2f} miles de $")


def main():
    for forma, horas in FORMAS.items():
        print("=" * 60)
        print(f"Forma {forma}: {horas['horas_maquina']} horas de máquina, "
              f"{horas['horas_hombre']} horas-hombre")
        print("=" * 60)
        imprimir("Con costos fijos:", resolver(**horas))
        imprimir("Sin costos fijos (comparación):", resolver(**horas, con_costo_fijo=False))
        print()


if __name__ == "__main__":
    main()
