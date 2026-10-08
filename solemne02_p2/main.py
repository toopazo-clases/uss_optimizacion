"""
Solemne 02, Problema 2 - Planificación militar con decisiones binarias.

Adaptado de Taha, "Investigación de Operaciones", problema 2.4F-8 (el mismo
de solemne01_p1), simplificado a UNA batalla (sin frentes norte/sur) con 4
líneas de defensa (I, II, III, IV), y extendido con dos técnicas de
modelado con variables binarias (Hillier & Lieberman, sección 11.3):

  - "Deben cumplirse K de N restricciones": idealmente cada línea resiste
    al menos DURACION_MINIMA días, pero solo se exige en K de las 4 líneas.
    Con w_j binaria (1 = se exige en la línea j):
        d_j >= DURACION_MINIMA - M (1 - w_j),   sum_j w_j >= K,
    y basta M = DURACION_MINIMA porque d_j >= a_j > 0 siempre.
  - "Funciones con N valores posibles": la reserva solo actúa en las
    líneas III y IV, en grupos de GRUPOS_RESERVA unidades (o nada). Con
    y_jk binaria (1 = se despliega el grupo k en la línea j):
        r_j = sum_k g_k y_jk,   sum_k y_jk <= 1.

Duración de la línea j:  d_j = a_j + b_j * (x_j + r_j) / u_j, con x_j las
unidades regulares (continuas) y u_j las unidades rojas que la atacan.
Objetivo: maximizar la duración total, sum_j d_j.

Las Formas A y B difieren SOLO en las tropas disponibles (ver FORMAS).

Uso:
    python main.py
"""

import pulp

LINEAS = ("I", "II", "III", "IV")
LINEAS_CON_RESERVA = ("III", "IV")

A = {"I": 0.5, "II": 0.75, "III": 1.1, "IV": 1.3}
B = {"I": 8.8, "II": 7.9, "III": 10.2, "IV": 9.2}
U = {"I": 30, "II": 60, "III": 20, "IV": 40}

DURACION_MINIMA = 5
K = 2
GRUPOS_RESERVA = (10, 50, 70)
M = DURACION_MINIMA

FORMAS = {
    "A": {"regulares": 300, "reserva": 200},
    "B": {"regulares": 200, "reserva": 100},
}


def resolver(regulares, reserva, grupos=GRUPOS_RESERVA, k_de_n=True):
    """Resuelve el modelo. Con k_de_n=False exige el mínimo en las 4
    líneas (sin la técnica K de N), para comparar."""
    modelo = pulp.LpProblem("solemne02_p2", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in LINEAS}
    w = {j: pulp.LpVariable(f"w_{j}", cat="Binary") for j in LINEAS}
    y = {
        (j, k): pulp.LpVariable(f"y_{j}_{k + 1}", cat="Binary")
        for j in LINEAS_CON_RESERVA
        for k in range(len(grupos))
    }
    r = {j: pulp.lpSum(g * y[(j, k)] for k, g in enumerate(grupos)) for j in LINEAS_CON_RESERVA}
    d = {j: A[j] + B[j] / U[j] * (x[j] + r.get(j, 0)) for j in LINEAS}

    modelo += pulp.lpSum(d.values()), "duracion_total"
    for j in LINEAS:
        if k_de_n:
            modelo += d[j] >= DURACION_MINIMA - M * (1 - w[j]), f"minimo_{j}"
        else:
            modelo += d[j] >= DURACION_MINIMA, f"minimo_{j}"
    if k_de_n:
        modelo += pulp.lpSum(w.values()) >= K, "k_de_n"
    for j in LINEAS_CON_RESERVA:
        modelo += pulp.lpSum(y[(j, k)] for k in range(len(grupos))) <= 1, f"un_grupo_{j}"
    modelo += pulp.lpSum(x.values()) <= regulares, "regulares"
    modelo += pulp.lpSum(r.values()) <= reserva, "reserva"

    modelo.solve(pulp.HiGHS(msg=False))
    estado = pulp.LpStatus[modelo.status]
    if estado != "Optimal":
        return {"estado": estado}
    return {
        "estado": estado,
        "z": pulp.value(modelo.objective),
        "x": {j: x[j].value() for j in LINEAS},
        "w": {j: round(w[j].value()) for j in LINEAS} if k_de_n else None,
        "r": {j: round(pulp.value(r[j])) for j in LINEAS_CON_RESERVA},
        "d": {j: pulp.value(d[j]) for j in LINEAS},
    }


def imprimir(titulo, res):
    print(titulo)
    if res["estado"] != "Optimal":
        print("  estado:", res["estado"])
        return
    print(f"  {'línea':6}{'regulares':>10}{'reserva':>9}{'duración':>10}{'w':>4}")
    for j in LINEAS:
        w = res["w"][j] if res["w"] else "-"
        print(f"  {j:6}{res['x'][j]:>10.2f}{res['r'].get(j, 0):>9}{res['d'][j]:>10.2f}{w:>4}")
    print(f"  z = {res['z']:.2f} días")


def main():
    for forma, tropas in FORMAS.items():
        print("=" * 60)
        print(f"Forma {forma}: {tropas['regulares']} regulares, {tropas['reserva']} de reserva")
        print("=" * 60)
        imprimir("Modelo completo (K de N + grupos de reserva):",
                 resolver(tropas["regulares"], tropas["reserva"]))
        imprimir("Comparación: exigiendo las 4 líneas (sin K de N):",
                 resolver(tropas["regulares"], tropas["reserva"], k_de_n=False))
        print()


if __name__ == "__main__":
    main()
