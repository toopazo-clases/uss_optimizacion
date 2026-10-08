"""
Evaluación Parcial 02 - Operación Cupido: costo fijo + decisiones contingentes.

Un estudiante quiere maximizar las citas que consigue en un mes, con un
presupuesto y horas libres limitadas. Puede dedicar horas a 5 actividades;
cada una deja c_i citas esperadas por hora, cuesta g_i por hora, tiene un
tope de u_i horas al mes y, SI se usa, paga un costo fijo k_i (técnica 4,
"problema de costo fijo", Hillier & Lieberman 11.3). Además, dos decisiones
"padre" sin horas habilitan a sus actividades "hijas" (técnica 6, decisiones
contingentes): la sesión de fotos habilita Tinder y Hinge, y el curso de
salsa habilita las noches de salsa.

Notación de la solución de la EP02: actividades i = 1..5 (Tinder, Hinge,
Gimnasio, Salsa, Voluntariado).

    x_i >= 0 (continua): horas al mes en la actividad i
    y_i en {0, 1}, i = 1..5: 1 si se usa la actividad i (paga k_i)
    y_6, y_7 en {0, 1}: 1 si se paga la sesión de fotos / el curso de salsa

    max  z = sum_i c_i x_i                                   (citas al mes)
    s.a. sum_i (g_i x_i + k_i y_i) + K_FOTOS y_6 + K_CURSO y_7 <= PRESUPUESTO
         sum_i x_i <= HORAS
         x_i <= u_i y_i                 (tope y vínculo de costo fijo: M = u_i)
         y_1 <= y_6,  y_2 <= y_6,  y_4 <= y_7               (contingentes)

Los datos salieron de una búsqueda aleatoria filtrada: el óptimo usa 3
actividades con horas enteras, compra las fotos pero no el curso (aunque la
salsa es la actividad con más citas por hora), el segundo mejor conjunto de
actividades queda 0,3 citas abajo y sin costos fijos el plan cambia.

Pregunta 7 (cambio de una restricción, cada caso por separado):
  a) HORAS = 30: el plan NO cambia (no queda plata para otro costo fijo).
  b) PRESUPUESTO = 110000: entra la salsa (paga el curso) y sale el
     voluntariado; la salsa entra desde un presupuesto de 103000.

Uso:
    python main.py
"""

import itertools

import pulp

ACTIVIDADES = ("Tinder", "Hinge", "Gimnasio", "Salsa", "Voluntariado")
CITAS_POR_HORA = {"Tinder": 0.25, "Hinge": 0.3, "Gimnasio": 0.15, "Salsa": 0.4, "Voluntariado": 0.2}
COSTO_FIJO = {"Tinder": 10000, "Hinge": 10000, "Gimnasio": 20000, "Salsa": 20000, "Voluntariado": 15000}
GASTO_POR_HORA = {"Tinder": 0, "Hinge": 2000, "Gimnasio": 0, "Salsa": 2000, "Voluntariado": 2000}
TOPE_HORAS = {"Tinder": 4, "Hinge": 10, "Gimnasio": 6, "Salsa": 4, "Voluntariado": 6}

PADRES = ("Fotos", "Curso")
COSTO_PADRE = {"Fotos": 10000, "Curso": 25000}
PADRE_DE = {"Tinder": "Fotos", "Hinge": "Fotos", "Salsa": "Curso"}

PRESUPUESTO = 80000  # $ al mes
HORAS = 20  # horas libres al mes


ESCENARIOS_PREGUNTA_7 = {
    "a) 30 horas libres": {"presupuesto": 80000, "horas": 30},
    "b) presupuesto de $110.000": {"presupuesto": 110000, "horas": 20},
}


def resolver(costo_padre=COSTO_PADRE, con_costo_fijo=True, forzar=(),
             presupuesto=None, horas=None):
    """Con forzar=S usa EXACTAMENTE las actividades de S (al menos media hora
    en cada una y nada en las demás), para comparar con el mejor plan que
    usa otro conjunto de actividades."""
    presupuesto = PRESUPUESTO if presupuesto is None else presupuesto
    horas = HORAS if horas is None else horas
    modelo = pulp.LpProblem("ep02", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in ACTIVIDADES}
    y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in ACTIVIDADES}
    p = {q: pulp.LpVariable(f"p_{q}", cat="Binary") for q in PADRES}
    fijo = pulp.lpSum(COSTO_FIJO[j] * y[j] for j in ACTIVIDADES) + pulp.lpSum(
        costo_padre[q] * p[q] for q in PADRES)
    modelo += pulp.lpSum(CITAS_POR_HORA[j] * x[j] for j in ACTIVIDADES), "citas"
    modelo += pulp.lpSum(GASTO_POR_HORA[j] * x[j] for j in ACTIVIDADES) + (
        fijo if con_costo_fijo else 0) <= presupuesto, "presupuesto"
    modelo += pulp.lpSum(x.values()) <= horas, "horas"
    for j in ACTIVIDADES:
        modelo += x[j] <= TOPE_HORAS[j] * y[j], f"tope_y_costo_fijo_{j}"
    for j, q in PADRE_DE.items():
        modelo += y[j] <= p[q], f"{j}_requiere_{q}"
    if forzar:
        for j in ACTIVIDADES:
            if j in forzar:
                modelo += x[j] >= 0.5, f"forzar_{j}"
            else:
                modelo += x[j] == 0, f"excluir_{j}"
    modelo.solve(pulp.HiGHS(msg=False))
    estado = pulp.LpStatus[modelo.status]
    if estado != "Optimal":
        return {"estado": estado}
    xv = {j: x[j].value() for j in ACTIVIDADES}
    yv = {j: round(y[j].value()) for j in ACTIVIDADES}
    pv = {q: round(p[q].value()) for q in PADRES}
    gasto = sum(GASTO_POR_HORA[j] * xv[j] + COSTO_FIJO[j] * yv[j] for j in ACTIVIDADES) + sum(
        costo_padre[q] * pv[q] for q in PADRES)
    return {"estado": estado, "z": pulp.value(modelo.objective), "x": xv, "y": yv, "p": pv,
            "gasto": gasto, "horas": sum(xv.values()),
            "presupuesto": presupuesto, "horas_disponibles": horas}


def segundo_mejor(res, **kwargs):
    """Mejor z entre los planes que usan un conjunto de actividades distinto."""
    usadas = {j for j in ACTIVIDADES if res["x"][j] > 1e-6}
    mejor = None
    for r in range(1, len(ACTIVIDADES) + 1):
        for conjunto in itertools.combinations(ACTIVIDADES, r):
            if set(conjunto) == usadas:
                continue
            otro = resolver(forzar=conjunto, **kwargs)
            if otro["estado"] == "Optimal" and (mejor is None or otro["z"] > mejor[1]):
                mejor = (conjunto, otro["z"])
    return mejor


def imprimir(titulo, res, costo_padre=COSTO_PADRE):
    print(titulo, f"(estado: {res['estado']})")
    print(f"  {'actividad':14}{'horas':>7}{'citas':>7}{'gasto':>9}")
    for j in ACTIVIDADES:
        if res["y"][j]:
            gasto = GASTO_POR_HORA[j] * res["x"][j] + COSTO_FIJO[j]
            print(f"  {j:14}{res['x'][j]:>7.2f}{CITAS_POR_HORA[j] * res['x'][j]:>7.2f}{gasto:>9.0f}")
    for q in PADRES:
        if res["p"][q]:
            print(f"  {q:14}{'-':>7}{'-':>7}{costo_padre[q]:>9.0f}")
    print(f"  z = {res['z']:.2f} citas al mes; gasto = {res['gasto']:.0f} de {res['presupuesto']}; "
          f"horas = {res['horas']:.2f} de {res['horas_disponibles']}")


def main():
    base = resolver()
    imprimir("Modelo completo:", base)
    conjunto, z2 = segundo_mejor(base)
    print(f"  Segundo mejor conjunto de actividades: {', '.join(conjunto)} -> z = {z2:.2f}")
    print()
    for titulo, cambio in ESCENARIOS_PREGUNTA_7.items():
        res = resolver(**cambio)
        imprimir(f"Pregunta 7 {titulo}:", res)
        conjunto, z2 = segundo_mejor(res, **cambio)
        print(f"  Segundo mejor conjunto de actividades: {', '.join(conjunto)} -> z = {z2:.2f}")
        print()
    sin_fijos = resolver(con_costo_fijo=False)
    print("Comparación: sin costos fijos (ni de actividades ni de padres):")
    print("  horas:", {j: round(v, 2) for j, v in sin_fijos["x"].items() if v > 1e-6},
          f"z = {sin_fijos['z']:.2f}")


if __name__ == "__main__":
    main()
