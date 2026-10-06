"""
Solemne 01, Problema 1 (Forma A) - Planificación militar (Taha, "Investigación
de Operaciones", problema 2.4F-8; ver Shepard and Associates, 1988).

Enunciado (resumen): el ejército azul (B) defiende su territorio con 200
unidades regulares y una reserva de 200 unidades adicionales, repartidas en
3 líneas de defensa este-oeste (I, II, III) y 2 frentes (Norte, Sur). La
reserva solo puede usarse en las líneas II y III. El ejército rojo (R) ataca
cada línea/frente con una cantidad fija de unidades. El tiempo que tarda el
rojo en romper una línea es:

    d_ij = a_ij + b_ij * (unidades azules asignadas a i,j) / (unidades rojas
                                                                atacantes i,j)

donde i es el frente (N, S) y j la línea (I, II, III). El azul debe asegurar
d >= 4 días en las líneas I y II (en ambos frentes); no hay mínimo para la
línea III, pero también aporta a la duración total.

Objetivo: maximizar la duración total de la batalla, sumando las 6 d_ij,
sujeto a los 2 topes de disponibilidad (regulares y reserva) y a los 4
mínimos de 4 días. Formulación completa en
`\\subsubsection{Problema de programación lineal}` de
`Evaluaciones/Solemne01/contenido-A.tex`.

Mapeo con las x_i del enunciado (10 variables, todas >= 0):
    x1..x6  -> unidades regulares en (N,I) (N,II) (N,III) (S,I) (S,II) (S,III)
    x7..x10 -> unidades de reserva en (N,II) (N,III) (S,II) (S,III)
    (no hay reserva en la línea I: el enunciado no la permite ahí)

Para probar otros valores de a, b o de las tropas atacantes (rojas, u):
edita directamente las constantes A, B, U más abajo y vuelve a ejecutar
`python main.py`. Los valores originales del enunciado quedan comentados
justo al lado, para poder volver a ellos.

Este script resuelve DOS versiones del modelo (ver
`\\subsubsection{Extensión: formulación minimax (simultaneidad de los
frentes)}` en `contenido-A.tex`):

  1. "Suma" (el modelo original): maximiza z = D_N + D_S, sumando la
     duración total del frente norte y del frente sur como si el ejército
     rojo los atacara uno después del otro.
  2. "Minimax": maximiza t = min(D_N, D_S), reconociendo que el rojo ataca
     ambos frentes EN PARALELO, por lo que la duración real de la batalla
     la determina el frente que cae primero. Sigue siendo un LP: se agrega
     la variable t y las restricciones de epígrafe t <= D_N, t <= D_S.

Donde D_N = d_NI + d_NII + d_NIII y D_S = d_SI + d_SII + d_SIII (la suma
DENTRO de un frente sí es válida, porque las líneas I, II, III son
secuenciales dentro de un mismo frente).
"""

import pulp

FRENTES = ("N", "S")
LINEAS = ("I", "II", "III")
LINEAS_CON_MINIMO = ("I", "II")  # deben durar >= DURACION_MINIMA días
LINEAS_CON_RESERVA = ("II", "III")  # únicas líneas donde se puede usar reserva

# --------------------------------------------------------------------------
# Parámetros editables -- cambia estos valores para probar otros escenarios
# --------------------------------------------------------------------------

# a_ij: duración base (Tabla "Constantes a y b de la fórmula de duración de
# la batalla" del enunciado).
A = {
    "N": {"I": 0.5, "II": 0.75, "III": 0.55},
    "S": {"I": 1.1, "II": 1.3, "III": 1.5},
}

# b_ij: coeficiente que multiplica la razón unidades_azules/unidades_rojas.
B = {
    "N": {"I": 8.8, "II": 7.9, "III": 10.2},
    "S": {"I": 10.5, "II": 8.1, "III": 9.2},
}

# u_ij: unidades de ataque del ejército rojo (Tabla "Cantidad de unidades de
# ataque del ejército rojo" del enunciado). Esto es lo que en el enunciado
# se llama "tropas atacantes (rojas)".
U = {
    "N": {"I": 30, "II": 60, "III": 20},
    "S": {"I": 30, "II": 40, "III": 20},
}

REGULARES_DISPONIBLES = 200
RESERVA_DISPONIBLE = 200
DURACION_MINIMA = 4

# Valores originales del enunciado, por si A/B/U de arriba se modifican
# mucho y hace falta recordar con qué se empezó:
#
# A_ENUNCIADO = {"N": {"I": 0.5, "II": 0.75, "III": 0.55},
#                "S": {"I": 1.1, "II": 1.3,  "III": 1.5}}
# B_ENUNCIADO = {"N": {"I": 8.8, "II": 7.9, "III": 10.2},
#                "S": {"I": 10.5, "II": 8.1, "III": 9.2}}
# U_ENUNCIADO = {"N": {"I": 30, "II": 60, "III": 20},
#                "S": {"I": 30, "II": 40, "III": 20}}
# REGULARES_DISPONIBLES = 200
# RESERVA_DISPONIBLE = 200
# DURACION_MINIMA = 4


# --------------------------------------------------------------------------
# Modelo (pulp)
# --------------------------------------------------------------------------


def _variables_y_duraciones(a, b, u):
    """Crea las variables de decisión (regulares, reserva) y las expresiones
    de duración d_ij = a_ij + (b_ij/u_ij)*(regulares_ij + reserva_ij).

    Común a ambas variantes del modelo (suma y minimax): sólo cambia la
    función objetivo y, en el caso minimax, dos restricciones adicionales.
    """
    regulares = {
        (f, l): pulp.LpVariable(f"regulares_{f}_{l}", lowBound=0)
        for f in FRENTES
        for l in LINEAS
    }
    reserva = {
        (f, l): pulp.LpVariable(f"reserva_{f}_{l}", lowBound=0)
        for f in FRENTES
        for l in LINEAS_CON_RESERVA
    }

    def unidades_azules(f, l):
        total = regulares[(f, l)]
        if (f, l) in reserva:
            total = total + reserva[(f, l)]
        return total

    duraciones = {
        (f, l): a[f][l] + (b[f][l] / u[f][l]) * unidades_azules(f, l)
        for f in FRENTES
        for l in LINEAS
    }
    return regulares, reserva, duraciones


def _restricciones_comunes(modelo, regulares, reserva, duraciones, regulares_disponibles, reserva_disponible, duracion_minima):
    """Agrega las restricciones que comparten ambas variantes del modelo:
    los 4 mínimos de 4 días (líneas I y II, ambos frentes) y los 2 topes de
    disponibilidad de tropas (regulares y reserva)."""
    for f in FRENTES:
        for l in LINEAS_CON_MINIMO:
            modelo += duraciones[(f, l)] >= duracion_minima, f"minimo_{f}_{l}"

    modelo += pulp.lpSum(regulares.values()) <= regulares_disponibles, "regulares_disponibles"
    modelo += pulp.lpSum(reserva.values()) <= reserva_disponible, "reserva_disponible"


def resolver_pulp(
    a,
    b,
    u,
    regulares_disponibles=REGULARES_DISPONIBLES,
    reserva_disponible=RESERVA_DISPONIBLE,
    duracion_minima=DURACION_MINIMA,
):
    """Arma y resuelve el modelo "suma": maximiza z = D_N + D_S, la suma de
    la duración total del frente norte y del frente sur, como si el rojo
    atacara un frente después del otro.

    a, b, u: diccionarios anidados {frente: {linea: valor}} para las
    constantes a_ij, b_ij y las unidades de ataque rojas u_ij.

    No lee las constantes globales A/B/U directamente, así que también sirve
    para probar valores propios desde otro script o una consola interactiva:
        from main import resolver_pulp
        resultado = resolver_pulp(mi_a, mi_b, mi_u)

    Devuelve un dict con el status del solver, las asignaciones óptimas de
    unidades regulares y de reserva por (frente, línea), las duraciones d_ij
    y la duración total z.
    """
    modelo = pulp.LpProblem("Planificacion_militar_suma", pulp.LpMaximize)
    regulares, reserva, duraciones = _variables_y_duraciones(a, b, u)

    modelo += pulp.lpSum(duraciones.values()), "duracion_total"

    _restricciones_comunes(
        modelo, regulares, reserva, duraciones, regulares_disponibles, reserva_disponible, duracion_minima
    )

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "status": pulp.LpStatus[modelo.status],
        "regulares": {k: pulp.value(v) for k, v in regulares.items()},
        "reserva": {k: pulp.value(v) for k, v in reserva.items()},
        "duraciones": {k: pulp.value(v) for k, v in duraciones.items()},
        "z": pulp.value(modelo.objective),
    }


def resolver_pulp_minimax(
    a,
    b,
    u,
    regulares_disponibles=REGULARES_DISPONIBLES,
    reserva_disponible=RESERVA_DISPONIBLE,
    duracion_minima=DURACION_MINIMA,
):
    """Arma y resuelve el modelo "minimax": maximiza t = min(D_N, D_S),
    reconociendo que el rojo ataca ambos frentes EN PARALELO, por lo que la
    duración real de la batalla la determina el frente que cae primero.

    Sigue siendo un LP: se linealiza con la técnica de epígrafe, agregando
    la variable t y las restricciones t <= D_N, t <= D_S. Mismos parámetros
    y mismo retorno que `resolver_pulp`, más las claves "D" (duración total
    por frente) y "t" (el mínimo entre ambos frentes, valor del objetivo).
    """
    modelo = pulp.LpProblem("Planificacion_militar_minimax", pulp.LpMaximize)
    regulares, reserva, duraciones = _variables_y_duraciones(a, b, u)

    duracion_por_frente = {
        f: pulp.lpSum(duraciones[(f, l)] for l in LINEAS) for f in FRENTES
    }
    t = pulp.LpVariable("t", lowBound=0)

    modelo += t, "min_duracion_frente"

    for f in FRENTES:
        modelo += t <= duracion_por_frente[f], f"epigrafe_{f}"

    _restricciones_comunes(
        modelo, regulares, reserva, duraciones, regulares_disponibles, reserva_disponible, duracion_minima
    )

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "status": pulp.LpStatus[modelo.status],
        "regulares": {k: pulp.value(v) for k, v in regulares.items()},
        "reserva": {k: pulp.value(v) for k, v in reserva.items()},
        "duraciones": {k: pulp.value(v) for k, v in duraciones.items()},
        "D": {f: pulp.value(v) for f, v in duracion_por_frente.items()},
        "t": pulp.value(modelo.objective),
    }


# --------------------------------------------------------------------------
# Presentación del resultado
# --------------------------------------------------------------------------


def _imprimir_tabla_asignacion(resultado, u):
    """Imprime la tabla de asignación de tropas por frente y línea, común a
    ambas variantes del modelo. No imprime status ni el valor del objetivo
    (eso lo hace cada llamador, porque el objetivo se llama distinto en cada
    variante: z en la "suma", t en la "minimax")."""
    print("Asignación de unidades azules por frente y línea de defensa:")
    print(
        f"{'frente':<8}{'línea':<8}{'regulares':>11}{'reserva':>10}"
        f"{'u (rojas)':>11}{'duración':>11}"
    )
    for f in FRENTES:
        for l in LINEAS:
            reg = resultado["regulares"][(f, l)]
            res = resultado["reserva"].get((f, l))
            res_str = f"{res:>10.2f}" if res is not None else f"{'-':>10}"
            marca = " (>=4 requerido)" if l in LINEAS_CON_MINIMO else ""
            print(
                f"{f:<8}{l:<8}{reg:>11.2f}{res_str}"
                f"{u[f][l]:>11}{resultado['duraciones'][(f, l)]:>11.2f}{marca}"
            )

    total_regulares = sum(resultado["regulares"].values())
    total_reserva = sum(resultado["reserva"].values())
    print(f"\nTotal unidades regulares usadas: {total_regulares:.2f} / {REGULARES_DISPONIBLES}")
    print(f"Total unidades de reserva usadas: {total_reserva:.2f} / {RESERVA_DISPONIBLE}")


def imprimir_resultado(resultado, u, duracion_minima=DURACION_MINIMA):
    """Imprime el resultado del modelo "suma" (objetivo z = D_N + D_S)."""
    print(f"Status: {resultado['status']}\n")

    if resultado["status"] != "Optimal":
        print("No se encontró un óptimo (revisa si los datos hacen el modelo")
        print("infactible, por ejemplo unidades rojas u_ij demasiado bajas).")
        return

    _imprimir_tabla_asignacion(resultado, u)
    print(f"\nDuración total de la batalla, suma norte+sur (z): {resultado['z']:.4f} días")


def imprimir_resultado_minimax(resultado, u):
    """Imprime el resultado del modelo "minimax" (objetivo t = min(D_N, D_S))."""
    print(f"Status: {resultado['status']}\n")

    if resultado["status"] != "Optimal":
        print("No se encontró un óptimo (revisa si los datos hacen el modelo")
        print("infactible, por ejemplo unidades rojas u_ij demasiado bajas).")
        return

    _imprimir_tabla_asignacion(resultado, u)
    for f in FRENTES:
        print(f"Duración total frente {f} (D_{f}): {resultado['D'][f]:.4f} días")
    print(f"\nPeor caso entre ambos frentes, min(D_N, D_S) (t): {resultado['t']:.4f} días")


def imprimir_comparacion(resultado_suma, resultado_minimax):
    """Tabla resumen comparando ambas variantes, igual a la Tabla de
    comparación que aparece en `contenido-A.tex`."""
    d_suma = {f: sum(resultado_suma["duraciones"][(f, l)] for l in LINEAS) for f in FRENTES}
    d_minimax = resultado_minimax["D"]

    print(f"{'modelo':<12}{'D_N':>10}{'D_S':>10}{'min(D_N,D_S)':>16}")
    print(
        f"{'suma':<12}{d_suma['N']:>10.2f}{d_suma['S']:>10.2f}"
        f"{min(d_suma.values()):>16.2f}"
    )
    print(
        f"{'minimax':<12}{d_minimax['N']:>10.2f}{d_minimax['S']:>10.2f}"
        f"{min(d_minimax.values()):>16.2f}"
    )


def main():
    print("=" * 78)
    print("Modelo 1: SUMA -- maximiza z = D_N + D_S")
    print("=" * 78)
    resultado_suma = resolver_pulp(
        A, B, U, REGULARES_DISPONIBLES, RESERVA_DISPONIBLE, DURACION_MINIMA
    )
    imprimir_resultado(resultado_suma, U, DURACION_MINIMA)

    print("\n" + "=" * 78)
    print("Modelo 2: MINIMAX -- maximiza t = min(D_N, D_S)")
    print("=" * 78)
    resultado_minimax = resolver_pulp_minimax(
        A, B, U, REGULARES_DISPONIBLES, RESERVA_DISPONIBLE, DURACION_MINIMA
    )
    imprimir_resultado_minimax(resultado_minimax, U)

    if resultado_suma["status"] == "Optimal" and resultado_minimax["status"] == "Optimal":
        print("\n" + "=" * 78)
        print("Comparación")
        print("=" * 78)
        imprimir_comparacion(resultado_suma, resultado_minimax)


if __name__ == "__main__":
    main()
