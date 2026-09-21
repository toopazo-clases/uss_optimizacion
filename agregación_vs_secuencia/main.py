"""
Diagrama de Gantt de una línea de flujo de 3 máquinas (M1 -> M2 -> M3).

Contexto (Taha, problema 2.2A-4, "fábrica secuencial/paralela"): dos
productos x1 y x2, cada unidad debe pasar por M1, luego M2, luego M3.
El LP de capacidad agregada (ver pulp_LpProblem_2d/main.py, problemas
"fabrica_secuencial_10hr" y "fabrica_paralela_10hr") solo garantiza que la
carga TOTAL de cada máquina quepa en 600 minutos (10 hr); no garantiza que
exista una SECUENCIA de fabricación que permita terminar dentro de esa
ventana. Ese es justamente el punto que este script visualiza: el LP
resuelve un problema de capacidad agregada, no de secuenciación.

Para una secuencia de unidades dada (el orden en que entran a M1), calcula
inicio/fin de cada unidad en cada máquina y grafica una línea de tiempo por
máquina: rojo = fabricando x1, verde = fabricando x2, blanco = tiempo
ocioso.

Supuesto de cálculo: buffer infinito entre máquinas, sin bloqueo (una
unidad que ya terminó en una máquina puede esperar en cola indefinidamente
a que la siguiente máquina se desocupe; M1 nunca espera, siempre tiene la
próxima unidad lista). Es el escenario más favorable posible para
cualquier secuencia.

Uso:
    Editar la sección "elegir aquí la secuencia" al final del archivo y
    correr:
        python main.py
    Genera out.png. Para comparar dos secuencias, renombrar out.png a mano
    entre una corrida y otra.
"""

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Datos del problema
# --------------------------------------------------------------------------
# Tiempos de proceso (minutos) por producto y máquina.
TIEMPOS = {
    "x1": {"M1": 10, "M2": 6, "M3": 8},
    "x2": {"M1": 5, "M2": 20, "M3": 10},
}

MAQUINAS = ["M1", "M2", "M3"]
COLORES = {"x1": "tab:red", "x2": "tab:green"}
LIMITE_JORNADA = 600  # minutos (10 horas)


# --------------------------------------------------------------------------
# Generadores de secuencia
# --------------------------------------------------------------------------
def secuencia_por_bloque(n1, n2):
    """Todo x1 primero, luego todo x2. Ej. de secuencia "mala"."""
    return ["x1"] * n1 + ["x2"] * n2


def secuencia_por_ciclo(patron, n1, n2):
    """Repite `patron` (ej. ["x1", "x2"]) hasta agotar uno de los dos
    productos, y rellena el resto con lo que sobre del otro. Ej. de
    secuencia "buena" (intercalada)."""
    restante = {"x1": n1, "x2": n2}
    secuencia = []
    while restante["x1"] > 0 or restante["x2"] > 0:
        avanzo = False
        for producto in patron:
            if restante.get(producto, 0) > 0:
                secuencia.append(producto)
                restante[producto] -= 1
                avanzo = True
        if not avanzo:
            break
    return secuencia


# --------------------------------------------------------------------------
# Cálculo del horario (recursión de flow-shop de permutación)
# --------------------------------------------------------------------------
def calcular_horario(secuencia, tiempos=TIEMPOS):
    """Calcula inicio/fin de cada unidad en cada máquina.

    Recursión estándar de flow-shop de permutación: mismo orden de unidades
    en las 3 máquinas, buffer infinito, sin bloqueo.
        fin(i, M1) = fin(i-1, M1) + t(i, M1)
        fin(i, Mk) = max(fin(i, M_{k-1}), fin(i-1, Mk)) + t(i, Mk)

    A cada unidad también se le asigna un número de orden DENTRO de su
    propio producto (el 5to x1 fabricado, el 6to, etc.), para poder
    etiquetarla en el gráfico (ej. "x1-5") y así seguirle la pista a la
    misma unidad a través de M1, M2 y M3.

    Retorna {maquina: [(inicio, fin, producto, numero), ...]} en el orden
    de fabricación.
    """
    fin_anterior = {m: 0.0 for m in MAQUINAS}
    contador_producto = {}
    horario = {m: [] for m in MAQUINAS}

    for producto in secuencia:
        contador_producto[producto] = contador_producto.get(producto, 0) + 1
        numero = contador_producto[producto]
        fin_prev_maquina = 0.0  # fin de esta unidad en la máquina anterior
        for m in MAQUINAS:
            inicio = max(fin_prev_maquina, fin_anterior[m])
            fin = inicio + tiempos[producto][m]
            horario[m].append((inicio, fin, producto, numero))
            fin_anterior[m] = fin
            fin_prev_maquina = fin

    return horario


def calcular_metricas(horario):
    """Ocioso total y makespan por máquina, más el makespan global."""
    resumen = {}
    makespan = max(fin for m in MAQUINAS for (_, fin, _, _) in horario[m])
    for m in MAQUINAS:
        ocupado = sum(fin - inicio for inicio, fin, _, _ in horario[m])
        fin_maquina = horario[m][-1][1] if horario[m] else 0.0
        resumen[m] = {
            "ocupado": ocupado,
            "fin": fin_maquina,
            "ocioso": fin_maquina - ocupado,
        }
    resumen["makespan"] = makespan
    return resumen


# --------------------------------------------------------------------------
# Gráfico
# --------------------------------------------------------------------------
def graficar(horario, resumen, titulo, archivo_salida="out.png"):
    # Filas bien separadas verticalmente: cada fila necesita espacio para
    # la marca de inicio (más alta que la barra) y, encima, la etiqueta de
    # texto rotada. Con solo 3 filas sobra ancho de sobra para esto.
    # Tamaños de texto al doble de la versión anterior, PERO sin tocar el
    # lienzo ni los espaciados: se pidió expresamente que el texto ocupe
    # más área de la misma imagen, así que aquí puede haber traslapes de
    # nuevo si quedan muy apretados (a revisar).
    fontsize_etiqueta = 52
    fontsize_ejes = 72
    fontsize_titulo = 96
    fontsize_leyenda = 64

    y_por_maquina = {"M1": 32, "M2": 16, "M3": 0}
    alto = 3.2
    tick_extra_corto = 3.2
    tick_extra_largo = 5.6
    espacio_etiqueta = 0.6

    # El gráfico se hace tan ancho como haga falta para que las etiquetas
    # (una por unidad, hasta 67 por fila) no se encimen. Se navega
    # haciendo scroll/zoom sobre la imagen.
    ancho_in = max(120, resumen["makespan"] * 0.36)
    alto_in = 44
    fig, ax = plt.subplots(figsize=(ancho_in, alto_in))

    for m in MAQUINAS:
        y = y_por_maquina[m]
        barras = {"x1": [], "x2": []}
        for inicio, fin, producto, _ in horario[m]:
            barras[producto].append((inicio, fin - inicio))
        for producto, xranges in barras.items():
            if xranges:
                ax.broken_barh(
                    xranges,
                    (y - alto / 2, alto),
                    facecolors=COLORES[producto],
                    edgecolor="none",
                )

        # Marca de inicio de cada unidad (línea vertical más alta que la
        # barra) con su etiqueta "productoX-N" rotada encima. Se alterna
        # una marca corta y una larga para separar visualmente unidades
        # consecutivas cuando quedan muy pegadas en el tiempo.
        for i, (inicio, fin, producto, numero) in enumerate(horario[m]):
            extra = tick_extra_corto if i % 2 == 0 else tick_extra_largo
            techo = y + alto / 2 + extra
            ax.plot(
                [inicio, inicio],
                [y - alto / 2, techo],
                color="black",
                linewidth=0.6,
                zorder=3,
            )
            ax.text(
                inicio,
                techo + espacio_etiqueta,
                f"{producto}-{numero}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=fontsize_etiqueta,
            )

    tope_y = y_por_maquina["M1"] + alto / 2 + tick_extra_largo + 8.8
    piso_y = y_por_maquina["M3"] - alto / 2 - 3.2

    ax.axvline(LIMITE_JORNADA, color="black", linestyle="--", linewidth=1)
    ax.text(
        LIMITE_JORNADA,
        tope_y - 1.2,
        f"{LIMITE_JORNADA} min",
        ha="center",
        fontsize=fontsize_ejes,
    )

    etiquetas = []
    for m in MAQUINAS:
        r = resumen[m]
        etiquetas.append(
            f"{m}\nocioso: {r['ocioso']:.0f} min\nfin: {r['fin']:.0f} min"
        )
    ax.set_yticks([y_por_maquina[m] for m in MAQUINAS])
    ax.set_yticklabels(etiquetas, fontsize=fontsize_ejes)
    ax.set_ylim(piso_y, tope_y)

    ax.set_xlabel("Tiempo (minutos)", fontsize=fontsize_ejes)
    ax.set_xlim(0, resumen["makespan"] * 1.02)
    ax.set_xticks(range(0, int(resumen["makespan"]) + 10, 10))
    ax.tick_params(axis="x", labelsize=fontsize_ejes, rotation=90)

    exceso = resumen["makespan"] - LIMITE_JORNADA
    signo = "+" if exceso >= 0 else ""
    ax.set_title(
        f"{titulo}\nmakespan = {resumen['makespan']:.0f} min "
        f"({signo}{exceso:.0f} min vs jornada de {LIMITE_JORNADA} min)",
        fontsize=fontsize_titulo,
    )

    leyenda = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORES["x1"], label="x1"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORES["x2"], label="x2"),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=fontsize_leyenda)

    fig.tight_layout()
    fig.savefig(archivo_salida, dpi=75)
    plt.close(fig)

    print(f"Gráfico guardado en: {archivo_salida}")
    for m in MAQUINAS:
        r = resumen[m]
        print(
            f"  {m}: ocupado={r['ocupado']:.1f} min  "
            f"ocioso={r['ocioso']:.1f} min  fin={r['fin']:.1f} min"
        )
    print(
        f"  Makespan total: {resumen['makespan']:.1f} min "
        f"(límite jornada: {LIMITE_JORNADA} min)"
    )


if __name__ == "__main__":
    # ---- elegir aquí la secuencia a graficar -----------------------------

    # secuencia = secuencia_por_bloque(53, 14)
    # titulo = "Secuencia por bloque: 53x x1, luego 14x x2"

    secuencia = secuencia_por_ciclo(["x1", "x2"], 53, 14)
    titulo = "Secuencia por ciclo: x1, x2 alternado"

    # -----------------------------------------------------------------------

    horario = calcular_horario(secuencia)
    resumen = calcular_metricas(horario)
    graficar(horario, resumen, titulo)
