"""
Fuerza bruta (enumeración exhaustiva) para un problema de programación
entera pura que crece de dimensión: muestra por qué enumerar la grilla de
puntos enteros no escala y motiva Ramificación y Acotamiento (Unidad_2_Clase,
láminas que siguen a la portadilla "Métodos de resolución").

El caso n = 2 reproduce la figura "Figura-4-Soluciones-de-la-programacion-
lineal-y-entera.png" (lámina "PL vs. PLE: solución entera vs. relajación").
La figura no trae números; estos se reconstruyeron a partir de la grilla y se
verificaron por enumeración:

    max z = x1 + x2
    R1: x2 <= 6
    R2: x1 <= 7
    R3: 2 x1 + 7 x2 <= 48
    R4: 5 x1 + 2 x2 <= 39
    x1, x2 >= 0 enteras

    -> A(3,6), B(4,5), C(5,4), D(6,3), E(7,2) en z = 9
    -> óptimo entero F(5,5) y G(6,4) con z = 10 (empate)
    -> relajación de PL: R = (5.71, 5.23), z = 10.94

Cada variable nueva x_k (k >= 3) agrega 2 restricciones, en cadena con la
anterior (R1..R4 quedan intactas, así el piso x3 = 0 del caso 3D es la figura
original):
    cota:   x_k <= 7 (k impar) o x_k <= 6 (k par)
    acople: 2 x_{k-1} + 5 x_k <= 39 (k impar) o 7 x_{k-1} + 2 x_k <= 48 (k par)
y la función objetivo sigue siendo la suma simple z = x1 + ... + xn (los
empates se muestran a propósito). Con n variables hay 2n restricciones (más
las n de no negatividad).

Uso:
    python main.py tabla <n_desde> <n_hasta>   # agrega/actualiza resultados/tiempos.csv
    python main.py figura2d
    python main.py figura3d
    python main.py curva                       # gráfico semilog a partir del CSV

La enumeración es deliberadamente "honesta": recorre toda la caja
0 <= x_j <= u_j en Python puro (itertools.product), evalúa cada restricción
completa (sin aprovechar que casi todos los coeficientes son cero) y no poda
nada. pulp/HiGHS se usa solo para verificar z* y comparar tiempos.
"""

import csv
import itertools
import math
import operator
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pulp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
RUTA_CSV = os.path.join(CARPETA_RESULTADOS, "tiempos.csv")
COLUMNAS_CSV = [
    "n",
    "restricciones",
    "no_negatividad",
    "vertices",
    "puntos_revisados",
    "factibles",
    "z_opt",
    "empates",
    "optimos",
    "t_fuerza_bruta_s",
    "z_pulp",
    "t_pulp_s",
]

# Puntos rotulados en la figura original (n = 2)
PUNTOS_FIGURA = {
    "A": (3, 6),
    "B": (4, 5),
    "C": (5, 4),
    "D": (6, 3),
    "E": (7, 2),
    "F": (5, 5),
    "G": (6, 4),
}

AZUL = "#4A66AC"  # USSaccent1
NAVY = "#052A4A"  # USSdk2
GRIS = "#595959"  # USSgray
ROJO = "#C00000"

EPS = 1e-9


# --------------------------------------------------------------------------
# Problema
# --------------------------------------------------------------------------


def construir_problema(n):
    """Devuelve {"n", "cotas", "restricciones", "objetivo"} con
    restricciones = lista de (coeficientes, rhs, etiqueta), todas "<=".
    Las cotas x_j <= u_j están incluidas como restricciones (R1, R2, R5, ...)
    y además definen la caja que recorre la fuerza bruta."""
    if n < 2:
        raise ValueError("n >= 2")

    def fila(coefs):
        a = [0] * n
        for j, v in coefs.items():
            a[j] = v
        return tuple(a)

    cotas = [7, 6] + [7 if (k + 1) % 2 == 1 else 6 for k in range(2, n)]
    restricciones = [
        (fila({1: 1}), 6, "R1"),
        (fila({0: 1}), 7, "R2"),
        (fila({0: 2, 1: 7}), 48, "R3"),
        (fila({0: 5, 1: 2}), 39, "R4"),
    ]
    for k in range(2, n):  # k es el índice 0-based de x_{k+1}
        r = len(restricciones) + 1
        restricciones.append((fila({k: 1}), cotas[k], f"R{r}"))
        if (k + 1) % 2 == 1:
            restricciones.append((fila({k - 1: 2, k: 5}), 39, f"R{r + 1}"))
        else:
            restricciones.append((fila({k - 1: 7, k: 2}), 48, f"R{r + 1}"))
    return {
        "n": n,
        "cotas": cotas,
        "restricciones": restricciones,
        "objetivo": (1,) * n,
    }


def _matrices(problema, con_no_negatividad=True):
    """A, b de A x <= b (incluye -x_j <= 0 si se pide)."""
    n = problema["n"]
    A = [list(a) for a, _b, _et in problema["restricciones"]]
    b = [rhs for _a, rhs, _et in problema["restricciones"]]
    if con_no_negatividad:
        for j in range(n):
            fila = [0] * n
            fila[j] = -1
            A.append(fila)
            b.append(0)
    return np.array(A, dtype=float), np.array(b, dtype=float)


# --------------------------------------------------------------------------
# Fuerza bruta
# --------------------------------------------------------------------------


def fuerza_bruta(problema):
    """Enumera toda la caja 0 <= x_j <= u_j (Python puro). Devuelve
    (puntos_revisados, factibles, z_opt, optimos, segundos)."""
    filas = [(a, rhs) for a, rhs, _et in problema["restricciones"]]
    c = problema["objetivo"]
    rangos = [range(u + 1) for u in problema["cotas"]]
    mul = operator.mul

    t0 = time.perf_counter()
    factibles = 0
    z_opt = None
    optimos = []
    for x in itertools.product(*rangos):
        for a, rhs in filas:
            if sum(map(mul, a, x)) > rhs:
                break
        else:
            factibles += 1
            z = sum(map(mul, c, x))
            if z_opt is None or z > z_opt:
                z_opt = z
                optimos = [x]
            elif z == z_opt:
                optimos.append(x)
    segundos = time.perf_counter() - t0

    revisados = math.prod(len(r) for r in rangos)
    return revisados, factibles, z_opt, optimos, segundos


# --------------------------------------------------------------------------
# Poliedro (relajación de PL)
# --------------------------------------------------------------------------


def vertices(problema):
    """Vértices del poliedro factible de la relajación de PL (incluye la no
    negatividad), vía intersección de semiespacios (qhull) desde el centro
    de Chebyshev."""
    A, b = _matrices(problema)
    n = problema["n"]
    normas = np.linalg.norm(A, axis=1)
    res = linprog(
        np.r_[np.zeros(n), -1.0],
        A_ub=np.c_[A, normas],
        b_ub=b,
        bounds=[(None, None)] * n + [(0, None)],
        method="highs",
    )
    centro = res.x[:n]
    hs = HalfspaceIntersection(np.c_[A, -b], centro)
    return np.unique(np.round(hs.intersections, 6), axis=0)


def _modelo_pulp(problema, relajar):
    n = problema["n"]
    modelo = pulp.LpProblem("fuerza_bruta", pulp.LpMaximize)
    cat = "Continuous" if relajar else "Integer"
    x = [pulp.LpVariable(f"x{j + 1}", lowBound=0, cat=cat) for j in range(n)]
    modelo += pulp.lpSum(cj * xj for cj, xj in zip(problema["objetivo"], x)), "z"
    for a, rhs, et in problema["restricciones"]:
        modelo += pulp.lpSum(aj * xj for aj, xj in zip(a, x) if aj) <= rhs, et
    return modelo, x


def resolver_pulp(problema, relajar=False):
    """Resuelve con pulp/HiGHS. Devuelve (x, z, segundos)."""
    modelo, x = _modelo_pulp(problema, relajar)
    t0 = time.perf_counter()
    modelo.solve(pulp.HiGHS(msg=False))
    segundos = time.perf_counter() - t0
    if pulp.LpStatus[modelo.status] != "Optimal":
        raise RuntimeError(pulp.LpStatus[modelo.status])
    return [v.value() for v in x], pulp.value(modelo.objective), segundos


# --------------------------------------------------------------------------
# Tabla de tiempos
# --------------------------------------------------------------------------


def _leer_csv():
    if not os.path.exists(RUTA_CSV):
        return {}
    with open(RUTA_CSV, newline="") as f:
        return {int(fila["n"]): fila for fila in csv.DictReader(f)}


def _escribir_csv(filas):
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    with open(RUTA_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        w.writeheader()
        for n in sorted(filas):
            w.writerow(filas[n])


def tabla(n_desde, n_hasta):
    """Corre la fuerza bruta para cada n y guarda una fila por n en el CSV
    apenas termina (si el proceso se corta, lo anterior queda guardado)."""
    print(f"{'n':>2} {'restr':>5} {'vert':>6} {'revisados':>12} {'factibles':>12} "
          f"{'z*':>4} {'empates':>7} {'t_fb [s]':>10} {'t_pulp [s]':>10}")
    for n in range(n_desde, n_hasta + 1):
        problema = construir_problema(n)
        n_vert = len(vertices(problema))
        revisados, factibles, z_opt, optimos, t_fb = fuerza_bruta(problema)
        _x, z_pulp, t_pulp = resolver_pulp(problema)
        if abs(z_pulp - z_opt) > 1e-6:
            raise RuntimeError(f"n={n}: fuerza bruta z*={z_opt} != pulp z={z_pulp}")
        fila = {
            "n": n,
            "restricciones": len(problema["restricciones"]),
            "no_negatividad": n,
            "vertices": n_vert,
            "puntos_revisados": revisados,
            "factibles": factibles,
            "z_opt": z_opt,
            "empates": len(optimos),
            "optimos": " ".join("(" + ",".join(map(str, x)) + ")" for x in optimos),
            "t_fuerza_bruta_s": f"{t_fb:.6f}",
            "z_pulp": round(z_pulp, 6),
            "t_pulp_s": f"{t_pulp:.6f}",
        }
        filas = _leer_csv()
        filas[n] = fila
        _escribir_csv(filas)
        print(f"{n:>2} {fila['restricciones']:>5} {n_vert:>6} {revisados:>12} {factibles:>12} "
              f"{z_opt:>4} {len(optimos):>7} {t_fb:>10.3f} {t_pulp:>10.3f}   óptimos: {fila['optimos']}",
              flush=True)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------


def _guardar(fig, nombre):
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre)
    fig.savefig(ruta, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("Guardado:", ruta)


def _ordenar_en_plano(puntos, normal):
    """Ordena puntos coplanares alrededor de su centroide."""
    centro = puntos.mean(axis=0)
    normal = normal / np.linalg.norm(normal)
    u = puntos[0] - centro
    if np.linalg.norm(u) < EPS:
        u = puntos[1] - centro
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    ang = np.arctan2((puntos - centro) @ v, (puntos - centro) @ u)
    return puntos[np.argsort(ang)]


def figura2d():
    problema = construir_problema(2)
    V = vertices(problema)
    _revisados, _fact, z_opt, optimos, _t = fuerza_bruta(problema)
    x_lp, z_lp, _ = resolver_pulp(problema, relajar=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    centro = V.mean(axis=0)
    V = V[np.argsort(np.arctan2(V[:, 1] - centro[1], V[:, 0] - centro[0]))]
    ax.fill(V[:, 0], V[:, 1], color=AZUL, alpha=0.12)
    ax.plot(np.r_[V[:, 0], V[0, 0]], np.r_[V[:, 1], V[0, 1]], color=AZUL, lw=1.5)

    # grilla de la caja: factibles llenos, infactibles huecos
    for x in itertools.product(range(8), range(7)):
        ok = all(a[0] * x[0] + a[1] * x[1] <= rhs for a, rhs, _ in problema["restricciones"])
        ax.plot(*x, "o", ms=5, color=AZUL if ok else "none", mec=AZUL if ok else "#BBBBBB")

    # rectas R1..R4
    xs = np.linspace(-0.3, 8.6, 200)
    estilos = dict(color="#2E9E3E", lw=1.8)
    ax.axhline(6, **estilos)
    ax.axvline(7, **estilos)
    ax.plot(xs, (48 - 2 * xs) / 7, **estilos)
    ax.plot(xs, 39 / 2 - 2.5 * xs, **estilos)
    for et, (px, py) in {"R1": (0.15, 6.15), "R2": (7.1, 0.15), "R3": (0.15, 7.0),
                         "R4": (5.0, 7.3)}.items():
        ax.text(px, py, et, color="#2E9E3E", fontsize=12, fontweight="bold")

    # curvas de nivel de z
    for z, estilo in [(9, ":"), (z_opt, "--"), (z_lp, ":")]:
        ax.plot(xs, z - xs, estilo, color=ROJO, lw=1.2)
    for z in (9, z_opt):
        ax.text(8.65, z - 8.6, f"z = {z}", color=ROJO, fontsize=11, va="center")

    for et, (px, py) in PUNTOS_FIGURA.items():
        es_opt = (px, py) in optimos
        ax.plot(px, py, "o", ms=9 if es_opt else 7, color=ROJO)
        ax.text(px - 0.32, py - 0.42, et, fontsize=12, fontweight="bold", color=NAVY)
    ax.plot(*x_lp, "o", ms=9, mfc="white", mec=ROJO, mew=2)
    ax.text(x_lp[0] + 0.15, x_lp[1] + 0.15,
            f"R ({x_lp[0]:.2f}, {x_lp[1]:.2f})\nz = {z_lp:.2f}", fontsize=11, color=ROJO)

    ax.set_xlim(-0.3, 9.7)
    ax.set_ylim(-0.3, 8.0)
    ax.set_xticks(range(10))
    ax.set_yticks(range(9))
    ax.set_xlabel("$x_1$", fontsize=13)
    ax.set_ylabel("$x_2$", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    _guardar(fig, "figura2d.png")


def figura3d():
    problema = construir_problema(3)
    V = vertices(problema)
    _revisados, _fact, z_opt, optimos, _t = fuerza_bruta(problema)
    x_lp, z_lp, _ = resolver_pulp(problema, relajar=True)
    A, b = _matrices(problema)

    # caras del poliedro: agrupar los triángulos de ConvexHull por plano
    hull = ConvexHull(V)
    caras = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        clave = tuple(np.round(eq, 6))
        caras.setdefault(clave, set()).update(simplex.tolist())
    poligonos = [
        _ordenar_en_plano(V[sorted(idx)], np.array(clave[:3])) for clave, idx in caras.items()
    ]

    # intersección del plano x1+x2+x3 = z* con el poliedro: cortar las aristas
    corte = []
    for pol in poligonos:
        for p, q in zip(pol, np.roll(pol, -1, axis=0)):
            sp, sq = p.sum() - z_opt, q.sum() - z_opt
            if abs(sp) < EPS:
                corte.append(p)
            elif sp * sq < 0:
                corte.append(p + (q - p) * sp / (sp - sq))
    corte = np.unique(np.round(corte, 6), axis=0)
    corte = _ordenar_en_plano(corte, np.ones(3))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(
        poligonos, facecolor=AZUL, alpha=0.07, edgecolor=AZUL, linewidth=0.8))
    ax.add_collection3d(Poly3DCollection(
        [corte], facecolor=ROJO, alpha=0.35, edgecolor=ROJO, linewidth=1.5))

    # grilla entera factible
    puntos = np.array([
        x for x in itertools.product(*[range(u + 1) for u in problema["cotas"]])
        if np.all(A @ np.array(x, dtype=float) <= b + EPS)
    ])
    ax.scatter(*puntos.T, s=4, color=AZUL, alpha=0.25, depthshade=False)

    # piso x3 = 0: la figura original
    piso = V[np.abs(V[:, 2]) < EPS]
    piso = _ordenar_en_plano(piso, np.array([0, 0, 1.0]))
    ax.plot(*np.r_[piso, piso[:1]].T, color=NAVY, lw=2)
    for et, (px, py) in PUNTOS_FIGURA.items():
        ax.scatter(px, py, 0, s=18, color=GRIS, depthshade=False)
        ax.text(px + 0.15, py - 0.55, 0, et, fontsize=13, color=NAVY, fontweight="bold")
    for et in ("F", "G"):
        px, py = PUNTOS_FIGURA[et]
        ax.scatter(px, py, 0, s=40, facecolor="none", edgecolor=ROJO, linewidth=1.5,
                   depthshade=False)

    # óptimos 3D y su proyección al piso
    for x in optimos:
        ax.plot([x[0], x[0]], [x[1], x[1]], [0, x[2]], "--", color=ROJO, lw=1.3)
        ax.scatter(*x, s=80, color=ROJO, depthshade=False)
        dx, dy, dz = (0.3, -1.6, 0.6) if x[1] < 3 else (-0.3, 0.4, 0.9)
        ax.text(x[0] + dx, x[1] + dy, x[2] + dz, f"({x[0]},{x[1]},{x[2]})",
                fontsize=15, color=ROJO, fontweight="bold")
    ax.scatter(*x_lp, s=70, facecolor="white", edgecolor=ROJO, linewidth=2, depthshade=False)
    ax.text(x_lp[0] + 0.3, x_lp[1] + 0.6, x_lp[2] + 0.2, "R", fontsize=15, color=ROJO,
            fontweight="bold")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_zlim(0, 8)
    ax.set_xlabel("$x_1$", fontsize=16)
    ax.set_ylabel("$x_2$", fontsize=16)
    ax.set_zlabel("$x_3$", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_box_aspect((8, 7, 8))
    ax.view_init(elev=20, azim=-35)
    _guardar(fig, "figura3d.png")


def curva():
    filas = _leer_csv()
    if not filas:
        print("No hay resultados: correr primero 'python main.py tabla 2 8'")
        return
    ns = sorted(filas)
    t_fb = [float(filas[n]["t_fuerza_bruta_s"]) for n in ns]
    t_pulp = [float(filas[n]["t_pulp_s"]) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(ns, t_fb, "o-", color=ROJO, lw=2, ms=7, label="Fuerza bruta (Python puro)")
    ax.semilogy(ns, t_pulp, "s-", color=AZUL, lw=2, ms=6,
                label="pulp / HiGHS (ramificación y acotamiento)")
    for n, t in zip(ns, t_fb):
        ax.annotate(_fmt_t(t), (n, t), textcoords="offset points", xytext=(8, -4),
                    ha="left", va="top", fontsize=11, color=ROJO)

    # proyección de la siguiente dimensión (si no está medida)
    if len(ns) >= 2:
        factor = t_fb[-1] / t_fb[-2]
        n_sig, t_sig = ns[-1] + 1, t_fb[-1] * factor
        ax.semilogy([ns[-1], n_sig], [t_fb[-1], t_sig], ":", color=ROJO, lw=1.5)
        ax.semilogy(n_sig, t_sig, "o", mfc="white", mec=ROJO, mew=2, ms=7)
        ax.annotate(f"{_fmt_t(t_sig)}\n(proyectado)", (n_sig, t_sig), textcoords="offset points",
                    xytext=(8, -4), ha="left", va="top", fontsize=11, color=ROJO)

    for t, et in [(1, "1 s"), (60, "1 min"), (600, "10 min")]:
        ax.axhline(t, color=GRIS, lw=0.8, ls="--", alpha=0.5)
        ax.text(ns[0] - 0.4, t * 1.15, et, fontsize=10, color=GRIS)
    ax.set_xticks(range(ns[0], ns[-1] + 2))
    ax.set_xlim(ns[0] - 0.5, ns[-1] + 2.0)
    ax.set_xlabel("Número de variables enteras n", fontsize=13)
    ax.set_ylabel("Tiempo [s] (escala logarítmica)", fontsize=13)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=11, loc="lower right")
    _guardar(fig, "curva_tiempos.png")


def _fmt_t(t):
    if t < 1e-3:
        return f"{t * 1e6:.0f} µs"
    if t < 1:
        return f"{t * 1e3:.0f} ms"
    if t < 120:
        return f"{t:.1f} s"
    return f"{t / 60:.1f} min"


def main():
    uso = (
        "Uso:\n"
        "  python main.py tabla <n_desde> <n_hasta>\n"
        "  python main.py figura2d\n"
        "  python main.py figura3d\n"
        "  python main.py curva"
    )
    if len(sys.argv) < 2:
        print(uso)
        return
    comando = sys.argv[1]
    if comando == "tabla" and len(sys.argv) == 4:
        tabla(int(sys.argv[2]), int(sys.argv[3]))
    elif comando == "figura2d":
        figura2d()
    elif comando == "figura3d":
        figura3d()
    elif comando == "curva":
        curva()
    else:
        print(uso)


if __name__ == "__main__":
    main()
