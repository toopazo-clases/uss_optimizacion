"""
Ramificación y acotamiento paso a paso sobre el problema 2D de la figura
"PL vs. PLE" de Unidad_2_Clase (el mismo de fuerzabruta_LpProblem_entera):

    max z = x1 + x2
    R1: x2 <= 6
    R2: x1 <= 7
    R3: 2 x1 + 7 x2 <= 48
    R4: 5 x1 + 2 x2 <= 39
    x1, x2 >= 0 enteras

Genera el material del "Ejemplo 2" de la clase y de
Guias/Guia_6_Ramificación_y_acotamiento:

1. El árbol de referencia, con las reglas que se les dan a los estudiantes:
     - se ramifica la PRIMERA variable fraccionaria (x1 antes que x2), en
       x_j <= floor(v) (hijo "-1") y x_j >= floor(v) + 1 (hijo "-2");
     - los dos hijos se resuelven (acotamiento) y se prueban APENAS se
       crean, primero el hijo "-1"; las pruebas se aplican en este orden:
         1) infactible                         -> podar (rojo)
         2) Z <= mejor solución entera conocida -> podar por cota (amarillo)
         3) solución entera                    -> podar (verde); pasa a ser
                                                  la mejor conocida
       si ninguna aplica, el nodo queda abierto. Cuando cambia la mejor
       conocida, se vuelve a aplicar la prueba 2 a los nodos abiertos;
     - se ramifica el nodo abierto de MEJOR COTA (mayor Z; si empatan, el
       creado primero);
     - la cota es el Z de la relajación, sin redondear.
2. La "tabla de subproblemas": TODAS las cajas de cotas a las que se puede
   llegar ramificando en cualquier variable fraccionaria y en cualquier
   orden (sin podar), cada una con su relajación resuelta con pulp/HiGHS.
   Los estudiantes no resuelven las relajaciones: plantean el subproblema
   (cotas acumuladas) y leen su solución en esta tabla. Las filas van
   ordenadas por cotas, no por el árbol, y la mayoría no se usa en el
   árbol de referencia (sirven, p. ej., para "¿y si se ramifica en x2?").

Uso:
    python main.py arbol   # árbol de referencia, paso a paso
    python main.py tabla   # tabla de subproblemas (y resultados/*.csv, *.tex)
"""

import csv
import math
import os
import sys

import pulp

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")

OBJETIVO = (1, 1)
RESTRICCIONES = [
    ((0, 1), 6, "R1"),
    ((1, 0), 7, "R2"),
    ((2, 7), 48, "R3"),
    ((5, 2), 39, "R4"),
]
N = 2
EPS = 1e-6

ROJO, AMARILLO, VERDE = "infactible", "por cota", "entera"


# --------------------------------------------------------------------------
# Relajación de un subproblema (caja de cotas lo <= x <= hi)
# --------------------------------------------------------------------------


def relajacion(lo, hi):
    """Resuelve la relajación de PL con cotas lo[j] <= x_j <= hi[j]
    (hi[j] = None: sin cota superior). Devuelve None si es infactible, o
    (x, z) con valores redondeados a 6 decimales."""
    modelo = pulp.LpProblem("subproblema", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{j + 1}", lowBound=lo[j], upBound=hi[j]) for j in range(N)]
    modelo += pulp.lpSum(c * v for c, v in zip(OBJETIVO, x))
    for a, rhs, et in RESTRICCIONES:
        modelo += pulp.lpSum(aj * v for aj, v in zip(a, x)) <= rhs, et
    modelo.solve(pulp.HiGHS(msg=False))
    if pulp.LpStatus[modelo.status] != "Optimal":
        return None
    valores = tuple(round(v.value(), 6) + 0.0 for v in x)
    return valores, round(pulp.value(modelo.objective), 6)


def es_entero(v):
    return abs(v - round(v)) < EPS


def fraccionarias(x):
    return [j for j in range(N) if not es_entero(x[j])]


def hijos(lo, hi, j, v):
    """Cajas de los dos hijos al ramificar x_j = v: x_j <= floor(v) y
    x_j >= floor(v) + 1."""
    f = math.floor(v)
    hi1 = list(hi)
    hi1[j] = f
    lo2 = list(lo)
    lo2[j] = f + 1
    return (tuple(lo), tuple(hi1)), (tuple(lo2), tuple(hi))


# --------------------------------------------------------------------------
# Árbol de referencia
# --------------------------------------------------------------------------


def ramificacion_y_acotamiento():
    """Devuelve (nodos, pasos, incumbente). nodos: dict nombre -> datos;
    pasos: lista de textos en orden; incumbente: nombre del nodo."""
    nodos = {}
    pasos = []
    abiertos = []
    incumbente = None
    orden = [0]

    def crear(nombre, padre, agregada, lo, hi):
        r = relajacion(lo, hi)
        nodo = {
            "nombre": nombre, "padre": padre, "agregada": agregada,
            "lo": lo, "hi": hi, "x": r[0] if r else None, "z": r[1] if r else None,
            "estado": None, "orden": orden[0],
        }
        orden[0] += 1
        nodos[nombre] = nodo
        return nodo

    def probar(nodo):
        nonlocal incumbente
        txt = f"{nodo['nombre']} ({describir_cotas(nodo['lo'], nodo['hi'])}): "
        if nodo["x"] is None:
            nodo["estado"] = ROJO
            pasos.append(txt + "infactible -> podar (rojo)")
            return
        txt += f"x = {fmt_x(nodo['x'])}, Z = {fmt(nodo['z'])}"
        if incumbente and nodo["z"] <= nodos[incumbente]["z"] + EPS:
            nodo["estado"] = AMARILLO
            pasos.append(txt + f" <= {fmt(nodos[incumbente]['z'])} -> podar por cota (amarillo)")
            return
        if not fraccionarias(nodo["x"]):
            nodo["estado"] = VERDE
            incumbente = nodo["nombre"]
            pasos.append(txt + " entera -> podar (verde); nueva mejor conocida")
            for otro in list(abiertos):
                if nodos[otro]["z"] <= nodo["z"] + EPS:
                    abiertos.remove(otro)
                    nodos[otro]["estado"] = AMARILLO
                    pasos.append(f"  {otro}: Z = {fmt(nodos[otro]['z'])} <= {fmt(nodo['z'])}"
                                 " -> podar por cota (amarillo)")
            return
        abiertos.append(nodo["nombre"])
        pasos.append(txt + " -> queda abierto")

    raiz = crear("P", None, None, (0,) * N, (None,) * N)
    probar(raiz)
    while abiertos:
        elegido = max(abiertos, key=lambda n: (nodos[n]["z"], -nodos[n]["orden"]))
        abiertos.remove(elegido)
        padre = nodos[elegido]
        j = fraccionarias(padre["x"])[0]
        v = padre["x"][j]
        padre["estado"] = f"ramificado en x{j + 1}"
        pasos.append(f"Ramificar {elegido} (mejor cota, Z = {fmt(padre['z'])}) en x{j + 1} = {fmt(v)}")
        base = "S" if elegido == "P" else elegido + "-"
        caja1, caja2 = hijos(padre["lo"], padre["hi"], j, v)
        f = math.floor(v)
        h1 = crear(f"{base}1", elegido, f"x{j + 1} <= {f}", *caja1)
        h2 = crear(f"{base}2", elegido, f"x{j + 1} >= {f + 1}", *caja2)
        probar(h1)
        probar(h2)
    return nodos, pasos, incumbente


# --------------------------------------------------------------------------
# Tabla de subproblemas (todas las cajas alcanzables, sin podar)
# --------------------------------------------------------------------------


def todas_las_cajas():
    """Cierre de la ramificación: desde la raíz, se ramifica cada caja
    factible y fraccionaria en CADA una de sus variables fraccionarias.
    Devuelve dict caja -> relajación (o None si infactible)."""
    raiz = ((0,) * N, (None,) * N)
    cajas = {}
    pendientes = [raiz]
    while pendientes:
        caja = pendientes.pop()
        if caja in cajas:
            continue
        r = relajacion(*caja)
        cajas[caja] = r
        if r is None:
            continue
        for j in fraccionarias(r[0]):
            pendientes.extend(hijos(caja[0], caja[1], j, r[0][j]))
    return cajas


def distractores(cajas):
    """Cajas que salen de errores típicos al plantear un subproblema, para
    que la tabla no regale el árbol y un planteamiento equivocado lleve a
    una fila que existe (con otros números):
      - redondear al revés: x_j <= ceil(v) o x_j >= floor(v);
      - olvidar una restricción heredada: quitar las cotas de una variable."""
    nuevas = set()
    for (lo, hi), r in cajas.items():
        if r is not None:
            for j in fraccionarias(r[0]):
                f = math.floor(r[0][j])
                hi1 = list(hi)
                hi1[j] = f + 1
                lo2 = list(lo)
                lo2[j] = f
                nuevas.add((tuple(lo), tuple(hi1)))
                nuevas.add((tuple(lo2), tuple(hi)))
        acotadas = [j for j in range(N) if lo[j] != 0 or hi[j] is not None]
        if len(acotadas) > 1:
            for j in acotadas:
                lo3, hi3 = list(lo), list(hi)
                lo3[j], hi3[j] = 0, None
                nuevas.add((tuple(lo3), tuple(hi3)))
    return {c: relajacion(*c) for c in nuevas if c not in cajas}


def clave_orden(caja):
    lo, hi = caja
    return tuple(v for j in range(N) for v in (lo[j], math.inf if hi[j] is None else hi[j]))


def tabla():
    cajas = todas_las_cajas()
    cajas.update(distractores(cajas))
    filas = []
    for i, caja in enumerate(sorted(cajas, key=clave_orden), start=1):
        r = cajas[caja]
        lo, hi = caja
        filas.append({
            "id": f"T{i:02d}",
            "cotas": [cota_variable(j, lo[j], hi[j]) for j in range(N)],
            "x": r[0] if r else None,
            "z": r[1] if r else None,
            "caja": caja,
        })
    return filas


# --------------------------------------------------------------------------
# Formato
# --------------------------------------------------------------------------


def fmt(v):
    if es_entero(v):
        return str(int(round(v)))
    return f"{v:.2f}"


def fmt_x(x):
    return "(" + ", ".join(fmt(v) for v in x) + ")"


def cota_variable(j, lo, hi):
    """Texto de las cotas de x_{j+1}: '—' si no tiene cotas adicionales."""
    nombre = f"x{j + 1}"
    if lo == 0 and hi is None:
        return "—"
    if hi is not None and lo == hi:
        return f"{nombre} = {lo}"
    if lo == 0:
        return f"{nombre} <= {hi}"
    if hi is None:
        return f"{nombre} >= {lo}"
    return f"{lo} <= {nombre} <= {hi}"


def describir_cotas(lo, hi):
    partes = [c for c in (cota_variable(j, lo[j], hi[j]) for j in range(N)) if c != "—"]
    return ", ".join(partes) if partes else "sin cotas adicionales"


def a_latex(texto):
    texto = texto.replace("<=", r"\le").replace(">=", r"\ge")
    texto = texto.replace("x1", "x_1").replace("x2", "x_2")
    return "$" + texto + "$" if texto != "—" else "---"


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------


def comando_arbol():
    nodos, pasos, incumbente = ramificacion_y_acotamiento()
    print("== Pasos ==")
    for p in pasos:
        print(p)
    print("\n== Nodos (en orden de creación) ==")
    print(f"{'nodo':10} {'agregada':10} {'cotas acumuladas':22} {'x':16} {'Z':7} estado")
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        x = fmt_x(n["x"]) if n["x"] else "—"
        z = fmt(n["z"]) if n["z"] is not None else "—"
        print(f"{n['nombre']:10} {n['agregada'] or '—':10} {describir_cotas(n['lo'], n['hi']):22} "
              f"{x:16} {z:7} {n['estado']}")
    inc = nodos[incumbente]
    print(f"\nÓptimo: {incumbente}, x = {fmt_x(inc['x'])}, Z = {fmt(inc['z'])}"
          f" ({len(nodos)} nodos)")

    # cada nodo del árbol debe estar en la tabla de subproblemas
    ids = {f["caja"]: f["id"] for f in tabla()}
    print("\n== Fila de la tabla de cada nodo ==")
    for n in sorted(nodos.values(), key=lambda n: n["orden"]):
        print(f"{n['nombre']:10} -> {ids[(n['lo'], n['hi'])]}")


def comando_tabla():
    filas = tabla()
    usados = {(n["lo"], n["hi"]) for n in ramificacion_y_acotamiento()[0].values()}
    print(f"{'id':4} {'cotas x1':14} {'cotas x2':14} {'Z':7} {'x1':6} {'x2':6} en árbol")
    for f in filas:
        x1, x2 = (fmt(f["x"][0]), fmt(f["x"][1])) if f["x"] else ("—", "—")
        z = fmt(f["z"]) if f["z"] is not None else "infactible"
        marca = "*" if f["caja"] in usados else ""
        print(f"{f['id']:4} {f['cotas'][0]:14} {f['cotas'][1]:14} {z:10} {x1:6} {x2:6} {marca}")
    print(f"\n{len(filas)} subproblemas; {len(usados)} aparecen en el árbol de referencia (*)")

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta_csv = os.path.join(CARPETA_RESULTADOS, "tabla_subproblemas.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "cotas_x1", "cotas_x2", "z", "x1", "x2"])
        for f in filas:
            if f["x"]:
                w.writerow([f["id"], *f["cotas"], fmt(f["z"]), fmt(f["x"][0]), fmt(f["x"][1])])
            else:
                w.writerow([f["id"], *f["cotas"], "infactible", "", ""])
    ruta_tex = os.path.join(CARPETA_RESULTADOS, "tabla_subproblemas.tex")
    with open(ruta_tex, "w", encoding="utf-8") as fh:
        fh.write("% Generado por uss_optimizacion/ramificacion_acotamiento_2d/main.py tabla\n")
        for f in filas:
            c1, c2 = (a_latex(c) for c in f["cotas"])
            if f["x"]:
                resto = f"{fmt(f['z'])} & {fmt(f['x'][0])} & {fmt(f['x'][1])}"
            else:
                resto = r"\multicolumn{3}{c}{infactible}"
            fh.write(f"{f['id']} & {c1} & {c2} & {resto} \\\\\n")
    print("Guardado:", ruta_csv)
    print("Guardado:", ruta_tex)


def main():
    comandos = {"arbol": comando_arbol, "tabla": comando_tabla}
    if len(sys.argv) != 2 or sys.argv[1] not in comandos:
        print("Uso: python main.py arbol | tabla")
        return
    comandos[sys.argv[1]]()


if __name__ == "__main__":
    main()
