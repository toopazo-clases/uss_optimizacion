# uss_optimizacion

Scripts de ejemplo del curso de Optimización (INGE E003, USS). Todos resuelven con
pulp; los bancos de problemas se usan como fuente de verdad de los números que
aparecen en clases, guías y laboratorios.

## Estructura

- `ejemplo_reddy_mikks/main_simple.py` — ejemplo 2.2-1 (Reddy Mikks) de Taha, todo en
  un solo archivo (pulp + gráfico). Pensado como introducción.
- `pulp_LpProblem_2d/main.py` — banco de problemas de PL de 2 variables. Resuelve con
  pulp (único método usado para encontrar el óptimo) y grafica región factible +
  curvas de nivel + óptimo; también calcula la sensibilidad (precio sombra y rango
  del lado derecho) con HiGHS. Requiere indicar qué problema del banco correr.
- `pulp_LpProblem_3d/main.py` — banco de problemas de PL de 3 variables: óptimo, precio
  sombra y su rango de validez, y costo reducido de cada variable. No grafica.
- `pulp_LpProblem_entera/main.py` — banco de problemas de programación entera
  (binaria, entera pura y mixta; Hillier & Lieberman, cap. 11): óptimo entero,
  relajación de PL, verificación por enumeración exhaustiva y, para el ejemplo
  prototipo (California Manufacturing), el árbol de ramificación y acotamiento.
  Incluye un ejemplo de cada técnica de la sección 11.3 (alternativas
  mutuamente excluyentes, K de N restricciones, función con N valores
  posibles, costo fijo, representación binaria de enteros) y de decisiones
  contingentes (sección 11.1); usado en Unidad_2_Clase y en
  Guia_5_Programacion_Entera_Binaria_Mixta.
- `fuerzabruta_LpProblem_entera/main.py` — enumeración exhaustiva (fuerza bruta, Python
  puro) del problema entero de la figura "PL vs. PLE" de Unidad_2_Clase, extendido a n
  variables (2 restricciones más por variable, z = x1 + ... + xn). Genera la tabla de
  tiempos (restricciones, vértices, puntos revisados, soluciones factibles, empates,
  tiempo; verificada contra pulp/HiGHS), las figuras 2D y 3D y la curva de tiempos.
- `agregación_vs_secuencia/main.py` — diagrama de Gantt de una línea de flujo de 3
  máquinas (Taha 2.2A-4): muestra que el LP de capacidad agregada no garantiza que
  exista una secuencia de fabricación factible.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python ejemplo_reddy_mikks/main_simple.py

python pulp_LpProblem_2d/main.py                       # sin argumento: lista los problemas disponibles
python pulp_LpProblem_2d/main.py reddy_mikks           # resuelve y grafica ese problema

python pulp_LpProblem_3d/main.py florista_ambulante_3d # óptimo, sensibilidad y costo reducido

python pulp_LpProblem_entera/main.py california_manufacturing

python fuerzabruta_LpProblem_entera/main.py tabla 2 8   # tiempos -> resultados/tiempos.csv
python fuerzabruta_LpProblem_entera/main.py figura3d    # también: figura2d, curva, curva_puntos

python agregación_vs_secuencia/main.py                 # editar la secuencia al final del archivo
```
