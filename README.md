# uss_optimizacion

Scripts de ejemplo del curso de optimización.

## Estructura

- `reddy_mikks/main_simple.py` — ejemplo 2.2-1 (Reddy Mikks) de Taha, todo en
  un solo archivo (pulp + gráfico). Pensado como introducción.
- `ejemplos_2d/main.py` — banco de problemas LP de 2 variables. Resuelve con
  pulp (único método usado para encontrar el óptimo) y grafica región
  factible + curvas de nivel + óptimo. Requiere indicar qué problema del
  banco correr.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python reddy_mikks/main_simple.py

python ejemplos_2d/main.py                # sin argumento: lista los problemas disponibles
python ejemplos_2d/main.py reddy_mikks    # resuelve y grafica ese problema
```
