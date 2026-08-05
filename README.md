# uss_optimizacion

Scripts de ejemplo del curso de optimización.

## Estructura

- `reddy_mikks/main_simple.py` — ejemplo 2.2-1 (Reddy Mikks) de Taha, todo en
  un solo archivo (pulp + gráfico). Pensado como introducción.
- `ejemplos_2d/` — versión modular del mismo ejemplo (`problema.py`,
  `solver_scipy.py`, `solver_pulp.py`, `visualizacion.py`, `main.py`), más
  `banco_problemas.py`: un banco de problemas LP de 2 variables para
  practicar y verificar visualmente (grilla 2x2 con curvas de nivel).

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python reddy_mikks/main_simple.py
python ejemplos_2d/main.py
python ejemplos_2d/banco_problemas.py
```
