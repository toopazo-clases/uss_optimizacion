# uss_optimizacion

Scripts de ejemplo del curso de optimización.

Ejemplo 2.2-1 (Reddy Mikks) de Taha, resuelto con `scipy` y `pulp`, con
visualización 3D interactiva (`plotly`).

## Estructura

- `problema.py` — modelo (restricciones, objetivo, vértices).
- `solver_scipy.py` — resolución con `scipy.optimize.linprog`.
- `solver_pulp.py` — resolución con `pulp`.
- `visualizacion_3d.py` — gráfico 3D interactivo.
- `main.py` — punto de entrada: corre todo lo anterior.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
