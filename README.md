# uss_optimizacion

Script de ejemplo para resolver un problema de programación lineal del libro
*Operations Research: An Introduction* de Hamdy A. Taha, usando `scipy` y `pulp`.

## Problema

Modelo de la empresa Reddy Mikks (mezcla de pinturas), correspondiente al
ejemplo 2.2-1 del libro (la numeración puede variar según la edición).

```
max z = 5*x1 + 4*x2

sujeto a:
6*x1 + 4*x2 <= 24
  x1 + 2*x2 <= 6
 -x1 +   x2 <= 1
           x2 <= 2
      x1, x2 >= 0
```

Solución óptima esperada: x1 = 3, x2 = 1.5, z = 21.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

El script resuelve el problema con `scipy.optimize.linprog` y con `pulp`, y
muestra ambos resultados por pantalla.
