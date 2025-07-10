# fast_forecast

Modelos rápidos de pronóstico lineal con tendencia (incluye LOWESS), estacionalidad y variables exógenas.

## Ejemplo

```python
from fast_forecast import mco_forecast

mco = mco_forecast(tendencia="lowess", estacionalidad=['dayofweek'], log_y=False)
mco.fit(data, y_col='MX', exog=['MX_festivo'])
preds = mco.forecast(14, exog=exog_futuro)
````
