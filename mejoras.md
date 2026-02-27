# Mejoras del Proyecto

## Ampliaci├│n de l├¡mites de trades para el backtest
Se aument├│ el l├¡mite m├íximo de d├¡as permitidos para ejecutar un backtest de `500` a `100,000`. 
Esto se realiz├│ modificando la variable `MAX_DAYS` en el archivo `backend/routers/backtest.py`, evitando as├¡ el error `HTTPException 400` que bloqueaba la ejecuci├│n con datasets m├ís extensos.

## Curva PNL b├ísica corregida
Se reescribi├│ la funci├│n `_compute_global_equity_and_drawdown` en `backend/services/backtest_service.py` para calcular correctamente la equity curve como P&L acumulado diario:
1. Se parte del capital inicial (`init_cash`).
2. Se agrupan los trades por fecha y se suma el P&L de cada d├¡a.
3. Se acumula d├¡a a d├¡a: `equity[hoy] = equity[ayer] + pnl_de_hoy`.

Adem├ís, se corrigi├│ el eje X de los gr├íficos de Equity Curve y Drawdown (`EquityCurveTab.tsx`, `DrawdownTab.tsx`) para mostrar **fechas reales** del dataset en lugar de timestamps ficticios secuenciales. Se activ├│ `timeVisible: true` y se elimin├│ c├│digo de debug residual de sesiones anteriores.

## PNL junto con DrawDown
Se ha unificado la visualizaci├│n del rendimiento en una sola pesta├▒a mejorada:
1. **Integraci├│n**: El gr├ífico de Drawdown se ha movido dentro de la pesta├▒a "Equity Curve", situ├índose justo debajo de la curva de capital.
2. **Dimensiones**: El cuadro de Drawdown se ajust├│ a un tama├▒o menor (1/3 del principal) para optimizar el espacio.
3. **Sincronizaci├│n Total**: Se implement├│ una sincronizaci├│n bidireccional de scroll y cursor (crosshair). Al interactuar con cualquiera de los dos gr├íficos, el otro se desplaza y marca el punto temporal exacto de forma simult├ínea.
4. **Simplificaci├│n**: Se elimin├│ la pesta├▒a redundante de Drawdown y se optimizaron las importaciones del componente para mejorar la estabilidad con Turbopack.

## definici├│n de R y aplicaci├│n de distintas visiones del equity curve (R, % y $)
Se ha implementado un sistema para gestionar el riesgo de las operaciones de manera absoluta y visualizar el rendimiento bajo diferentes perspectivas:
1. **Riesgo 1R Parametrizable**: Se ha a├▒adido un nuevo campo `Riesgo 1R ($)` en el panel de Configuraci├│n (por defecto 100$). En el backend, el motor de simulaci├│n (`portfolio_sim.py`) se ha actualizado para invertir por operaci├│n el m├¡nimo entre el capital disponible y el riesgo R especificado.
2. **Nuevos Modos de Visualizaci├│n del Eje Y**: En los gr├íficos de Equity Curve y Drawdown, se ha a├▒adido un panel de control con 3 botones que calculan y transforman los valores en tiempo real:
   - **[$] (D├│lares)**: Muestra el valor absoluto de la cuenta y el drawdown en d├│lares.
   - **[%] (Porcentaje)**: Muestra el crecimiento relativo de la cuenta y la ca├¡da en porcentaje.
   - **[R] (M├║ltiplos de R)**: Divide las ganancias, p├®rdidas y el drawdown por el `Riesgo 1R` definido, mostrando cu├íntas unidades de riesgo se han ganado o perdido.

## Cambio de la pesta├▒a preformance
Se ha redise├▒ado por completo la pesta├▒a de "Performance" para ofrecer una visualizaci├│n mucho m├ís anal├¡tica, compacta y profesional del rendimiento de la estrategia:

1. **Matriz de Rendimiento Mensual**:
   - Se ha implementado una tabla que cruza **A├▒os (eje Y)** y **Meses (eje X)**, a├▒adiendo una columna final para el acumulado anual (**YTD**).
   - **Selector Din├ímico**: Se a├▒adi├│ un panel superior que permite cambiar instant├íneamente la m├®trica calculada que se muestra dentro de cada celdilla de la cuadr├¡cula. Las opciones disponibles son: `PnL %`, `PnL $`, `PnL R`, `Win Rate`, `Trades` y `Profit Factor`.
   - **Coloraci├│n Inteligente**: Las celdas se iluminan din├ímicamente en tonos verdes o rojos dependiendo de si el mes fue positivo o negativo seg├║n la m├®trica elegida (incluyendo el Win Rate y el Profit Factor).

2. **Gr├ífico Combinado de Evoluci├│n Temporal**:
   - Debajo de la cuadr├¡cula, se reemplazaron las tablas antiguas por un ├║nico gr├ífico avanzado de *lightweight-charts* que resume el rendimiento temporal.
   - **Ejes Duales (Doble Eje Y)**: Para evitar que valores muy dispares se solapen o compriman, el gr├ífico utiliza m├║ltiples ejes.
   - **Trades (Histograma Izquierdo)**: Muestra en formato de barras el volumen de operaciones mensuales.
   - **Win Rate (L├¡nea Derecha Visible)**: Muestra la evoluci├│n del % de acierto mensual (escala de 0 a 100).
   - **Profit Factor (L├¡nea Derecha Invisible)**: Muestra la m├®trica de Beneficio/P├®rdida con su propia escala adaptada para convivir armoniosamente junto a la l├¡nea del Win Rate sin colapsarla.
   - Se ha fijado la temporalidad de este gr├ífico estrictamente en **Mensual** para alinearse de manera coherente con la matriz superior.
