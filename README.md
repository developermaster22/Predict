# BTC/EUR Quant Bot

Bot de analisis tecnico y simulacion para `BTC/EUR` con:

- datos historicos de Binance
- indicadores tecnicos
- prediccion Monte Carlo a 30 dias
- backtesting de 1 ano
- interfaz web con Flask
- integracion opcional de noticias via CryptoPanic o NewsAPI
- traduccion de titulares al espanol
- despliegue compatible con Vercel

## Requisitos

- Python 3.12 o similar
- acceso a internet
- entorno virtual recomendado

## Instalacion

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Ejecucion

### Interfaz web local

```bash
cd /home/sm-des/Predict
.venv/bin/flask --app app run
```

Luego abre en el navegador:

```text
http://127.0.0.1:5000
```

### Modo terminal

```bash
cd /home/sm-des/Predict
.venv/bin/python predict.py
```

## Parametros de entrada

Puedes cargarlos desde la interfaz web:

- `Par a analizar`: por defecto `BTCEUR`
- `Binance API Key`: opcional
- `Binance API Secret`: opcional
- `Capital inicial (EUR)`
- `Porcentaje de capital por operacion`
- `Proveedor de noticias`: `none`, `cryptopanic`, `newsapi`
- `API Key noticias`: opcional, segun proveedor

## Uso de Binance

La API de Binance no es obligatoria para:

- descargar velas historicas
- consultar el precio actual

Eso funciona con endpoints publicos.

La `API Key` y `Secret` quedan preparadas para una futura integracion de:

- saldo de cuenta
- ordenes reales
- trading automatico

## Noticias

### CryptoPanic

El proyecto soporta CryptoPanic.

Debes:

1. crear una cuenta
2. obtener tu `auth_token`
3. elegir `cryptopanic` en la interfaz
4. pegar la key en `API Key noticias`

Nota importante:

- CryptoPanic anuncio que el plan gratis `DEVELOPER` se elimina el `1 de abril de 2026`
- segun su documentacion mostrada en tu cuenta, los planes pagos indican historial de `1 mes`

## Traduccion de noticias

Los titulares se traducen al espanol para visualizacion.

- `title`: titular original
- `title_es`: traduccion al espanol

El sentimiento sigue evaluandose sobre el titular original para no perder contexto.

## Salidas del sistema

Al ejecutar el analisis se generan:

- `btc_eur_trading_system.html`: grafico interactivo
- `btc_eur_operaciones.csv`: operaciones simuladas
- `btc_eur_equity_curve.csv`: curva de equity
- `btc_eur_noticias.csv`: noticias descargadas y traducidas, si aplica

## Estructura del proyecto

- `app.py`: interfaz web Flask
- `predict.py`: entrada principal del flujo
- `config.py`: configuracion general
- `data_loader.py`: carga de datos desde Binance
- `indicators.py`: indicadores y Monte Carlo
- `signals.py`: scoring y senales
- `backtest.py`: simulacion y metricas
- `news.py`: carga y traduccion de noticias
- `charting.py`: visualizacion Plotly

## Dependencias

Instaladas desde `requirements.txt`:

- `python-binance`
- `pandas`
- `numpy`
- `plotly`
- `requests`
- `pandas_ta`
- `flask`
- `deep-translator`

## Notas

- El backtest actual usa `1 ano`
- El historico de precios consulta hasta `5 anos`
- Si cambias reglas de senales o stops, debes volver a ejecutar el analisis
- En Vercel, los archivos exportados se escriben en `/tmp` por restricciones del entorno serverless

## Deploy en Vercel

El proyecto incluye [vercel.json](/home/sm-des/Predict/vercel.json) para desplegar la app Flask sin cambiar la logica cuantitativa.

Pasos:

1. sube este repo a GitHub
2. importa el repo en Vercel
3. Vercel detectara el runtime Python por `vercel.json`
4. si necesitas claves por defecto, configurarlas como variables de entorno en Vercel

Variables utiles:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `CRYPTOPANIC_API_KEY`
- `NEWSAPI_API_KEY`
