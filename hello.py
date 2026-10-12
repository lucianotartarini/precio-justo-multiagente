# Script mínimo de ejemplo para aprender a desplegar código en Modal
# Modal permite ejecutar funciones Python en la nube (con su propio contenedor/imagen)
# sin tener que gestionar servidores manualmente
import modal
from modal import App, Image

# Setup
# Se crea la "app" de Modal (unidad organizativa de despliegue) con nombre "hello"
app = modal.App("hello")
# Se define la imagen del contenedor: parte de debian_slim y se le instala la librería requests
image = Image.debian_slim().pip_install("requests")

# Hello!

# El decorador @app.function indica que esta función se ejecutará remotamente en Modal,
# usando la imagen definida arriba. Modal serializa la función y la ejecuta en su infraestructura.
@app.function(image=image)
def hello() -> str:
    # El import se hace dentro de la función porque se ejecuta en el contenedor remoto,
    # no en el entorno local
    import requests

    # Llamada a un servicio externo para averiguar la geolocalización del contenedor remoto
    response = requests.get('https://ipinfo.io/json')
    data = response.json()
    city, region, country = data['city'], data['region'], data['country']
    return f"Hola desde {city}, {region}, {country}!!"
