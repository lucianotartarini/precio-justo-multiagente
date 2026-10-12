# Script utilitario para mantener "caliente" el servicio Modal que sirve el modelo fine-tuneado.
# Los contenedores de Modal se "duermen" tras un periodo de inactividad; la primera llamada tras
# dormirse (cold start) es lenta porque hay que descargar y cargar el modelo en GPU. Este script
# hace ping periódicamente para evitar esos cold starts durante demos o pruebas en vivo.
import time
import modal
from datetime import datetime

# Se obtiene una referencia a la clase remota "Pricer" ya desplegada en la app "pricer-service"
Pricer = modal.Cls.lookup("pricer-service", "Pricer")
pricer = Pricer()
# Bucle infinito: cada 30 segundos invoca el método remoto wake_up para mantener el contenedor activo
while True:
    reply = pricer.wake_up.remote()
    print(f"{datetime.now()}: {reply}")
    time.sleep(30)