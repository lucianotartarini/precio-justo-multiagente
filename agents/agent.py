# Clase base de la que heredan TODOS los agentes del sistema (ScannerAgent, PlanningAgent,
# EnsembleAgent, FrontierAgent, SpecialistAgent, RandomForestAgent, MessagingAgent).
# No implementa lógica de negocio: solo estandariza cómo cada agente se identifica en los logs
# (nombre + color), lo que permite seguir en consola/UI qué agente está haciendo qué en cada
# momento del pipeline (scanner -> planning -> ensemble [specialist+frontier+random_forest] -> messaging).
import logging

class Agent:
    """
    Una superclase abstracta para agentes
    Se utiliza para registrar mensajes de forma que se pueda identificar a cada agente
    """

    # Foreground colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background color
    BG_BLACK = '\033[40m'
    
    # Código de reinicio para volver al color predeterminado
    RESET = '\033[0m'

    name: str = ""
    color: str = '\033[37m'

    def log(self, message):
        """
        Registre esto como un mensaje de información, identificando al agente
        """
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)