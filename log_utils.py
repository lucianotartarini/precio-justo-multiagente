# Utilidad para convertir los códigos de color ANSI (usados en los logs de consola de los agentes,
# ver agents/agent.py) en HTML, de modo que los logs se puedan mostrar con color dentro de la
# interfaz Gradio (ver price_is_right_final.py, que usa reformat() para pintar el panel de logs)

# Foreground colors (códigos ANSI de color de texto para consola)
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# Background color
BG_BLACK = '\033[40m'
BG_BLUE = '\033[44m'

# Reset code to return to default color
RESET = '\033[0m'

# Mapa que traduce cada combinación de color ANSI (fondo+texto) usada por los agentes
# a un color hexadecimal equivalente para HTML/CSS
mapper = {
    BG_BLACK+RED: "#dd0000",
    BG_BLACK+GREEN: "#00dd00",
    BG_BLACK+YELLOW: "#dddd00",
    BG_BLACK+BLUE: "#0000ee",
    BG_BLACK+MAGENTA: "#aa00dd",
    BG_BLACK+CYAN: "#00dddd",
    BG_BLACK+WHITE: "#87CEEB",
    BG_BLUE+WHITE: "#ff7800"
}


def reformat(message):
    """
    Reemplaza los códigos de color ANSI de un mensaje de log por etiquetas <span> con estilo CSS,
    de modo que se pueda insertar directamente en un componente HTML de Gradio
    """
    for key, value in mapper.items():
        message = message.replace(key, f'<span style="color: {value}">')
    message = message.replace(RESET, '</span>')
    return message
    
    