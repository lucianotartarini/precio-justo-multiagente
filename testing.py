# Utilidad de evaluación (reutilizada de semanas anteriores) para medir qué tan bien predice
# precios cualquier función/modelo (frontier, specialist, random forest o el ensemble completo).
# Ejecuta la función predictora sobre un conjunto de datos de prueba, calcula el error de cada
# predicción y muestra un resumen + gráfico de dispersión Predicción vs Realidad.
import math
import matplotlib.pyplot as plt

# Colores de consola para imprimir cada predicción según su nivel de acierto
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

class Tester:

    def __init__(self, predictor, data, title=None, size=250):
        """
        predictor: función que recibe un Item y devuelve el precio estimado
        data: conjunto de datos (Items) de prueba
        size: cuántos puntos del conjunto de datos evaluar
        """
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error, truth):
        """
        Clasifica el acierto de una predicción por color según el error absoluto
        y el error relativo: verde (buena), naranja (aceptable), rojo (mala)
        """
        if error<40 or error/truth < 0.2:
            return "green"
        elif error<80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        """
        Ejecuta la predicción sobre un único punto de datos, calcula el error absoluto
        y el error logarítmico al cuadrado (SLE, para el RMSLE), y lo imprime en consola
        """
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        """
        Dibuja un scatter plot de precio real (eje X) vs precio estimado (eje Y),
        con una línea diagonal de referencia (predicción perfecta) y puntos coloreados por acierto
        """
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        """
        Calcula las métricas agregadas (error medio, RMSLE, % de aciertos "verdes")
        y muestra el gráfico final con esas métricas en el título
        """
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        """
        Recorre los primeros `size` puntos de datos, evaluando y acumulando resultados,
        y termina generando el reporte final
        """
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        """
        Método de conveniencia: crea el Tester y lo ejecuta en una sola línea
        """
        cls(function, data).run()