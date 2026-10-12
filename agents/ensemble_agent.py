# El "ensemble" es la pieza que combina las tres estrategias de estimación de precio del
# proyecto (fine-tuning open-source con QLoRA -> SpecialistAgent, RAG + modelo Frontier ->
# FrontierAgent, ML clásico -> RandomForestAgent) en una única predicción final, en vez de
# quedarse con una sola. La idea de un ensemble es que combinar varios modelos "débiles" o
# distintos entre sí suele dar una predicción más robusta que confiar en uno solo.
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

class EnsembleAgent(Agent):

    name = "Agente de Conjunto"
    color = Agent.YELLOW
    
    def __init__(self, collection):
        """
        Crea una instancia de Ensemble, creando cada uno de los modelos
        y cargando los pesos del Ensemble
        """
        self.log("Inicializando Agente de Conjunto")
        # Cada uno de los tres modelos base se inicializa como su propio agente: el Ensemble
        # los orquesta pero no reimplementa su lógica interna
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.random_forest = RandomForestAgent()
        # ensemble_model.pkl es una regresión lineal ya entrenada (en un notebook aparte, no
        # mostrado aquí) que aprendió los PESOS óptimos para combinar las 3 predicciones + min/max
        self.model = joblib.load('ensemble_model.pkl')
        self.log("Agente de Conjunto listo")

    def price(self, description: str) -> float:
        """
        Ejecuta este modelo de conjunto
        Pide a cada uno de los modelos que fije el precio del producto
        Luego, utiliza el modelo de regresión lineal para obtener el precio ponderado
        :param description: la descripción de un producto
        :return: una estimación de su precio
        """
        self.log("Ejecutando Agente de Ensemble: colaboración con agentes de bosque aleatorio, especialistas y fronterizos")
        # Se piden las 3 estimaciones de forma secuencial (podría paralelizarse, pero así es más simple)
        specialist = self.specialist.price(description)
        frontier = self.frontier.price(description)
        random_forest = self.random_forest.price(description)
        # Min y Max se añaden como features extra: le dan a la regresión lineal información sobre
        # cuánto "discrepan" los 3 modelos entre sí, no solo sus valores individuales
        X = pd.DataFrame({
            'Specialist': [specialist],
            'Frontier': [frontier],
            'RandomForest': [random_forest],
            'Min': [min(specialist, frontier, random_forest)],
            'Max': [max(specialist, frontier, random_forest)],
        })
        # La regresión lineal ya entrenada combina estas 5 features en una única predicción final
        y = self.model.predict(X)[0]
        self.log(f"El Agente de Conjunto ha terminado - predicción ${y:.2f}")
        return y