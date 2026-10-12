# imports
# Este agente es el baseline de "ML clásico" del ensemble (equivalente al Random Forest sobre
# embeddings word2vec de la semana 6, pero aquí usando embeddings de SentenceTransformer)

import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent

class RandomForestAgent(Agent):

    name = "Agente de Bosque Aleatorio"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Inicialice este objeto cargando los pesos del modelo guardado
        y el modelo de codificación vectorial SentenceTransformer
        """
        self.log("El Agente de Bosque Aleatorio se está inicializando")
        # Mismo modelo de embeddings (MiniLM) que usa FrontierAgent para la búsqueda RAG en Chroma:
        # así los vectores usados aquí son comparables/consistentes con el resto del sistema
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # random_forest_model.pkl es un RandomForestRegressor de sklearn ya entrenado y serializado
        # (entrenado offline sobre los mismos embeddings, no en este archivo)
        self.model = joblib.load('random_forest_model.pkl')
        self.log("El Agente de Bosque Aleatorio está listo")

    def price(self, description: str) -> float:
        """
        Inicializa este objeto cargando los pesos del modelo guardado
        Usa un modelo de Bosque aleatorio para estimar el precio del artículo descrito
        :param description: el producto que se va a estimar
        :return: el precio en coma flotante
        """        
        self.log("El Agente de Bosque aleatorio ha empezado una predicción")
        # Convierte la descripción a un embedding y deja que el Random Forest prediga el precio;
        # max(0, ...) evita devolver precios negativos (posibles con un regresor sin restricciones)
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])
        self.log(f"El agente de Random Forest ha terminado - predicción ${result:.2f}")
        return result