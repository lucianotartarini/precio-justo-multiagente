# Punto central de orquestación de todo el sistema multi-agente ("El Precio Justo").
# Esta clase NO contiene lógica de negocio de los agentes en sí; su trabajo es:
#  1) Preparar el entorno (logging, variables de entorno, conexión al vectorstore Chroma)
#  2) Crear (de forma perezosa) el PlanningAgent, que a su vez orquesta scanner/ensemble/messenger
#  3) Persistir en memory.json las oportunidades encontradas entre ejecuciones
#  4) Exponer utilidades para las interfaces Gradio (price_is_right.py / price_is_right_final.py),
#     como la reducción de dimensionalidad del vectorstore para graficarlo en 3D
import os
import sys
import logging
import json
from typing import List, Optional
from twilio.rest import Client
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from sklearn.manifold import TSNE
import numpy as np


# Colors for logging
# Colores ANSI usados para resaltar los mensajes de log propios del framework en la consola
BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Colors for plot
# Categorías de producto presentes en el vectorstore y un color fijo asignado a cada una,
# usados para colorear el gráfico 3D de embeddings en la interfaz Gradio
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

def init_logging():
    """
    Configura el logger raíz de Python para que todos los agentes (que heredan de Agent.log())
    escriban sus mensajes por stdout con un formato consistente (fecha, nivel, mensaje)
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

class DealAgentFramework:
    """
    Orquestador de más alto nivel del sistema: conecta el vectorstore de Chroma,
    gestiona la memoria persistente de oportunidades y delega el trabajo real en PlanningAgent
    """

    DB = "products_vectorstore"          # Carpeta donde vive la base de datos vectorial Chroma
    MEMORY_FILENAME = "memory.json"       # Archivo donde se persisten las oportunidades ya detectadas

    def __init__(self):
        """
        Prepara logging, variables de entorno (.env) y la conexión al vectorstore.
        El PlanningAgent (y con él, todos los demás agentes) se crea más tarde y de forma perezosa,
        para no cargar modelos pesados hasta que realmente se necesite ejecutar el framework
        """
        init_logging()
        load_dotenv()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection('products')
        self.planner = None

    def init_agents_as_needed(self):
        """
        Crea el PlanningAgent (y, en cascada, ScannerAgent/EnsembleAgent/MessagingAgent)
        solo la primera vez que se necesita, para evitar el coste de inicialización si
        únicamente se quiere consultar la memoria existente
        """
        if not self.planner:
            self.log("Inicializando el Framework de Agentes")
            self.planner = PlanningAgent(self.collection)
            self.log("El framework de agentes está listo")

    def read_memory(self) -> List[Opportunity]:
        """
        Carga desde disco (memory.json) las oportunidades detectadas en ejecuciones anteriores,
        para no repetir alertas ya enviadas y para poblar la interfaz al arrancar
        """
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            opportunities = [Opportunity(**item) for item in data]
            return opportunities
        return []

    def write_memory(self) -> None:
        """
        Guarda en disco (memory.json) el estado actual de la memoria de oportunidades
        """
        data = [opportunity.dict() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str):
        """
        Emite un mensaje de log identificado como proveniente del Framework de Agentes (no de un agente específico)
        """
        text = BG_BLUE + WHITE + "[Framework de Agentes] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """
        Ejecuta un ciclo completo del sistema: inicializa los agentes si hace falta,
        delega en PlanningAgent.plan() (que a su vez hace scan -> ensemble -> messaging),
        y si se encontró una nueva oportunidad, la añade a la memoria y la persiste en disco.
        Este es el método que llaman tanto el modo CLI (__main__) como las interfaces Gradio.
        """
        self.init_agents_as_needed()
        logging.info("Puesta en marcha del agente de planificación")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"La agente de planificación ha completado y regresado: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        """
        Recupera embeddings, documentos y categorías desde Chroma, y reduce los vectores
        de alta dimensión a 3 dimensiones con t-SNE, para poder graficarlos en un scatter 3D
        en la interfaz Gradio (ver price_is_right_final.py)
        """
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection('products')
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        # t-SNE reduce la dimensionalidad de los embeddings a 3D preservando similitud relativa
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


# Punto de entrada CLI: ejecuta una única pasada del framework (útil para probar sin la interfaz Gradio)
if __name__=="__main__":
    DealAgentFramework().run()
    