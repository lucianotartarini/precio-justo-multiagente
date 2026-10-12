# El PlanningAgent es el "cerebro" que orquesta el pipeline completo de punta a punta:
# Scanner (busca ofertas) -> Ensemble (estima el precio real) -> Messaging (avisa si vale la pena).
# Es el agente que usa deal_agent_framework.py para ejecutar un ciclo completo de detección de ofertas.
from typing import Optional, List
from agents.agent import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class PlanningAgent(Agent):

    name = "Agente Planificador"
    color = Agent.GREEN
    # Solo se avisa de una oferta si el descuento estimado (precio_estimado - precio_anunciado)
    # supera este umbral en dólares; evita notificaciones por diferencias poco significativas
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Crea instancias de los 3 agentes que este planificador coordina
        """
        self.log("Inicializando el Agnete Planificador.")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("El Agente Planificador está listo.")

    def run(self, deal: Deal) -> Opportunity:
        """
        Ejecutar el flujo de trabajo para una oferta en particular
        :param deal: la oferta, resumida a partir de un extracto RSS
        :returns: una oportunidad que incluye el descuento
        """
        self.log("La agente de planificación está valorando un acuerdo potencial.")
        # El "descuento" no es el que anuncia el vendedor, sino la diferencia entre lo que el
        # Ensemble cree que vale realmente el producto y el precio al que se está ofreciendo:
        # así se detectan gangas reales, no solo ofertas que se autodenominan "descuento"
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"El agente de planificación ha procesado un trato con descuento ${discount:.2f}.")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Ejecute el flujo de trabajo completo:
        1. Use ScannerAgent para buscar ofertas en los feeds RSS
        2. Use EnsembleAgent para estimarlas
        3. Use MessagingAgent para enviar una notificación de ofertas
        :param memory: una lista de URL que han aparecido en el pasado
        :return: una oportunidad si se encontró una, de lo contrario, Ninguna
        """
        self.log("La agente de planificación está iniciando una carrera.")
        # 1. El scanner filtra el ruido de los feeds RSS y devuelve como mucho unas pocas ofertas
        #    ya preseleccionadas por tener buena descripción y precio claro
        selection = self.scanner.scan(memory=memory)
        if selection:
            # 2. Se valora cada oferta candidata (máx. 5) con el Ensemble
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            # 3. Se ordenan de mayor a menor descuento estimado, y solo se actúa sobre la mejor
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"El agente de planificación ha identificado la mejor oferta con descuento: ${best.discount:.2f}.")
            # 4. Solo se envía notificación (y se devuelve la oportunidad) si el descuento supera
            #    el umbral definido; si no, se descarta aunque técnicamente fuera "la mejor"
            if best.discount > self.DEAL_THRESHOLD:
                self.messenger.alert(best)
            self.log("El agente de planificación ha completado una ejecución.")
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None