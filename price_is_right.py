# Primera versión (simple) de la interfaz Gradio del producto final "El Precio Justo".
# Muestra una tabla con las oportunidades detectadas y refresca automáticamente cada 60s
# ejecutando de nuevo el framework de agentes. Ver price_is_right_final.py para la versión
# más completa (con logs en vivo y gráfico 3D del vectorstore).
import gradio as gr
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity, Deal

class App:

    def __init__(self):
        # El framework de agentes se crea de forma perezosa (lazy) en start(), no aquí
        self.agent_framework = None

    def run(self):
        # gr.Blocks define el layout de la interfaz; fill_width hace que ocupe todo el ancho
        with gr.Blocks(title="El Precio Justo", fill_width=True) as ui:

            def table_for(opps):
                """Convierte una lista de Opportunity en filas para el componente Dataframe"""
                return [[opp.deal.product_description, f"${opp.deal.price:.2f}", f"${opp.estimate:.2f}", f"${opp.discount:.2f}", opp.deal.url] for opp in opps]

            def start():
                """Se ejecuta al cargar la página: instancia el framework y muestra la memoria existente"""
                self.agent_framework = DealAgentFramework()
                opportunities = self.agent_framework.memory
                table = table_for(opportunities)
                return table

            def go():
                """Se ejecuta en cada tick del timer: lanza una nueva pasada del framework de agentes"""
                self.agent_framework.run()
                new_opportunities = self.agent_framework.memory
                table = table_for(new_opportunities)
                return table

            def do_select(selected_index: gr.SelectData):
                """Al seleccionar una fila de la tabla, se reenvía manualmente la alerta de esa oportunidad"""
                opportunities = self.agent_framework.memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.agent_framework.planner.messenger.alert(opportunity)

            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px">El precio justo: agencia de búsqueda de ofertas con IA</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Marco de agente autónomo que encuentra ofertas en línea, colaborando con un LLM patentado y optimizado implementado en Modal, y un pipeline RAG con un modelo de frontera y Chroma.</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Ofertas encontradas hasta el momento:</div>')
            with gr.Row():
                # Tabla principal donde se listan las oportunidades (ofertas con buen descuento estimado)
                opportunities_dataframe = gr.Dataframe(
                    headers=["Descripción", "Precio", "Estimación", "Descuento", "URL"],
                    wrap=True,
                    column_widths=[4, 1, 1, 1, 2],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )

            # Al cargar la UI se ejecuta start() para poblar la tabla inicial
            ui.load(start, inputs=[], outputs=[opportunities_dataframe])

            # Timer que dispara go() cada 60 segundos para buscar nuevas ofertas automáticamente
            timer = gr.Timer(value=60)
            timer.tick(go, inputs=[], outputs=[opportunities_dataframe])

            # Al hacer clic/seleccionar una fila de la tabla se dispara una notificación manual
            opportunities_dataframe.select(do_select)

        # Lanza el servidor de Gradio; share=False evita crear una URL pública
        ui.launch(share=False, inbrowser=True)

if __name__=="__main__":
    App().run()
    