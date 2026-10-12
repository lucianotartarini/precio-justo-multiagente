# Versión final y completa de la interfaz Gradio del producto "The Price is Right".
# Respecto a price_is_right.py, esta versión añade:
#  - Un panel de logs en vivo (streaming de los mensajes de log de los agentes hacia la UI)
#  - Un gráfico 3D interactivo (Plotly) que visualiza el vectorstore de productos (Chroma + t-SNE)
#  - Ejecución del framework en un hilo aparte para no bloquear la interfaz mientras trabaja
import logging
import queue
import threading
import time
import gradio as gr
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity, Deal
from log_utils import reformat
import plotly.graph_objects as go


class QueueHandler(logging.Handler):
    """
    Handler de logging personalizado que, en lugar de escribir a consola/archivo,
    coloca cada mensaje de log en una cola (queue). Esto permite "puentear" los logs
    generados por los agentes (que corren en un hilo aparte) hacia la interfaz Gradio,
    que los va leyendo de la cola para mostrarlos en tiempo real.
    """
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

def html_for(log_data):
    """
    Construye el bloque HTML con scroll que muestra las últimas 18 líneas de log
    """
    output = '<br>'.join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """

def setup_logging(log_queue):
    """
    Conecta el QueueHandler al logger raíz para capturar todos los logs (incluidos los de los agentes)
    """
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class App:

    def __init__(self):
        # El framework de agentes se crea de forma perezosa (lazy), ver get_agent_framework()
        self.agent_framework = None

    def get_agent_framework(self):
        """
        Devuelve la instancia del framework de agentes, creándola la primera vez que se necesita
        """
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):
        with gr.Blocks(title="El Precio Justo", fill_width=True) as ui:

            # Estado de Gradio que persiste el historial de logs entre renders del componente
            log_data = gr.State([])

            def table_for(opps):
                """Convierte una lista de Opportunity en filas para el componente Dataframe"""
                return [[opp.deal.product_description, f"${opp.deal.price:.2f}", f"${opp.estimate:.2f}", f"${opp.discount:.2f}", opp.deal.url] for opp in opps]

            def update_output(log_data, log_queue, result_queue):
                """
                Generador que va vaciando la cola de logs y, en paralelo, vigila la cola de resultados.
                Cada vez que hay un mensaje nuevo (de log o de resultado final), hace yield para que
                Gradio actualice la interfaz en tiempo real (streaming), hasta que el resultado final llega.
                """
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield log_data, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            # Si ya llegó el resultado final y no hay más logs pendientes, se termina el streaming
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def get_initial_plot():
                """Figura placeholder mostrada mientras se calcula el gráfico 3D real"""
                fig = go.Figure()
                fig.update_layout(
                    title='Cargando la base de datos de vectores...',
                    height=400,
                )
                return fig

            def get_plot():
                """
                Obtiene los embeddings de productos desde Chroma, los reduce a 3D con t-SNE
                (ver DealAgentFramework.get_plot_data) y construye el scatter 3D interactivo
                """
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=1000)
                # Create the 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=vectors[:, 0],
                    y=vectors[:, 1],
                    z=vectors[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=colors, opacity=0.7),
                )])

                fig.update_layout(
                    scene=dict(xaxis_title='x',
                               yaxis_title='y',
                               zaxis_title='z',
                               aspectmode='manual',
                               aspectratio=dict(x=2.2, y=2.2, z=1),  # Make x-axis twice as long
                               camera=dict(
                                   eye=dict(x=1.6, y=1.6, z=0.8)  # Adjust camera position
                               )),
                    height=400,
                    margin=dict(r=5, b=1, l=5, t=2)
                )

                return fig

            def do_run():
                """Ejecuta una pasada completa del framework de agentes (scan + plan + notificar) y arma la tabla"""
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table

            def run_with_logging(initial_log_data):
                """
                Lanza do_run() en un hilo (thread) separado para no bloquear la interfaz de Gradio,
                mientras el hilo principal va consumiendo la cola de logs con update_output() para
                mostrar el progreso en vivo
                """
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    result = do_run()
                    result_queue.put(result)

                thread = threading.Thread(target=worker)
                thread.start()

                for log_data, output, final_result in update_output(initial_log_data, log_queue, result_queue):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData):
                """Al seleccionar una fila de la tabla, se reenvía manualmente la alerta de esa oportunidad"""
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.get_agent_framework().planner.messenger.alert(opportunity)

            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px"><strong>The Price is Right</strong> - Autonomous Agent Framework that hunts for deals</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">A proprietary fine-tuned LLM deployed on Modal and a RAG pipeline with a frontier model collaborate to send push notifications with great online deals.</div>')
            with gr.Row():
                # Tabla principal con las oportunidades encontradas hasta el momento
                opportunities_dataframe = gr.Dataframe(
                    headers=["Descuentos hasta el momento", "Precio", "Estimación", "Descuento", "URL"],
                    wrap=True,
                    column_widths=[3, 1, 1, 1, 3],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    # Panel de logs en vivo, alimentado por run_with_logging
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    # Gráfico 3D del vectorstore de productos
                    plot = gr.Plot(value=get_plot(), show_label=False)

            # Al cargar la página se ejecuta automáticamente una primera pasada del framework, con logging en vivo
            ui.load(run_with_logging, inputs=[log_data], outputs=[log_data, logs, opportunities_dataframe])

            # Timer que repite el ciclo completo (scan+plan+notify) cada 300 segundos (5 minutos)
            timer = gr.Timer(value=300, active=True)
            timer.tick(run_with_logging, inputs=[log_data], outputs=[log_data, logs, opportunities_dataframe])

            # Selección manual de una fila para reenviar la notificación de esa oportunidad
            opportunities_dataframe.select(do_select)

        # Lanza el servidor Gradio; share=False evita crear una URL pública
        ui.launch(share=False, inbrowser=True)

if __name__=="__main__":
    App().run()
    