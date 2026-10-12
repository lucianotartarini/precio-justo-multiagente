# El Precio Justo — sistema multiagente de detección de ofertas

Sistema multiagente que rastrea ofertas online, estima su precio real combinando tres modelos y avisa por notificación cuando encuentra una ganga.

> **Proyecto de curso — código sin modificaciones.**
> Este repositorio reúne el proyecto final (Semana 8) del curso Ingeniería de LLM, tomado del repositorio del curso [`joanby/llm_engineering`](https://github.com/joanby/llm_engineering) (versión en español de Juan Gabriel Gomila). El código original es de **Ed Donner** y se distribuye bajo licencia MIT (ver [`LICENSE`](LICENSE)). Se publica como constancia del proyecto trabajado en el curso; la autoría del código no es mía.

> **Este repositorio, tal como está, no se ejecuta de punta a punta:** faltan artefactos pesados que el curso excluye de git (ver "No incluido").

## Qué hace

```
ScannerAgent ──► PlanningAgent ──► EnsembleAgent ──► MessagingAgent
(RSS + LLM)       (orquesta)        ├─ SpecialistAgent  (Llama 3.1 8B + adaptador QLoRA en Modal)
                                    ├─ FrontierAgent    (RAG con Chroma + gpt-4o-mini)
                                    └─ RandomForestAgent (embeddings MiniLM + Random Forest)
                                    → regresión lineal pre-entrenada que combina las tres estimaciones
```

- **ScannerAgent:** lee cinco feeds RSS de DealNews y usa `gpt-4o-mini` con salidas estructuradas (`DealSelection`, Pydantic) para quedarse con las 5 ofertas con mejor descripción y precio claro.
- **EnsembleAgent:** pide una estimación a cada modelo y las combina (más mínimo y máximo) con una regresión lineal ya entrenada.
- **PlanningAgent:** calcula `descuento = estimación − precio ofrecido` y alerta si supera 50 USD.
- **MessagingAgent:** notificación por Pushover (por defecto) o SMS por Twilio (desactivado por defecto).
- **DealAgentFramework:** orquesta todo, persiste las oportunidades en `memory.json` y usa un vectorstore Chroma (`products_vectorstore`).
- **Interfaz Gradio:** `price_is_right.py` (simple) y `price_is_right_final.py` (logs en vivo y gráfico 3D t-SNE del vectorstore).

## Archivos

| Archivo | Rol |
|---|---|
| `agents/` | Los agentes (`scanner`, `planning`, `ensemble`, `specialist`, `frontier`, `random_forest`, `messaging`), la clase base `agent.py` y los modelos de datos `deals.py`. |
| `deal_agent_framework.py` | Orquestación, memoria persistente y datos del gráfico 3D. |
| `price_is_right.py`, `price_is_right_final.py`, `log_utils.py` | Interfaces Gradio y formato de logs. |
| `pricer_service2.py` | Servicio final en Modal: clase con `@modal.enter()` que carga el modelo una sola vez. |
| `pricer_service.py`, `pricer_ephemeral.py`, `llama.py`, `hello.py`, `keep_warm.py` | Versiones previas y utilidades de despliegue en Modal. |
| `items.py`, `testing.py` | Curación de datos y arnés de evaluación (error medio, RMSLE, aciertos). |

## Servicios y dependencias externas

- OpenAI (`gpt-4o-mini`), Hugging Face (el tokenizer de Llama 3.1, de acceso restringido, se carga al importar `items.py`), Modal (GPU T4) y Pushover. Twilio es opcional.
- `requirements.txt` es el del curso. Además, el código importa `twilio`, `feedparser` y `sentence-transformers`, que **no** figuran en ese archivo.
- El modelo del agente especialista es un adaptador publicado por el autor del curso (`ed-donner/pricer-2024-09-13_13.04.39`, con revisión fija). En este repositorio no se entrena ningún modelo propio.

## Pasos de ejecución (según el curso)

1. `pip install -r requirements.txt` y `pip install twilio feedparser sentence-transformers`.
2. Copiar `.env.example` a `.env` y completarlo.
3. `modal token new` y crear en Modal el secreto `hf-secret` con tu token de Hugging Face.
4. `modal deploy pricer_service2.py`.
5. Colocar en la raíz los artefactos que faltan (ver abajo).
6. `python price_is_right_final.py`.

## No incluido

- `ensemble_model.pkl` y `random_forest_model.pkl`: el `.gitignore` del curso excluye `*.pkl`. Se obtienen o generan con el material de la Semana 8.
- `products_vectorstore/` (base Chroma) y `memory.json`.
- Los notebooks de la semana y los datos de entrenamiento.

## Limitaciones conocidas

Observadas al leer el código; no se corrigieron porque este repositorio es una copia fiel del material del curso.

- No hay tests automatizados.
- Las tres estimaciones del ensemble se piden en secuencia (el propio código indica que podrían paralelizarse).
- No hay manejo explícito de errores en las llamadas remotas ni en el scraping.
- Los pesos del ensemble provienen de un entrenamiento hecho fuera de este repositorio.

## Licencia y créditos

MIT. Copyright (c) 2024 Ed Donner. Material de curso distribuido en español por Juan Gabriel Gomila (`joanby/llm_engineering`).
