# Despliegue "efímero" (ephemeral) en Modal: cada llamada a esta función descarga el modelo base
# y aplica los pesos LoRA del fine-tuning DESDE CERO (no hay estado persistente entre llamadas).
# Es simple pero lento (cold start en cada invocación). Comparar con pricer_service.py, donde se usa
# una clase (@app.cls) que carga el modelo una sola vez en @modal.enter() y lo reutiliza entre llamadas.
import modal
from modal import App, Image

# Setup
app = modal.App("pricer")
# Imagen del contenedor con las librerías necesarias para cargar y ejecutar un modelo LLM
# cuantizado (bitsandbytes) con adaptador LoRA (peft)
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
# Secreto con el token de HuggingFace para poder descargar el modelo base (Llama) y el fine-tuneado
secrets = [modal.Secret.from_name("hf-secret")]

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "ed-donner" # tu nombre de HF debe ir aquí! O bien usa el mío si solo quieres reproducir mis resultados
RUN_NAME = "2024-09-13_13.04.39"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
# Revisión/commit específico del repo de HuggingFace donde están los pesos fine-tuneados (para reproducibilidad)
REVISION = "e8d637df551603dc86cd7a1598a8f44af4d7ae36"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


# Función remota de Modal: cada invocación arranca un contenedor (o reutiliza uno "caliente")
# y ejecuta este cuerpo por completo, incluyendo la carga del modelo
@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    # Imports dentro de la función porque solo se ejecutan en el entorno remoto de Modal
    import os
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    QUESTION = "How much does this cost to the nearest dollar?"
    PREFIX = "Price is $"

    # Se construye el mismo formato de prompt usado durante el entrenamiento (ver items.py)
    prompt = f"{QUESTION}\n{description}\n{PREFIX}"

    # Quant Config
    # Cuantización en 4 bits para que el modelo (8B parámetros) quepa en una GPU T4
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Se carga el modelo base cuantizado...
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    # ...y se le aplica el adaptador LoRA fine-tuneado (PEFT) sobre ese modelo base
    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    # Solo se generan unos pocos tokens nuevos: basta con el número del precio
    outputs = fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    result = tokenizer.decode(outputs[0])

    # Extrae el número de precio de la respuesta del modelo (texto después de "Price is $")
    contents = result.split("Price is $")[1]
    contents = contents.replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0
