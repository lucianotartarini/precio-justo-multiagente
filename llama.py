# Ejemplo de despliegue en Modal del modelo base Llama 3.1 8B (SIN fine-tuning), cuantizado en 4-bit.
# Sirve como paso intermedio de aprendizaje antes de desplegar la versión fine-tuneada (ver pricer_service.py)
import modal
from modal import App, Volume, Image

# Setup
# Se define la app de Modal y la imagen del contenedor con las librerías necesarias para
# cargar un modelo de HuggingFace cuantizado (bitsandbytes) y ejecutarlo en GPU (accelerate)
app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate")
# Secreto de Modal con el token de HuggingFace, necesario para descargar el modelo (Llama requiere licencia)
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "T4"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # "google/gemma-2-2b"


# Función remota: se ejecuta en la infraestructura de Modal, no localmente.
# image/secrets/gpu/timeout configuran el contenedor y los recursos de la ejecución.
@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    # Imports dentro de la función porque solo existen en el contenedor remoto
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    # Quant Config
    # Configuración de cuantización en 4 bits (nf4) para que el modelo quepa en una GPU T4 (16GB)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Cargamos model y tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )

    # Semilla fija para reproducibilidad de la generación
    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    # Genera solo unos pocos tokens nuevos (ejemplo simple de inferencia)
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    return tokenizer.decode(outputs[0])
