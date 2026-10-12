# Versión mejorada del servicio de precios en Modal: en vez de una función suelta que recarga
# el modelo en cada invocación (ver pricer_service.py / pricer_ephemeral.py), aquí se define una
# CLASE de Modal (@app.cls) con métodos de ciclo de vida:
#   - @modal.enter(): se ejecuta UNA SOLA VEZ cuando el contenedor arranca (descarga y carga el modelo)
#   - @modal.method(): son los métodos invocables remotamente y reutilizan el estado ya cargado
# Esto reduce drásticamente la latencia de llamadas sucesivas, ya que el modelo queda "caliente"
# en memoria mientras el contenedor siga vivo. SpecialistAgent (agents/specialist_agent.py) usa
# exactamente esta clase "Pricer" del app "pricer-service" para hacer sus predicciones remotas.
import modal
from modal import App, Volume, Image

# Setup - define our infrastructure with code!
app = modal.App("pricer-service")
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "ed-donner" # your HF name here! Or use mine if you just want to reproduce my results.
RUN_NAME = "2024-09-13_13.04.39"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "e8d637df551603dc86cd7a1598a8f44af4d7ae36"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
# Carpeta local (dentro del contenedor) donde se cachean los pesos descargados de HuggingFace
MODEL_DIR = "hf-cache/"
BASE_DIR = MODEL_DIR + BASE_MODEL
FINETUNED_DIR = MODEL_DIR + FINETUNED_MODEL

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"


# @app.cls despliega una CLASE completa en Modal (en vez de una única función),
# lo que permite mantener estado (self.tokenizer, self.fine_tuned_model, etc.) entre llamadas
@app.cls(image=image, secrets=secrets, gpu=GPU, timeout=1800)
class Pricer:
    @modal.enter()
    def download_model_to_folder(self):
        """
        Hook de ciclo de vida ejecutado al arrancar el contenedor: descarga a disco local
        tanto el modelo base como el modelo fine-tuneado (snapshot_download), para que
        setup() los pueda cargar desde disco en vez de descargarlos en cada arranque
        """
        from huggingface_hub import snapshot_download
        import os
        os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
        snapshot_download(FINETUNED_MODEL, revision=REVISION, local_dir=FINETUNED_DIR)

    @modal.enter()
    def setup(self):
        """
        Segundo hook de ciclo de vida: carga en memoria (GPU) el modelo base cuantizado
        y le aplica el adaptador LoRA fine-tuneado. Se ejecuta una sola vez por contenedor,
        dejando el modelo listo ("caliente") para atender múltiples llamadas a price()
        """
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel

        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # Load model and tokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR,
            quantization_config=quant_config,
            device_map="auto"
        )

        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_DIR, revision=REVISION)

    @modal.method()
    def price(self, description: str) -> float:
        """
        Método remoto invocado por SpecialistAgent.price(): genera la estimación de precio
        para la descripción de producto dada, usando el modelo ya cargado en setup()
        """
        import os
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0])

        # Extrae el número de precio del texto generado
        contents = result.split("Price is $")[1]
        contents = contents.replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0

    @modal.method()
    def wake_up(self) -> str:
        """
        Método ligero usado solo para "despertar"/mantener caliente el contenedor
        (ver keep_warm.py), sin necesidad de ejecutar una inferencia completa
        """
        return "ok"
