from transformers import AutoProcessor, AutoModelForImageTextToText,BitsAndBytesConfig
import torch
from PIL import Image
import os
from pathlib import Path
class VLM:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Configure 4-bit quantization with bitsandbytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
        )

        adapter_path = r"U:\Fraunhofer Waldbrand\Fine_tuning\qwen3-8b-instruct-fine-tune-V3"
        self.model.load_adapter(adapter_path)
        self._sync_lora_to_base_devices()

    def _sync_lora_to_base_devices(self):
        # Ensure LoRA A/B projection weights live on the same device as the wrapped base layer.
        for module in self.model.modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue

            base_weight = getattr(module, "weight", None)
            if base_weight is None:
                continue

            target_device = base_weight.device
            for adapter_name, lora_a in module.lora_A.items():
                lora_b = module.lora_B[adapter_name] if adapter_name in module.lora_B else None
                if lora_a.weight.device != target_device:
                    lora_a.to(target_device)
                if lora_b is not None and lora_b.weight.device != target_device:
                    lora_b.to(target_device)

    def analyze(self, image):
        # prompt = "Is the smoke in the middle of the image coming from burning vegetation (trees, forest, grass)? Answer ONLY with the letter:\
        #             D: can't be determined with certainty\
        #             B: no (chimney, cloud, fog, industrial)\
        #             C: there is no smoke in the image.\
        #             A: yes (wildfire/vegetation fire)"
        # prompt = "You are tasked with detecting wildfire smoke. Is there smoke in the image? If yes, is it coming from burning vegetation (trees, forest, grass)? Answer ONLY with the letter:\
        #             B: no there is no smoke in the image\
        #             C: there is smoke in the image, but it's not coming from burning vegetation\
        #             A: there is smoke in the image and it's coming from burning vegetation\
        #             D: it can't be determined whether there is smoke or not, or where it's coming from"
        # prompt = """You are tasked with detecting wildfire smoke.

        #         Determine:
        #         1. Is there visible smoke in the image?
        #         2. If yes, is the smoke coming from burning vegetation (trees, forest, grass)?

        #         Answer with ONLY the letter:
        #         A: Smoke is present and is coming from burning vegetation
        #         B: No smoke is present. This includes:
        #         - images with nothing resembling smoke
        #         - images with fog, clouds
        #         C: Smoke is present, but NOT from burning vegetation (e.g., chimney industrial)
        #         D: It cannot be determined whether there is smoke or its source
        #         """
        prompt = """You are tasked with detecting wildfire smoke.

                Determine:
                Is there smoke in the image coming from burning vegetation (trees, forest, grass)?

                Answer with ONLY the letter:
                A: yes, Smoke is present and is coming from burning vegetation
                B: no (e.g., there is no smoke, or there is smoke but it's from a non-vegetation source like a chimney or industrial)
                """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=2, do_sample=False)

        answer = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return answer.strip()


if __name__ == "__main__":
    vlm = VLM()
    forestfireinsight_folder = r"U:\Fraunhofer Waldbrand\Testbilder\Kamera_Balkon\Images_smoke_cropped"
    positives = 0
    for image_path in Path(forestfireinsight_folder).iterdir():
        if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image = Image.open(image_path).convert("RGB")
            print(f"Analyzing {image_path.name}...")
            result = vlm.analyze(image)
            if result == "A":
                positives += 1
            print(f"Result: {result}")
    print(f"Total positives: {positives}")
    print(f"rate of positives: {positives/len(list(Path(forestfireinsight_folder).iterdir())):.2%}")