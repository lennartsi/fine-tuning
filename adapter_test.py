from transformers import AutoProcessor, AutoModelForImageTextToText,BitsAndBytesConfig
import torch
from PIL import Image
import os
from pathlib import Path
import json
import re
from pydantic import BaseModel, ValidationError
from typing import Optional


class SmokeAnalysisAnswer(BaseModel):
    """Schema for smoke detection response"""
    reasoning: str
    decision: Optional[bool]
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

        # adapter_path = r"U:\Fraunhofer Waldbrand\Fine_tuning\qwen3-8b-instruct-fine-tune-V3"
        # self.model.load_adapter(adapter_path)
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

    def _parse_json_response(self, text: str) -> dict:
        """Extract and parse JSON from model output, handling various formats"""
        text = text.strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text (handles markdown code blocks, extra text, etc.)
        try:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        
        raise ValueError(f"Could not extract valid JSON from model output: {text}")
    
    def _validate_response(self, data: dict) -> SmokeAnalysisAnswer:
        """Validate response against schema and return typed object"""
        try:
            return SmokeAnalysisAnswer.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Response validation failed: {e}")
    
    def analyze(self, image) -> SmokeAnalysisAnswer:
        prompt = """You are tasked with detecting wildfire smoke in images.

Question: Is there smoke in this image coming from burning vegetation (trees, forest, grass)?

Respond with ONLY valid JSON in this exact format, no markdown, no extra text:
{
  "reasoning": "Your analysis of why this is true or false (mention presence/absence of smoke, characteristics, confidence). Keep the reasoning brief, ideally one sentence.",
    "decision": true, false, or null
}

Return ONLY the JSON object. true = smoke from vegetation, false = no smoke or non-vegetation smoke (chimney, clouds, industrial), null = unsure/cannot determine."""
        
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
            # Increased max_new_tokens to allow full JSON response (e.g., ~150 chars)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100,  # Changed from 2 to allow JSON output
                do_sample=False  # Deterministic for consistent format
            )

        decoded_text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        print(f"Decoded model output: {decoded_text}")
        # Parse and validate JSON response
        try:
            json_data = self._parse_json_response(decoded_text)
            validated_response = self._validate_response(json_data)
            return validated_response
        except (ValueError, ValidationError) as e:
            print(f"Warning: Failed to parse/validate response: {e}")
            print(f"Raw response: {decoded_text}")
            # Return a default response on parse failure
            return SmokeAnalysisAnswer(
                reasoning=f"Model output: {decoded_text[:100]}",
                decision=None
            )


if __name__ == "__main__":
    vlm = VLM()
    forestfireinsight_folder = r"U:\Fraunhofer Waldbrand\Testbilder\Reasoning_test"
    positives = 0
    for image_path in Path(forestfireinsight_folder).iterdir():
        if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image = Image.open(image_path).convert("RGB")
            print(f"Analyzing {image_path.name}...")
            result = vlm.analyze(image)
            # result is now a SmokeAnalysisAnswer object with reasoning and decision fields
            if result.decision:  # true = smoke from vegetation
                positives += 1
            print(f"Result: decision={result.decision}, reasoning={result.reasoning}")
    print(f"Total positives: {positives}")
    print(f"Rate of positives: {positives/len(list(Path(forestfireinsight_folder).iterdir())):.2%}")