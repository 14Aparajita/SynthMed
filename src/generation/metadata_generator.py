import json
import torch
import logging
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("synthmed.generation")

class MetadataGenerator:
    """
    Generate synthetic clinical metadata using DistilGPT-2.
    Generates schema-structured JSON records.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "cpu",
        max_length: int = 256,
        temperature: float = 0.7
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        logger.info(f"Loading metadata model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate metadata records from prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length // 2
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                num_return_sequences=num_return_sequences,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def generate_structured(
        self,
        dr_grade: int,
        context: Optional[str] = None,
        num_records: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate schema-structured metadata records."""
        base_prompt = self._build_prompt(dr_grade, context)
        
        records = []
        for _ in range(num_records):
            raw_text = self.generate(base_prompt)[0]
            parsed = self._parse_json_record(raw_text)
            records.append(parsed)
        
        return records
    
    def _build_prompt(
        self,
        dr_grade: int,
        context: Optional[str] = None
    ) -> str:
        """Build generation prompt with DR grade and optional context."""
        grade_descriptions = {
            0: "No diabetic retinopathy",
            1: "Mild non-proliferative diabetic retinopathy",
            2: "Moderate non-proliferative diabetic retinopathy",
            3: "Severe non-proliferative diabetic retinopathy",
            4: "Proliferative diabetic retinopathy"
        }
        
        prompt = (
            f"Generate a clinical metadata record for a patient with "
            f"{grade_descriptions.get(dr_grade, 'unknown DR grade')}.\n"
            f"DR Grade: {dr_grade}\n"
        )
        
        if context:
            prompt += f"Clinical Context: {context}\n"
        
        prompt += (
            "Output format (JSON):\n"
            '{"patient_id": "P00000", "age": 50, "sex": "M", '
            '"dr_grade": 0, "image_quality": 0.8, "left_eye": true, '
            '"anatomical_findings": {"microaneurysms": "none", '
            '"hemorrhages": "none", "exudates": "none"}}\n'
        )
        
        return prompt
    
    def _parse_json_record(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from generated text."""
        # Find JSON-like structure
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return basic structure
        return {
            "patient_id": "P00000",
            "age": 50,
            "sex": "M",
            "dr_grade": 0,
            "image_quality": 0.5,
            "left_eye": True,
            "anatomical_findings": {
                "microaneurysms": "none",
                "hemorrhages": "none",
                "exudates": "none"
            }
        }