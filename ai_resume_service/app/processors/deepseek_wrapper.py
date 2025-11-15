"""
DeepSeek LLM Wrapper Module

Provides interface to local DeepSeek model using HuggingFace transformers.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DeepSeekLLM:
    """DeepSeek 7B chat model wrapper."""
    
    def __init__(self):
        """Initialize DeepSeek model and tokenizer."""
        logger.info("Loading DeepSeek model (7B)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-llm-7b-chat",
            trust_remote_code=True
        )

        offload_path = "./deepseek_offload"
        os.makedirs(offload_path, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-llm-7b-chat",
            trust_remote_code=True,
            device_map="auto",
            offload_folder=offload_path,
            torch_dtype=torch.float32
        )

    def ask(self, prompt: str, max_tokens: int = 128):
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only after <|assistant|>
        if "<|assistant|>" in output:
            output = output.split("<|assistant|>")[-1].strip()

        return output.strip()

