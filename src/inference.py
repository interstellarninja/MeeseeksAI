import re
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from src.utils import (
    inference_logger,
)

class ModelInference:
    def __init__(self, model_path, load_in_4bit):
        self.bnb_config = None

        if load_in_4bit == "True":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.float32,
            #attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
        inference_logger.info(self.model.config)
        inference_logger.info(self.model.generation_config)
        inference_logger.info(self.tokenizer.special_tokens_map)

    
    def run_inference(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1500,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True)

        assistant_message = self.get_assistant_message(completion, "chatml", self.tokenizer.eos_token)
        return assistant_message
    
    def get_assistant_message(self, completion, chat_template, eos_token):
        """define and match pattern to find the assistant message"""
        completion = completion.strip()
        if chat_template == "chatml":
            assistant_pattern = re.compile(r'<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$', re.DOTALL)
        else:
            raise NotImplementedError(f"Handling for chat_template '{chat_template}' is not implemented.")
        
        assistant_match = assistant_pattern.search(completion)
        if assistant_match:
            assistant_content = assistant_match.group(1).strip()
            return assistant_content.replace(eos_token, "")
        else:
            assistant_content = None
            inference_logger.info("No match found for the assistant pattern")
            return assistant_content