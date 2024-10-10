import os
import time
from openai import OpenAI
import json
import tools
from typing import *
from dotenv import load_dotenv
load_dotenv()#no tools because i dont know:(
model = os.getenv("Gmodel")
maxtoken = os.getenv("Gmodel_maxtokens")
import time
import torch
from bitsandbytes import *
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
model_name = model
cache_dir="./model_cache"
pipe = None
def chat(messages,input):
    global pipe
    messages.append({
		"role": "user",
		"content": input,	
	})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = time.time()
    if pipe == None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            ),
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        generate_kwargs = {
            "max_new_tokens": 1000 
        }
        pipe = pipeline("text-generation",torch_dtype="auto",model=model, tokenizer=tokenizer,
            trust_remote_code=True,**generate_kwargs)

    return_mes = pipe(messages)
    assistant_message_raw = return_mes["generated_text"][-1]
    assistant_message = assistant_message_raw['content']
    print(time.time()-s)
    messages.append(assistant_message_raw)
    return (assistant_message,messages)