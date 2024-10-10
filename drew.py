#from tkinter import Image
import os
import time
from diffusers import StableDiffusionPipeline,UNet2DConditionModel,AutoencoderKL,DiffusionPipeline,StableDiffusionUpscalePipeline
import torch
from transformers import BitsAndBytesConfig, CLIPTextModel
import numpy as np
import ipywidgets as widgets
cache_dir = "./model_cache"
#pipe = 
# 加载模型
# text_encoder = CLIPTextModel.from_pretrained("CompVis/clip-vit-base-patch32")
# Use a pipeline as a high-level helper
# from transformers import pipeline

# text_encoder = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
# # 使用文本编码器和单文件加载管道
# pipe = StableDiffusionPipeline.from_single_file("sd2.1\\keai_V1.0.safetensors", text_encoder=text_encoder).to("cuda")
# 指定模型文件路径


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = None
def ai(pro,negpro=''):
    global pipe
    s = time.time()
    # 测试分割函数
    if pipe == None:
        pipe = DiffusionPipeline.from_pretrained("eienmojiki/Anything-XL",cache_dir=cache_dir,allow_pickle=False,torch_dtype=torch.float16,)
        pipe.safety_checker = lambda images, clip_input: (images, None)
        pipe = pipe.to(device)
        # upscale = StableDiffusionUpscalePipeline.from_pretrained(
        #     model_id, variant="fp16", torch_dtype=torch.float16
        # ).to("cuda")
    negative_prompt = negpro
    print(int(time.time()))
    print(pro +'\n'+negpro)
    generated_image = pipe(prompt=pro,negative_prompt=negative_prompt,num_inference_steps=35,width=512, height=768, generator=torch.manual_seed(int(time.time()))).images[0]
    generated_image.save('result.png')
    current_working_directory = os.getcwd()
    print(current_working_directory)
    absolute_path = os.path.abspath(os.path.join(current_working_directory, "result.png"))
    os.chdir("Real-ESRGAN")
    os.system("python inference_realesrgan.py -n RealESRGAN_x2plus -i "+absolute_path+" -o " + current_working_directory)
    os.chdir(current_working_directory)
    os.remove("result.png")
    os.rename("result_out.png","result.png")#fix shit
    print(time.time() - s)
#ai("NSFW:3,explicit:3,sensitive\nbest quality, masterpiece, nekonya, catgirl, cute, loyal, 11-12 years old, 139cm, 36kg, M-shaped bangs, long hair, light green, pink eyes, cat ears, white fur inside ears","lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry，safe")