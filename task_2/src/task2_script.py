import os
from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *

# 文件夹路径
background_folder = 'E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_2/base'

# 获取背景和掩码图片文件列表
background_images = [f for f in os.listdir(background_folder) if f.endswith('.jpg') or f.endswith('.png')]

with Workflow():
    model = UNETLoader('flux1-kontext-dev.safetensors', 'default')
    clip = DualCLIPLoader('clip_l.safetensors', 't5xxl_fp8_e4m3fn_scaled.safetensors', 'flux', 'default')
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('remove the crack on the concrete road', clip)
    for background_file in background_images:
        background_path = os.path.join(background_folder, background_file)
        image, _ = LoadImage(background_path)
        image = ImageStitch(image, 'right', True, 0, 'white', None)
        image = FluxKontextImageScale(image)
        vae = VAELoader('ae.safetensors')
        latent = VAEEncode(image, vae)
        conditioning = ReferenceLatent(clip_text_encode_positive_prompt_conditioning, latent)
        conditioning = FluxGuidance(conditioning, None)
        conditioning2 = ConditioningZeroOut(clip_text_encode_positive_prompt_conditioning)
        latent2 = RepeatLatentBatch(latent, 3)
        latent2 = KSampler(model, 114514, 20, 1, 'euler', 'simple', conditioning, conditioning2, latent2, 1)
        image2 = VAEDecode(latent2, vae)
        save_name = f"base{background_file.split('.')[0]}"
        SaveImage(image2, f'CrackRemove/{save_name}')

