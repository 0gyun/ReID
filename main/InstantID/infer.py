import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
# from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from pipeline_reid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers import DDIMScheduler

from sklearn.metrics.pairwise import cosine_similarity

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":
    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Load pipeline
    # IdentityNet을 위한 ControlNet
    scheduler = DDIMScheduler.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            subfolder="scheduler",
        )
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = scheduler
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    face_image = load_image("../../images/testImage.png")
    face_image = resize_img(face_image)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])
    face_embed = torch.Tensor(face_emb)
    face_embed = [face_embed.reshape([1, 1, -1])]

    mask_image = load_image("../../images/face_mask_image.jpg")
    face_image = face_image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    image, face_embed = pipe(
        prompt='',
        num_inference_steps=30,
        guidance_scale=0,
        init_image = face_image, # 원본 이미지
        mask_image = mask_image, # 마스크 이미지1
        ctrl_image_embeds=face_emb, # face embedding
        ctrl_image=face_kps, # keypoints 그린 이미지
        controlnet_conditioning_scale=1.0,
        ip_adapter_scale=1,
        # ip_adapter_image = face_image,
        ip_adapter_image_embeds=face_embed,
        strength=0.1,
    )

    # image = pipe(
    #     prompt='',
    #     image_embeds=face_emb,
    #     image=face_kps,
    #     controlnet_conditioning_scale=0.8,
    #     ip_adapter_scale=1,
    #     num_inference_steps=50,
    #     guidance_scale=5,
    # ).images[0]

    image = image.images[0]
    # image1 = image1.images[0]
    image.save('./results/reid_result.jpg')
    # image1.save('./results/reid_result_비교.jpg')
    # import pdb; pdb.set_trace()
    # face_embed_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    # face_embed_info = sorted(face_embed_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    # face_embed = face_embed_info['embedding']
    
    # face_embed = face_embed.reshape(-1,1)
    # face_emb = face_emb.reshape(-1,1)
    # similarity = cosine_similarity(face_embed, face_emb)[0]
    # print(f"\nSimilarity between Input image's face embedding and Optimized face embedding is : {similarity[0]}")