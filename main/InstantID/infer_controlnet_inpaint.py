import cv2
import math
from controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
from diffusers import DDIMScheduler
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from PIL import Image
import PIL
import numpy as np
import torch
from insightface.app import FaceAnalysis

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps) # keypoints

    w, h = image_pil.size # 이미지 사이즈
    out_img = np.zeros([h, w, 3]) # 이미지 크기에 rgb 3채널의 numpy array

    for i in range(len(limbSeq)): # 0~3 아마 키포인트 찍고 색 채우는 for문?
        index = limbSeq[i] # [0,2] , [1,2], [3,2] ,[4,2]
        color = color_list[index[0]] # color_list[0, 1, 3, 4]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5 # 두 점 사이의 거리
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1])) # 두 점 사이의 각도
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps): # 키포인트 동그라미 그리는?
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

if __name__=="__main__":
    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    generator = torch.Generator(device="cpu").manual_seed(1)

    init_image = load_image("../../images/testImage.png").resize((1024,1024))
    mask_image = load_image("../../images/face_mask_image.jpg").resize((1024,1024))
    import pdb
    face_info = app.get(cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding'] # Face embedding
    face_embed = torch.Tensor(face_emb)
    face_embed = [face_embed.reshape([1, 1, -1])]
    face_kps = draw_kps(init_image, face_info['kps'])

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()


    image = pipe(
        prompt='',
        num_inference_steps=30,
        generator=generator,
        guidance_scale=0.0,
        eta=0.0,
        image=init_image,
        mask_image=mask_image,
        control_image=face_kps,
        # ip_adapter_image=init_image,
        ip_adapter_image_embeds=face_embed,
        strength=0.1,
    ).images[0]

    image.save('./results/ControlNetInpaintPipeline_result.jpg')
