import cv2
import torch
import numpy as np
from PIL import Image


from insightface.app import FaceAnalysis
from diffusers.utils import load_image
from diffusers import StableDiffusionXLInpaintPipeline, DDIMScheduler
# from diffusers1.src.diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline

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

    # Vanila Inpainting
    model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    scheduler = DDIMScheduler.from_pretrained(
        model,
        subfolder="scheduler"
    )
    # inpainting = StableDiffusionXLInpaintPipeline.from_pretrained(
    inpainting = StableDiffusionXLInpaintPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        scheduler=scheduler
    ).to("cuda")

    face_image = load_image("../../images/testImage.png")
    face_image = resize_img(face_image)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']

    mask_image = load_image("../../images/face_mask_image.jpg")
    face_image = face_image.convert("RGB")
    mask_image = mask_image.convert("RGB")
    face_embed = torch.Tensor(face_emb)
    face_embed = [face_embed.reshape([1, 1, -1])]
    import pdb; pdb.set_trace()
    vanila_image = inpainting(
        prompt='',
        image=face_image,
        mask_image=mask_image,
        num_inference_steps=100,
        eta=0.0,
        guidance_scale=0,
        strength=0.01,
        ip_adapter_image_embeds=face_embed
    ).images[0]
    vanila_image.save('./results/vanila_result.jpg')
