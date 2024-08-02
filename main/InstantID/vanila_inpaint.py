import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers import StableDiffusionXLInpaintPipeline, DDIMScheduler

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
    # Vanila Inpainting
    model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    scheduler = DDIMScheduler.from_pretrained(
        model,
        subfolder="scheduler"
    )
    inpainting = StableDiffusionXLInpaintPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        scheduler=scheduler
    ).to("cuda")

    face_image = load_image("../../images/testImage.png")
    face_image = resize_img(face_image)

    mask_image = load_image("../../images/face_mask_image.jpg")
    face_image = face_image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    vanila_image = inpainting(
        prompt='',
        image=face_image,
        mask_image=mask_image,
        num_inference_steps=100,
        eta=0.0,
        guidance_scale=7,
        strength=0.01
    ).images[0]
    vanila_image.save('vanila_result.jpg')
