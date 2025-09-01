import os
import torch
from skimage import io


def save_adv_example(adv_images, img_path, save_path):
    # img_path es tipo: ["video/frame.jpg"]
    video_dir, frame_name = img_path[0].split('/')

    # Normalización inversa (ImageNet)
    #mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(adv_images.device)
    #std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(adv_images.device)

    #adv_denorm = adv_images.squeeze() * std + mean
    #adv_denorm = torch.clamp(adv_denorm, 0, 1)  # [0,1]
    adv_denorm = (adv_images[0] * 255).byte().cpu().permute(1,2,0).numpy()  # HWC uint8

    # Crear directorio de guardado
    save_dir = os.path.join(save_path, video_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Definir nombre del archivo
    formated_img_name = f"{video_dir.split('_')[-1]}_{int(frame_name[:-4])}.jpg"
    complete_saving_path = os.path.join(save_dir, frame_name)
    
    # Guardar usando skimage
    io.imsave(complete_saving_path, adv_denorm)

    return formated_img_name