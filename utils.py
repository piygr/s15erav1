import torch

def load_model_from_checkpoint(device, file_name='ckpt_light.pth'):
    checkpoint = torch.load(file_name, map_location=device)

    return checkpoint
