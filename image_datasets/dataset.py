import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from utils.fast_loading import load_embeddings_safetensors
import torch.nn.functional as F

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized image latents.

    When using precomputed embeddings, images may have different aspect ratios,
    resulting in latents with different spatial dimensions. This function pads
    them to the maximum dimensions in the batch.

    Args:
        batch: List of tuples (img_latent, text_embed, text_mask) or (img, prompt)

    Returns:
        Batched tensors with padding applied to image latents
    """
    if len(batch[0]) == 2:
        # Non-embedded case: (img, prompt)
        imgs, prompts = zip(*batch)
        # Check if imgs are tensors (latents) or need processing
        if torch.is_tensor(imgs[0]):
            # Handle variable-sized latents
            if len(imgs[0].shape) == 4:  # [C, T, H, W] format
                # Find max dimensions
                max_h = max(img.shape[2] for img in imgs)
                max_w = max(img.shape[3] for img in imgs)

                # Pad each latent to max dimensions
                padded_imgs = []
                for img in imgs:
                    c, t, h, w = img.shape
                    pad_h = max_h - h
                    pad_w = max_w - w
                    # Pad: (left, right, top, bottom)
                    padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    padded_imgs.append(padded)

                imgs = torch.stack(padded_imgs, dim=0)
            else:
                imgs = torch.stack(imgs, dim=0)

        return imgs, prompts

    elif len(batch[0]) == 3:
        # Embedded case: (img_latent, text_embed, text_mask)
        imgs, text_embeds, text_masks = zip(*batch)

        # Handle variable-sized image latents
        if len(imgs[0].shape) == 4:  # [C, T, H, W] format
            # Find max dimensions
            max_h = max(img.shape[2] for img in imgs)
            max_w = max(img.shape[3] for img in imgs)

            # Pad each latent to max dimensions
            padded_imgs = []
            for img in imgs:
                c, t, h, w = img.shape
                pad_h = max_h - h
                pad_w = max_w - w
                # Pad: (left, right, top, bottom)
                padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                padded_imgs.append(padded)

            imgs = torch.stack(padded_imgs, dim=0)
        else:
            imgs = torch.stack(imgs, dim=0)

        # Stack text embeddings and masks
        text_embeds = torch.stack(text_embeds, dim=0)
        text_masks = torch.stack(text_masks, dim=0)

        return imgs, text_embeds, text_masks

    else:
        raise ValueError(f"Unexpected batch format with {len(batch[0])} elements")


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, cached_text_embeddings=None,
                 cached_image_embeddings=None, txt_cache_dir=None, img_cache_dir=None):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.txt_cache_dir = txt_cache_dir
        self.img_cache_dir = img_cache_dir
        print('cached_text_embeddings', type(cached_text_embeddings))
        
    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.images) - 1)
            if self.cached_image_embeddings is None and self.img_cache_dir is None:
                img = Image.open(self.images[idx]).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img = crop_to_aspect_ratio(img, ratio)
                img = image_resize(img, self.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1)
            elif self.img_cache_dir is not None:
                # Try .safetensors first (new format), fallback to .pt (old format)
                img_name = self.images[idx].split('/')[-1]
                safetensors_path = os.path.join(self.img_cache_dir, img_name + '.safetensors')
                pt_path = os.path.join(self.img_cache_dir, img_name + '.pt')

                if os.path.exists(safetensors_path):
                    img_dict = load_embeddings_safetensors(safetensors_path)
                    img = img_dict['latent']
                else:
                    img = torch.load(pt_path)
            else:
                # Clone to ensure resizable storage for DataLoader batching
                img = self.cached_image_embeddings[self.images[idx].split('/')[-1]].clone()
            txt_path = self.images[idx].rsplit('.', 1)[0] + '.' + self.caption_type
            if self.cached_text_embeddings is None and self.txt_cache_dir is None:
                prompt = open(txt_path, encoding='utf-8').read()
                if throw_one(self.caption_dropout_rate):
                    return img, " "
                else:
                    return img, prompt
            elif self.txt_cache_dir is not None:
                if throw_one(self.caption_dropout_rate):
                    # Try .safetensors first (new format), fallback to .pt (old format)
                    safetensors_path = os.path.join(self.txt_cache_dir, 'empty_embedding.safetensors')
                    pt_path = os.path.join(self.txt_cache_dir, 'empty_embedding.pt')

                    if os.path.exists(safetensors_path):
                        txt_embs = load_embeddings_safetensors(safetensors_path)
                    else:
                        txt_embs = torch.load(pt_path)
                    return img, txt_embs['prompt_embeds'], txt_embs['prompt_embeds_mask']
                else:
                    # Try .safetensors first (new format), fallback to .pt (old format)
                    txt_name = txt_path.split('/')[-1]
                    safetensors_path = os.path.join(self.txt_cache_dir, txt_name + '.safetensors')
                    pt_path = os.path.join(self.txt_cache_dir, txt_name + '.pt')

                    if os.path.exists(safetensors_path):
                        txt_embs = load_embeddings_safetensors(safetensors_path)
                    else:
                        txt_embs = torch.load(pt_path)

                    return img, txt_embs['prompt_embeds'], txt_embs['prompt_embeds_mask']
            else:
                # Clone cached embeddings to ensure resizable storage for DataLoader batching
                txt = txt_path.split('/')[-1]
                if throw_one(self.caption_dropout_rate):
                    return img, self.cached_text_embeddings['empty_embedding']['prompt_embeds'].clone(), self.cached_text_embeddings['empty_embedding']['prompt_embeds_mask'].clone()
                else:
                    return img, self.cached_text_embeddings[txt]['prompt_embeds'].clone(), self.cached_text_embeddings[txt]['prompt_embeds_mask'].clone()
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))

def loader(train_batch_size, num_workers, pin_memory=True, **args):
    """
    Create a DataLoader for the custom image dataset.

    Args:
        train_batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to use pinned memory (set to False for unified memory systems)
        **args: Additional arguments passed to CustomImageDataset

    Returns:
        DataLoader instance
    """
    dataset = CustomImageDataset(**args)
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn  # Use custom collate to handle variable-sized latents
    )
