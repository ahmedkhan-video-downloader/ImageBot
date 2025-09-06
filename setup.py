#!/usr/bin/env python3
"""
Setup script for pre-downloading models
"""

import os
import sys
from diffusers import StableDiffusionPipeline
import torch

def download_model():
    print("📥 Downloading Stable Diffusion model...")
    try:
        # Download a small model for faster deployment
        model_id = "OFA-Sys/small-stable-diffusion-v0"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir="./models"
        )
        print("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

if __name__ == "__main__":
    download_model()
