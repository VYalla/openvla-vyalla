"""
Simple script to run inference with OpenVLA model on a single image.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run inference with OpenVLA model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--instruction", type=str, default="pick up the object", 
                        help="Instruction for the robot")
    parser.add_argument("--model_name", type=str, default="openvla/openvla-7b", 
                        help="Model name or path")
    args = parser.parse_args()
    
    print(f"Loading model {args.model_name}...")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with appropriate settings
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else None,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    # Load and process image
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    
    # Format prompt
    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"
    print(f"Using prompt: {prompt}")
    
    # Process inputs
    inputs = processor(prompt, image)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Convert to appropriate dtype if using CUDA
    if device.startswith("cuda"):
        inputs = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float32 else v 
                 for k, v in inputs.items()}
    
    print("Running inference...")
    # Generate action
    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    print("\nPredicted Action (7-DoF):")
    print(action)
    
    # If the action is a numpy array, format it nicely
    if isinstance(action, np.ndarray):
        print("\nAction components:")
        components = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        for i, comp in enumerate(components):
            if i < len(action):
                print(f"  {comp}: {action[i]:.4f}")

if __name__ == "__main__":
    main()
