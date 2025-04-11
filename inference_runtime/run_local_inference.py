"""
Script to run the entire OpenVLA model pipeline locally without downloading from Hugging Face.
This uses the local implementation of the model components.
"""

import torch
import argparse
import os
import numpy as np
from PIL import Image
import sys
import logging
import json
import datetime
from typing import Any, Dict

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Store the original print function
original_print = print

# Custom print function that also logs to file
def tee_print(*args, **kwargs):
    # Call the original print function
    original_print(*args, **kwargs)
    
    # Also log to file if we have a log file
    if hasattr(tee_print, 'log_file') and tee_print.log_file:
        message = ' '.join(str(arg) for arg in args)
        with open(tee_print.log_file, 'a') as f:
            f.write(f"PRINT: {message}\n")

def setup_logging(log_file=None):
    """Set up logging to both console and file."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]: 
        logger.removeHandler(handler)
    
    # Create console handler and set level
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console)
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')  # Start with a fresh file
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Set up direct file logging for print statements
        tee_print.log_file = log_file
        
        # Override the built-in print function
        import builtins
        builtins.print = tee_print
    
    return logger

def log_object(logger, name, obj, level=logging.DEBUG):
    """Log detailed information about an object."""
    message = f"\n=== {name} ===\n"
    
    try:
        # Log type information
        message += f"Type: {type(obj).__name__}\n"
        
        # Handle different types of objects
        if hasattr(obj, 'shape'):  # For numpy arrays and tensors
            message += f"Shape: {obj.shape}\n"
            if hasattr(obj, 'dtype'):
                message += f"Dtype: {obj.dtype}\n"
            
            # Sample values
            if hasattr(obj, 'flatten') and callable(obj.flatten):
                flattened = obj.flatten()
                if len(flattened) > 0:
                    sample_size = min(5, len(flattened))
                    message += f"Sample values: {flattened[:sample_size]}\n"
                    if hasattr(obj, 'tolist') and callable(obj.tolist):
                        try:
                            message += f"Full data: {obj.tolist()}\n"
                        except:
                            message += f"Full data: [too large to display]\n"
            
        elif isinstance(obj, dict):
            message += f"Keys: {list(obj.keys())}\n"
            # Sample values from dict
            message += "Contents:\n"
            for k, v in obj.items():  # Show all items
                if hasattr(v, 'shape'):
                    message += f"  {k}: {type(v).__name__} with shape {v.shape}\n"
                    if hasattr(v, 'tolist') and callable(v.tolist):
                        try:
                            if v.size < 100:  # Only show small tensors
                                message += f"    Data: {v.tolist()}\n"
                        except:
                            pass
                else:
                    message += f"  {k}: {type(v).__name__}\n"
                    try:
                        message += f"    Value: {str(v)[:100]}...\n"
                    except:
                        pass
        
        elif hasattr(obj, '__len__'):
            message += f"Length: {len(obj)}\n"
            # Sample values
            if len(obj) > 0:
                sample_size = min(5, len(obj))
                message += f"Sample values: {obj[:sample_size]}\n"
                # Full data for small collections
                if len(obj) < 100:
                    message += f"Full data: {obj}\n"
        
        # For other types, just log the string representation
        else:
            message += f"Value: {obj}\n"
    
    except Exception as e:
        message += f"Error logging object: {e}\n"
    
    logger.log(level, message)

def main():
    parser = argparse.ArgumentParser(description="Run local OpenVLA model inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--instruction", type=str, default="pick up the object", 
                        help="Instruction for the robot")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the local model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to log file (default: auto-generated)")
    args = parser.parse_args()
    
    # Set up logging
    if not args.log_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"openvla_detailed_execution_{timestamp}.txt"
    
    # Make sure the log file has a .txt extension
    if not args.log_file.endswith('.txt'):
        args.log_file += '.txt'
    
    logger = setup_logging(args.log_file)
    logger.info("=== OpenVLA Local Inference Started ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Import local model components
    logger.info("Importing model components...")
    from prismatic.models.vlas.openvla import OpenVLA
    from prismatic.models.vlms.prismatic import PrismaticVLM
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.load import load_vla
    
    print(f"Loading local OpenVLA model from {args.checkpoint_path}...")
    
    # Load the model from local checkpoint
    logger.info("Starting model loading...")
    with open(args.log_file, 'a') as f:
        f.write("\n\n===== STARTING MODEL LOADING =====\n\n")
    
    model = load_vla(args.checkpoint_path, load_for_training=False)
    logger.info(f"Model loaded: {type(model).__name__}")
    
    # Log model details
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Vision backbone: {type(model.vision_backbone).__name__}")
    logger.info(f"LLM backbone: {type(model.llm_backbone).__name__}")
    logger.info(f"Action tokenizer: {type(model.action_tokenizer).__name__}")
    
    # Log dataset statistics
    logger.info(f"Normalization stats keys: {list(model.norm_stats.keys())}")
    
    # Write model structure to file
    with open(args.log_file, 'a') as f:
        f.write("\n\n===== MODEL STRUCTURE =====\n\n")
        f.write(f"Model type: {type(model).__name__}\n")
        f.write(f"Model device: {next(model.parameters()).device}\n")
        f.write(f"Vision backbone: {type(model.vision_backbone).__name__}\n")
        f.write(f"LLM backbone: {type(model.llm_backbone).__name__}\n")
        f.write(f"Action tokenizer: {type(model.action_tokenizer).__name__}\n")
        f.write(f"Normalization stats keys: {list(model.norm_stats.keys())}\n")
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Load and process image
    logger.info(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    logger.info(f"Image loaded: {image.size} (WxH), Mode: {image.mode}")
    log_object(logger, "Image", np.array(image))
    
    # Run inference
    logger.info(f"Running inference with instruction: '{args.instruction}'...")
    
    # Write directly to log file
    with open(args.log_file, 'a') as f:
        f.write("\n==============================================================\n")
        f.write("                DETAILED EXECUTION FLOW                  \n")
        f.write("==============================================================\n\n")
        f.write(f"[1] Starting inference with instruction: '{args.instruction}'\n")
    
    print("\n=== DETAILED EXECUTION FLOW ===")
    print(f"[1] Starting inference with instruction: '{args.instruction}'")
    
    # Log input image details
    log_object(logger, "Input Image", image, level=logging.INFO)
    
    # Use a more direct approach to avoid compatibility issues
    try:
        # First try the standard approach
        logger.info("Attempting standard predict_action approach...")
        print("[2] Attempting to use model.predict_action() method")
        print(f"    - Model type: {type(model).__name__}")
        print(f"    - Model device: {next(model.parameters()).device}")
        print(f"    - Vision backbone: {type(model.vision_backbone).__name__}")
        print(f"    - LLM backbone: {type(model.llm_backbone).__name__}")
        
        action = model.predict_action(
            image=image,
            instruction=args.instruction,
            unnorm_key="bridge_orig",  # Use the appropriate dataset key
            do_sample=False
        )
    except AttributeError as e:
        logger.warning(f"Encountered error: {e}")
        logger.info("Trying alternative approach...")
        print(f"[3] Encountered error: {e}")
        print("[4] Switching to alternative approach (manual pipeline)")
        
        # Get the image transform and tokenizer
        logger.info("Getting image transform and tokenizer...")
        print("[5] Getting image transform and tokenizer components")
        image_transform = model.vision_backbone.image_transform
        tokenizer = model.llm_backbone.tokenizer
        print(f"    - Image transform type: {type(image_transform).__name__}")
        print(f"    - Tokenizer type: {type(tokenizer).__name__}")
        log_object(logger, "Image transform", image_transform)
        log_object(logger, "Tokenizer", tokenizer)
        
        # Build prompt
        prompt_text = f"In: What action should the robot take to {args.instruction.lower()}?\nOut: "
        logger.info(f"Prompt: '{prompt_text}'")
        print(f"[6] Building prompt: '{prompt_text}'")
        
        # Process inputs
        logger.info("Tokenizing prompt...")
        print("[7] Tokenizing prompt text")
        tokenizer_output = tokenizer(prompt_text, truncation=True, return_tensors="pt")
        print(f"    - Tokenizer output type: {type(tokenizer_output).__name__}")
        print(f"    - Tokenizer output keys: {list(tokenizer_output.keys())}")
        log_object(logger, "Tokenizer output", tokenizer_output)
        
        input_ids = tokenizer_output.input_ids.to(model.device)
        print(f"[8] Moving input_ids to device: {model.device}")
        print(f"    - Input IDs shape: {input_ids.shape}")
        print(f"    - Token sequence length: {input_ids.shape[1]}")
        log_object(logger, "Input IDs", input_ids)
        
        # Handle special token for Llama models if needed
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 2:
            logger.info(f"Handling special tokens for Llama model (eos_token_id={tokenizer.eos_token_id})")
            print(f"[9] Handling special tokens for Llama model (eos_token_id={tokenizer.eos_token_id})")
            if not torch.all(input_ids[:, -1] == 29871):
                logger.info("Adding special empty token (29871) to match training inputs")
                print("[10] Adding special empty token (29871) to match training inputs")
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                print(f"    - Updated input IDs shape: {input_ids.shape}")
                log_object(logger, "Updated input IDs", input_ids)
        
        # Process image
        logger.info("Processing image with vision backbone...")
        print("[11] Processing image with vision backbone")
        pixel_values = image_transform(image)
        print(f"    - Processed image type: {type(pixel_values).__name__}")
        if isinstance(pixel_values, torch.Tensor):
            print(f"    - Pixel values shape: {pixel_values.shape}")
        elif isinstance(pixel_values, dict):
            print(f"    - Pixel values keys: {list(pixel_values.keys())}")
        log_object(logger, "Processed image (pixel_values)", pixel_values)
        
        if isinstance(pixel_values, torch.Tensor):
            logger.info("Pixel values is a tensor, adding batch dimension...")
            print("[12] Adding batch dimension to pixel values tensor")
            pixel_values = pixel_values[None, ...].to(model.device)
            print(f"    - Batched pixel values shape: {pixel_values.shape}")
        elif isinstance(pixel_values, dict):
            logger.info("Pixel values is a dict, adding batch dimension to each value...")
            print("[12] Adding batch dimension to each value in pixel values dict")
            pixel_values = {k: v[None, ...].to(model.device) for k, v in pixel_values.items()}
            for k, v in pixel_values.items():
                print(f"    - {k} shape: {v.shape}")
        
        log_object(logger, "Batched pixel_values", pixel_values)
        
        # Use forward instead of generate
        logger.info("Running model forward pass...")
        print("[13] Running model forward pass (instead of generate)")
        print(f"    - Input IDs shape: {input_ids.shape}")
        if isinstance(pixel_values, torch.Tensor):
            print(f"    - Pixel values shape: {pixel_values.shape}")
        else:
            print(f"    - Pixel values type: {type(pixel_values).__name__}")
        
        print("[14] Starting torch.no_grad() context for inference")
        with torch.no_grad():
            print("    - Calling model.__call__ with input_ids and pixel_values")
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            print("    - Forward pass complete")
        
        print(f"[15] Model outputs type: {type(outputs).__name__}")
        if hasattr(outputs, 'logits'):
            print(f"    - Output logits shape: {outputs.logits.shape}")
        log_object(logger, "Model outputs", outputs)
        log_object(logger, "Output logits", outputs.logits)
            
        # Extract token IDs and decode to actions
        logger.info("Extracting token IDs and decoding to actions...")
        print("[16] Extracting token IDs and decoding to actions")
        action_dim = model.get_action_dim("bridge_orig")
        logger.info(f"Action dimension: {action_dim}")
        print(f"    - Action dimension: {action_dim}")
        
        print("[17] Taking argmax over vocabulary dimension to get predicted token IDs")
        predicted_token_ids = outputs.logits.argmax(dim=-1)[0, -action_dim:]
        print(f"    - Predicted token IDs shape: {predicted_token_ids.shape}")
        print(f"    - Token IDs: {predicted_token_ids.tolist()}")
        log_object(logger, "Predicted token IDs", predicted_token_ids)
        
        logger.info("Decoding token IDs to normalized actions...")
        print("[18] Decoding token IDs to normalized actions using ActionTokenizer")
        normalized_actions = model.action_tokenizer.decode_token_ids_to_actions(predicted_token_ids.cpu().numpy())
        print(f"    - Normalized actions shape: {normalized_actions.shape}")
        print(f"    - Normalized actions: {normalized_actions}")
        log_object(logger, "Normalized actions", normalized_actions)
        
        # Un-normalize actions
        logger.info("Un-normalizing actions...")
        print("[19] Un-normalizing actions to real-world values")
        action_stats = model.get_action_stats("bridge_orig")
        print(f"    - Action stats keys: {list(action_stats.keys())}")
        log_object(logger, "Action stats", action_stats)
        
        # Convert lists to numpy arrays if needed
        q01 = np.array(action_stats["q01"])
        q99 = np.array(action_stats["q99"])
        mask = action_stats.get("mask", np.ones_like(q01, dtype=bool))
        action_high, action_low = q99, q01
        
        print(f"    - Action mask type: {type(mask).__name__}")
        print(f"    - Action mask length: {len(mask) if hasattr(mask, '__len__') else 'N/A'}")
        print(f"    - Action high (q99): {action_high}")
        print(f"    - Action low (q01): {action_low}")
        log_object(logger, "Action mask", mask)
        log_object(logger, "Action high (q99)", action_high)
        log_object(logger, "Action low (q01)", action_low)
        
        print("[20] Applying un-normalization formula: 0.5 * (norm_actions + 1) * (high - low) + low")
        action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        print(f"    - Final action shape: {action.shape}")
        print(f"    - Final action values: {action}")
        logger.info("Action un-normalization complete")
        print("[21] Action prediction pipeline complete")
    
    log_object(logger, "Final predicted action", action, level=logging.INFO)
    
    print("\nPredicted Action (7-DoF):")
    print(action)
    logger.info("\nPredicted Action (7-DoF):")
    logger.info(str(action))
    
    # Print the action components with labels
    print("\nAction components:")
    logger.info("\nAction components:")
    components = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i, (component, value) in enumerate(zip(components, action)):
        component_str = f"  {component}: {value:.4f}"
        print(component_str)
        logger.info(component_str)
    
    # Log a summary of the execution directly to file
    with open(args.log_file, 'a') as f:
        f.write("\n==============================================================\n")
        f.write("                   EXECUTION SUMMARY                     \n")
        f.write("==============================================================\n\n")
        f.write(f"Model: OpenVLA\n")
        f.write(f"Vision backbone: {type(model.vision_backbone).__name__}\n")
        f.write(f"LLM backbone: {type(model.llm_backbone).__name__}\n")
        f.write(f"Input image: {args.image_path}\n")
        f.write(f"Instruction: '{args.instruction}'\n")
        f.write(f"Output action: {action}\n")
        
    # Also log to the standard logger
    logger.info("\n=== EXECUTION SUMMARY ===")
    logger.info(f"Model: OpenVLA")
    logger.info(f"Vision backbone: {type(model.vision_backbone).__name__}")
    logger.info(f"LLM backbone: {type(model.llm_backbone).__name__}")
    logger.info(f"Input image: {args.image_path}")
    logger.info(f"Instruction: '{args.instruction}'")
    logger.info(f"Output action: {action}")
    
    logger.info(f"Detailed execution log saved to: {args.log_file}")
    print(f"\nDetailed execution log saved to: {args.log_file}")

if __name__ == "__main__":
    main()
