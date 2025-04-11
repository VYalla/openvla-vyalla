# Running OpenVLA Model Locally

This document provides a detailed explanation of how to run the OpenVLA model locally, including the key components involved in loading and running the model pipeline.

## Model Loading Architecture

The OpenVLA model uses a hierarchical loading system with several key components:

1. **OpenVLA Class**: The main model class that inherits from `PrismaticVLM` and adds action prediction capabilities
2. **Vision Backbone**: Handles image processing and feature extraction
3. **LLM Backbone**: Processes text and generates outputs
4. **Action Tokenizer**: Converts between discrete tokens and continuous robot actions

## Key Components and Their Locations

### Core Model Loading Functions

The main functions for loading the model are located in:
- **`prismatic/models/load.py`**
  - `load()`: Loads a pretrained PrismaticVLM (base vision-language model)
  - `load_vla()`: Loads a pretrained OpenVLA model (vision-language-action model)

### Model Implementation

- **`prismatic/models/vlas/openvla.py`**: Contains the `OpenVLA` class implementation
  - Inherits from `PrismaticVLM`
  - Adds the `predict_action()` method for generating robot actions
  - Handles action normalization/denormalization

### Backbone Components

- **Vision Backbone**: 
  - Located in `prismatic/models/backbones/vision/`
  - Loaded via `get_vision_backbone_and_transform()` in `prismatic/models/materialize.py`
  
- **LLM Backbone**: 
  - Located in `prismatic/models/backbones/llm/`
  - Includes `llama2.py` for the Llama 2 model implementation
  - Loaded via `get_llm_backbone_and_tokenizer()` in `prismatic/models/materialize.py`

### Action Handling

- **`prismatic/vla/action_tokenizer.py`**: Handles conversion between discrete tokens and continuous actions
- **Dataset Statistics**: Stored in `dataset_statistics.json` in the model checkpoint directory
  - Used for denormalizing predicted actions to real-world values

## Loading Process Explained

When you call `load_vla()`, the following steps occur:

1. **Checkpoint Identification**:
   - Validates the checkpoint path
   - Loads configuration from `config.json` and statistics from `dataset_statistics.json`

2. **Vision Backbone Loading**:
   - Loads the vision model specified in the config (e.g., `dinosiglip-vit-so-224px`)
   - Sets up the image transformation pipeline

3. **LLM Backbone Loading**:
   - Loads the language model specified in the config (e.g., `llama2-7b-pure`)
   - **Note**: This requires access to the Llama 2 model on Hugging Face
   - Creates the tokenizer for processing text

4. **Action Tokenizer Creation**:
   - Creates an `ActionTokenizer` instance using the LLM tokenizer
   - This handles conversion between discrete tokens and continuous actions

5. **OpenVLA Model Creation**:
   - Instantiates the `OpenVLA` class with the loaded components
   - Loads weights from the checkpoint file

## Running Inference Locally

To run inference with a local OpenVLA model, you need:

1. **Local Checkpoint**: The `.pt` file containing model weights
2. **Configuration Files**: `config.json` and `dataset_statistics.json`
3. **Access to Llama 2**: The model requires access to the Llama 2 model on Hugging Face

### Example Usage

```python
from prismatic.models.load import load_vla

# Load the model from checkpoint
model = load_vla("/path/to/checkpoint.pt")

# Run inference
action = model.predict_action(
    image=image,               # PIL Image
    instruction="pick up the object",
    unnorm_key="bridge_orig"   # Dataset key for denormalization
)
```

## Requirements for Local Execution

1. **Hugging Face Account**: You need to be logged in with `huggingface-cli login`
2. **Llama 2 Access**: You need to request and be granted access to the Llama 2 model
3. **Python Environment**: All required packages must be installed

## Alternative Approaches

If you don't have access to the Llama 2 model, you can:

1. Use the `run_inference.py` script which downloads the OpenVLA model directly
2. Modify the model to use an open-source LLM instead of Llama 2
3. Use the simulation environment with the pre-downloaded model

## Troubleshooting

### Common Issues

1. **Gated Repository Error**:
   ```
   huggingface_hub.errors.GatedRepoError: Cannot access gated repo
   ```
   - Solution: Request access to the Llama 2 model at https://huggingface.co/meta-llama/Llama-2-7b-hf

2. **Authentication Error**:
   ```
   You must be logged in to access this resource
   ```
   - Solution: Run `huggingface-cli login` and enter your access token

3. **Memory Issues**:
   - The model requires significant RAM/VRAM
   - Consider using quantization or a smaller model variant
