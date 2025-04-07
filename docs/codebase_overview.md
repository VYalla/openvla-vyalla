# OpenVLA Codebase Overview

## Introduction

OpenVLA (Open Vision-Language-Action) is an open-source framework for training and fine-tuning vision-language-action models for robotic manipulation. It builds on top of the Prismatic VLMs (Vision-Language Models) codebase and extends it to enable robots to perform actions based on visual input and language instructions.

The codebase provides a scalable architecture for training VLA models with different dataset mixtures, easy scaling capabilities, and native fine-tuning support. This document provides an overview of the codebase structure and explains the purpose of key components.

## Project Structure

### Root Directory

- `run_inference.py`: A simple script for running inference with OpenVLA models on a single image.
- `README.md`: Documentation for the project, including installation instructions, usage examples, and fine-tuning guides.
- `Makefile`: Contains commands for common operations in the project.
- `LICENSE`: MIT license for the codebase (note that model weights may have different licensing).

### Scripts Directory

- `scripts/pretrain.py`: Main script for pretraining Prismatic VLMs using PyTorch FSDP (Fully-Sharded Data Parallel).
- `scripts/preprocess.py`: Script for preprocessing datasets.
- `scripts/generate.py`: Script for generating outputs from trained models.
- `scripts/additional-datasets/`: Contains scripts for additional datasets like LRVIS and LVIS.
- `scripts/extern/`: Contains utility scripts for converting and verifying Prismatic weights.

### VLA Scripts Directory

- `vla-scripts/finetune.py`: Script for fine-tuning OpenVLA models for new tasks and embodiments, supporting different fine-tuning modes including LoRA.
- `vla-scripts/deploy.py`: Lightweight script for serving OpenVLA models over a REST API.

### Prismatic Core

The `prismatic/` directory contains the core components of the framework:

#### Configuration (`prismatic/conf/`)

- `models.py`: Configuration for different model architectures.
- `datasets.py`: Configuration for different datasets.
- `vla.py`: Configuration for VLA models, including training parameters, architecture settings, and dataset mixtures.

#### Models (`prismatic/models/`)

- `backbones/`: Contains implementations of vision and language model backbones.
- `vlas/`: Vision-Language-Action model implementations.
- `vlms/`: Vision-Language model implementations.
- `load.py`: Utilities for loading models.
- `materialize.py`: Utilities for materializing model configurations.
- `registry.py`: Registry of available models.

#### VLA Components (`prismatic/vla/`)

- `action_tokenizer.py`: Extension class that wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
- `datasets/`: Contains dataset implementations for VLA training.

#### Training (`prismatic/training/`)

- `strategies/`: Different training strategies (e.g., FSDP).
- `materialize.py`: Utilities for materializing training configurations.
- `metrics.py`: Metrics for evaluating model performance.

#### Preprocessing (`prismatic/preprocessing/`)

- `datasets/`: Dataset preprocessing utilities.
- `download.py`: Utilities for downloading datasets.
- `materialize.py`: Utilities for materializing preprocessing configurations.

#### Utilities (`prismatic/util/`)

- `batching_utils.py`: Utilities for batching data.
- `data_utils.py`: Utilities for data processing.
- `nn_utils.py`: Neural network utilities.
- `torch_utils.py`: PyTorch-specific utilities.

#### Overwatch (`prismatic/overwatch/`)

- `overwatch.py`: Logging and monitoring utilities.

### Experiments

- `experiments/robot/`: Contains utilities for robot experiments, including:
  - `bridge/`: Bridge dataset utilities.
  - `libero/`: LIBERO simulation benchmark utilities.
  - `openvla_utils.py`: Utility functions for OpenVLA.
  - `robot_utils.py`: Utility functions for robot control.

## Key Components

### Action Tokenizer

The `ActionTokenizer` class in `prismatic/vla/action_tokenizer.py` is a core component that discretizes continuous robot actions into N bins per dimension and maps them to tokens in the vocabulary. This enables the model to generate discrete action tokens that can be converted back to continuous actions for robot control.

### VLA Configurations

The `VLAConfig` class in `prismatic/conf/vla.py` defines the configuration for VLA models, including:
- Base VLM model
- Freezing options for vision and language backbones
- Data mixture parameters
- Optimization parameters
- Training strategy

The file also defines various experiment configurations for different datasets and model architectures.

### Inference Pipeline

The `run_inference.py` script demonstrates how to use OpenVLA models for inference:
1. Load the model and processor
2. Process an input image and instruction
3. Generate an action prediction
4. Convert the action tokens back to continuous actions for robot control

### Fine-tuning

OpenVLA supports different fine-tuning approaches:
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning that requires less compute.
- **Full Fine-tuning**: Complete fine-tuning of the model, which requires more compute but can achieve better performance.
- **Partial Fine-tuning**: Freezing certain components (e.g., vision backbone) while fine-tuning others.

## Model Architecture

OpenVLA models consist of three main components:
1. **Vision Encoder**: Processes input images (e.g., SigLIP, DINOv2)
2. **Language Model**: Processes text instructions (e.g., Llama-2)
3. **Action Generation**: Generates discretized actions that can be converted to continuous robot commands

The architecture follows a vision-language-action paradigm where:
- Visual features are extracted from input images
- Language features are extracted from instructions
- The combined features are used to predict appropriate actions

## Available Models

OpenVLA provides two main pretrained models:
- `openvla-7b`: The flagship model trained from the Prismatic `prism-dinosiglip-224px` VLM (based on a fused DINOv2 and SigLIP vision backbone, and Llama-2 LLM).
- `openvla-v01-7b`: An early development model trained from the Prismatic `siglip-224px` VLM (singular SigLIP vision backbone, and a Vicuña v1.5 LLM).

## Training and Fine-tuning

The codebase supports:
- Pretraining VLA models from scratch
- Fine-tuning existing models for new tasks
- Parameter-efficient fine-tuning via LoRA
- Evaluation on benchmark datasets like BridgeData V2 and LIBERO

## Conclusion

OpenVLA provides a comprehensive framework for training and deploying vision-language-action models for robotic manipulation. The modular architecture allows for easy experimentation with different model components, training strategies, and datasets.
