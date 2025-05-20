# Continual Learning for Multilingual Indic ASR on IndicSUPERB

WandB Results - [https://wandb.ai/frozenwolf/CL-ASR/](https://wandb.ai/frozenwolf/CL-ASR/)

This project investigates and implements Continual Learning (CL) strategies for Automatic Speech Recognition (ASR) systems, specifically targeting the linguistic diversity of India. Traditional multilingual models, requiring simultaneous access to all language data, are often impractical due to sequential data arrival and privacy constraints. CL enables models to learn new languages sequentially without catastrophically forgetting prior knowledge.

This work focuses on ASR for Indian languages using a subset of the **IndicSUPERB benchmark**. We employ a **Conformer-based hybrid RNNT-CTC model**, initially pretrained on Hindi, which is subsequently trained incrementally on eight additional Indian languages, for a sequence of nine languages in total. We evaluate three prominent regularization and distillation-based CL strategies: **Elastic Weight Consolidation (EWC)**, **Memory Aware Synapses (MAS)**, and **Learning without Forgetting (LwF)**, chosen for their suitability in no-replay, privacy-conscious scenarios.

Performance is analyzed using Word Error Rate (WER) for both RNNT and CTC paths on clean/noisy data, and knowledge retention via Backward Transfer (BWT). We explore varying training epochs (1, 2, 5, and 10) per task. The results, compared against naive fine-tuning, demonstrate CL’s efficacy in mitigating forgetting for scalable ASR in diverse Indian languages under realistic constraints.

This repository contains the code and configurations used for these experiments, leveraging the NVIDIA NeMo toolkit.

## 🌟 Key Features

*   **Continual Learning Strategies:**
    *   Elastic Weight Consolidation (EWC)
    *   Learning without Forgetting (LwF) - Note it uses online learning setup (thus extremely slow and need to be changed for batch runs)
    *   Memory Aware Synapses (MAS)
    *   A general CL baseline framework.
*   **Multilingual Indic ASR:**
    *   Focus on a sequence of 9 Indian languages from the IndicSUPERB benchmark.
    *   Initial pretraining on Hindi, followed by incremental learning of 8 other languages.
*   **Model Architecture:**
    *   Conformer-based hybrid RNNT-CTC model.
    *   Custom model definitions likely in `custom_model.py` and explored in `IndicConformerASR.ipynb`.
*   **Privacy-Conscious Approach:** Utilizes "no-replay" CL strategies suitable for scenarios with data privacy concerns.
*   **NVIDIA NeMo Integration:** Leverages the NeMo toolkit for building, training, and evaluating ASR models.
*   **Fine-tuning & Baselines:** Scripts and configurations for standard fine-tuning (as a baseline) and CL methods.
*   **Comprehensive Evaluation:**
    *   Word Error Rate (WER) for both RNNT and CTC decoders.
    *   Evaluation on clean and noisy speech data.
    *   Backward Transfer (BWT) to measure knowledge retention.
    *   Analysis of varying training epochs (1, 2, 5, 10) per language task.
*   **Experiment Management:**
    *   Configuration-driven experiments (`config.yaml`, `finetune_config.yaml`).
    *   Results logging and analysis (`results/`, `results.ipynb`, `results.py`).
    *   Weights & Biases (`wandb/`) integration for experiment tracking.
*   **HPC Ready:** Includes `sbatch.sh` for submitting jobs to Slurm-managed clusters.
*   **Dataset Handling:** Notebook for dataset generation/preprocessing (`dataset_gen.ipynb`).

## 📂 Project Structure

```
├── cl_baseline_ewc.py # Continual Learning with Elastic Weight Consolidation (EWC)
├── cl_baseline_lwf.py # Continual Learning with Learning without Forgetting (LwF)
├── cl_baseline_mas.py # Continual Learning with Memory Aware Synapses (MAS)
├── cl_baseline.py # Base script/framework for CL baselines (and naive fine-tuning)
├── custom_model.py # Definition of custom ASR models (e.g., Conformer RNNT-CTC)
├── NeMo/ # Potentially NVIDIA NeMo toolkit or related custom modules/fork
├── runs/ # Directory containing all .sh scripts
├── dataset_gen.ipynb # Jupyter notebook for dataset generation or preprocessing (e.g., IndicSUPERB subset)
├── output/ # General output directory for trained models, checkpoints
├── sbatch.sh # Slurm batch script for running jobs on an HPC cluster
├── finetune_config.yaml # Configuration for naive fine-tuning experiments
├── finetune.py # Script for naive fine-tuning ASR models - Non CL setup
├── results/ # Directory contains all the plots in pdf
├── utils.py # Utility functions and helper scripts
├── config.yaml # Main configuration file for CL experiments
├── IndicConformerASR.ipynb # Jupyter notebook explorin the Conformer ASR model
├── results.ipynb # Jupyter notebook for analyzing and visualizing experiment results
├── results.py # Python script for processing and aggregating results
└── wandb/ # Directory for Weights & Biases logs and artifacts


```


## ⚙️ Setup



# Installation Instructions

This document outlines the steps to set up the environment and install necessary dependencies for this project.

## 1. Create and Activate Conda Environment

```bash
conda create -n indiaai python=3.8 # Or your preferred Python version (e.g., 3.9)
conda activate indiaai
```
## 2. Install System Dependencies

These are required for audio file handling and other utilities.
```bash
sudo apt-get update
sudo apt-get install libsndfile-dev libsndfile1
sudo apt-get install ffmpeg # For general audio/video processing
```

## 3. Install PyTorch

Install PyTorch with CUDA 11.8 support. Adjust the CUDA version (cu118) if your GPU requires a different one (e.g., cu117, cu121).
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 4. Install Python Packages
```bash
pip install huggingface-hub==0.23.2
pip install transformers==4.40.0
pip install sentencepiece
pip install ffmpeg-python # Python bindings for ffmpeg
pip install inflect
pip install "sacremoses>=0.0.43"
pip install omegaconf wandb editdistance linecache py3nvml librosa
```

## 5. Set up Weights & Biases (W&B)

Log in to your Weights & Biases account.
```bash
# You will be prompted to enter your API key, or you can provide it directly:
# wandb login YOUR_API_KEY
wandb login 283c41dda88b658ba85c2d8ee7d37230f3341d8c
```
(Note: The API key 283c41dda88b658ba85c2d8ee7d37230f3341d8c was in your log. Replace it if it's a personal/sensitive key or use interactive login wandb login.)

## 6. Install Project Code (Editable Mode)

Install the current project and the NeMo subdirectory in editable mode. This allows changes in the source code to be reflected immediately without reinstalling.

Ensure you are in the root directory of this project.
```
cd NeMo
pip install -e .
cd .. # Return to the project root directory
```


## 7. Dataset Setup

Download the dataset.pkl files from the following Hugging Face Dataset link:

Download Link: [https://huggingface.co/datasets/FrozenWolf/Indic-CL-ASR](https://huggingface.co/datasets/FrozenWolf/Indic-CL-ASR)

or you can generate the dataset from [https://github.com/AI4Bharat/IndicSUPERB](IndicSUPERB) and run dataset_gen.ipynb


After downloading, place these files in your desired project location and update their paths accordingly in your ```config.yaml``` file.

## 🚀 Usage

### 1. Configuration

*   **`config.yaml`**: Main configuration for CL experiments.
    *   Define the sequence of languages.
    *   Set paths to datasets for each language/task.
    *   Specify model parameters (e.g., Conformer configuration).
    *   Set CL strategy hyperparameters (e.g., EWC lambda, LwF alpha).
    *   Define number of training epochs per task.
*   **`finetune_config.yaml`**: Configuration for naive fine-tuning baseline experiments.

### 2. Running Experiments

The CL experiments involve training the model sequentially on a series of language tasks.

#### a. Naive Fine-tuning (Baseline):

This typically involves training on each new language task without any CL mechanism.
```bash
# Example for fine-tuning on a new language task
python finetune.py --config finetune_config.yaml --task_id <task_index> --output_dir ./output/finetune_task_<task_index>
```

(The finetune.py script would need to handle loading the model from the previous task).

#### b. Continual Learning Baselines:

Execute the desired CL baseline script. These scripts should handle the sequential training paradigm.

```bash
# Example for EWC across a sequence of tasks
# (The script likely iterates through tasks internally based on config.yaml)
python cl_baseline_ewc.py --config config.yaml --output_dir ./output/ewc_experiment
```

```bash
# Example for LwF
python cl_baseline_lwf.py --config config.yaml --output_dir ./output/lwf_experiment
```

```bash
# Example for MAS
python cl_baseline_mas.py --config config.yaml --output_dir ./output/mas_experiment
```

(The exact command-line arguments and execution flow will depend on your script implementations. The scripts should be designed to handle the sequence of tasks described in the abstract.)

#### c. Using Slurm (HPC):

Modify sbatch.sh with the correct paths, module loads, and Python execution commands for your CL or fine-tuning experiments. Then submit:
```bash
sbatch sbatch.sh
```

### 3. Using Jupyter Notebooks

dataset_gen.ipynb: Open and run cells to prepare or generate your IndicSUPERB subset and other language data.

IndicConformerASR.ipynb: Explore the Conformer ASR model, potentially for initial model checks or demonstrations.

results.ipynb: Load and analyze experiment results (WER, BWT) stored in results/, runs/, or from W&B.

### 4. Analyzing Results

Check the runs/ directory for detailed logs (e.g., TensorBoard).

Check the wandb/ directory for W&B sync data or view results on the W&B dashboard.

Use results.py to process raw output files into aggregated metrics.
Use results.ipynb for custom analysis, plotting WER, BWT, and comparing CL strategies against naive fine-tuning. Output artifacts might be saved in results/.

![Screenshot from 2025-05-20 12-17-23](https://github.com/user-attachments/assets/d9d5e172-94df-453b-a373-947b5e446288)
![Screenshot from 2025-05-20 12-17-30](https://github.com/user-attachments/assets/3e12df17-8480-401a-931f-895c2ecbea35)
![Screenshot from 2025-05-20 12-17-42](https://github.com/user-attachments/assets/9b04c671-0793-48ee-b09e-d9e63e08987b)
![Screenshot from 2025-05-20 12-17-59](https://github.com/user-attachments/assets/3382c9e8-362a-4ebb-b4c3-855408e7b34e)
![Screenshot from 2025-05-20 12-18-09](https://github.com/user-attachments/assets/4a21f5bb-0871-433c-b93d-cccdbea52241)
![Screenshot from 2025-05-20 12-18-22](https://github.com/user-attachments/assets/71e8e19d-4b6c-4d97-b7e0-6f56e48e69b3)
![Screenshot from 2025-05-20 12-18-36](https://github.com/user-attachments/assets/6c515414-cb06-45b4-8b3d-bf39f3ae0e69)
![Screenshot from 2025-05-20 12-18-49](https://github.com/user-attachments/assets/ebde61c6-56aa-4be9-974d-1efe25153c9d)
![Screenshot from 2025-05-20 12-19-00](https://github.com/user-attachments/assets/98380e93-3c68-46e9-80c4-7c542938a1fe)
![Screenshot from 2025-05-20 12-19-09](https://github.com/user-attachments/assets/590d9122-2f2c-4f81-86c2-8f53ba9b9082)
![Screenshot from 2025-05-20 12-19-17](https://github.com/user-attachments/assets/1c674c13-9ab9-49aa-9e30-c800aa61f2e4)
![Screenshot from 2025-05-20 12-19-25](https://github.com/user-attachments/assets/2c494b82-f916-4260-94f3-e595ceb9b99b)
![Screenshot from 2025-05-20 12-19-51](https://github.com/user-attachments/assets/43786d69-36f3-4052-ace1-98492339e11e)
![Screenshot from 2025-05-20 12-16-48](https://github.com/user-attachments/assets/ec89b2c7-4a91-4e61-98ba-dcaa48fe511d)


