# ARC
An attempt at solving the ARC-AGI challenge. Done during the Meta-Learning course offered at IIIT-Delhi by Dr. Gautam Shroff.

## Approach
The solution employs a **Qwen2.5-0.5B-Instruct** language model with a combination of meta-learning techniques:

- **In-Context Learning**: Few-shot prompting where the model adapts to tasks using provided examples without gradient updates
- **Fine-tuning with LoRA**: Transductive learning using Low-Rank Adaptation (rank 32) for efficient parameter updates
- **Data Augmentation**: Geometric transformations and color permutations to improve generalization
- **Test-Time Adaptation**: Fine-tuning on test examples during inference, allowing the model to adapt to new task patterns on-the-fly

This approach combines the model's pre-trained reasoning capabilities with task-specific adaptation, addressing the challenge of solving diverse abstract reasoning tasks with limited examples.

## Installation
In order to run the project, simply run pre-installs.py
This will create the data and saved_models directory and populate them accordingly.
Make sure to switch to GPU before running train/test.
Everything should run smoothly.
