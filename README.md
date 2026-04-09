# Ribosome-Cascade

A Priority-Driven, Non-Autoregressive Hierarchical Decoder. Instead of processing all tokens equally, the model isolates semantic boundaries, groups them into Metatokens, and processes the heaviest/most important concepts first to establish a semantic anchor, reducing context-loss over long windows.

## Overview

The Ribosome-Cascade architecture introduces a novel approach to natural language processing by acting as a "Ribosome Scorer" alongside a standard language model (like GPT-2). This mechanism helps determine the "weight" or "importance" of each token in a sequence, creating semantic boundaries.

By multiplying the hidden states by these importance scores, the model pays more attention to high-scoring words and builds "Metatokens"—grouped tokens around a semantic anchor.

## Key Concepts

* **Ribosome Scorer**: A lightweight neural network (e.g., sequence of Linear layers and GELU activation) that takes in contextual embeddings and outputs a score between 0 and 1 for each token, signifying its semantic importance.
* **Metatokens**: Chunks of text constructed around "peaks" (local maxima) in the Ribosome Scorer's output. Text is grouped by these boundaries, clustering less important filler words with their heavy anchor words.
* **Custom Loss Function**: Training is governed by two main components:
  * **Reconstruction Loss**: Standard cross-entropy loss to predict the next word.
  * **Sparsity Penalty**: A penalty on the average importance score, which prevents the Ribosome from assigning high importance to every token and encourages compression.

## Installation & Setup

Ensure you have a GPU-enabled environment (e.g., Google Colab with T4 GPU) and install the necessary libraries:

```bash
pip install torch transformers datasets accelerate
```

## Usage

The project is demonstrated in the `Project_Ribosome.ipynb` notebook. The pipeline generally follows these steps:

1. **Initialize the Base Model**: Load a standard pre-trained model (e.g., `gpt2`).
2. **Initialize the Ribosome**: Define the `RibosomeScorer` and append it to the base model.
3. **Data Preparation**: Load a dataset (e.g., `wikitext-2-raw-v1`), tokenize it, and add labels.
4. **Training**: Run the custom training loop using Hugging Face's `Trainer`. The combined model processes inputs, generates importance scores, scales the hidden states (Cascade Simulation), and computes the custom loss.
5. **Inference**: Given a sentence, the trained Ribosome evaluates token importance, identifying peaks and assembling Metatokens for optimized downstream processing.

```python
# Example inference snippet (simplified)
from transformers import AutoTokenizer

test_sentence = "The mechanism is absolutely revolutionary."
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = trainable_model.base_model(**inputs)
    hidden_states = outputs.last_hidden_state

    # Get importance scores
    scores = trainable_model.ribosome(hidden_states)

# Scores dictate Metatoken boundaries and semantic anchors
```
