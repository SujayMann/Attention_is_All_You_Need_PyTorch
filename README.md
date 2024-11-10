# Research_Paper_Implementation
Implementation of the research paper **'Attention is All You Need'** in **PyTorch**.

This project implements the Transformer model for machine translation between English and German using the WMT 2014 dataset. The architecture follows the 'Attention is All You Need' paper and is implemented in PyTorch.

# Overview
The transformer model is a deep learning architecture for sequence-to-sequence modelling tasks and relies solely on the attention mechanism.

Components of architecture:
* Position Encoding
* Multi-head Self Attention (implements Scaled Dot Product Attention)
* Feed Forward Network
* Encoder
* Decoder
* Transformer

A custom class `TranslationDataset` is used to tokenize the dataset using `MarianTokenizer`.

A custom class `CustomScheduler` is used for learning rate scheduling based on the implementation in the research paper. The difference being `warmup_steps=2000` instead of 4000.

Utility functions for creating masks, training, evaluating, calculating BLEU-score and plotting loss curves are used.

# Dataset

Dataset used is the [WMT 2014 English-German Dataset.](https://huggingface.co/datasets/wmt/wmt14/viewer/de-en)

Load the dataset using the code below:

```python
dataset = load_dataset("wmt14", "de-en")
train_dataset = dataset['train']
test_dataset = dataset['test']
val_dataset = dataset['validation']
```

Due to hardware limitations, I have only used 1% of the data which is roughly 45000 rows instead of 4.5 Million rows.

# Hyperparameters
* batch_size: 32
* learning_rate: 1e-3
* num_epochs: 10 (due to hardware limitations)
* max_grad_norm: 1.0
* warmup_steps: 2000
* embedding dimension: 512
* num_heads: 8
* feed forward dimension: 2048
* num_layers: 6 (for encoder and decoder)
* dropout: 0.1

# File Overview
* `modules`: Contains various layers used in the Transformer (MultiHeadAttention, Encoder, Decoder...).
* `main.py`: Script for loading data, preprocessing, training and evaluating the model.
* `scheduler.py`: Script for custom learning rate scheduler.
* `tokenizer.py`: Script for tokenizing the data.
* `utils.py`: Contains various utility functions for training, evaluating, plotting.
