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

