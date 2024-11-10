# Research_Paper_Implementation
Implementation of the research paper **'Attention is All You Need'** in **PyTorch**.

This project implements the Transformer model for machine translation between English and German using the WMT 2014 dataset. The architecture follows the 'Attention is All You Need' paper and is implemented in PyTorch.

Originally developed in Google Colab, leveraging Colab AI whenever possible, then shifted to Kaggle for training.

**Training time:** Approx. 10 hours

# Overview
The transformer model is a deep learning architecture for sequence-to-sequence modelling tasks and relies solely on the attention mechanism.

Components of architecture:
* Positional Encoding
* Multi-head Self Attention (implements Scaled Dot Product Attention)
* Feed Forward Network
* Encoder
* Decoder
* Transformer

## Positional Encoding
Positional Encodings are added to the input embeddings to retain the positional information of each word in the sequence.

Formula for positional encoding in the paper:

$$
PE(pos, 2i) = \sin \left( \frac{pos}{10000^{\frac{2i}{\verb|d_model|}}} \right)
$$

$$
PE(pos, 2i+1) = \cos \left( \frac{pos}{10000^{\frac{2i}{\verb|d_model|}}} \right)
$$

where:
* `pos` is the position of the word in sequence.
* `i` is the dimension of current embedding.
* `d_model` is embedding dimension (512).

## Scaled Dot Product Attention
Attention mechanism works by computing a set of attention weights based on query, key and value vectors.

Formula: 

$$
Attention(Q, K, V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

where:
* Q, K and V represent query, key and value matrices.
* d_k is dimension of key vectors used to scale the dot product to avoid large values.

## Multi Head Attention
The model used multiple attention heads to capture different types of relationships in the sequence. The outputs are concatenated and passed to a final linear layer.

Formula:

$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
$$

$$
where \\ head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

## Feed Forward Network
Each Feed Forward Network layer contains a Linear layer followed by a ReLU activation followed by another Linear layer.

The FFN helps the model learn complex transformations and representations for each word in the sequence.

Formula:

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

The dimensionality of input and output is $`d_{model} = 512`$, hidden layer has dimensionality $`d_{ff} = 2048`$

## Layer Normalization and Residual Connection
Each sub-layer (self-attention and feedforward network) has a residual connection around it, followed by layer normalization. This helps in improving convergence by allowing gradients to flow more easily through the network.

Formula:

$$
Output = LayerNorm(x + Sublayer(x))
$$

## Encoder
The encoder is responsible for processing the input sequence and generating a sequence of hidden states that the decoder will use for producing the output. It consists of a stack of 6 Encoder Layers which each contain: 
* `MultiHeadAttention` block followed by LayerNorm
* `FeedForwardNetwork` block followed by LayerNorm

## Decoder
The decoder generates the output sequence based on the encoded representations of the input sequence. Like the encoder, the decoder consists of a stack of 6 identical layers. Each Decoder Layer contains:
* `Masked MultiHeadAttention` block followed by LayerNorm
* `MultiheadAttention` block followed by LayerNorm
* `FeedForwardNetwork` block followed by LayerNorm

## Linear Output Layer and Softmax
After the final layer of the Decoder, a linear output layer projects a vector which contains likelihood of each word in the vocabulary.

Softmax function is applied get the probabilities and the word with highest probability is selected.

## Data Preprocessing
A custom class `TranslationDataset` is used to tokenize the dataset using `MarianTokenizer` from Hugging face `transformers` library. The tokenization includes padding, truncation, and converting text into numerical tokens.

The `TranslationDataset` class handles the tokenization process, and it ensures that both the source and target sequences are appropriately preprocessed for the model.

## Learning Rate Scheduler
A custom class `CustomScheduler` is used for learning rate scheduling based on the implementation in the research paper. The difference being `warmup_steps=2000` instead of 4000.

## Utility Functions

Some utility functions used for:
- **Creating masks**: For attention mechanisms.
- **Training**: Functions to train the model over multiple epochs.
- **Evaluation**: Functions to evaluate the model on the validation set.
- **BLEU score calculation**: Using the `nltk.translate` library to calculate BLEU score on the translated outputs.
- **Plotting loss curves**: Visualize the training and validation loss over epochs.

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
* max_len: 128

# Key differences

| Feature | Original paper | My implementation |
| --- | --- | --- |
| Dataset size | 4.5 Million rows | 45000k rows (1 percent of original) |
| Batch size | 2048 | 32 |
| Training steps | 100000 | 10 |
| GPUs | 8 P100s | 1 P100 (Kaggle environment) |
| Tokenization | Byte Pair Encoding | Marian Tokenizer |
| Warmup steps | 4000 | 2000 |
| max_len | 10000 | 128 |
| Bleu score | - | - |

# File Overview
* `modules`: Contains various layers used in the Transformer (MultiHeadAttention, Encoder, Decoder...).
* `main.py`: Script for loading data, preprocessing, training and evaluating the model.
* `scheduler.py`: Script for custom learning rate scheduler.
* `tokenizer.py`: Script for tokenizing the data.
* `utils.py`: Contains various utility functions for training, evaluating, plotting.

# Run the project

* Run the notebook `research-paper-implementation-project.ipynb`.

OR

* Install dependencies mentioned in `requirements.txt`
```cmd
pip install -r requirements.txt
```

* Run main.py
```cmd
python main.py
```
**Note:** Use a machine with a GPU and sufficient RAM.

Feel free to change the hyperparameters and the dataset size.
