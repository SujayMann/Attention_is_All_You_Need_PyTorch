import torch
from torch import nn
from transformers import MarianTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizer import TranslationDataset
from modules.transformer import Transformer
from scheduler import CustomScheduler
from utils import train, evaluate, plot_loss_graphs

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

dataset = load_dataset("wmt14", "de-en")
train_dataset = dataset['train']
test_dataset = dataset['test']
val_dataset = dataset['validation']

one_percent_train = int(len(train_dataset) * 0.01) # 1% data
train_dataset = train_dataset.shuffle().select(range(one_percent_train))

# Create tokenized train and test datasets
train_dataset_tokenized = TranslationDataset(train_dataset, tokenizer)
test_dataset_tokenized = TranslationDataset(test_dataset, tokenizer)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset_tokenized, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset_tokenized, batch_size=32)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
learning_rate = 1e-3
num_epochs = 10
max_grad_norm = 1.0  # Gradient clipping

# Initialize model, optimizer, loss function and learning rate scheduler
model = Transformer(d_model=512, d_ff=2048, num_heads=8, target_size=len(tokenizer), num_layers=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scheduler = CustomScheduler(optimizer, d_model=512, warmup_steps=2000)

train_losses = []
val_losses = []
bleu_scores = []

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Training
    train_loss = train(model, train_dataloader, optimizer, criterion, scheduler, device, epoch, max_grad_norm)
    train_losses.append(train_loss)

    # Evaluation
    val_loss, bleu_score = evaluate(model, test_dataloader, criterion, device, tokenizer)
    val_losses.append(val_loss)
    bleu_scores.append(bleu_score)

    # Save the best model (based on validation loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_transformer_model_epoch_{epoch+1}.pt")

# Plot loss curves and BLEU score
plot_loss_graphs(train_losses, val_losses, bleu_scores)