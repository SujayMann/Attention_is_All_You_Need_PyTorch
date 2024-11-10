import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from transformers import MarianTokenizer
import matplotlib.pyplot as plt
from typing import Tuple

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Function to generate a square subsequent mask for the decoder
def generate_square_subsequent_mask(sz) -> torch.Tensor:
    """Generates look-ahead mask for decoder."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create source and target masks."""
    tgt_seq_len = tgt.shape[1]
    # Encoder mask (padding mask)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
    # Decoder mask (padding mask + look-ahead mask)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
    seq_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    # Expand seq_mask to have the same dimensions as tgt_mask before the bitwise operation
    seq_mask = seq_mask.unsqueeze(0).expand(tgt_mask.shape[0], -1, -1)  # Remove one dimension here
    # Convert seq_mask to bool before the bitwise operation
    seq_mask = seq_mask.type(torch.bool)
    return src_mask, seq_mask  # Return the correct masks

def calculate_bleu_score(references, candidate):
    """Calculate BLEU score for translations."""
    references = [ref.split() for ref in references]
    candidate = candidate.split()
    bleu = sentence_bleu(references, candidate)
    return bleu

# Training function
def train(model, train_dataloader, optimizer, criterion, scheduler, device, epoch, max_grad_norm=1.0) -> float:
    """Train model for a single epoch."""
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False)):
        src = batch['input_ids'].to(device)
        tgt = batch['labels'].to(device)

        # Create masks
        src_mask, tgt_mask = create_mask(src, tgt[:, :-1], tokenizer.pad_token_id)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # Forward pass
        output = model(src, tgt_input, src_mask,tgt_mask)  # Pass the new masks

        # Calculate the loss
        loss = criterion(output.view(-1, output.size(-1)), tgt_labels.contiguous().view(-1))

        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()

        # Update scheduler
        scheduler.step(epoch=None)

        # Accumulate the loss for monitoring
        total_loss += loss.item()

        # Print every 200 batches
        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Compute average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
    return avg_loss

# Evaluation function
def evaluate(model, val_dataloader, criterion, device, tokenizer) -> Tuple[float, float]:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0
    all_references = []
    all_candidates = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating", leave=False)):
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            src_mask = (src != tokenizer.pad_token_id).to(device)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            # Forward pass
            output = model(src, tgt_input, src_mask[:, None, None, :], src_mask[:, None, None, :])

            # Calculate the loss
            loss = criterion(output.view(-1, output.size(-1)), tgt_labels.contiguous().view(-1))
            total_loss += loss.item()

            predicted_ids = torch.argmax(output, dim=-1)
            predicted_translations = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            reference_translations = tokenizer.batch_decode(tgt_labels, skip_special_tokens=True)

            all_references.extend(reference_translations)
            all_candidates.extend(predicted_translations)

    avg_loss = total_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    bleu_score = 0
    for ref, cand in zip(all_references, all_candidates):
        bleu_score += calculate_bleu_score([ref], cand)

    bleu_score /= len(all_references)
    print(f"BLEU score: {bleu_score:.4f}")
    return avg_loss, bleu_score

# Plotting
def plot_loss_graphs(train_losses, val_losses, bleu_scores) -> None:
    """Plot loss and BLEU score curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(epochs, train_losses, label='Training Loss')
    ax1.plot(epochs, val_losses, label='Validation Loss')
    ax2.plot(epochs, bleu_scores, label='BLEU Score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax1.set_title('Training and Validation Loss over Epochs')
    ax2.set_title('BLEU Score over Epochs')
    ax1.legend()
    ax2.legend()
    plt.show()
