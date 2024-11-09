from torch.utils.data import Dataset
from torch import nn

# Custom Dataset class for tokenization
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length: int=128, d_model: int=512) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding = nn.Embedding(len(tokenizer), d_model)  # Add embedding layer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text = self.dataset[idx]['translation']['de']
        labels = self.dataset[idx]['translation']['en']
        model_input = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(labels, truncation=True, max_length=self.max_length + 1,  # Add 1 for <start> token
                                    padding='max_length', return_tensors='pt')

        # Embed input and ensure correct dimensions
        embeddings = self.embedding(model_input['input_ids'].squeeze())
        
        return {
            'input_ids': embeddings,
            'labels': labels['input_ids'].squeeze()
        }
