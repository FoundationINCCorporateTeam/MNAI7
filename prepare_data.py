import json
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

# Load corpus
with open('corpus.json', 'r') as f:
    corpus = json.load(f)

class WikiDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        input_ids = tokens.ids[:-1]
        target_ids = tokens.ids[1:]
        return input_ids, target_ids

wiki_dataset = WikiDataset(corpus, tokenizer, max_length=512)
dataloader = DataLoader(wiki_dataset, batch_size=8, shuffle=True)

# Save dataloader to a file for later use
torch.save(dataloader, 'dataloader.pth')
