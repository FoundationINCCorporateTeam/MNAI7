from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
import json

# Load corpus
with open('corpus.json', 'r') as f:
    corpus = json.load(f)

# Initialize a tokenizer
tokenizer = Tokenizer(BPE())

# Customize normalizers
tokenizer.normalizer = NFD() + Lowercase() + StripAccents()

# Customize pre-tokenizers
tokenizer.pre_tokenizer = Whitespace()

# Customize the post-processors
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

# Train the tokenizer
trainer = BpeTrainer(special_tokens=["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"])
tokenizer.train_from_iterator(corpus, trainer)

# Save the tokenizer
tokenizer.save("custom_tokenizer.json")
