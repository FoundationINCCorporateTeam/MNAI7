import datasets

# Load the Wikipedia dataset
dataset = datasets.load_dataset('wikipedia', '20200501.en', split='train')
print(f"Dataset size: {len(dataset)} articles")

def get_texts(dataset, num_texts=10000):
    texts = []
    for i, data in enumerate(dataset):
        texts.append(data['text'])
        if i >= num_texts - 1:
            break
    return texts

corpus = get_texts(dataset)

# Save corpus to a file for later use
import json
with open('corpus.json', 'w') as f:
    json.dump(corpus, f)
