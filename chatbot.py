import torch
from build_model import TransformerModel
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

# Model hyperparameters
VOCAB_SIZE = tokenizer.get_vocab_size()
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
MAX_SEQ_LENGTH = 512

# Load the model
model = TransformerModel(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH)
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = input_ids.shape[1] + 50  # Allow the model to generate 50 tokens more than the input
    with torch.no_grad():
        output = model(input_ids, input_ids)
        output_ids = torch.argmax(output, dim=-1)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def chat():
    print("Chat with the bot (type 'exit' to stop)")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
