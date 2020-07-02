import torch
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

text = 'what is the fastest car in the'
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])

model.eval()

tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

print(predicted_text)
