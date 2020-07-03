import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')      # 加载预训练模型tokenizer（vocabulary）
model = GPT2LMHeadModel.from_pretrained('gpt2')        # 加载预训练模型(weights)

text = 'hey jude, do not make it bad. take a sad song and make it'
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])

model.eval()                                           # 将模型设置为evaluation模式，关闭dropout模块

tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

print(predicted_text)
