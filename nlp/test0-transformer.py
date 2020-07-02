from transformers import pipeline

print(pipeline('sentiment-analysis')('never say never.'))
