from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# inputs = tokenizer("We are very happy to show you the Transformers library.")
# print(inputs)

pt_batch = tokenizer(
    ["We are very happy to show you the Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")

pt_outputs = pt_model(**pt_batch)
print(pt_outputs)
# pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# classifier = pipeline('sentiment-analysis')
# results = classifier(["We are very happy to show you the Transformers library.", "We hope you don't hate it."])
#
# for result in results:
#     print(f"label:{result['label']}, with score:{round(result['score'], 4)}")
