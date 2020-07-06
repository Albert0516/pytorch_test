import torch
from transformers import pipeline

nlp_sentence_classification = pipeline('sentiment-analysis')
sc_result = nlp_sentence_classification('Such a nice weather outside!')

print("sentence classification:{}".format(sc_result))

nlp_token_class = pipeline('ner')
ner_result = nlp_token_class('Hugging Face is a French company based in New-York.')
print("NER Classification:{}".format(ner_result))

nlp_qa = pipeline('question-answering')
qa_answer = nlp_qa(context='Hugging Face is a French company based in New-York.', question='Where is based Hugging Face?')
print("QA Answer:{}".format(qa_answer))

nlp_fill = pipeline('fill-mask')
fill_mask = nlp_fill('Hugging Face is a French company based in'+nlp_fill.tokenizer.mask_token)
print('fill-mask result:{}'.format(fill_mask))



