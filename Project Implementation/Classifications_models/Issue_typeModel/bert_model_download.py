# How to load it in your local VS Code later:
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained("mohamedelkady0/support-ticket-classifier")
model = DistilBertForSequenceClassification.from_pretrained("mohamedelkady0/support-ticket-classifier")