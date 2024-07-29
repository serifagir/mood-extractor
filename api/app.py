import gradio as gr
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn


df = pd.read_csv("data/moods.csv")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

NUM_CLASSES = len(label_encoder.classes_)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv_layer_1 = nn.Conv2d(1, 100, (3, embed_size))
        self.conv_layer_2 = nn.Conv2d(1, 100, (4, embed_size))
        self.conv_layer_3 = nn.Conv2d(1, 100, (5, embed_size))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)

        x1 = torch.relu(self.conv_layer_1(x)).squeeze(3)
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)

        x2 = torch.relu(self.conv_layer_2(x)).squeeze(3)
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = torch.relu(self.conv_layer_3(x)).squeeze(3)
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)

        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)

        return self.fc(x)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_path = 'mood_extractor_model_v1.pkl'
model = CNNModel(len(tokenizer.vocab), 128, NUM_CLASSES)  # Use the same vocab size and embedding size
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

MAX_LEN = 128

def preprocess_input(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()

def predict(text, model, tokenizer, max_len, label_encoder):
    input_ids, attention_mask = preprocess_input(text, tokenizer, max_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        _, preds = torch.max(outputs, dim=1)
        predicted_label = label_encoder.inverse_transform(preds.cpu().numpy())[0]

    return predicted_label

def predict_output(input_text):
    predicted_label = predict(input_text, model, tokenizer, MAX_LEN, label_encoder)
    return {predicted_label}

# input_text = "i have an exam tomorrow and i feel i will have low grade"
# predicted_label = predict(input_text, model, tokenizer, MAX_LEN, label_encoder)
# print(f'Predicted label: {predicted_label}')

# def greet(name):
#     return "Hello " + name + "!!"


demo = gr.Interface(fn=predict_output, inputs="textbox", outputs="text")
demo.launch()
