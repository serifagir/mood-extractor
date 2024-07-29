import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
import numpy as np

df = pd.read_csv("training.csv")

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split dataset and reset index
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.text[index])
        target = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


train_dataset = TextDataset(train_df, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_df, tokenizer, MAX_LEN)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
  
VOCAB_SIZE = len(tokenizer.vocab)
EMBED_SIZE = 128
NUM_CLASSES = len(label_encoder.classes_)

model = CNNModel(VOCAB_SIZE, EMBED_SIZE, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train_model(model, data_loader, loss_fn, optimizer, device):
  model.train()
  losses = []
  correct_predictions = 0

  for data in data_loader:
    input_ids = data["input_ids"].to(device)
    targets = data["targets"].to(device)

    outputs = model(input_ids)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
  model.eval()
  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for data in data_loader:
      input_ids = data["input_ids"].to(device)
      targets = data["targets"].to(device)

      outputs = model(input_ids)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

EPOCHS = 10

for epoch in range(EPOCHS):
   print(f'Epoch {epoch + 1}/{EPOCHS}')
   print('-' * 10)

   train_acc, train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
   print(f'Train loss {train_loss} accuracy {train_acc}')

   val_acc, val_loss = eval_model(model, val_loader, loss_fn, device)
   print(f'Validation loss {val_loss} accuracy {val_acc}')



model_path = 'mood_extractor_model.pkl'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")