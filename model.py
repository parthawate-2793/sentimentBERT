import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# %%
# import pandas as pd
# Load the CSV dataset

dataset_path = r'C:\Users\tejna\OneDrive\Desktop\project2.0\Reviews.csv'  # Replace with the actual path to your dataset
df = pd.read_csv(dataset_path)
data= df.sample(30)

# Get the shape of the DataFrame
dataset_size = data.shape

# Print the number of rows and columns
print(f"Number of rows: {dataset_size[0]}")
print(f"Number of columns: {dataset_size[1]}")


# Function to map the Score to sentiment
def map_score_to_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score in [3, 4, 5]:
        return 'positive'
    else:
        return 'unknown'  # Handle any other values gracefully

# Apply the mapping function to create the 'sentiment' column
data['sentiment'] = data['Score'].apply(map_score_to_sentiment)

# Save the modified dataset to a new CSV file or overwrite the existing one
# output_csv_path = '/kaggle/working/amazon-product-reviews-modified.csv'  # Use '/kaggle/working/' as a writable directory
output_csv_path = r'C:\Users\tejna\OneDrive\Desktop\project2.0\Reviews_with_sentiment.csv'
data.to_csv(output_csv_path, index=False)

# Display the first few rows of the modified dataset
print(data.head())


# %%
def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['Text'].tolist()
    labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]
    return texts, labels

# %%
data_file = r'C:\Users\tejna\OneDrive\Desktop\project2.0\Reviews_with_sentiment.csv'
texts, labels = load_imdb_data(data_file)

# %%
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


# %%
# class BERTClassifier(nn.Module):
#     def _init_(self, bert_model_name, num_classes):
#         super(BERTClassifier, self)._init_()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         x = self.dropout(pooled_output)
#         logits = self.fc(x)
#         return logits



class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits



# %%
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

# %%
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

#%%
# def predict_sentiment(text, model, tokenizer, device, max_length=128):
#     model.eval()
#     encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         _, preds = torch.max(outputs, dim=1)
#     return "positive" if preds.item() == 1 else "negative"


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Ensure token_type_ids is present in the input
    token_type_ids = encoding.get('token_type_ids', None)
    
    with torch.no_grad():
        if token_type_ids is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        _, preds = torch.max(outputs, dim=1)
    
    return "positive" if preds.item() == 1 else "negative"







# %%
# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 6
batch_size = 8
num_epochs = 3
learning_rate = 2e-5

# %%
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# %%
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

# %%
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# %%
for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)


model_path = r'C:\Users\tejna\OneDrive\Desktop\project2.0\bert_classifier_model.pth'
torch.save(model.state_dict(), model_path)

tokenizer_path = r'C:\Users\tejna\OneDrive\Desktop\project2.0\bert_tokenizer.pkl'
with open(tokenizer_path, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
# %%
torch.save(model.state_dict(), "bert_classifier.pth")

# %%
# Test sentiment prediction
test_text = "the product built quality is good,the price is not ok,the design is not great,battery life is not great, but the functionality is not great.overall it is not a good product"
sentiment = predict_sentiment(test_text, model, tokenizer, device)
print(test_text)
print(f"Predicted sentiment: {sentiment}")

# import pickle
# with open('pick.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)


# Load the Python script
script_path = 'model.py'
with open(script_path, 'r') as script_file:
    kaggle_code = script_file.read()

# Save the code in a pickle file
pickle_path = 'kaggle_code.pkl'
with open(pickle_path, 'wb') as pickle_file:
    pickle.dump(kaggle_code, pickle_file)

print(f"Kaggle notebook code has been saved in '{pickle_path}'")