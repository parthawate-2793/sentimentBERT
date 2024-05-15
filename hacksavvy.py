import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch.optim as optim

# Load the dataset
dataset_path = r"C:\Users\tejna\OneDrive\Desktop\training_set_rel3.csv"
df = pd.read_csv(dataset_path)

class BERTRegressor(nn.Module):
    def __init__(self, bert_model_name):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)  # Output a single value for regression

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        score = self.fc(x)
        return score

class EssayDataset(Dataset):
    def __init__(self, essays, scores, tokenizer, max_length):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        essay = self.essays[idx]
        score = self.scores[idx]
        encoding = self.tokenizer(essay, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'score': torch.tensor(score, dtype=torch.float)}

# Function to train the regression model
def train_regression(model, data_loader, optimizer, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(1)
        loss = nn.MSELoss()(predictions, scores)
        loss.backward()
        optimizer.step()

# Function to predict the score and generate feedback
def predict_score_and_feedback(model, tokenizer, essay_text, device):
    encoding = tokenizer(essay_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        score_prediction = model(input_ids=input_ids, attention_mask=attention_mask).item()
    feedback = generate_feedback(score_prediction)
    return score_prediction, feedback

# Function to generate feedback based on the predicted score
def generate_feedback(predicted_score):
    if predicted_score >= 4:
        return "Excellent work! Your essay demonstrates strong critical thinking and clear communication."
    elif predicted_score >= 3:
        return "Good effort! You've made some valid points, but there's room for improvement in organization and clarity."
    else:
        return "Your essay needs improvement. Focus on developing stronger arguments and organizing your ideas more clearly."

# Load the pre-trained BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Instantiate the BERTRegressor model
model = BERTRegressor(bert_model_name)

# Define optimizer and move model to device
optimizer = optim.AdamW(model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the regression model (You can add your own training logic here)

# Take input from the user
essay_text = input("Enter your essay: ")

# Predict the score and generate feedback
score_prediction, feedback = predict_score_and_feedback(model, tokenizer, essay_text, device)

# Output the predicted score and feedback
print(f"Predicted Score: {score_prediction}")
print("Feedback:")
print(feedback)