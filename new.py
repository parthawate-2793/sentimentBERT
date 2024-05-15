import pickle
import torch
from transformers import BertTokenizer
import streamlit as st


# Replace 'your_model_file' with the actual file name
from model import BERTClassifier  

# Load the saved model and tokenizer
model_path = r'C:\Users\tejna\OneDrive\Desktop\project2.0\bert_classifier_model.pth'
tokenizer_path = r'C:\Users\tejna\OneDrive\Desktop\project2.0\bert_tokenizer.pkl'


loaded_model = BERTClassifier(bert_model_name='bert-base-uncased', num_classes=2)
loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
loaded_model.eval()

with open(tokenizer_path, 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# Function to preprocess the input text and make predictions
# Function to preprocess the input text and make predictions
def predict_sentiment(text):
    tokenizer = loaded_tokenizer
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

    # Check if token_type_ids is present in the tokenizer output
    token_type_ids = inputs.get('token_type_ids', None)

    with torch.no_grad():
        outputs = loaded_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=token_type_ids)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Assuming 0 is negative and 1 is positive
    return "positive" if predicted_class == 1 else "negative"


# Streamlit web app
def main():
    st.title("Product Review Sentiment Predictor")

    # Input text area for the user to enter the review
    review_text = st.text_area("Enter your product review:")

    if st.button("Predict Sentiment"):
        # Make predictions when the button is clicked
        if review_text:
            sentiment = predict_sentiment(review_text)
            st.success(f"Predicted Sentiment: {sentiment.capitalize()}")
        else:
            st.warning("Please enter a review before predicting.")

# Run the Streamlit app
if __name__ == "__main__":
    main()