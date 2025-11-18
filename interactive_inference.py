import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

# Set hyperparameters and choose full-training output model best_bert_model(3).bin
# Initialize parameters consistent with training
MODEL_NAME = 'bert-base-uncased'
MODEL_PATH = 'best_bert_model(3).bin'
MAX_LENGTH = 512

#Select the device to use for computation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Text preprocessing function, processes user input. Since input is checked for non-empty values, there is no need to filter out empty values.
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)          # Remove HTML tags
    text = re.sub(r'http\S+', '', text)        # Remove URL
    text = re.sub(r'@\w+', '', text)           # Remove @mention
    text = re.sub(r'\s+', ' ', text).strip()   # Merge spaces
    return text

#Load the pre-trained model from disk
print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"The model has been loaded onto the device:{DEVICE}")

# Use functions to predict results after loading the model
def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "Positive" if prediction == 1 else "Negtive"

#Main Loop
print("\nWelcome to the IMDB sentiment analysis system!\n")
print("Please enter an English movie review, and the model will determine its sentiment.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Please enter a comment:").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    if not user_input:
        print("Input cannot be empty, please try again.")
        continue

    try:
        result = predict_sentiment(user_input)
        print(f"Prediction result：{result}\n")
    except Exception as e:
        print(f"Reasoning error：{e}\n")