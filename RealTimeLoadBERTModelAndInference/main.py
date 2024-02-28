from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the dataset
df = pd.read_csv('./corrected_dummy_app_store_data_short.csv')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


# Compute embeddings for the app descriptions
df['embedding'] = df['SampleSearchKeyword'].apply(lambda x: get_embedding(x).numpy())


def suggest_apps(search_term):
    search_embedding = get_embedding(search_term).numpy()
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_embedding)[0][0])
    return df.sort_values(by='similarity', ascending=False).head(5)['AppInfo']


# Example search term
search_term = "Neural Network"
suggestions = suggest_apps(search_term)
print(suggestions)
