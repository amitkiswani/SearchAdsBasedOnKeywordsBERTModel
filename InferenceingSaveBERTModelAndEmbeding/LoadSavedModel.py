from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast  # For safely evaluating strings containing Python expressions

df = pd.read_csv('./model/embeddings.csv')
df['embedding'] = df['dim_0'].apply(ast.literal_eval)
model_path = './model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def suggest_apps(search_term):
    search_embedding = get_embedding(search_term).numpy()
    # Here df suppose to be 2d array but file only had 1 d so updated x to [x]
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], search_embedding)[0][0])
    return df.sort_values(by='similarity', ascending=True).head(5)['identifier']


# Example search term
search_term = "Shares"
suggestions = suggest_apps(search_term)
print(suggestions)
