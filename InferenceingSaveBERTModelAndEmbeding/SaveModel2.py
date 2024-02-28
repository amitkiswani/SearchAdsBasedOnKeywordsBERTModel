#### Note in this code we are downloading the BERT model from hugging face and we save it into the file system.
####

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv

# Suppose model and tokenizer are your trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Saving
model_path = './model'
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

df = pd.read_csv('./corrected_dummy_app_store_data_short.csv')
model_path = './model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


identifier = df['AppInfo']
df['embedding'] = df['SampleSearchKeyword'].apply(lambda x: get_embedding(x).numpy())

# Open a CSV file to write
with open('./model/embeddings.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row, if desired
    embedding_dim = len(df['embedding'].iloc[0])
    header = ['identifier'] + [f'dim_{i}' for i in range(embedding_dim)]
    writer.writerow(header)

    # Write embeddings to the CSV file
    for index, row in df.iterrows():
        indent = identifier[index]
        row_item = row['embedding'].tolist()  # Convert numpy array to list if using numpy
        writer.writerow([indent] + row_item)
