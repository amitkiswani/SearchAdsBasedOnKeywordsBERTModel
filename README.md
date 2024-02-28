# SearchAdsBasedOnKeywordsBERTModel
Search Ads Based On the Keywords BERT Model and utilized cosign similarity

This code snippet demonstrates how to use BERT (Bidirectional Encoder Representations from Transformers) to generate embeddings for app descriptions (or in this specific case, for sample search keywords) and then find the most similar apps to a given search term based on cosine similarity. Here's a step-by-step breakdown:

**Imports and Setup:**

The script imports necessary libraries: BertTokenizer and BertModel from transformers for working with a pre-trained BERT model, torch for tensor operations, cosine_similarity from sklearn to compute similarity between embeddings, and pandas to handle data in DataFrame format.

**Load Dataset:**

A DataFrame df is created by reading data from a CSV file, which presumably contains app information.
Initialize BERT Tokenizer and Model:
The BERT tokenizer (BertTokenizer) and model (BertModel) are loaded with a pre-trained 'bert-base-uncased' version. The tokenizer converts text into tokens that BERT understands, and the model generates embeddings for the tokens.

**Define get_embedding Function:**

This function takes a piece of text as input, tokenizes it, passes it through the BERT model, and returns the mean of the last hidden state as the representation (embedding) of the input text. This embedding captures the semantic meaning of the text in a high-dimensional space.
Compute Embeddings for Sample Search Keywords:
The script computes embeddings for each sample search keyword in the DataFrame. It applies the get_embedding function to the 'SampleSearchKeyword' column and stores the result in a new 'embedding' column. Note that the embeddings are converted to numpy arrays for compatibility with sklearn functions.

**Define suggest_apps Function:**

This function takes a search term as input, computes its embedding, and then calculates the cosine similarity between this embedding and the embeddings of the sample search keywords in the DataFrame.
It applies cosine similarity to each embedding in the 'embedding' column, comparing them with the search term's embedding, and stores the similarity scores in a new 'similarity' column.
The apps are then sorted by their similarity scores in descending order, and the function returns the top 5 apps' information from the 'AppInfo' column as suggestions.

**Example Usage:**

The script demonstrates how to use the suggest_apps function with "Song app" as the search term. It prints the top 5 suggested apps based on their semantic similarity to the search term.

Current Issues 
There are 2 implementation approaches of this project RealTimeLoadBERTModelAndInference and InferenceingSaveBERTModelAndEmbeding. Though both approaches work, however, RealTimeLoadBERTModelAndInference is working as expected and the Saved model approach needs more tunning. 
