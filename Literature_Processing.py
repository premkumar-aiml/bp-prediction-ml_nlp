import os
import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from bp_category import get_bp_category
import spacy
from nltk.corpus import stopwords

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model.eval()
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Function to get BioBERT embedding of a text chunk
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        # Keep periods, lemmatize others
        if token.text == ".":
            tokens.append(".")
        elif token.is_alpha or token.like_num and token.text not in stop_words:
            tokens.append(token.lemma_)

    return ' '.join(tokens)

# Step 1: Load and chunk text documents
def load_and_chunk_texts(folder_path, chunk_size=1):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                raw_text = file.read()
                clean_text = preprocess_text(raw_text)
                # Split into sentences or paragraphs (naive splitting)
                chunks = clean_text.split('. ')
                # Group into larger chunks
                for i in range(0, len(chunks), chunk_size):
                    chunk = '. '.join(chunks[i:i+chunk_size]).strip()
                    if len(chunk) > 50:  # Only keep meaningful size
                        documents.append(chunk)
    return documents

# Step 2: Embed all document chunks
def build_document_embeddings(chunks):
    # print("Creating embeddings for document chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb)
        except:
            print(f"Skipping chunk {i} due to token length")
    return embeddings

# Step 3: Search for relevant chunks given a query
def search_documents(query, doc_chunks, doc_embeddings, top_k=1):
    query_emb = get_embedding(query)
    similarities = cosine_similarity([query_emb], doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append((doc_chunks[i], round(similarities[i], 3)))
    return results

def get_first_sentences(text, max_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:max_sentences])

def retrieve_relevant_passages(diastolic_bp,age, gender, doc_chunks, doc_embeddings, top_k=1):
    bp_category = get_bp_category(diastolic_bp)
    query = (
        f"A {gender} patient aged {age} when diastolic_bp is around {diastolic_bp}mm hg, "
        f"which falls under {bp_category['status']}. The recommended precaution is: {bp_category['precaution']} "
        f"What advice, treatment, or risk is associated?"
    )
    results = search_documents(query, doc_chunks, doc_embeddings, top_k=top_k)
    print("Clinical Guidelines:")
    print(f"Based on your current diastolic pressure we noticed that, {bp_category['status']}")
    print(f"Recommended Precaution : {bp_category['precaution']}")

    # print("\nTop relevant interventions:")
    for i, (text, score) in enumerate(results, 1):
        short_summary = get_first_sentences(text, max_sentences=1)
        print(f"Additional Information: {short_summary}")

    return results, bp_category
