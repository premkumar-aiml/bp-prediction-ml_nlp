from Literature_Processing import load_and_chunk_texts, build_document_embeddings, retrieve_relevant_passages
from diastolic_predection import predict_diastolic_bp

# 1. Predict the diastolic BP from user input
predicted_dbp,age, gender = predict_diastolic_bp()

# Load your local PubMed documents
text_folder_path = "E:\\BITS_Masters_In_DataScience\\Dissertation\\Practice\\pubmedData"
doc_chunks = load_and_chunk_texts(text_folder_path)
doc_embeddings = build_document_embeddings(doc_chunks)

# Use BioBERT to retrieve relevant suggestions
retrieve_relevant_passages(predicted_dbp,age, gender, doc_chunks, doc_embeddings)

