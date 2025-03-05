import fitz  # for reading the pdf
import pytesseract #Li OCR for text extraction
from PIL import Image # pdf to image
from pdf2image import convert_from_path # pdf to image
import nltk # text processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer # ML library for vectorization
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np #numerical operations

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

def get_text_from_invoice(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        text = extract_text_with_ocr(pdf_path)
    return text

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Removing punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens), set(tokens)  # Return both processed text and unique tokens

def vectorize_documents(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

def calculate_cosine_similarity(new_doc, existing_docs, vectorizer):
    new_doc_tfidf = vectorizer.transform([new_doc])
    similarity_scores = cosine_similarity(new_doc_tfidf, existing_docs)
    return similarity_scores

def calculate_jaccard_similarity(set_a, set_b):
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if len(union) > 0 else 0

def find_most_similar_invoice(new_invoice_path, existing_invoices_paths):
    # Extract and preprocess text from new invoice
    new_invoice_text, new_invoice_tokens = preprocess_text(get_text_from_invoice(new_invoice_path))
    
    # Extract and preprocess text from existing invoices
    existing_invoices_texts = []
    existing_invoices_tokens = []
    
    for path in existing_invoices_paths:
        text, tokens = preprocess_text(get_text_from_invoice(path))
        existing_invoices_texts.append(text)
        existing_invoices_tokens.append(tokens)
    
    # Vectorize existing invoices
    tfidf_matrix, vectorizer = vectorize_documents(existing_invoices_texts)
    
    # Calculate Cosine Similarity
    cosine_similarity_scores = calculate_cosine_similarity(new_invoice_text, tfidf_matrix, vectorizer)
    
    # Calculate Jaccard Similarity
    jaccard_similarity_scores = [
        calculate_jaccard_similarity(new_invoice_tokens, tokens) for tokens in existing_invoices_tokens
    ]

    # Determine the most similar invoice based on cosine similarity
    most_similar_index_cosine = np.argmax(cosine_similarity_scores)
    most_similar_index_jaccard = np.argmax(jaccard_similarity_scores)

    # Compare the highest scores from both methods
    if cosine_similarity_scores[most_similar_index_cosine] > jaccard_similarity_scores[most_similar_index_jaccard]:
        return (existing_invoices_paths[most_similar_index_cosine], 
                cosine_similarity_scores[0][most_similar_index_cosine], 
                'cosine')
    else:
        return (existing_invoices_paths[most_similar_index_jaccard], 
                jaccard_similarity_scores[most_similar_index_jaccard], 
                'jaccard')

def main():
    # Paths to existing invoices
    existing_invoices = [
        # r'C:\Users\rohit\OneDrive\Desktop\Project 2.0\PDF reader\invoice_102857.pdf',
        r'C:\Users\rohit\OneDrive\Desktop\Project 2.0\PDF reader\invoice_77098.pdf'
    ]
    
    # Path to new incoming invoice
    new_invoice = r'C:\Users\rohit\OneDrive\Desktop\Project 2.0\PDF reader\invoice_77098.pdf'  # Example path to a new invoice

    # Find the most similar invoice
    most_similar_invoice, similarity_score, similarity_method = find_most_similar_invoice(new_invoice, existing_invoices)
    print(f"Most similar invoice: {most_similar_invoice}")
    print(f"Similarity score: {similarity_score} (using {similarity_method} similarity)")

if __name__ == "__main__":
    main()
