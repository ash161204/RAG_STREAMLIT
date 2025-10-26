# rag_core.py
import time
import os
import re
import fitz  # PyMuPDF
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import streamlit as st

# Imports for file extraction
import pytesseract
from PIL import Image
import io
import docx

# --- 1. INITIALIZE EVERYTHING ---

# Load API Key from .env file
# Load API Key
load_dotenv() # This still works for local testing

# Try to get API_KEY from Streamlit secrets first, then .env
try:
    # This is for Streamlit Cloud deployment
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # This is for local testing (using your .env file)
    API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Add it to .env or Streamlit secrets.")

genai.configure(api_key=API_KEY)

# Initialize models and DB
try:
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading generative model...")
    # Using a standard, working model name
    model = genai.GenerativeModel('gemini-2.0-flash-lite') 
    
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db") 
    
    # --- Create TWO collections ---
    doc_collection = chroma_client.get_or_create_collection(name="study_buddy_doc_store")
    chat_collection = chroma_client.get_or_create_collection(name="study_buddy_chat_history")
    
    print("Models and DB initialized successfully.")

except Exception as e:
    print(f"Error during initialization: {e}")

# --- 2. FULL "extract_text" FUNCTION ---
# (Handles PDF, TXT, DOCX, and OCR)

def extract_text(file_path):
    chunks = []
    
    if file_path.endswith('.txt'):
        print("Processing .txt file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        for i, paragraph in enumerate(text.split('\n\n')):
            if paragraph.strip():
                chunks.append({'page_number': 1, 'content': paragraph})
    
    elif file_path.endswith('.pdf'):
        print("Processing .pdf file (with OCR)...")
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                ocr_text = ""
                image_list = page.get_images(full=True)
                
                if image_list:
                    # print(f"Page {page_num+1} - Found {len(image_list)} image(s). Running Pytesseract...")
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        try:
                            image = Image.open(io.BytesIO(image_bytes))
                            ocr_text += pytesseract.image_to_string(image) + "\n"
                        except Exception as e:
                            print(f"Warning: Could not OCR image on page {page_num+1}: {e}")
                
                combined_content = page_text + "\n" + ocr_text
                if combined_content.strip():
                    chunks.append({
                        'page_number': page_num + 1,
                        'content': combined_content
                    })
            doc.close()
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return []
            
    elif file_path.endswith('.docx'):
        print("Processing .docx file...")
        try:
            doc_obj = docx.Document(file_path)
            for i, para in enumerate(doc_obj.paragraphs):
                if para.text.strip():
                    chunks.append({
                        'page_number': 1, # .docx doesn't have pages
                        'content': para.text
                    })
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {e}")
            return []
    
    else:
        print(f"Error: Unsupported file type: {file_path}")
        return []

    print(f"Successfully extracted {len(chunks)} chunks.")
    return chunks

# --- 3. DOCUMENT RETRIEVAL FUNCTION ---
# (Queries the doc_collection)
def get_answer(query_text):
    context_str = ""
    sources = []
    
    match = re.search(r"page (\d+)", query_text.lower())
    
    if match:
        page_number_to_find = int(match.group(1))
        print(f"--- Detected page lookup: {page_number_to_find} ---")
        results = doc_collection.get(
            where={"page": page_number_to_find}
        )
        
        if results['documents']:
            for doc, meta in zip(results['documents'], results['metadatas']):
                context_str += f"Context from Page {meta['page']}:\n{doc}\n---\n"
                sources.append(meta)
        else:
            context_str = f"I looked for page {page_number_to_find}, but I could not find any content for that page."
            
    else:
        print("--- Detected semantic search ---")
        try:
            query_embedding = embedding_model.encode([query_text]).tolist()
            results = doc_collection.query(
                query_embeddings=query_embedding,
                n_results=3 
            )

            if results['documents']:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    context_str += f"Context from Page {meta['page']}:\n{doc}\n---\n"
                    sources.append(meta)
            
            if not context_str:
                context_str = "No relevant context found in the document."
        
        except Exception as e:
            print(f"Error in vector search: {e}")
            context_str = "Error retrieving context."

    return context_str, sources

# --- 4. QUERY REWRITER FUNCTION ---
def _rewrite_query(query_text, chat_history):
    """Uses an LLM to rewrite the user's query."""
    if not chat_history:
        print("--- No history, using original query. ---")
        return query_text
    
    formatted_history = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in chat_history])
    prompt = f"""
    You are a query rewriter. Based on the chat history below and the
    user's new question, rewrite the question into a standalone query.
    CHAT HISTORY:
    {formatted_history}
    NEW QUESTION:
    {query_text}
    REWRITTEN STANDALONE QUERY:
    """
    print("--- Rewriting query... ---")
    try:
        # Use the global 'model' object
        response = model.generate_content(prompt) 
        rewritten = response.text.strip()
        print(f"--- Rewritten query: {rewritten} ---")
        return rewritten
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return query_text

# --- 5. LONG-TERM MEMORY RETRIEVAL ---
def _retrieve_relevant_history(query_text, k=3):
    """Retrieves relevant past conversations from the CHAT collection."""
    print(f"--- Chat Retrieval: Semantic search for: '{query_text}' ---")
    try:
        query_embedding = embedding_model.encode([query_text]).tolist()
        results = chat_collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        if results['documents']:
            return "\n---\n".join(results['documents'][0])
        return ""
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return ""

# --- 6. LONG-TERM MEMORY STORAGE ---
def _store_in_long_term_memory(user_msg, bot_msg):
    """Stores a chat turn in the CHAT collection."""
    print("--- Storing chat in long-term memory... ---")
    text_to_store = f"User: {user_msg}\nAssistant: {bot_msg}"
    chat_id = f"chat_{int(time.time())}"
    
    chat_collection.add(
        documents=[text_to_store],
        metadatas=[{"page": 0}], # Dummy metadata
        ids=[chat_id]
    )

# --- 7. MAIN CONVERSATIONAL FUNCTION ---
# (This replaces the old 'get_final_answer')
def get_conversational_answer(query_text, chat_history):
    """Main function for the conversational RAG."""
    
    # 1. Rewrite query
    rewritten_query = _rewrite_query(query_text, chat_history)
    
    # 2. Get DOCUMENT context
    document_context, sources = get_answer(rewritten_query)
    
    # 3. Get LONG-TERM MEMORY context
    long_term_memory = _retrieve_relevant_history(rewritten_query)
    
    # 4. Get SHORT-TERM MEMORY context
    formatted_history = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in chat_history])

    # 5. Build the final prompt
    prompt = f"""
    You are a helpful study assistant.
    DOCUMENT CONTEXT:
    {document_context}
    RELEVANT PAST CONVERSATIONS (Long-term memory):
    {long_term_memory}
    RECENT CHAT HISTORY (Short-term memory):
    {formatted_history}
    USER'S LATEST QUESTION:
    {query_text}
    
    YOUR TASK: Answer the user's LATEST question.
    Base your answer *only* on the DOCUMENT CONTEXT.
    Use the chat histories to understand the question.
    Cite page numbers from the document context.
    If the answer isn't in the document context, say so.
    """

    print("--- Generating final answer... ---")
    try:
        response = model.generate_content(prompt)
        return {
            'answer': response.text,
            'sources': sources 
        }
    except Exception as e:
        print(f"!!!!!!!!!!!!!! API CALL FAILED !!!!!!!!!!!!!!")
        print(f"THE FULL ERROR IS: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {'answer': f"Error generating response: {e}", 'sources': []}