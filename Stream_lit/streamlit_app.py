# streamlit_app.py
import streamlit as st
import os
import tempfile

# Import all your core logic functions from rag_core.py
# (Make sure rag_core.py is in the same folder)
from test_core_core import (
    extract_text,
    get_conversational_answer,
    _store_in_long_term_memory, # We can still use this
    doc_collection,
    chat_collection
)

# --- 1. Configure Page ---
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="🤖",
    layout="wide"
)

# --- 2. Initialize Session State ---
# This is Streamlit's way of "remembering" things
# It replaces your JavaScript `chatHistory = []`
if "history" not in st.session_state:
    st.session_state.history = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# --- 3. Sidebar for File Upload ---
with st.sidebar:
    st.title("Smart Study Buddy 🤖")
    st.write("Upload your document and ask questions.")
    
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf, .txt, .docx)",
        type=["pdf", "txt", "docx"]
    )
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a moment."):
                
                # Save the uploaded file to a temporary location
                # so `extract_text` can read it from a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    filepath = tmp_file.name
                
                try:
                    # --- This is your /upload logic ---
                    print("Clearing old collections...")
                    doc_collection.delete(where={"page": {"$gte": 0}})
                    chat_collection.delete(where={"page": {"$gte": 0}})
                    print("Collections cleared.")

                    # Reset chat history in UI
                    st.session_state.history = []

                    print(f"Extracting text from {filepath}...")
                    chunks = extract_text(filepath)
                    
                    if not chunks:
                        st.error("File processed, but no text was extracted. Please try another file.")
                    else:
                        print(f"Adding {len(chunks)} chunks to ChromaDB...")
                        documents_to_add = [chunk['content'] for chunk in chunks]
                        metadatas_to_add = [{'page': chunk['page_number']} for chunk in chunks]
                        ids_to_add = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
                        
                        doc_collection.add(
                            documents=documents_to_add,
                            metadatas=metadatas_to_add,
                            ids=ids_to_add
                        )
                        
                        # --- End of /upload logic ---
                        
                        st.session_state.file_processed = True
                        st.session_state.file_name = uploaded_file.name
                        st.success(f"Processed '{uploaded_file.name}' ({len(chunks)} chunks added). Ready to chat!")
                        print("Processing complete.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    print(f"Error: {e}")
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(filepath):
                        os.remove(filepath)

# --- 4. Main Chat Interface ---
st.header(f"Chat with: {st.session_state.file_name}" if st.session_state.file_processed else "Chat")

if not st.session_state.file_processed:
    st.info("Please upload and process a document in the sidebar to begin.")

# Display existing chat history
# `st.chat_message` creates the nice chat bubbles
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"])

# Get new user input
# `st.chat_input` is the text box at the bottom
if prompt := st.chat_input("Ask a question about your document..."):
    
    if not st.session_state.file_processed:
        st.warning("Please upload and process a document first.")
    else:
        # 1. Add user message to UI and history
        st.chat_message("user").markdown(prompt)
        st.session_state.history.append({"role": "user", "parts": prompt})

        # 2. Get bot's response
        # This is your /ask logic
        with st.spinner("Thinking..."):
            try:
                # Pass the history *before* this latest prompt
                history_for_llm = st.session_state.history[:-1]
                
                result = get_conversational_answer(prompt, history_for_llm)
                answer = result['answer']
                sources = result['sources']

                # Also store in long-term memory (optional, but you have it)
                if not answer.startswith("Error"):
                    _store_in_long_term_memory(prompt, answer)
            
            except Exception as e:
                answer = f"Error generating response: {e}"
                sources = []
                print(f"Error in get_conversational_answer: {e}")
        
        # 3. Add bot's response to UI and history
        with st.chat_message("model"):
            st.markdown(answer)
            # Display sources if available
            if sources:
                source_str = ", ".join([f"Page {s['page']}" for s in sources])
                st.caption(f"Sources: {source_str}")
        
        st.session_state.history.append({"role": "model", "parts": answer})