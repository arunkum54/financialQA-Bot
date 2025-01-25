import os
import tempfile
import streamlit as st
from backend import extract_text_from_pdf, structure_extracted_text, chunk_structured_data, init_pinecone_index, generate_response
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def main():
    st.title("Financial QA Bot")
    st.sidebar.header("Documentation")
    st.sidebar.markdown("""
    **How to Use:**
    1. Upload a PDF financial document
    2. Wait for document processing (≈30 sec)
    3. Ask financial questions in natural language
    4. View answers with supporting evidence
    
    **Example Questions:**
    - What is the total revenue for 2024?
    - Show me the operating expenses breakdown
    - Compare net income between 2023 and 2024
    """)

    # Initializing session state variables
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        st.session_state.index = None
        st.session_state.embedder = None
        st.session_state.current_file = None

    # To File upload section
    uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Checking if new file is uploaded or not
        if st.session_state.current_file != uploaded_file.getvalue():
            st.session_state.processed = False
            st.session_state.current_file = uploaded_file.getvalue()
            
            # Saving uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            # Processing document
            with st.spinner("Processing document..."):
                # Cleaning previous state
                st.session_state.index = None
                
                # Processing PDF
                raw_text = extract_text_from_pdf(pdf_path)
                structured_data = structure_extracted_text(raw_text)
                
                # Generating chunks with different strategies so that the LLM can get best one chuck and understand particular context properly
                small_chunks = chunk_structured_data(structured_data, chunk_size=906, overlap=380)
                medium_chunks = chunk_structured_data(structured_data, chunk_size=960, overlap=412)
                large_chunks = chunk_structured_data(structured_data, chunk_size=1005, overlap=501)

                # Combining all chunks
                all_chunks = small_chunks + large_chunks+ medium_chunks
                # Initializing Pinecone to store chucks
                st.session_state.index = init_pinecone_index()
                st.session_state.embedder = load_embedder()

                # Upserting vectors into pinecone
                batch_size = 10
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i + batch_size]
                    vectors = []
                    for j, chunk in enumerate(batch):
                        embedding = st.session_state.embedder.encode(chunk['text']).tolist()
                        vectors.append({
                            "id": f"chunk_{i + j}",
                            "values": embedding,
                            "metadata": {"text": chunk['text'][:2000]}
                        })
                    st.session_state.index.upsert(vectors=vectors)
                
                os.unlink(pdf_path)  # Cleaning temp file
                st.session_state.processed = True

    # Defining Query interface
    if st.session_state.processed and st.session_state.index and st.session_state.embedder:
        query = st.text_input("Ask a financial question:")
        if query:
            with st.spinner("Analyzing..."):
                # Retrieving context using existing index and embedder
                query_embedding = st.session_state.embedder.encode(query).tolist()
                
                results = st.session_state.index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                # Generating and display got answer
                answer = generate_response(query, st.session_state.index, st.session_state.embedder)
                st.subheader("Generated Answer:")
                st.success(answer)
                
                # Displaying supporting evidence, matches segments for answer
                st.subheader("Relevant Financial Data Segments:")
                for match in results.matches:
                    with st.expander(f"Source excerpt (score: {match.score:.2f})"):
                        st.write(match.metadata['text'])
if __name__ == "__main__":
    main()