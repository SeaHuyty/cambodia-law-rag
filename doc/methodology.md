## Methodology

### Step 1: Data Collection & Preprocessing
- Gather official Cambodian Road Traffic Laws (ច្បាប់ស្តីពីចរាចរណ៍ផ្លូវគោក) and related penalty sub-decrees.
- Extract text from official PDFs and remove unnecessary noise (page headers, footers, scanning artifacts).


### Step 2: Document Chunking
Legal texts have an inherent structure (Chapters, Articles, and Clauses). Instead of splitting text randomly, documents will be segmented by Article / Clause boundaries (e.g., មាត្រា) with moderate overlap to preserve legal context.


### Step 3: Dense Vector Embedding & Indexing (IR Component)
- Each chunk is converted into a 1024-dimensional embedding vector using a multilingual sentence transformer (e.g., BAAI/bge-m3).
- Vectors and their metadata (Article number, Chapter, Page reference) are indexed into a vector database (such as ChromaDB or FAISS).


### Step 4: Semantic Retrieval via Cosine Similarity
- The user inputs a question in Khmer through the web interface.
- The query is transformed into a 1024-dimensional vector using the same embedding model.
- The system computes the Cosine Similarity between the query vector Q and all documents vectors D
​- ​The system retrieves the Top-K most similar legal chunks (e.g., Top 3 to 5 chunks).


### Step 5: Plain-Language Answer Generation (ML Component)
- The retrieved legal text chunks and the user's question are passed into an LLM with a strict prompt instruction: "Answer the user's question in natural Khmer based ONLY on the provided legal context. Clearly state the relevant Article number (មាត្រា). If the context does not contain the answer, state that you do not know."


### Step 6: Web Interface Development
Develop a lightweight, responsive web application where users can:
- Type questions in Khmer.
- Receive an easy-to-read answer.
- Expand a "Sources / References" tab to view the exact original legal article retrieved.


### Step 7: System Evaluation
Test retrieval accuracy using metrics like Recall@K and Mean Reciprocal Rank (MRR) on a sample set of typical traffic law questions. Assess response faithfulness (ensuring the model does not invent laws not present in the retrieved context).
