# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Workshop 
# MAGIC
# MAGIC This notebook processes all PDFs from the Sample_PDF folder and creates a RAG system using Databricks Vector Search.

# COMMAND ----------

# MAGIC %pip install pypdf databricks-vectorsearch

# COMMAND ----------

# MAGIC %md
# MAGIC # Create volume rawdata
# MAGIC # Store Pdf folder into volume

# COMMAND ----------

from pypdf import PdfReader

pdf_folder = "/Volumes/workspace/default/rawdata/Sample_PDF"

# List PDF files using dbutils (standard Databricks approach for Volumes)
files = dbutils.fs.ls(pdf_folder)
pdf_files = [f.name for f in files if f.name.endswith('.pdf')]

print(f"Found {len(pdf_files)} PDF files to process:")
for pdf_file in pdf_files:
    print(f"  - {pdf_file}")

# Process all PDFs and extract text with metadata
pdf_data = []

if not pdf_files:
    raise ValueError(f"No PDF files found in {pdf_folder}. Please verify the path and file permissions.")

for pdf_file in pdf_files:
    
    pdf_path = f"{pdf_folder}/{pdf_file}"
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            print(f"Warning: {pdf_file} contains no extractable text, skipping...")
            continue
            
        pdf_data.append({
            "filename": pdf_file,
            "text": text.strip(),
            "num_pages": len(reader.pages)
        })
        print(f"Processed {pdf_file}: {len(reader.pages)} pages, {len(text)} characters")
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        continue

print(f"\nTotal PDFs processed: {len(pdf_data)}")
if pdf_data:
    print(f"\nSample text from first PDF ({pdf_data[0]['filename']}):")
    print(pdf_data[0]['text'][:500])

# COMMAND ----------

def chunk_text(text, chunk_size=300, overlap=50, filename=None):
    """Chunk text and return list of dictionaries with text and metadata"""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_content = text[start:end]
        chunks.append({
            "text": chunk_text_content,
            "filename": filename,
            "chunk_id": chunk_id
        })
        chunk_id += 1
        start = end - overlap
    return chunks

# Chunk all PDFs
all_chunks = []

if not pdf_data:
    raise ValueError("No PDF data to chunk. Please check PDF processing step.")

for pdf_info in pdf_data:
    chunks = chunk_text(pdf_info["text"], filename=pdf_info["filename"])
    all_chunks.extend(chunks)

if not all_chunks:
    raise ValueError("No chunks created from PDFs. Please check chunking parameters.")

print(f"Total chunks created: {len(all_chunks)}")
print(f"Average chunks per PDF: {len(all_chunks) / len(pdf_data) if pdf_data else 0:.1f}")

# COMMAND ----------

# DBTITLE 1,Embedding function with correct attribute access
from databricks.sdk import WorkspaceClient

ws_client = WorkspaceClient()

embedding_endpoint = "databricks-qwen3-embedding-0-6b"

def embed_text(text: str):
    """
    Create embeddings using Databricks serving endpoint.
    """
    try:
        response = ws_client.serving_endpoints.query(
            name=embedding_endpoint,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise Exception(
                f"Embedding endpoint '{embedding_endpoint}' not found. "
                "For Free Edition, consider using sentence-transformers library instead."
            ) from e
        raise


# COMMAND ----------

# DBTITLE 1,Embedding for each chunk

# Create embeddings for all chunks
print(f"Creating embeddings for {len(all_chunks)} chunks...")
embeddings = []

if not all_chunks:
    raise ValueError("No chunks available for embedding. Please check previous steps.")

for i, chunk_info in enumerate(all_chunks):
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(all_chunks)} chunks...")
    try:
        embedding = embed_text(chunk_info["text"])
        if not embedding or len(embedding) == 0:
            raise ValueError(f"Empty embedding returned for chunk {i}")
        embeddings.append(embedding)
    except Exception as e:
        print(f"Error creating embedding for chunk {i} ({chunk_info.get('filename', 'unknown')}): {str(e)}")
        raise

if len(embeddings) != len(all_chunks):
    raise ValueError(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(all_chunks)} chunks")

print(f"Completed creating {len(embeddings)} embeddings")


# COMMAND ----------

import pandas as pd

# Create DataFrame with all chunks, embeddings, and metadata
df = pd.DataFrame({
    "id": range(len(all_chunks)),
    "text": [chunk["text"] for chunk in all_chunks],
    "filename": [chunk["filename"] for chunk in all_chunks],
    "chunk_id": [chunk["chunk_id"] for chunk in all_chunks],
    "embedding": embeddings
})

print(f"DataFrame created with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFiles indexed:")
print(df["filename"].value_counts())

spark_df = spark.createDataFrame(df)
spark_df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable("pdf_rag_table")
print("\nData saved to Delta table: pdf_rag_table")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

try:
    client = VectorSearchClient()
    print("Vector Search client initialized successfully")
except Exception as e:
    print(f"Warning: Vector Search may not be available in Free Edition: {e}")
    print("Consider using FAISS or ChromaDB for vector storage instead")
    raise

# COMMAND ----------

# Create endpoint if it doesn't exist
endpoint_name = "pdf-rag-endpoint"
try:
    client.create_endpoint(
        name=endpoint_name,
        endpoint_type="STANDARD"
    )
    print(f"Created endpoint: {endpoint_name}")
except Exception as e:
    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
        print(f"Endpoint {endpoint_name} already exists, using existing endpoint")
    else:
        raise

# COMMAND ----------

# Create index if it doesn't exist
endpoint_name = "pdf-rag-endpoint"
index_name = "workspace.default.idx_pdf_rag_index"

try:
    client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name="workspace.default.pdf_rag_table",
        pipeline_type="TRIGGERED",
        embedding_vector_column="embedding",
        embedding_dimension=1024,
        primary_key="id"
    )
    print(f"Created index: {index_name}")
except Exception as e:
    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
        print(f"Index {index_name} already exists, using existing index")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC # Below Cell will take around 10- 15 mins to get Vector store index in Online State

# COMMAND ----------

import time

endpoint_name = "pdf-rag-endpoint"
index_name = "workspace.default.idx_pdf_rag_index"

index = client.get_index(
    endpoint_name=endpoint_name,
    index_name=index_name
)

# Wait for index to be ready with timeout (max 30 minutes)
print("Waiting for index to be ready...")
max_wait_time = 1800  # 30 minutes in seconds
start_time = time.time()
check_interval = 10  # Check every 10 seconds

while not index.describe().get('status', {}).get('ready', False):
    elapsed = time.time() - start_time
    if elapsed > max_wait_time:
        raise TimeoutError(f"Index {index_name} did not become ready within {max_wait_time} seconds")
    
    print(f"Index is still provisioning... (elapsed: {int(elapsed)}s)")
    time.sleep(check_interval)
    index = client.get_index(
        endpoint_name=endpoint_name,
        index_name=index_name
    )

print(f"Index is ready! (took {int(time.time() - start_time)} seconds)")

# COMMAND ----------

index.describe()

# COMMAND ----------

# DBTITLE 1,RAG answer function minimal fix

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

llm_endpoint = "databricks-meta-llama-3-1-8b-instruct"

def rag_answer(question: str, num_results: int = 3) -> str:
    """
    Answer a question using RAG (Retrieval Augmented Generation).
    
    Args:
        question: The question to answer
        num_results: Number of similar chunks to retrieve (default: 3)
    
    Returns:
        Answer string based on retrieved context
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Embed query
    query_embedding = embed_text(question)

    # Vector search - include filename in results for better context
    try:
        results = index.similarity_search(
            query_vector=query_embedding,
            columns=["text", "filename"],
            num_results=num_results
        )
    except Exception as e:
        raise Exception(f"Vector search failed: {str(e)}. Ensure index is ready and accessible.") from e
    
    if not results or "result" not in results or "data_array" not in results["result"]:
        raise ValueError("No results returned from vector search")

    # Build context with source information
    context_parts = []
    for result in results["result"]["data_array"]:
        text_content = result[0]
        filename = result[1] if len(result) > 1 else "Unknown"
        context_parts.append(f"[Source: {filename}]\n{text_content}")

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question:
{question}
"""

    try:
        response = ws_client.serving_endpoints.query(
            name=llm_endpoint,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=prompt
                )
            ],
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise Exception(
                f"LLM endpoint '{llm_endpoint}' not found. "
                "For Free Edition, consider using Hugging Face Inference API, Ollama, or other free LLM services."
            ) from e
        raise




# COMMAND ----------

# Test the RAG system with a sample question
response = rag_answer("What are key features of techmobile smartphone?")
print("Question: What are key features of techmobile smartphone?")
print("\nAnswer:")
print(response)

# COMMAND ----------

# Test the RAG system with a sample question
response = rag_answer("What are bluetooth and wifi connectivity spec of techmobile smartphone?")
print("Question: What are bluetooth and wifi connectivity spec of techmobile smartphone?")
print("\nAnswer:")
print(response)

# COMMAND ----------

# DBTITLE 1,MLflow logging with correct python_model
import mlflow
import time
import pandas as pd
from mlflow.models import infer_signature

mlflow.set_experiment("/rag_workshop")

# Calculate total metrics across all PDFs
total_pages = sum(pdf["num_pages"] for pdf in pdf_data)
total_chunks = len(all_chunks)
avg_chunk_length = sum(len(chunk["text"]) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0

with mlflow.start_run(run_name="rag_pdf_data_sample_pdf"):
    mlflow.log_metric("num_pdfs", len(pdf_data))
    mlflow.log_metric("num_pages", total_pages)
    mlflow.log_metric("num_chunks", total_chunks)
    mlflow.log_metric("avg_chunk_length", avg_chunk_length)
    mlflow.log_param("chunk_size", 300)
    mlflow.log_param("chunk_overlap", 50)
    mlflow.log_param("embedding_model", "databricks-qwen3-embedding-0-6b")
    mlflow.log_param("top_k", 3)
    mlflow.log_param("pdf_folder", pdf_folder)
    mlflow.log_param("pdf_files", ", ".join(pdf_files))
    
    start = time.time()
    answer = rag_answer("What products are covered in these documents?")
    latency = time.time() - start

    mlflow.log_metric("latency_seconds", latency)
    
    mlflow.end_run()

print(f"MLflow run completed. Logged metrics for {len(pdf_data)} PDFs with {total_chunks} chunks.")


# COMMAND ----------



# COMMAND ----------


