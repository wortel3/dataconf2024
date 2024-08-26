from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

# Milvus and LLM configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "document_collection"
EMBEDDING_DIM = 768
import openai

OPENAI_API_KEY = "your_openai_api_key"  # Replace with your OpenAI API key

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Step 1: Connect to Milvus
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Step 2: Set up SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Define Milvus collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]
schema = CollectionSchema(fields, description="Document Embeddings")

# Step 4: Create collection
collection = Collection(name=COLLECTION_NAME, schema=schema)

# Step 5: Insert data into Milvus (Example Data)
documents = [
    "Team A is frustrated after losing their match. The players are disappointed and morale is low.",
    "Team B is thrilled after breaking a world record during practice. The team is full of excitement and confidence.",
    "Team C is feeling a bit anxious about their upcoming event. There’s a mix of nervousness and anticipation.",
    "Team D is disappointed after their recent performance but remains determined to improve.",
    "Team E is relaxed and enjoying their time in the Olympic Village. They’re using this downtime to bond and recover.",
    "Team F is concerned about an injury that one of their key players sustained.",
]

# Generate embeddings
embeddings = model.encode(documents)

# Insert data into Milvus
import numpy as np

data = [list(range(len(documents))), [np.array(embedding) for embedding in embeddings]]
collection.insert(data)

# Step 6: Build an index on the embedding field
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    },
)


# Step 7: Define the RAG pipeline function
def rag_pipeline(query):
    # Convert query to embedding
    query_embedding = model.encode([query])

    # Search in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding], anns_field="embedding", param=search_params, limit=3
    )

    # Retrieve top-k documents
    top_k_docs = [documents[hit.entity.get("id")] for hit in results.result()[0]]
    context = " ".join(top_k_docs)

    # Generate response using LLM
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Based on the following documents: {context}, answer the question: {query}",
        max_tokens=150,
    )

    return response.choices[0].text.strip()


# Step 8: Test the RAG pipeline
if __name__ == "__main__":
    query = "What is the sentiment of Team A's mood after the match?"
    response = rag_pipeline(query)
    print("Generated Response:", response)
