import chromadb

client = chromadb.PersistentClient(path="/Users/kere/igbo_vector_db")
col = client.get_collection("igbo_translations")

# Collection metadata
print("Collection metadata:", col.metadata)

# Peek at schema
sample = col.peek(limit=3)
print("\nSample IDs:", sample['ids'])
print("\nSample documents:", sample['documents'])
print("\nSample metadatas:", sample['metadatas'])

# Check embedding dimensions
results = col.query(query_texts=["hello"], n_results=1, include=["embeddings"])
print("\nEmbedding dimensions:", len(results['embeddings'][0][0]))