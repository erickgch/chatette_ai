from ingestion import vectorstore
results = vectorstore.get(where={"collection": "lists"})
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(meta)
    print(doc[:200])
    print("---")