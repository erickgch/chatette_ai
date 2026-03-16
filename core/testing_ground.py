from ingestion import vectorstore

for collection in ["calendar", "emails", "notes", "documents"]:
    results = vectorstore.get(where={"collection": collection})
    print(f"\n📁 {collection}: {len(results['documents'])} chunks")
    for doc in results["documents"]:
        print(f"  - {doc[:80]}")