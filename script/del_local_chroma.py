import chromadb
import sys

client = chromadb.PersistentClient()
collection_name = sys.argv[1]
client.delete_collection(collection_name)