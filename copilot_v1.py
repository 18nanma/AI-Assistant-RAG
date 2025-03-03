import os
import time
import hashlib
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ollama import chat
import threading

# Directory to monitor
DATA_DIR = "./data"

# Paths to store FAISS index and metadata
INDEX_SAVE_PATH = "faiss_index"
METADATA_SAVE_PATH = "faiss_metadata.pkl"

vector_store = None
file_cache = {}

def compute_file_hash(content):
    """Generate a unique hash for a file's content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def save_vector_store():
    """Save FAISS index and metadata."""
    if vector_store:
        vector_store.save_local(INDEX_SAVE_PATH)
        with open(METADATA_SAVE_PATH, "wb") as f:
            pickle.dump(file_cache, f)
        print("Vector store saved.")


def load_vector_store():
    """Load FAISS index and metadata if available."""
    global vector_store, file_cache
    if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(METADATA_SAVE_PATH):
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.load_local(INDEX_SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
            with open(METADATA_SAVE_PATH, "rb") as f:
                file_cache = pickle.load(f)
            print("Vector store loaded successfully.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            vector_store = None
            file_cache = {}


def load_files(directory):
    """Load text files and compute their hashes."""
    texts = {}
    file_hashes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".log") or filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts[filename] = content
                    file_hashes[filename] = compute_file_hash(content)
            except UnicodeDecodeError:
                print(f"Skipping file {filename} due to encoding error.")
    return texts, file_hashes


def update_vector_store():
    """Update FAISS index incrementally."""
    global vector_store, file_cache
    print("Checking for file changes...")

    texts, file_hashes = load_files(DATA_DIR)

    # Identify added or modified files
    new_or_modified_files = {
        filename: content
        for filename, content in texts.items()
        if filename not in file_cache or file_hashes[filename] != file_cache[filename]
    }

    # Identify deleted files
    deleted_files = set(file_cache.keys()) - set(texts.keys())

    # If no changes, do nothing
    if not new_or_modified_files and not deleted_files:
        print("No changes detected.")
        return

    print("Updating FAISS index...")

    # Process new or modified files
    new_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename, content in new_or_modified_files.items():
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            new_documents.append(Document(page_content=chunk, metadata={"source": filename}))

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if new_documents:
        print(f"Adding {len(new_documents)} new documents to FAISS index.")
        if vector_store:
            new_vectors = FAISS.from_documents(new_documents, embeddings)
            vector_store.merge_from(new_vectors)
        else:
            vector_store = FAISS.from_documents(new_documents, embeddings)

    # Update file cache with new hashes
    for filename in new_or_modified_files:
        file_cache[filename] = file_hashes[filename]

    # Remove deleted files from FAISS index
    if deleted_files and vector_store:
        vector_store.delete_by_metadata_field("source", list(deleted_files))
        for filename in deleted_files:
            del file_cache[filename]

    # Save the updated vector store
    save_vector_store()
    print("Index updated.")


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory or not (event.src_path.endswith(".log") or event.src_path.endswith(".txt")):
            return
        update_vector_store()

    def on_created(self, event):
        if event.is_directory or not (event.src_path.endswith(".log") or event.src_path.endswith(".txt")):
            return
        update_vector_store()

    def on_deleted(self, event):
        if event.is_directory or not (event.src_path.endswith(".log") or event.src_path.endswith(".txt")):
            return
        update_vector_store()


def start_file_monitoring():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=False)
    observer.start()
    print(f"Watching {DATA_DIR} for changes...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main():
    global vector_store
    print("\nInitializing Log/Text File Q&A System...")
    
    # Load vector store if available
    load_vector_store()

    # If no existing store, initialize it
    if not vector_store:
        update_vector_store()

    # Start file monitoring in a separate thread
    monitoring_thread = threading.Thread(target=start_file_monitoring, daemon=True)
    monitoring_thread.start()

    while True:
        query = input("Ask a question (or 'quit' to exit): ").strip()
        if not query or query.lower() == "quit":
            break

        if not vector_store:
            print("No indexed files found.")
            continue

        # Retrieve relevant text chunks
        results = vector_store.similarity_search(query, k=3)

        if not results:
            print("No relevant information found.")

        # Collect unique sources of retrieved documents
        sources = set(doc.metadata.get("source", "Unknown File") for doc in results)

        # Prepare context for querying the model
        context_text = "\n\n".join([doc.page_content for doc in results])
        prompt = f"Use the following context to answer the question:\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

        try:
            answer = chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])['message']['content']
            print(f"Answer:\n{answer}\n")

            # Display the unique sources
            print("Sources used:")
            for source in sources:
                print(f"- {source}")
            print()

        except Exception as e:
            print(f"Error querying the model: {e}")


if __name__ == "__main__":
    main()
