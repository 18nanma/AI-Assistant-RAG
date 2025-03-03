## AI-Assistant-RAG

A simple RAG (Retrieval-Augmented Generation) tool that lets you query your local documents using a Large Language Model via Ollama. The script indexes text files from a folder, finds the most relevant chunks for your questions (using vector similarity search), and then uses a local LLM to generate answers with the retrieved content as context.

## Prerequisites

- **Python 3.x** installed on your system.
- **Ollama** installed and running locally. ([Installation guide](https://ollama.com/docs/installation))
- **LLM model pulled in Ollama**: Ensure you have downloaded the model you want to use for Q&A. Example:
  ```bash
  ollama pull llama2

- Or use any other supported model as per your requirements.

- **(Optional)** A Python virtual environment to keep dependencies isolated:
  ```bash
  python -m venv env
  source env/bin/activate  # On macOS/Linux
  env\Scripts\activate  # On Windows

## Installation and Setup

- Clone the repository (or download the script files)
- Install Python dependencies
   ```bash
   pip install -r requirements.txt

- Prepare your documents: Put the text files you want to query into the designated folder (for example, a folder named data/ within the project). By default, the script is set to index .txt (and possibly .md) files in that folder. You can modify the script if you have a different folder name or file types.

## Usage

- Start the Ollama server (if not already running)
- Run the script:
  ```bash
    python copilot_v1.py
