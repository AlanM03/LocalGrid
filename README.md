# LocalGrid

A simple Python library to inspect and get metadata for local LLMs, like token limits and token counts.

I built this because I needed a way to get accurate info for local models without having to make any web calls. All the data and tokenizers are bundled directly into the package.

## Key Features

* **Fully Local:** No internet connection needed after installation.
* **Accurate Token Counting:** Uses the real tokenizer for a given model, not just a guess.
* **Bundled Tokenizers:** All the necessary tokenizer files are included in the package.
* **Simple API:** Just a few functions to get what you need.

## Data Source

The model data (like context limits) was gathered by scraping and formatting information from Ollama's model library and the LM Studio website. This data is saved in a JSON file (`localgrid_cache.json`) inside the package.

## Installation

```bash
pip install localgrid

## Quick Start & Usage

The library has three main functions you'll probably use.

### 1. `get_context_limit`

Gets the total context size (token limit) for a model.

**Python**

```python
import localgrid

# Get the context limit for llama3.1:latest
limit = localgrid.get_context_limit("llama3.1:latest")

print(f"llama3.1:latest limit: {limit}")
# Output: llama3.1:latest limit: 131072```