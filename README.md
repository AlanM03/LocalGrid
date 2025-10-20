# LocalGrid

A simple Python library to inspect and get metadata for local LLMs, like token limits and token counts.

I built this because I needed a way to get accurate info for local models without having to make any web calls. All the data and tokenizers are bundled directly into the package.

## Key Features

* **Fully Local:** No internet connection needed after installation.
* **Accurate Token Counting:** Uses the real tokenizer for a given model, not just a guess.
* **Bundled Tokenizers:** All the necessary tokenizer files are included in the package.
* **Simple API:** Just a few functions to get what you need.

## Data Source

The model data (like context limits) was gathered by scraping and formatting information from Ollama and lm-studios public model library's. This data is saved in a JSON file (`localgrid_cache.json`) inside the package.

## Installation

```bash
pip install localgrid
```

## Quick Start & Usage

The library has three main functions you'll probably use.

### 1. `get_context_limit`

Gets the total context size (token limit) for a model.


```python
from localgrid import get_context_limit

# Get the context limit for llama3.1:latest
limit = get_context_limit("llama3.1:latest")#ollama format

print(f"llama3.1:latest limit: {limit}")
# Output: llama3.1:latest limit: 131072
```

### 2. `count_tokens`

Counts the number of tokens in a string for a specific model. It loads the correct tokenizer to give you an accurate count.

```python
from localgrid import count_tokens

text = "This is a test sentence for my model."
model = "google/gemma-3-12b"#lm-studio format

token_count = count_tokens(text, model)

print(f"The text has {token_count} tokens according to {model}.")
# The text has 9 tokens according to google/gemma-3-12b.
```

### 3. `preload_tokenizers` (optional async)

Due to the decision to keep this package fully offline capable there
is a little delay when initially loading tokenizers from disk.

If you're using this in a server and want to avoid a tiny delay on the first call, you can preload the tokenizers into memory when your app starts.

```python
import asyncio
from localgrid import preload_tokenizers

async def main():
    # Preloads all 25 base tokenizers
    await preload_tokenizers()
    
    # Or just preload specific ones
    await localgrid.preload_tokenizers(families=["llama", "gemma"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Licensing & Included Tokenizers

All included tokenizer files are bundled with their original licenses (e.g., `LICENSE` and `tokenizer_config.json`). You can find these within the installed package.

**Note:** Due to licensing restrictions, the following tokenizer families are **not** included in this package:

* `command-r`
* `stablelm2`
* `codestral`

*As a result they default to generic tokenizers*

> *If someone knows more about me than this please let me know and I will add them ASAP*