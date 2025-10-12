import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoTokenizer
import json
from importlib import resources
import tiktoken

class LocalGrid:
    """A library to inspect and get metadata for local LLMs, with a focus on providing accurate, offline tokenization for various model families."""

    DEFAULT_TOKEN_LIMIT = 8192
    FALLBACK_TOKEN_RATIO = 4
    TOKENIZER_MAP = {
            "aya",
            "codellama",
            "codestral",
            "command-r",
            "dbrx",
            "deepseek",
            "falcon",
            "gemma",
            "granite",
            "llama",
            "mistral",
            "mixtral",
            "olmo",
            "phi",
            "qwen",
            "solar",
            "starcoder",
            "yi"
        }

    def __init__(self):
        """Initializes the database and tokenizer caches."""

        self._db = self._load_all_dbs()
        self._tokenizer_cache = {}#after a tokenizer is loaded from hf to an object we store it here for quick subsequent lookup

        try:
            self._default_tokenizer = tiktoken.get_encoding("cl100k_base")#tiktoken is a safe bet for a default tokenizer despite only being fully accurate for GPT family

        except Exception:
            self._default_tokenizer = None

    def _get_tokenizer(self, model_name: str):
        if ':' in model_name:
            family_name = model_name.split(':', 1)[0]
            tokenizer_dir_name = None

            for family in self.TOKENIZER_MAP:
                if family in family_name:
                    tokenizer_dir_name = family
                    break 

            if not tokenizer_dir_name:
                return self._default_tokenizer 

            if tokenizer_dir_name in self._tokenizer_cache:
                return self._tokenizer_cache[tokenizer_dir_name]

            try:
                base_path = os.path.dirname(os.path.abspath(__file__))
                tokenizer_path = os.path.join(base_path, "tokenizers", tokenizer_dir_name)

                if os.path.exists(tokenizer_path):
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    self._tokenizer_cache[tokenizer_dir_name] = tokenizer
                    return tokenizer

            except Exception as err:
                print(f"Warning: Could not load bundled tokenizer for {tokenizer_dir_name}: {err}")
                
        return self._default_tokenizer

    def count_tokens(self, text: str, model_name: str) -> int:
        """Provides an accurate token count by using the best available offline tokenizer."""

        tokenizer = self._get_tokenizer(model_name)

        if tokenizer:
            if isinstance(tokenizer, tiktoken.Encoding):
                return len(tokenizer.encode(text, disallowed_special=()))
            elif hasattr(tokenizer, 'encode'):
                return len(tokenizer.encode(text, add_special_tokens=False))
        
        return len(text) // self.FALLBACK_TOKEN_RATIO

    def _load_all_dbs(self) -> dict:
        """Scans the data directory, loads all provider JSON files (*_data.json) and organizes them into a single dictionary."""

        per_provider_db = {}

        try:
            for json_filename in resources.contents('localgrid.data'):

                if json_filename.endswith('_data.json'):
                    provider_name = json_filename.replace('_data.json', '')

                    with resources.path('localgrid.data', json_filename) as json_path:

                        with open(json_path, 'r', encoding='utf-8') as f:
                            model_list = json.load(f)
                        
                            per_provider_db[provider_name] = {#per file scanned we make that provider a key on our map that holds the file as data.
                                model['parent_model']: model for model in model_list
                            }
                            

        except (FileNotFoundError, ModuleNotFoundError):
            pass

        return per_provider_db

    def get_model_token_limit(self, model_name: str) -> int:
        """Gets the context size for a model using an optimized dictionary lookup."""
        if ':' in model_name:
            parent_name, tag = model_name.split(':', 1)
        
            for provider_data in self._db.values():
                tag_data = provider_data.get(parent_name, {}).get('variants', {}).get(tag)
                
                if tag_data:
                    context_str = tag_data.get('context', '8K')
                    try:
                        if 'K' in str(context_str).upper():
                            return int(str(context_str).upper().replace('K', '')) * 1024
                        
                        numeric_value = int(context_str)
                        if numeric_value <= 1000:#to handle errors scraping might have let in.
                            return numeric_value * 1024
                        else: #if the number is large enough with no 'K' we assume its taken literally.
                            return numeric_value

                    except (ValueError, TypeError):# Catches errors if context_str is "N/A" or other non-numeric text.
                        pass

        return self.DEFAULT_TOKEN_LIMIT
