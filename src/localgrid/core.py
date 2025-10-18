import os
import re
import json
from importlib import resources
import tiktoken
import asyncio

from .mappings import TOKENIZER_CONSOLIDATION_MAP

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

_models = None  # treat these as states
_tokenizer_cache = {}

DEFAULT_TOKEN_LIMIT = 8192
FALLBACK_TOKEN_RATIO = 4

# all unique 25 base tokenizers 
BASE_TOKENIZERS = sorted(set(TOKENIZER_CONSOLIDATION_MAP.values()))

try:
    _default_tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _default_tokenizer = None


def _load_cache() -> dict:
    """Loads the pre-processed unified database from the cache file."""
    global _models
    if _models is None:  # cold start
        try:
            with resources.path('localgrid.data', 'localgrid_cache.json') as cache_path:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    _models = json.load(f)
        except FileNotFoundError:
            print("Error: Cache file 'localgrid_cache.json' not found.")
            _models = {}
        except Exception as err:
            print(f"Error loading cache: {err}")
            _models = {}
    return _models


def _load_tokenizer_from_disk(tokenizer_dir_name: str):
    """Synchronous helper to load a tokenizer."""
    if tokenizer_dir_name in _tokenizer_cache:
        return _tokenizer_cache[tokenizer_dir_name]
    
    try:  # cold start
        from transformers import AutoTokenizer
        base_path = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir_path = os.path.join(base_path, "tokenizers", tokenizer_dir_name)
        
        if os.path.isdir(tokenizer_dir_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_path, trust_remote_code=True)
            _tokenizer_cache[tokenizer_dir_name] = tokenizer
            return tokenizer
    except Exception as err:
        print(f"Warning: Could not load bundled tokenizer for {tokenizer_dir_name}: {err}")
    
    return None


async def preload(families: list = None):
    """Asynchronously pre-loads tokenizers into the cache."""
    _load_cache()
    
    if families is None:
        families = BASE_TOKENIZERS
    else:
        families = list(set(TOKENIZER_CONSOLIDATION_MAP.get(family, family) for family in families))
    
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(None, _load_tokenizer_from_disk, family)
        for family in families
    ]
    await asyncio.gather(*tasks)


def _get_tokenizer(model_name: str):
    """Finds the appropriate tokenizer, loading it synchronously if not in cache."""
    models = _load_cache()
    model_data = models.get(model_name)
    
    if not model_data:
        family_to_match = model_name
    else:
        family_to_match = model_data.get('tokenizer_family', model_name)
    
    base_tokenizer = TOKENIZER_CONSOLIDATION_MAP.get(family_to_match)
    
    if base_tokenizer:
        return _load_tokenizer_from_disk(base_tokenizer) or _default_tokenizer
    
   #fallback logic trying to be dummy proof
    sorted_families = sorted(TOKENIZER_CONSOLIDATION_MAP.keys(), key=len, reverse=True)
    
    for family_key in sorted_families:
        if family_key in family_to_match:
            base_tokenizer = TOKENIZER_CONSOLIDATION_MAP[family_key]
            return _load_tokenizer_from_disk(base_tokenizer) or _default_tokenizer
    
    return _default_tokenizer


def count(text: str, model: str) -> int:
    """Provides an accurate token count for a given model and text."""
    tokenizer = _get_tokenizer(model)
    
    if tokenizer:
        if isinstance(tokenizer, tiktoken.Encoding):
            return len(tokenizer.encode(text, disallowed_special=()))
        elif hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text, add_special_tokens=False))
    
    # Fallback to character-based estimation
    return len(text) // FALLBACK_TOKEN_RATIO


def limit(model: str) -> int:
    """Gets the context size (token limit) for a given model."""
    models = _load_cache()
    model_data = models.get(model)
    
    if model_data:
        context_val = model_data.get('context')
        
        try:
            if isinstance(context_val, (int, float)):
                return int(context_val * 1024) if context_val < 1024 else int(context_val)

            if isinstance(context_val, str):
                val_str = context_val.upper().strip()
                
                if not val_str or val_str == "N/A":
                    return DEFAULT_TOKEN_LIMIT

                multiplier = 1
                if 'M' in val_str:
                    multiplier = 1024 * 1024
                    val_str = val_str.replace('M', '')
                elif 'K' in val_str:
                    multiplier = 1024
                    val_str = val_str.replace('K', '')
                
                # Clean out any non-numeric characters
                numeric_part = re.sub(r"[^0-9.]", "", val_str)
                
                if not numeric_part:
                    return DEFAULT_TOKEN_LIMIT

                number = float(numeric_part)
                result = int(number * multiplier)
                
                # Threshold rule for strings that had no 'K' or 'M'
                if multiplier == 1 and result < 1024:
                    return result * 1024
                
                return result

        except (ValueError, TypeError, AttributeError):
            pass 
    
    return DEFAULT_TOKEN_LIMIT


def usage(text: str, model: str) -> str:
    """Returns a string showing used tokens vs. the model's total limit."""
    model_limit = limit(model)
    current_tokens = count(text, model)
    return f'{current_tokens} / {model_limit}'