import pytest
import json
from importlib import resources

import localgrid
from localgrid.core import (
    BASE_TOKENIZERS, 
    TOKENIZER_CONSOLIDATION_MAP
)

@pytest.fixture(scope="session")
def real_model_cache():
    try:
        with resources.path('localgrid.data', 'localgrid_cache.json') as cache_path:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        print("ERROR: localgrid_cache.json not found or failed to load")
        return {}

@pytest.fixture(scope="session")
def all_model_names(real_model_cache):
    return list(real_model_cache.keys())

@pytest.mark.parametrize("tokenizer_name", BASE_TOKENIZERS)
def test_all_base_tokenizers_can_load(tokenizer_name):
    tokenizer = localgrid.core._load_tokenizer_from_disk(tokenizer_name)
    
    assert tokenizer is not None, f"Base tokenizer missing or failed to load: {tokenizer_name}"
    
    localgrid.core._tokenizer_cache = {}

def test_all_models_in_cache_are_valid(real_model_cache):
    validation_errors = []
    
    if not real_model_cache:
        pytest.fail("Model cache is empty. Cannot run test.")

    for model_name, model_data in real_model_cache.items():
        
        limit_val = localgrid.limit(model=model_name)
        if not (limit_val > 0):
            validation_errors.append(f"Model {model_name}: Invalid limit: {limit_val}")

        family_to_match = model_data.get('tokenizer_family', model_name)

        base_tokenizer = TOKENIZER_CONSOLIDATION_MAP.get(family_to_match)
        
        if not base_tokenizer:
            sorted_keys = sorted(TOKENIZER_CONSOLIDATION_MAP.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if key in family_to_match:
                    base_tokenizer = TOKENIZER_CONSOLIDATION_MAP[key]
                    break
        
        if base_tokenizer:
            if base_tokenizer not in BASE_TOKENIZERS:
                validation_errors.append(f"Model {model_name}: Family '{family_to_match}' maps to '{base_tokenizer}', which is not in BASE_TOKENIZERS")

    if validation_errors:
        error_report = "\n".join(validation_errors)
        pytest.fail(f"{len(validation_errors)} validation errors found:\n{error_report}")

@pytest.mark.asyncio
async def test_async_preload_all_tokenizers():
    localgrid.core._tokenizer_cache = {}

    await localgrid.preload()
    
    assert len(localgrid.core._tokenizer_cache) == len(BASE_TOKENIZERS)
    
    localgrid.core._tokenizer_cache = {}