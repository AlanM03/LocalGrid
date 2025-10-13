import pytest
from unittest.mock import MagicMock
from localgrid.core import LocalGrid

MOCK_DB_DATA = {
    "ollama": {
        "llama3.1": {
            "parent_model": "llama3.1",
            "variants": {
                "latest": {"context": "128K"},   # Checks strings like "128K"
                "8b": {"context": 128}          # Checks small numbers that should mean "K"
            }
        },
        "gemma": {
            "parent_model": "gemma",
            "variants": {
                "latest": {"context": 8192},    # Checks big numbers that are literal values
                "2b": {"context": "N/A"}        # Checks bad data like "N/A"
            }
        },
        "phi-3": {
            "parent_model": "phi-3",
            "variants": {
                "mini": {}                      # Checks what happens if 'context' is missing
            }
        }
    }
}

@pytest.fixture
def grid_instance(mocker) -> LocalGrid:
    mocker.patch.object(LocalGrid, '_load_all_dbs', return_value=MOCK_DB_DATA)
    return LocalGrid()


# 1. Tests for get_model_token_limit()
@pytest.mark.parametrize("model_name, expected_limit", [
    # Cases that should work correctly
    ("llama3.1:latest", 131072),      # checks "128K" string
    ("llama3.1:8b", 131072),          # checks small number 128 -> 128 * 1024
    ("gemma:latest", 8192),           # checks large number 8192 -> 8192
    
    # Cases that should fall back to the default value
    ("gemma:2b", 8192),               # checks bad data "N/A"
    ("phi-3:mini", 8192),             # checks a model with no context data
    ("unknown-model:latest", 8192),   # checks a model that's not in our database
    ("model-no-colon", 8192),         # checks a name with no ":"
])
def test_get_model_token_limit(grid_instance, model_name, expected_limit):
    # Make sure the function returns the right context size for every case.
    assert grid_instance.get_model_token_limit(model_name) == expected_limit

@pytest.mark.parametrize("model_family", [
    "codellama",
    "codestral",
    "deepseek",
    "falcon",
    "gemma",
    "granite",
    "llama",
    "mistral",
    "phi",
    "qwen",
    "starcoder",
    "yi",
])
def test_selects_correct_hf_tokenizer_family(mocker, grid_instance, model_family):
    # Make sure the code tries to load a Hugging Face tokenizer for known families.
    mock_from_pretrained = mocker.patch('localgrid.core.AutoTokenizer.from_pretrained')
    mocker.patch('os.path.exists', return_value=True) # Pretend the tokenizer folder exists

    grid_instance.count_tokens("some text", model_name=f"{model_family}:latest")

    # Check that it tried to load a tokenizer from the correct directory.
    # We get the path it was called with and make sure the family name is in it.
    call_args, _ = mock_from_pretrained.call_args
    tokenizer_path = call_args[0]
    assert f"tokenizers/{model_family}" in tokenizer_path

# 2. Tests for count_tokens()
def test_count_tokens_uses_hf_tokenizer(mocker, grid_instance):
    # Create a fake tokenizer that acts like a real Hugging Face one.
    mock_hf_tokenizer = MagicMock()
    mock_hf_tokenizer.encode.return_value = [1, 2, 3, 4, 5] # Pretend it found 5 tokens.
    mocker.patch.object(grid_instance, '_get_tokenizer', return_value=mock_hf_tokenizer)
    
    token_count = grid_instance.count_tokens("some text", model_name="llama3.1:latest")

    # Check that we got the 5 tokens we expected from our fake tokenizer.
    assert token_count == 5
    # Also check that the 'encode' function was called correctly.
    mock_hf_tokenizer.encode.assert_called_with("some text", add_special_tokens=False)

def test_count_tokens_uses_tiktoken_default(mocker, grid_instance):
    # Force the grid to return its default tokenizer.
    mocker.patch.object(grid_instance, '_get_tokenizer', return_value=grid_instance._default_tokenizer)
    # "gpt-4" isn't a known family, so it should use the default.
    token_count = grid_instance.count_tokens("hello world", model_name="gpt-4:latest")
    # The default tokenizer should count "hello world" as 2 tokens.
    assert token_count == 2 

def test_count_tokens_uses_final_fallback(mocker, grid_instance):
    # Force the grid to find no tokenizer at all.
    mocker.patch.object(grid_instance, '_get_tokenizer', return_value=None)
    # "hello world" has 11 characters. 11 // 4 = 2.
    token_count = grid_instance.count_tokens("hello world", model_name="any-model:latest")
    assert token_count == 2

# 3. Tests for Tokenizer Loading and Caching
def test_tokenizer_caching(mocker):
    # Fake the parts of the code that touch the file system.
    mocker.patch.object(LocalGrid, '_load_all_dbs', return_value={})
    mock_from_pretrained = mocker.patch('localgrid.core.AutoTokenizer.from_pretrained')
    mocker.patch('os.path.exists', return_value=True)

    grid = LocalGrid()
    
    # First call for a llama model should load from "disk".
    grid.count_tokens("text 1", model_name="llama3.1:latest")
    # Second call for another llama model should use the one we already loaded.
    grid.count_tokens("text 2", model_name="llama-3-8b:instruct")
    # Check that we only tried to load from disk one time.
    mock_from_pretrained.assert_called_once()

# 4. Integration Test
def test_count_tokens_llama_integration():
    
    grid = LocalGrid()
    test_string = "Hello im alan and today im going to make local grid!!!!!!"
    
    token_count = grid.count_tokens(test_string, model_name="llama3.1:latest")
    
    # Llama 3 tokenizer should classift this as 13 tokens
    assert token_count == 13