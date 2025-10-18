import os
import pytest
import tiktoken
from unittest.mock import MagicMock

import localgrid
from localgrid import limit, count, preload
from localgrid.core import (
    _default_tokenizer,
    FALLBACK_TOKEN_RATIO,
    BASE_TOKENIZERS,
    DEFAULT_TOKEN_LIMIT
)

MOCK_MODELS_DB = {
    "google/gemma-3n-e4b": {"tokenizer_family": "gemma", "context": "32K"},
    "granite3.1-moe:latest": {"tokenizer_family": "gemma", "context": "128K"},
    "mistral-small3.2:latest": {"tokenizer_family": "mistral", "context": "128K"},
    "llama4:latest": {"context": "N/A", "tokenizer_family": "llama"},
    "phi:latest": {"tokenizer_family": "phi", "context": "2K"},
    "unknown-family:latest": {"tokenizer_family": "N/A", "context": "8K"},
    "openthinker:7b": {"context": "32K", "tokenizer_family": "qwen"},

    "model-small-int": {"context": 128},
    "model-large-int": {"context": 8192},
    "model-small-str": {"context": "128"},
    "model-large-str": {"context": "8192"},
    "model-megatokens": {"context": "2M"},
    "model-float-k": {"context": "8.5K"},
    "model-bad-string": {"context": "Unknown"}
}


@pytest.fixture(autouse=True)
def setup_mock_db_and_clear_cache_before_each_test(mocker):
    mocker.patch('localgrid.core._load_cache', return_value=MOCK_MODELS_DB)
    mocker.patch.dict('localgrid.core._tokenizer_cache', {}, clear=True)

    if _default_tokenizer is None:
        mocker.patch('localgrid.core._default_tokenizer', tiktoken.get_encoding("cl100k_base"))


@pytest.fixture
def mock_hf_tokenizer_loader(mocker):
    real_isdir = os.path.isdir

    def isdir_mock_for_tokenizer_paths(path_arg):
        path_str = str(path_arg)
        is_tokenizer_dir = "tokenizers" in path_str and any(path_str.endswith(family) for family in BASE_TOKENIZERS)

        if is_tokenizer_dir:
            return True
        return real_isdir(path_arg)

    mocker.patch('os.path.isdir', side_effect=isdir_mock_for_tokenizer_paths)

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

    patched_autotokenizer_loader = mocker.patch(
        'transformers.AutoTokenizer.from_pretrained',
        return_value=fake_tokenizer
    )
    return patched_autotokenizer_loader


@pytest.mark.parametrize("model_name, expected_limit", [
    ("google/gemma-3n-e4b", 32768),
    ("granite3.1-moe:latest", 131072),
    ("phi:latest", 2048),
    ("openthinker:7b", 32768),
    ("llama4:latest", DEFAULT_TOKEN_LIMIT),
    ("model-not-in-db", DEFAULT_TOKEN_LIMIT),
    ("model-bad-string", DEFAULT_TOKEN_LIMIT),
    ("model-small-int", 131072),
    ("model-large-int", 8192),
    ("model-small-str", 131072),
    ("model-large-str", 8192),
    ("model-megatokens", 2097152),
    ("model-float-k", 8704),
])
def test_limit_function_parses_context_values_correctly(model_name, expected_limit):
    assert limit(model=model_name) == expected_limit


@pytest.mark.parametrize("model_name, text, expected_count, expected_tokenizer_family", [
    ("granite3.1-moe:latest", "Hello world this is gemma", 5, "gemma"),
    ("mistral-small3.2:latest", "Hello world this is mistral", 5, "mistral"),
    ("llama4:latest", "Hello from llama", 5, "llama"),
    ("openthinker:7b", "Hello from qwen", 5, "qwen"),
])
def test_count_uses_correct_hf_tokenizer_when_available(
    mock_hf_tokenizer_loader,
    model_name,
    text,
    expected_count,
    expected_tokenizer_family
):
    token_count = count(text=text, model=model_name)

    assert token_count == expected_count

    tokenizer_path_arg = mock_hf_tokenizer_loader.call_args[0][0]
    assert tokenizer_path_arg.endswith(expected_tokenizer_family)

    fake_tokenizer = mock_hf_tokenizer_loader.return_value
    fake_tokenizer.encode.assert_called_once_with(text, add_special_tokens=False)


def test_count_falls_back_to_tiktoken_if_hf_fails(mocker):
    mocker.patch('os.path.isdir', return_value=False)

    tiktoken_encode_spy = mocker.spy(localgrid.core._default_tokenizer, 'encode')

    token_count = count(text="hello world", model="llama4:latest")

    assert token_count == 2
    tiktoken_encode_spy.assert_called_once_with("hello world", disallowed_special=())


def test_count_falls_back_to_ratio_if_all_tokenizers_fail(mocker):
    mocker.patch('os.path.isdir', return_value=False)
    mocker.patch('localgrid.core._default_tokenizer', None)

    text = "This is a fallback test."
    token_count = count(text=text, model="llama4:latest")

    expected_count = len(text) // FALLBACK_TOKEN_RATIO
    assert token_count == expected_count


def test_tokenizer_is_cached_after_first_load(mock_hf_tokenizer_loader):
    count("text 1", model="google/gemma-3n-e4b")
    count("text 2", model="granite3.1-moe:latest")

    mock_hf_tokenizer_loader.assert_called_once()


@pytest.mark.asyncio
async def test_preload_loads_specified_families(mocker):
    disk_load_spy = mocker.spy(localgrid.core, '_load_tokenizer_from_disk')

    mocker.patch('transformers.AutoTokenizer.from_pretrained', return_value=MagicMock())
    mocker.patch('os.path.isdir', return_value=True)

    families_to_load = ['llama', 'gemma', 'phi']
    await preload(families=families_to_load)

    assert disk_load_spy.call_count == len(families_to_load)
    loaded_families = {call[0][0] for call in disk_load_spy.call_args_list}
    assert loaded_families == set(families_to_load)


@pytest.mark.asyncio
async def test_preload_loads_all_families_by_default(mocker):
    disk_load_spy = mocker.spy(localgrid.core, '_load_tokenizer_from_disk')

    mocker.patch('transformers.AutoTokenizer.from_pretrained', return_value=MagicMock())
    mocker.patch('os.path.isdir', return_value=True)

    await preload()

    assert disk_load_spy.call_count == len(BASE_TOKENIZERS)
    loaded_families = {call[0][0] for call in disk_load_spy.call_args_list}
    assert loaded_families == set(BASE_TOKENIZERS)