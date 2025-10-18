"""
This file contains the master TOKENIZER_CONSOLIDATION_MAP.
It maps ~193 raw tokenizer family names to ~25 standardized base tokenizer names.
This allows the LocalGrid package to support thousands of models while only needing
to bundle a small number of base tokenizer files.
"""

TOKENIZER_CONSOLIDATION_MAP = {
    # LLaMA Family (covers ~68 model families)
    'llama': 'llama', 'llama2': 'llama', 'llama3': 'llama', 'llama3.1': 'llama',
    'llama3.2': 'llama', 'llama3.2-vision': 'llama', 'llama3.3': 'llama', 'llama4': 'llama',
    'llama-pro': 'llama', 'llama-guard3': 'llama', 'llama2-uncensored': 'llama',
    'llama2-chinese': 'llama', 'llama3-chatqa': 'llama', 'llama3-gradient': 'llama',
    'llama3-groq-tool-use': 'llama', 'codellama': 'llama', 'phind-codellama': 'llama',
    'yarn-llama2': 'llama', 'llava': 'llama', 'llava-phi3': 'llama', 'llava-llama3': 'llama',
    'bakllava': 'llama', 'moondream': 'llama', 'tinyllama': 'llama', 'medllama2': 'llama',
    'vicuna': 'llama', 'wizard-vicuna': 'llama', 'wizard-vicuna-uncensored': 'llama',
    'orca-mini': 'llama', 'orca2': 'llama', 'open-orca-platypus2': 'llama',
    'wizardlm': 'llama', 'wizardlm2': 'llama', 'wizardlm-uncensored': 'llama',
    'wizard-math': 'llama', 'wizardcoder': 'llama', 'xwinlm': 'llama',
    'nous-hermes': 'llama', 'nous-hermes2': 'llama', 'openhermes': 'llama',
    'samantha-mistral': 'llama', 'stable-beluga': 'llama', 'starling-lm': 'llama',
    'neural-chat': 'llama', 'openchat': 'llama', 'goliath': 'llama',
    'meditron': 'llama', 'megadolphin': 'llama', 'tinydolphin': 'llama',
    'dolphin-llama3': 'llama', 'dolphin3': 'llama', 'tulu3': 'llama', 'hermes3': 'llama',
    'alfred': 'llama', 'magicoder': 'llama', 'codebooga': 'llama', 'solar-pro': 'llama',
    'solar': 'llama', 'duckdb-nsql': 'llama', 'everythinglm': 'llama',
    'nexusraven': 'llama', 'athene-v2': 'llama', 'r1-1776': 'llama',
    'reflection': 'llama', 'cogito': 'llama', 'deepscaler': 'llama', 'firefunction-v2': 'llama',
    'lfm2': 'llama',
    
    # Mistral/Mixtral Family (covers ~21 model families)
    'mistral': 'mistral', 'mixtral': 'mistral', 'mistral-small': 'mistral',
    'mistral-small3.1': 'mistral', 'mistral-small3.2': 'mistral', 'mistral-large': 'mistral',
    'mistral-nemo': 'mistral', 'mistral-openorca': 'mistral', 'mistrallite': 'mistral',
    'codestral': 'mistral', 'mathstral': 'mistral', 'devstral': 'mistral',
    'magistral': 'mistral', 'yarn-mistral': 'mistral', 'zephyr': 'mistral',
    'stablelm-zephyr': 'mistral', 'notus': 'mistral', 'notux': 'mistral',
    'dolphin-mistral': 'mistral', 'dolphin-mixtral': 'mistral', 'nous-hermes2-mixtral': 'mistral',
    
    # Qwen Family (covers ~22 model families)
    'qwen': 'qwen', 'qwen2': 'qwen', 'qwen2.5': 'qwen', 'qwen2.5-coder': 'qwen',
    'qwen2-math': 'qwen', 'qwen2vl': 'qwen', 'qwen2.5vl': 'qwen',
    'qwen3': 'qwen', 'qwen3moe': 'qwen', 'qwen3_moe': 'qwen', 'qwen3_next': 'qwen',
    'qwen3-coder': 'qwen', 'qwen3-embedding': 'qwen', 'codeqwen': 'qwen', 'qwq': 'qwen',
    'openthinker': 'qwen', 'sailor2': 'qwen', 'marco-o1': 'qwen', 'codeup': 'qwen',
    'reader-lm': 'qwen', 'smallthinker': 'qwen', 'opencoder': 'qwen',
    
    # Gemma Family (includes Granite - covers ~20 model families)
    'gemma': 'gemma', 'gemma2': 'gemma', 'gemma3': 'gemma', 'gemma3n': 'gemma',
    'codegemma': 'gemma', 'embeddinggemma': 'gemma', 'shieldgemma': 'gemma',
    'granite': 'gemma', 'granitehybrid': 'gemma', 'granite3.1-moe': 'gemma',
    'granite3.2': 'gemma', 'granite3.3': 'gemma', 'granite3-guardian': 'gemma',
    'granite3-moe': 'gemma', 'granite3.1-dense': 'gemma', 'granite3-dense': 'gemma',
    'granite3.2-vision': 'gemma', 'granite-code': 'gemma', 'granite-embedding': 'gemma',
    'granite4': 'gemma',
    
    # Phi Family (covers ~11 model families)
    'phi': 'phi', 'phi-4': 'phi', 'phi3': 'phi', 'phi3.5': 'phi', 'phi4': 'phi',
    'phi4-mini': 'phi', 'phi4-mini-reasoning': 'phi', 'phi4-reasoning': 'phi',
    'dolphin-phi': 'phi', 'nuextract': 'phi', 'bespoke-minicheck': 'phi',
    
    # Falcon Family (covers 3 model families)
    'falcon': 'falcon', 'falcon2': 'falcon', 'falcon3': 'falcon',
    
    # StarCoder Family (covers 4 model families)
    'starcoder': 'starcoder', 'starcoder2': 'starcoder', 'dolphincoder': 'starcoder',
    'sqlcoder': 'starcoder',
    
    # DeepSeek Family (covers 9 model families)
    'deepseek-coder': 'deepseek', 'deepseek-coder-v2': 'deepseek',
    'deepseek-v2': 'deepseek', 'deepseek-v2.5': 'deepseek', 'deepseek-v3': 'deepseek',
    'deepseek-v3.1': 'deepseek', 'deepseek-llm': 'deepseek', 'deepseek-r1': 'deepseek',
    'deepcoder': 'deepseek',
    
    # Yi Family (covers 2 model families)
    'yi': 'yi', 'yi-coder': 'yi',
    
    # Remaining base tokenizers (1-2 families each)
    'dbrx': 'dbrx',
    'glm4': 'glm4',
    'nemotron': 'nemotron', 'nemotron-mini': 'nemotron',
    'smollm': 'smollm', 'smollm2': 'smollm',
    'olmo2': 'olmo', 'olmo': 'olmo', 
    'internlm2': 'internlm', 'internlm': 'internlm',
    'exaone3.5': 'exaone', 'exaone-deep': 'exaone', 'exaone': 'exaone',
    'aya': 'aya', 'aya-expanse': 'aya',
    'codegeex4': 'glm4', 'codegeex': 'glm4',
    'minicpm-v': 'minicpm', 'minicpm': 'minicpm',
    
    # Embedding models
    'nomic-embed-text': 'nomic-embed', 'nomic-embed': 'nomic-embed',
    'all-minilm': 'all-minilm',
    'bge-large': 'bge', 'bge-m3': 'bge', 'bge': 'bge',
    'mxbai-embed-large': 'mxbai', 'mxbai': 'mxbai',
    'snowflake-arctic-embed': 'snowflake-arctic', 'snowflake-arctic-embed2': 'snowflake-arctic', 'snowflake-arctic': 'snowflake-arctic',
    'paraphrase-multilingual': 'sentence-transformers', 'sentence-transformers': 'sentence-transformers',
    
    # Special cases
    'ernie4_5': 'ernie', 'ernie': 'ernie',
}