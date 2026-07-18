from .llama_prompt_enhancer import LlamaPromptEnhancer

NODE_CLASS_MAPPINGS = {
    "LlamaPromptEnhancer": LlamaPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaPromptEnhancer": "Llama Prompt Enhancer",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
