from pathlib import Path
import json
import requests
import random
import time

SYSTEM_PROMPT = (
    Path(__file__).parent / "system_prompt.txt"
).read_text(encoding="utf-8")


class LlamaPromptEnhancer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080",
                        "multiline": False,
                    },
                ),

                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),

                "style": (
                    [
                        "Photographic",
                        "Fantasy",
                        "Anime",
                        "Portrait",
                        "Landscape",
                        "Product Photo",
                        "Concept Art",
                    ],
                ),

                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "round": 0.01,
                    },
                ),

                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 32,
                        "max": 4096,
                        "step": 32,
                    },
                ),

                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 5,
                        "max": 300,
                        "step": 5,
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2147483647,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)

    FUNCTION = "enhance"

    CATEGORY = "Prompt"

    def enhance(
        self,
        server_url,
        user_prompt,
        style,
        temperature,
        max_tokens,
        timeout,
        seed,
    ):

        endpoint = server_url.rstrip("/") + "/v1/chat/completions"

        user_message = f"""
Expand the following into a detailed Krea Image 2 prompt.

Preferred visual style: {style}

Image request:

{user_prompt}
"""

        payload = {
            "model": "default",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }

        try:

            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout,
            )

            response.raise_for_status()

            data = response.json()

            prompt = data["choices"][0]["message"]["content"].strip()

            print("\n========== Llama Prompt Enhancer ==========")
            print("Server :", endpoint)
            print("Style  :", style)
            print("Temp   :", temperature)
            print("Tokens :", max_tokens)
            print()
            print("Input:")
            print(user_prompt)
            print()
            print("Output:")
            print(prompt)
            print("===========================================\n")

            return (prompt,)

        except requests.exceptions.ConnectionError:
            return ("ERROR: Could not connect to llama.cpp server.",)

        except requests.exceptions.Timeout:
            return ("ERROR: Request timed out.",)

        except json.JSONDecodeError:
            return ("ERROR: Invalid JSON returned by llama.cpp.",)

        except Exception as e:
            return (f"ERROR: {e}",)
