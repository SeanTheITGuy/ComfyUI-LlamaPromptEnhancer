from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


NODE_DIRECTORY = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = NODE_DIRECTORY / "system_prompt.txt"
HISTORY_DIRECTORY = NODE_DIRECTORY / "history"
HISTORY_FILE = HISTORY_DIRECTORY / "prompt_history.txt"

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.1

USER_INSTRUCTION = """Expand the following basic image request into a polished, detailed image generation prompt.

The final prompt should describe only the visible contents of the image.

Basic request:
{user_prompt}
"""

class SystemPromptCache:
    """
    Loads system_prompt.txt and caches it in memory.

    The file is automatically reloaded when its modification time changes,
    allowing the prompt to be edited without restarting ComfyUI.
    """

    _lock = threading.Lock()
    _cached_text: str | None = None
    _cached_mtime_ns: int | None = None

    @classmethod
    def get(cls) -> str:
        try:
            stat = SYSTEM_PROMPT_PATH.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"System prompt file was not found: {SYSTEM_PROMPT_PATH}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not inspect the system prompt file: {exc}"
            ) from exc

        with cls._lock:
            if (
                cls._cached_text is None
                or cls._cached_mtime_ns != stat.st_mtime_ns
            ):
                try:
                    prompt = SYSTEM_PROMPT_PATH.read_text(
                        encoding="utf-8"
                    ).strip()
                except OSError as exc:
                    raise RuntimeError(
                        f"Could not read the system prompt file: {exc}"
                    ) from exc

                if not prompt:
                    raise RuntimeError(
                        "The system prompt file is empty."
                    )

                cls._cached_text = prompt
                cls._cached_mtime_ns = stat.st_mtime_ns

                print(
                    "[Llama Prompt Enhancer] "
                    "Loaded system_prompt.txt"
                )

            return cls._cached_text


class PromptHistory:
    """
    Appends successful prompt expansions to a local text file.

    History failures are deliberately non-fatal. A permissions problem should
    not prevent the enhanced prompt from being returned to the workflow.
    """

    _lock = threading.Lock()

    @classmethod
    def append(
        cls,
        *,
        server_url: str,
        user_prompt: str,
        enhanced_prompt: str,
    ) -> None:
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        entry = (
            "\n"
            + "=" * 80
            + "\n"
            f"Timestamp: {timestamp}\n"
            f"Server: {server_url}\n"
            "\n"
            "User prompt:\n"
            f"{user_prompt}\n"
            "\n"
            "Enhanced prompt:\n"
            f"{enhanced_prompt}\n"
        )

        try:
            with cls._lock:
                HISTORY_DIRECTORY.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with HISTORY_FILE.open(
                    "a",
                    encoding="utf-8",
                ) as history_file:
                    history_file.write(entry)

        except OSError as exc:
            print(
                "[Llama Prompt Enhancer] "
                f"Warning: could not save prompt history: {exc}"
            )


class LlamaPromptEnhancer:
    """
    ComfyUI node that sends a basic image request to a remote llama.cpp
    OpenAI-compatible chat-completions endpoint.
    """

    _session = requests.Session()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": DEFAULT_SERVER_URL,
                        "multiline": False,
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
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
        server_url: str,
        user_prompt: str,
    ) -> tuple[str]:
        normalized_url = self._normalize_server_url(server_url)
        cleaned_user_prompt = user_prompt.strip()

        if not cleaned_user_prompt:
            return (
                self._format_error(
                    title="Empty prompt",
                    server_url=normalized_url,
                    reason="Enter a basic image prompt before running the node.",
                ),
            )

        try:
            system_prompt = SystemPromptCache.get()
        except RuntimeError as exc:
            return (
                self._format_error(
                    title="System prompt error",
                    server_url=normalized_url,
                    reason=str(exc),
                ),
            )

        endpoint = normalized_url + CHAT_COMPLETIONS_PATH

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": USER_INSTRUCTION.format(
                        user_prompt=cleaned_user_prompt
                    ),
                },
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        self._log_request(
            server_url=normalized_url,
            user_prompt=cleaned_user_prompt,
        )

        try:
            response = self._session.post(
                endpoint,
                json=payload,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        except requests.exceptions.ConnectTimeout:
            return (
                self._format_error(
                    title="Connection timed out",
                    server_url=normalized_url,
                    reason=(
                        "The llama.cpp server did not accept the connection "
                        f"within {DEFAULT_TIMEOUT_SECONDS} seconds."
                    ),
                ),
            )

        except requests.exceptions.ReadTimeout:
            return (
                self._format_error(
                    title="Generation timed out",
                    server_url=normalized_url,
                    reason=(
                        "The server accepted the request but did not finish "
                        f"within {DEFAULT_TIMEOUT_SECONDS} seconds."
                    ),
                ),
            )

        except requests.exceptions.ConnectionError as exc:
            return (
                self._format_error(
                    title="Could not connect to llama.cpp",
                    server_url=normalized_url,
                    reason=self._short_exception(exc),
                ),
            )

        except requests.exceptions.RequestException as exc:
            return (
                self._format_error(
                    title="Network request failed",
                    server_url=normalized_url,
                    reason=self._short_exception(exc),
                ),
            )

        if not response.ok:
            return (
                self._format_http_error(
                    server_url=normalized_url,
                    response=response,
                ),
            )

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            preview = self._truncate(response.text.strip(), 500)

            return (
                self._format_error(
                    title="Invalid server response",
                    server_url=normalized_url,
                    reason=(
                        "The server returned HTTP success, but the response "
                        "was not valid JSON."
                        + (
                            f"\n\nResponse preview:\n{preview}"
                            if preview
                            else ""
                        )
                    ),
                ),
            )

        try:
            raw_prompt = self._extract_assistant_text(data)
        except ValueError as exc:
            return (
                self._format_error(
                    title="Unexpected response format",
                    server_url=normalized_url,
                    reason=str(exc),
                ),
            )

        enhanced_prompt = self._clean_model_output(raw_prompt)

        if not enhanced_prompt:
            return (
                self._format_error(
                    title="Empty model response",
                    server_url=normalized_url,
                    reason=(
                        "The llama.cpp server returned an empty assistant "
                        "message."
                    ),
                ),
            )

        self._log_success(enhanced_prompt)

        PromptHistory.append(
            server_url=normalized_url,
            user_prompt=cleaned_user_prompt,
            enhanced_prompt=enhanced_prompt,
        )

        return (enhanced_prompt,)

    @staticmethod
    def _normalize_server_url(server_url: str) -> str:
        url = server_url.strip()

        if not url:
            url = DEFAULT_SERVER_URL

        url = url.rstrip("/")

        # Permit users to paste the full endpoint without accidentally
        # producing /v1/chat/completions/v1/chat/completions.
        if url.endswith(CHAT_COMPLETIONS_PATH):
            url = url[: -len(CHAT_COMPLETIONS_PATH)]

        return url

    @staticmethod
    def _extract_assistant_text(data: Any) -> str:
        """
        Extract assistant text from common OpenAI-compatible response forms.

        llama.cpp normally returns:
        choices[0].message.content
        """

        if not isinstance(data, dict):
            raise ValueError(
                "The JSON response root was not an object."
            )

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            server_error = data.get("error")

            if server_error:
                if isinstance(server_error, dict):
                    message = (
                        server_error.get("message")
                        or json.dumps(
                            server_error,
                            ensure_ascii=False,
                        )
                    )
                else:
                    message = str(server_error)

                raise ValueError(
                    f"The server reported an error: {message}"
                )

            raise ValueError(
                "The response did not contain a non-empty 'choices' list."
            )

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise ValueError(
                "The first item in 'choices' was not an object."
            )

        message = first_choice.get("message")

        if isinstance(message, dict):
            content = message.get("content")

            if isinstance(content, str):
                return content

            # Some compatible servers represent content as structured parts.
            if isinstance(content, list):
                text_parts: list[str] = []

                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        text = part.get("text")

                        if isinstance(text, str):
                            text_parts.append(text)

                if text_parts:
                    return "".join(text_parts)

        # Fallback used by older text-completions-like wrappers.
        text = first_choice.get("text")

        if isinstance(text, str):
            return text

        raise ValueError(
            "No assistant text was found in "
            "'choices[0].message.content' or 'choices[0].text'."
        )

    @classmethod
    def _clean_model_output(cls, text: str) -> str:
        cleaned = text.strip()

        # Remove complete Markdown code fences such as:
        # ```text
        # prompt here
        # ```
        fenced_match = re.fullmatch(
            r"```(?:text|plaintext|markdown|md)?\s*\n?"
            r"(?P<body>.*?)"
            r"\n?```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if fenced_match:
            cleaned = fenced_match.group("body").strip()

        # Remove common headings instruction-tuned models occasionally add.
        cleaned = re.sub(
            r"^\s*(?:enhanced\s+prompt|image\s+prompt|prompt)\s*:\s*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

        # Remove one matching pair of surrounding quotation marks.
        quote_pairs = (
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
        )

        for opening, closing in quote_pairs:
            if (
                len(cleaned) >= 2
                and cleaned.startswith(opening)
                and cleaned.endswith(closing)
            ):
                cleaned = cleaned[
                    len(opening) : len(cleaned) - len(closing)
                ].strip()
                break

        # Normalize excessive blank lines without flattening ordinary prose.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @classmethod
    def _format_http_error(
        cls,
        *,
        server_url: str,
        response: requests.Response,
    ) -> str:
        reason = response.reason or "Unknown HTTP error"
        details = cls._extract_error_details(response)

        message = (
            f"HTTP {response.status_code} {reason}"
        )

        if details:
            message += f"\n\nServer response:\n{details}"

        return cls._format_error(
            title="llama.cpp request failed",
            server_url=server_url,
            reason=message,
        )

    @classmethod
    def _extract_error_details(
        cls,
        response: requests.Response,
    ) -> str:
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            return cls._truncate(response.text.strip(), 1000)

        if isinstance(data, dict):
            error = data.get("error")

            if isinstance(error, dict):
                message = error.get("message")

                if isinstance(message, str):
                    return cls._truncate(message.strip(), 1000)

                return cls._truncate(
                    json.dumps(
                        error,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    1000,
                )

            if isinstance(error, str):
                return cls._truncate(error.strip(), 1000)

            message = data.get("message")

            if isinstance(message, str):
                return cls._truncate(message.strip(), 1000)

        return cls._truncate(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            1000,
        )

    @staticmethod
    def _format_error(
        *,
        title: str,
        server_url: str,
        reason: str,
    ) -> str:
        message = (
            f"ERROR: {title}\n\n"
            f"Server:\n{server_url}\n\n"
            f"Reason:\n{reason}"
        )

        print("\n[Llama Prompt Enhancer] ERROR")
        print(message)
        print()

        return message

    @staticmethod
    def _log_request(
        *,
        server_url: str,
        user_prompt: str,
    ) -> None:
        print("\n[Llama Prompt Enhancer]")
        print(f"Server: {server_url}")
        print("\nUser prompt:")
        print(user_prompt)
        print("\nRequesting enhanced prompt...")

    @staticmethod
    def _log_success(enhanced_prompt: str) -> None:
        print("\nEnhanced prompt:")
        print(enhanced_prompt)
        print()

    @staticmethod
    def _short_exception(exc: Exception) -> str:
        text = str(exc).strip()
        return text or exc.__class__.__name__

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text

        return text[: limit - 1].rstrip() + "…"
