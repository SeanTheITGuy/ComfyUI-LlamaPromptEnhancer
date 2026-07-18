# ComfyUI Llama Prompt Enhancer

A small ComfyUI custom node that sends a basic image request to a remote
llama.cpp server and returns a detailed prompt optimized for Krea Image 2.

## Inputs

- `server_url`
  - Base URL of the llama.cpp server.
  - Example: `http://bot-farm.local:8080`
  - A full `/v1/chat/completions` URL is also accepted.

- `user_prompt`
  - A basic image description.
  - Example: `a golden retriever in a field`

## Output

- `enhanced_prompt`
  - The expanded image prompt returned by llama.cpp.

## llama.cpp requirements

The remote server must expose the OpenAI-compatible endpoint:

`/v1/chat/completions`

The request uses:

```json
{
  "model": "default",
  "temperature": 0.1,
  "max_tokens": 500
}
