from pathlib import Path
import base64
import io
import json
import requests

try:
    from PIL import Image
except ImportError:
    Image = None

PROMPT_FILES = {
    "Flux / Krea2 / ZIT (natural language)": "flux.txt",
    "LTX / H3 (video)": "h3.txt",
}


def load_system_prompt(prompt_type):
    prompt_file = Path(__file__).parent / "prompts" / PROMPT_FILES[prompt_type]
    return prompt_file.read_text(encoding="utf-8")


def comfy_image_to_data_url(image, max_side=1536, jpeg_quality=92):
    """
    Convert the first image in a ComfyUI IMAGE tensor (BHWC float 0..1)
    into a JPEG data URL suitable for llama.cpp's OpenAI-compatible
    multimodal chat-completions endpoint.
    """
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image-aware prompt enhancement. "
            "Install it with: pip install Pillow"
        )

    # Comfy IMAGE is normally [batch, height, width, channels].
    tensor = image[0] if len(image.shape) == 4 else image
    array = tensor.detach().cpu().clamp(0, 1).mul(255).byte().numpy()

    if array.shape[-1] == 1:
        array = array[..., 0]
        pil = Image.fromarray(array, mode="L").convert("RGB")
    elif array.shape[-1] == 4:
        pil = Image.fromarray(array, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(array[..., :3], mode="RGB")

    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        pil = pil.resize(
            (max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def h3_mode(first_frame, last_frame):
    if first_frame is not None and last_frame is not None:
        return "FL2V"
    if first_frame is not None:
        return "I2V"
    if last_frame is not None:
        return "L2V"
    return "T2V"


class LlamaPromptEnhancer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enhance_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
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
                "prompt_type": (
                    list(PROMPT_FILES.keys()),
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
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)

    FUNCTION = "enhance"
    CATEGORY = "Prompt"

    def enhance(
        self,
        enhance_prompt,
        server_url,
        user_prompt,
        prompt_type,
        style,
        temperature,
        max_tokens,
        timeout,
        seed,
        first_frame=None,
        last_frame=None,
    ):
        # If enhancement is disabled, preserve the old passthrough behavior.
        if not enhance_prompt:
            print("[Llama Prompt Enhancer] Disabled.")
            return (user_prompt,)

        endpoint = server_url.rstrip("/") + "/v1/chat/completions"
        system_prompt = load_system_prompt(prompt_type)

        if prompt_type == "Flux / Krea2 / ZIT (natural language)":
            # Image sockets are intentionally ignored for the still-image
            # prompt mode. They are H3 temporal frame anchors, not generic
            # image-reference inputs.
            user_content = f"""
Expand the following into a detailed natural language image prompt (for example: Flux, Z Image Turbo, Krea2, etc.)

Preferred visual style: {style}

Image request:

{user_prompt}
""".strip()
            mode = "IMAGE"

        else:
            mode = h3_mode(first_frame, last_frame)

            if mode == "T2V":
                instructions = """
MODE: T2V / T2VA
No frame images are supplied. Construct the video's initial visual state and its subsequent audiovisual action from the user's request.
""".strip()

            elif mode == "I2V":
                instructions = """
MODE: I2V / I2VA
The attached FIRST FRAME is the authoritative literal visual state at 0.00 seconds.

Inspect it before writing. Establish only the reference-derived identity, style, composition, pose, environment, objects, and spatial relationships needed for continuity, then describe how the requested action develops forward from that exact state.

Do not treat the first frame as merely an inspiration image or generic character reference. Do not invent an incompatible starting setup. Preserve unrelated visible details unless the user explicitly requests a change. Concentrate the prompt on observable motion, performance, camera behavior, physical consequences, and the resulting state.
""".strip()

            elif mode == "L2V":
                instructions = """
MODE: L2V / L2VA
The attached LAST FRAME is the authoritative literal final visual state of the generated video.

Inspect it before writing. Infer a plausible preceding visual state and physically coherent action that can converge onto the supplied final frame. Describe the sequence forward in time, progressively narrowing the meaningful visual differences until the action lands naturally on the final frame's subject state, pose, composition, objects, environment, and spatial relationships.

Do not treat the last frame as merely an inspiration image or generic character reference. Do not simply describe the static final image. The prompt's job is to construct the motion and causality that lead into it while respecting the user's requested action.
""".strip()

            else:  # FL2V
                instructions = """
MODE: FL2V / FL2VA
Two authoritative temporal frame anchors are attached:
- FIRST FRAME: the literal visual state at 0.00 seconds.
- LAST FRAME: the literal required visual state at the end of the generated video.

Inspect and compare BOTH images before writing. Determine the meaningful visual delta between them: subject position and pose, body orientation, hand occupancy, object state and manipulation, composition, environment, lighting, hairstyle/clothing state where relevant, and any other visible change that must occur.

Then solve that visual delta as a physically plausible continuous temporal path consistent with the user's requested action. Describe observable intermediate actions and consequences that progressively transform the first-frame state into the last-frame state.

Do NOT spend the prompt merely describing two static pictures. Spend it describing what changes between them and how those changes occur. Preserve properties that are consistent across both anchors. Do not invent changes unsupported by either the user's request or the need to bridge the two frames.

Prefer one continuous shot unless the user explicitly requests a cut or the two supplied frames genuinely require a scene transition. The ending should naturally converge on the last frame's pose, composition, object state, environment, and spatial relationships.
""".strip()

            text = f"""
Convert the following request into a production-ready MiniMax H3 video prompt.

Preferred visual style: {style}

{instructions}

IMPORTANT REFERENCE SEMANTICS:
The frame images attached to this message are temporal anchors for H3 generation. They are separate from ordinary <Picture N> character/style reference semantics. Use your vision capability to understand them and write the appropriate H3 prompt, but do not pretend that the output prompt itself has access to these chat attachments unless the user's downstream H3 workflow supplies equivalent frame conditioning.

Video request:

{user_prompt}
""".strip()

            # OpenAI-compatible multimodal content. Text first, then frame
            # images in deterministic temporal order.
            user_content = [{"type": "text", "text": text}]

            if first_frame is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": comfy_image_to_data_url(first_frame),
                        },
                    }
                )

            if last_frame is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": comfy_image_to_data_url(last_frame),
                        },
                    }
                )

        payload = {
            "model": "default",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
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
            print("Type   :", prompt_type)
            print("Prompt :", PROMPT_FILES[prompt_type])
            print("Mode   :", mode)
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
