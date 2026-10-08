"""Qwen3.5-9B 4bit inference wrapper for the harness (runs in `qwen35` env).

Model-loading recipe is migrated from qwen35_demo/chat.py (verified: 4bit NF4,
49.8 s load, 7.38 GiB peak on the RTX 2080 Ti). Two E5 findings are baked in:

  * `eos_token_id=tokenizer.eos_token_id` MUST be passed to generate(). The
    checkpoint's generation_config points at token 248044, but the true turn
    terminator is <|im_end|>=248046; without the override the model runs past
    its turn and hallucinates the following user/assistant exchanges.
  * replies are truncated to the first </tool_call> before entering history.

Image budget: the processor re-encodes every attached image on each request,
and the vision tower stays fp16 even with a 4bit LLM. MEASURED (M1 episode 0):
with keep=4 and 256px frames the run OOM'd at turn 15 — the 10.57 GiB card held
7.38 GiB of weights, leaving only ~3 GiB for vision activations + KV cache.
M1 therefore runs with keep=2, 160 max_new_tokens and an explicit
`torch.cuda.empty_cache()` after every turn.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

IMAGE_KEEP_DEFAULT = 2


class QwenAgentModel:
    def __init__(
        self,
        model_dir: str | Path,
        load: str = "4bit",
        gpu_mem_gib: float = 9.5,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        image_max_side: int = 160,
        n_gpu: int = 1,
    ) -> None:
        self.model_dir = str(model_dir)
        self.load = load
        self.gpu_mem_gib = gpu_mem_gib
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Image tokens and vision-tower activations scale with pixel count:
        # 256px frames measured OOM at turn 12-17 on the 10.57 GiB card, 160px
        # is ~2.6x cheaper and still legible for a 3D scene (see probe_gpu_memory).
        self.image_max_side = image_max_side
        self.n_gpu = max(1, int(n_gpu))
        self._tokenizer = None
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ load
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Multi-GPU: `device_map="auto"` shards by LAYER (one cross-GPU hop per
        # token at the boundary). On gpu026 the two 2080 Ti are joined by a PCIe
        # Host Bridge (measured: topo=PHB, nvlink all inActive), which is fine
        # for layer-sequential sharding but would be costly for tensor parallel.
        max_memory: dict[Any, str] = {i: f"{self.gpu_mem_gib}GiB" for i in range(self.n_gpu)}
        max_memory["cpu"] = "220GiB"
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "max_memory": max_memory,
            "trust_remote_code": True,
        }
        if self.load == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.load == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.load in ("fp16", "float16"):
            kwargs["dtype"] = torch.float16
        elif self.load in ("fp32", "float32"):
            kwargs["dtype"] = torch.float32
        elif self.load != "cpu":
            raise ValueError(f"unsupported load mode: {self.load}")

        model_cls = self._resolve_model_cls()
        t0 = time.time()
        self._model = model_cls.from_pretrained(self.model_dir, **kwargs)
        self._model.eval()
        print(f"[llm] loaded {self.load} in {time.time() - t0:.1f}s", flush=True)

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_dir)
        except Exception as exc:  # noqa: BLE001 - text-only fallback
            print(f"[llm] no processor ({exc}); text-only mode", flush=True)
            self._processor = None

    @staticmethod
    def _resolve_model_cls():
        import transformers

        try:
            from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

            return Qwen3_5ForConditionalGeneration
        except ImportError:
            return transformers.AutoModelForCausalLM

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        """One assistant turn. `messages` follow the chat-template schema.

        Image content items must be {"type": "image", "image": PIL.Image};
        they are flattened out for the processor and re-referenced positionally,
        which is the order the chat template emits vision placeholders in.
        """
        import torch

        assert self._tokenizer is not None and self._model is not None

        images = _collect_images(messages, max_side=self.image_max_side)
        text = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )

        if images and self._processor is not None:
            inputs = self._processor(text=[text], images=images, return_tensors="pt")
        else:
            inputs = self._tokenizer(text, return_tensors="pt")

        # Context/VRAM telemetry. On this Turing card attention is unfused, so the
        # N x N matrix (N = prompt tokens, 16 heads, fp16) is the thing that
        # actually OOMs -- and when v7 died at turn 63 the log contained no
        # trend to read, which turned the diagnosis into guesswork. Print the
        # token count and the live allocation every turn so the next ceiling is
        # visible BEFORE it bites (16 heads * 2 bytes / 2**20 = MiB per token^2).
        n_tok = int(inputs["input_ids"].shape[1])
        if torch.cuda.is_available():
            n_heads = int(getattr(self._model.config, "num_attention_heads", 16))
            matrix_mib = n_tok * n_tok * n_heads * 2 / 2**20
            print(
                f"[llm] ctx={n_tok}tok img={len(images)} "
                f"attn_matrix≈{matrix_mib:.0f}MiB "
                f"cuda_alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                flush=True,
            )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self._tokenizer.eos_token_id,  # E5: 248046, not the config's 248044
            "stop_strings": ["</tool_call>"],
            "tokenizer": self._tokenizer,
            "repetition_penalty": 1.05,
        }
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Free the per-turn vision/attention workspace immediately; without
        # this the fragmenting allocator OOMs mid-episode (measured).
        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text


def _collect_images(messages: list[dict[str, Any]], max_side: int = 160) -> list[Any]:
    """Gather PIL images, downscaling to at most `max_side` on the long edge.

    Downscaling is the cheapest lever on both token count and vision-tower
    activation memory: a 256px frame costs ~2.6x the pixels of a 160px one, and
    the vision tower stays fp16 regardless of the LLM's 4bit weight format.
    """
    images = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item["image"]
                    if max_side and max(image.size) > max_side:
                        scale = max_side / max(image.size)
                        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                        image = image.resize(new_size)
                    images.append(image)
    return images


def prepare_for_model(
    messages: list[dict[str, Any]],
    keep_recent_images: int = IMAGE_KEEP_DEFAULT,
) -> list[dict[str, Any]]:
    """Replace all but the newest `keep_recent_images` frames with a text stub.

    This is the M1 image-elision budget: the processor would otherwise re-encode
    the full image history on every turn. The stub keeps a breadcrumb (camera +
    the step it was taken) so the model knows a frame existed.
    """
    total = sum(
        sum(1 for item in message["content"] if isinstance(item, dict) and item.get("type") == "image")
        for message in messages
        if isinstance(message.get("content"), list)
    )
    if total <= keep_recent_images:
        return messages

    to_drop = total - keep_recent_images
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        new_items = []
        for item in content:
            if (
                to_drop > 0
                and isinstance(item, dict)
                and item.get("type") == "image"
            ):
                to_drop -= 1
                new_items.append({"type": "text", "text": "[较早的画面已省略，参考随后的状态数字]"})
            else:
                new_items.append(item)
        out.append({**message, "content": new_items})
    return out
