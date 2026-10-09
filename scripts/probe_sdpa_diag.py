import sys, torch
from pathlib import Path
sys.path.insert(0, "/lab/haoq_lab/cse12311731/arm_agent/src")
from arm_agent.agent.llm import QwenAgentModel
from arm_agent.agent import prompts as P

calls = {"n": 0, "gqa": 0, "shapes": []}
_orig = torch.nn.functional.scaled_dot_product_attention

def patched(q, k, v, *a, **kw):
    calls["n"] += 1
    if q.dim() == 4 and k.dim() == 4 and k.shape[1] != q.shape[1]:
        calls["gqa"] += 1
        if len(calls["shapes"]) < 3:
            am = kw.get("attn_mask", a[0] if a else None)
            info = {"q": tuple(q.shape), "k": tuple(k.shape), "kw": sorted(kw),
                    "q_dtype": str(q.dtype), "q_dev": str(q.device)}
            if am is not None:
                info["mask_shape"] = tuple(am.shape)
                info["mask_dtype"] = str(am.dtype)
                try:
                    info["mask_min_max"] = (float(am.min()), float(am.max()))
                except Exception:
                    info["mask_min_max"] = "n/a(bool)"
            calls["shapes"].append(info)
        rep = q.shape[1] // k.shape[1]
        k = k.unsqueeze(2).expand(-1, k.shape[1], rep, -1, -1).reshape(q.shape[0], q.shape[1], k.shape[2], k.shape[3])
        v = v.unsqueeze(2).expand(-1, v.shape[1], rep, -1, -1).reshape(q.shape[0], q.shape[1], v.shape[2], v.shape[3])
        kw.pop("enable_gqa", None)
    return _orig(q, k, v, *a, **kw)

torch.nn.functional.scaled_dot_product_attention = patched
print("patched torch.nn.functional.sdpa:", torch.nn.functional.scaled_dot_product_attention is patched)

m = QwenAgentModel("/lab/haoq_lab/cse12311731/qwen35_demo/models/Qwen3.5-9B",
                   load="4bit", gpu_mem_gib=9.5, image_max_side=160, temperature=0.0)
m.load_model()
print("attn impl:", getattr(m._model.config, "_attn_implementation", "?"))
print("modeling file:", sys.modules[type(m._model).__module__].__file__)

msgs=[{"role":"system","content":[{"type":"text","text":P.SYSTEM_PROMPT_PLAIN}]},
      {"role":"user","content":[{"type":"text","text":"任务：pick up the alphabet soup。请开始。"}]}]
for i in range(3):
    msgs += [{"role":"assistant","content":[{"type":"text","text":"<tool_call>\n<function=locate>\n<parameter=name>\nalphabet_soup\n</parameter>\n</function>\n</tool_call>"}]},
             {"role":"user","content":[{"type":"text","text":"[locate] alphabet_soup_1: (-0.1191, -0.2398, +0.0384) m"}]}]
text = m._tokenizer.apply_chat_template(msgs, tools=P.TOOL_SCHEMAS, tokenize=False, enable_thinking=False, add_generation_prompt=True)
enc = m._tokenizer(text, return_tensors="pt")
print("ctx_tok:", int(enc["input_ids"].shape[1]))

torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.cuda.reset_peak_memory_stats()
try:
    with torch.inference_mode():
        m._model.generate(**{k: v.to(m._model.device) for k, v in enc.items()},
                          max_new_tokens=8, do_sample=False,
                          eos_token_id=m._tokenizer.eos_token_id,
                          pad_token_id=m._tokenizer.eos_token_id)
    print("GENERATE OK with math disabled -> the patch IS in effect")
except Exception as e:
    print("GENERATE FAILED with math disabled ->", type(e).__name__, str(e)[:80])
    import traceback as _tb
    _tb.print_exc(limit=3)
print("sdpa calls:", calls["n"], " GQA calls:", calls["gqa"])
for s in calls["shapes"]:
    print("   ", s)
