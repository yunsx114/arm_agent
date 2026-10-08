"""Phase 0 · E6 probe: can Qwen3.5-4bit read the LIBERO scene image well enough
to answer spatial questions the harness will rely on? (runs on GPU)

GROUND TRUTH SOURCE — do not edit by eye:
The choice answers below were derived MECHANICALLY by
scripts/probe_camera_projection.py, which projects every object's world position
through the agentview camera extrinsics/intrinsics (cam pos (0.897, 0, 0.65),
looking down -x, fovy 45deg).

First pass of this probe used hand-derived answers ("y+ = away from camera") and
scored the model 2/4 — the projection check then proved the ANSWERS were wrong,
not the model: with the camera on the +x axis, "near the camera" means LARGE x,
not small y. Measured mapping in the saved (display-oriented) PNG:

    u_px increases with world y    (image left = -y, image right = +y)
    v_px increases with world x    (image bottom = near camera = +x)
    depth ordering (near->far): tomato_sauce 0.957, milk 0.980, salad_dressing
        1.012, basket 1.095, eef 1.102, alphabet_soup 1.185, cream_cheese 1.228,
        butter 1.274

Object pixel positions (u, v) for this init state:
    ROBOT0 eef      (126.2,  68.0)
    alphabet_soup   ( 65.5, 123.3)   left & below the eef
    basket          (199.2, 153.2)   right & below the eef
    butter          (109.7, 118.8)
    cream_cheese    (142.7, 125.5)
    milk            ( 66.2, 151.6)
    salad_dressing  ( 97.5, 135.2)
    tomato_sauce    (137.7, 168.2)   lowest in frame / NEAREST camera

Each question is multiple-choice (single letter) so the 4bit model has the best
chance and scoring is mechanical.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QWEN_DEMO = REPO.parent / "qwen35_demo"
sys.path.insert(0, str(QWEN_DEMO))

MODEL_DIR = QWEN_DEMO / "models" / "Qwen3.5-9B"
AGENTVIEW = REPO / "outputs" / "smoke" / "e1_libero_object_0_agentview.png"
SCENE_JSON = REPO / "outputs" / "smoke" / "e1_scene_state.json"

# (question, options, expected letter)  -- GT from probe_camera_projection.py
QUESTIONS = [
    (
        "白色机械爪（gripper，画面中上方那个白色夹爪）停在桌面上方。"
        "字母汤罐头（alphabet soup，蓝红色圆柱罐）相对于机械爪在什么方向？",
        [
            "A. 机械爪左侧偏下",
            "B. 机械爪右侧偏下",
            "C. 机械爪正下方",
        ],
        "A",
    ),
    (
        "编织篮（basket，褐色编织容器）相对于白色机械爪在什么方向？",
        [
            "A. 机械爪左侧偏上",
            "B. 机械爪正下方",
            "C. 机械爪右侧偏下",
        ],
        "C",
    ),
    (
        "白色机械爪现在是张开还是闭合状态？",
        [
            "A. 闭合",
            "B. 张开",
            "C. 无法判断",
        ],
        "B",
    ),
    (
        "画面中最靠近相机的物体是哪一个（在画面上位置最靠下、离镜头最近）？",
        [
            "A. 字母汤罐头（蓝红色圆柱罐）",
            "B. 牛奶罐（带蓝色标签的高罐）",
            "C. 番茄酱罐头（深红色圆柱罐，画面偏下偏右处）",
        ],
        "C",
    ),
]


def main() -> int:
    import chat

    tokenizer, model, processor = chat.load_model(str(MODEL_DIR), load="4bit", gpu_mem_gib=9.5)
    print("[e6] model loaded", flush=True)

    from PIL import Image

    image = Image.open(AGENTVIEW).convert("RGB")
    state = json.loads(SCENE_JSON.read_text())
    del state  # GT is baked into the QUESTIONS table above

    hits = 0
    for i, (question, options, expected) in enumerate(QUESTIONS, 1):
        messages = [
            {
                "role": "system",
                "content": "你是机器人视觉助手。只回答选项字母（A/B/C），不要解释。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question + "\n" + "\n".join(options)},
                ],
            },
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        t0 = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
        )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        letter = next((c for c in reply.strip() if c.upper() in "ABC"), "?")
        ok = letter == expected
        hits += ok
        print(f"[e6] Q{i} ({time.time() - t0:.1f}s) expected={expected} got={letter!r} "
              f"{'✅' if ok else '❌'} raw={reply.strip()[:60]!r}", flush=True)

    print(f"[e6] score: {hits}/{len(QUESTIONS)}")
    return 0 if hits == len(QUESTIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
