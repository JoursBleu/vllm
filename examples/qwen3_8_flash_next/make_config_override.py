#!/usr/bin/env python3
"""Rewrite a Qwen3.8-Flash-Next-FP8 checkpoint config for the vLLM PR16 branch.

The published checkpoint identifies itself as ``qwen4_exp`` (the name SGLang
registers).  vLLM's PR16 branch registers the same architecture as
``qwen3_8_flash_next``, so the config has to be renamed before serving.  Mount
the result over ``/model/config.json`` instead of editing the checkpoint.

    python3 make-config-override.py /model/config.json config.json
"""

import json
import sys

RENAMES = {
    "model_type": "qwen3_8_flash_next",
    "architectures": ["Qwen3_8FlashNextForConditionalGeneration"],
}


def main(src: str, dst: str) -> None:
    cfg = json.load(open(src))
    cfg.update(RENAMES)
    if "text_config" in cfg:
        cfg["text_config"]["model_type"] = "qwen3_8_flash_next_text"
    if "vision_config" in cfg:
        cfg["vision_config"]["model_type"] = "qwen3_8_flash_next"
    # quantization_config must survive verbatim: it carries weight_block_size
    # [128, 128] and ~943 modules_to_not_convert entries.
    json.dump(cfg, open(dst, "w"), ensure_ascii=False)
    q = cfg.get("quantization_config") or {}
    print("model_type      :", cfg["model_type"])
    print("architectures   :", cfg["architectures"])
    print("quant_method    :", q.get("quant_method"))
    print("weight_block    :", q.get("weight_block_size"))
    print("not_convert     :", len(q.get("modules_to_not_convert") or []))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
