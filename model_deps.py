r"""
model_deps.py — 从工作流 prompt 解析需要的模型文件(纯逻辑,无文件系统,可单测)。

两条互补的路:
  1) LOADER_MAP:已知 loader 节点 → 精确知道模型 type(ComfyUI 的 models/<type>/ 目录)。
  2) 通用兜底:扫所有节点的所有 string input,命中模型扩展名的文件名 —— 覆盖 LOADER_MAP
     之外的 loader(新自定义节点很常见)。扩展名集合与前端 Export API
     (web/modal_bridge.js 的 /\.(safetensors|ckpt|pt|pth|gguf|bin|sft|onnx)$/i)和
     modal_volume.MODEL_EXTS 三处保持一致,避免漂移导致"某处列了模型、另一处不上传"。

通用兜底拿到的是工作流里写的那个名字(保留子目录,如 "flux/x.gguf"),不知道 type。type 需要按本地
命中位置反推,涉及文件系统,放在 routes.py(用 folder_paths.get_full_path(t, 相对路径));这里只做纯解析。
"""
from pathlib import Path
import json

# 与前端 Export API 的扩展名集合严格对齐(见模块 docstring)。注意:不含图片扩展名,
# 所以 LoadImage 的 .png/.jpg 天然不会被通用兜底误中。
MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin", ".sft", ".onnx"}


# class_type → (模型 type, [输入字段名...])
LOADER_MAP = {
    "CheckpointLoaderSimple":  ("checkpoints",     ["ckpt_name"]),
    "CheckpointLoader":        ("checkpoints",     ["ckpt_name"]),
    "UNETLoader":              ("diffusion_models", ["unet_name"]),
    "DiffusionModelLoader":    ("diffusion_models", ["model_name"]),
    "VAELoader":               ("vae",             ["vae_name"]),
    "CLIPLoader":              ("text_encoders",   ["clip_name"]),
    "DualCLIPLoader":          ("text_encoders",   ["clip_name1", "clip_name2"]),
    "TripleCLIPLoader":        ("text_encoders",   ["clip_name1", "clip_name2", "clip_name3"]),
    "CLIPVisionLoader":        ("clip_vision",     ["clip_name"]),
    "StyleModelLoader":        ("style_models",    ["style_model_name"]),
    "LoraLoader":              ("loras",           ["lora_name"]),
    "LoraLoaderModelOnly":     ("loras",           ["lora_name"]),
    "ControlNetLoader":        ("controlnet",      ["control_net_name"]),
    "UpscaleModelLoader":      ("upscale_models",  ["model_name"]),
    "GLIGENLoader":            ("gligen",          ["gligen_name"]),
    "PulidFluxModelLoader":    ("pulid",           ["pulid_file"]),
}


def extract_loader_models(prompt: dict) -> list[dict]:
    """Known loader models plus LoRA files exposed by custom loader nodes.

    Custom LoRA nodes use many class and input names, so the built-in map alone
    misses them. A model filename on an input whose name or node class mentions
    LoRA is unambiguously a lora dependency; recording its type here also lets
    cloud-only LoRAs be checked against the Modal Volume without a local copy.
    Returns [{type, filename}, ...] with duplicates removed.
    """
    deps: list[dict] = []
    seen: set[tuple] = set()
    for node in (prompt or {}).values():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") or {}
        if not isinstance(ins, dict):
            continue
        class_type = str(node.get("class_type", ""))
        spec = LOADER_MAP.get(class_type)
        candidates = []
        if spec:
            type_, fields = spec
            candidates.extend((type_, ins.get(field)) for field in fields)

        # Pixaroma's LoRA Loader stores its dynamic rows in a JSON string rather
        # than separate prompt inputs: {"loras":[{"name": ..., "on": true}]}.
        # Its node JS injects this into the hidden LoraLoaderState input when
        # graphToPrompt runs, so extract only the enabled file rows here.
        state_value = ins.get("LoraLoaderState")
        if isinstance(state_value, str) and state_value.strip():
            try:
                state = json.loads(state_value)
            except (TypeError, ValueError):
                state = None
            rows = state.get("loras", []) if isinstance(state, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict) or not row.get("on"):
                        continue
                    name = row.get("name")
                    if isinstance(name, str) and Path(name).suffix.lower() in MODEL_EXTS:
                        candidates.append(("loras", name))

        # Custom node packs vary their class/input names (e.g. Power Lora Loader,
        # multi-LoRA stacks). Recognize model-file inputs from the node class or
        # input label, while ignoring strength/trigger/text fields.
        for field, value in ins.items():
            if not isinstance(value, str) or not value.strip():
                continue
            is_lora_loader = "lora" in class_type.casefold()
            is_lora_input = "lora" in str(field).casefold()
            if (is_lora_loader or is_lora_input) and Path(value).suffix.lower() in MODEL_EXTS:
                candidates.append(("loras", value))

        for type_, name in candidates:
            if isinstance(name, str) and name.strip():
                key = (type_, name)
                if key in seen:
                    continue
                seen.add(key)
                deps.append({"type": type_, "filename": name})
    return deps


def extract_generic_filenames(prompt: dict) -> set:
    """通用兜底:扫所有节点所有 string input,命中模型扩展名的文件名(**保留相对路径**,去重)。
    覆盖 LOADER_MAP 之外的 loader。返回 set[str]。

    ⚠ 以前只取 basename:ComfyUI 下拉框里的值本来就带子目录("flux/x.gguf"),剥掉之后
      get_full_path(t, "x.gguf") 找不到 → 推不出 type → 这个模型从需求里消失,既不报缺也不上传;
      只有它一个模型的工作流(例如只用 UnetLoaderGGUF)甚至因「扫不到模型」被路由到 CPU
      (2026-10-05 深度 review,契约 C15)。与 LOADER_MAP 那条一致,原样保留工作流里的写法。"""
    found: set = set()
    for node in (prompt or {}).values():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") or {}
        if not isinstance(ins, dict):
            continue
        for v in ins.values():
            if isinstance(v, str) and v.strip() and Path(v).suffix.lower() in MODEL_EXTS:
                found.add(v)
    return found
