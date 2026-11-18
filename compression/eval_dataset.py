"""Caption/image records shared by the CSA evaluation entry points."""
import json
import os


def load_caption_rows(caption_json: str, data_dir: str = "", fallback_prompt: str = "a clear photo"):
    rows = []
    if caption_json:
        with open(caption_json, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(data, dict):
            for path, caption in data.items():
                if isinstance(caption, list):
                    caption = caption[0] if caption else fallback_prompt
                rows.append({"img_path": str(path), "txt": str(caption or fallback_prompt)})
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                path = item.get("img_path") or item.get("image_path") or item.get("path")
                caption = item.get("txt") or item.get("caption") or item.get("text") or fallback_prompt
                if path:
                    rows.append({"img_path": str(path), "txt": str(caption or fallback_prompt)})
        else:
            raise ValueError(f"unsupported caption_json structure: {type(data)}")
    elif data_dir:
        paths = []
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                paths.append(os.path.join(data_dir, name))
        rows = [{"img_path": path, "txt": fallback_prompt} for path in paths]
    else:
        raise ValueError("either --caption_json or --data_dir must be provided")
    return rows
