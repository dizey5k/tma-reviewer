import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict

from .config import Settings
from .schemas import RESPONSE_JSON_SCHEMA


def analyze_diff_with_ollama(diff_text: str, system_prompt: str, cfg: Settings) -> Dict[str, Any]:
    """Отправляет git diff в локальный сервер Ollama и возвращает валидированный JSON."""
    if len(diff_text) > cfg.max_diff_chars:
        print(f"⚠️ Diff превышает лимит ({len(diff_text)} симв.), обрезаем до {cfg.max_diff_chars}...")
        diff_text = diff_text[:cfg.max_diff_chars] + "\n... [DIFF TRUNCATED]"

    payload = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Проанализируй следующий git diff на соответствие правилам TMA:\n\n```diff\n{diff_text}\n```",
            },
        ],
        "format": RESPONSE_JSON_SCHEMA,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_ctx": cfg.num_ctx,
        },
    }

    req = urllib.request.Request(
        cfg.ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_content = res_data.get("message", {}).get("content", "{}")
            return json.loads(raw_content)
    except urllib.error.URLError as e:
        print(f"❌ Не удалось подключиться к Ollama ({cfg.ollama_url}): {e}", file=sys.stderr)
        print("💡 Проверь, что запущена команда `ollama serve` и скачана модель.", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"❌ Модель вернула невалидный JSON: {e}", file=sys.stderr)
        sys.exit(2)