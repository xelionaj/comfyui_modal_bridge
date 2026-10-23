"""Download a tab-separated model manifest directly into the deployed Modal Volume.

Run from ComfyUI's custom_nodes directory:
  python -m comfyui_modal_bridge.download_models --manifest models-to-download.tsv

Each nonblank line is: HTTPS URL<TAB>Modal models folder<TAB>filename (optional).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from . import config, modal_client
except ImportError:  # python download_models.py from the plugin folder
    import config
    import modal_client


def _request(url, payload=None, headers=None, timeout=60):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, headers=headers or {}, method="POST" if body else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from None
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from None


def _read_manifest(path):
    items = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [part.strip() for part in raw.split("\t")]
        if len(cols) not in (2, 3) or not cols[0] or not cols[1]:
            raise ValueError(f"Line {line_no}: expected URL<TAB>folder<TAB>optional filename.")
        items.append({"url": cols[0], "model_type": cols[1], "filename": cols[2] if len(cols) == 3 else ""})
    if not items:
        raise ValueError("The manifest has no model rows.")
    if len(items) > 100:
        raise ValueError("A manifest can contain at most 100 models per run.")
    return items


def _download_one(cfg, item, overwrite):
    base = cfg.get("modal_endpoint_base", "").strip()
    key = cfg.get("bridge_api_key", "").strip()
    if not base or not key:
        raise RuntimeError("Deploy Modal Bridge first; its endpoint or bridge key is missing from config.")
    start_url = modal_client._endpoint(base, "download-model")
    status_url = modal_client._endpoint(base, "download-model-status")
    started = _request(start_url, {"auth_key": key, **item, "overwrite": overwrite},
                       {"Content-Type": "application/json"})
    download_id = started.get("download_id")
    if not download_id:
        raise RuntimeError(started.get("error", "Cloud downloader did not return a download ID."))
    print(f"  Started cloud download {download_id[:8]}…", flush=True)
    deadline = time.monotonic() + 6 * 60 * 60
    previous = None
    while time.monotonic() < deadline:
        query = urlencode({"download_id": download_id})
        state = _request(f"{status_url}?{query}", headers={"X-Bridge-Key": key})
        status = state.get("status")
        message = state.get("message") or status or "Waiting…"
        if message != previous:
            done = int(state.get("downloaded_bytes") or 0)
            total = int(state.get("total_bytes") or 0)
            progress = f" ({done / 1024**3:.2f}/{total / 1024**3:.2f} GiB)" if total else ""
            print(f"  {message}{progress}", flush=True)
            previous = message
        if status == "completed":
            return
        if status in ("failed", "not_found"):
            raise RuntimeError(state.get("error") or message)
        time.sleep(2)
    raise TimeoutError("Timed out waiting for the cloud download. Check Modal logs; it may still be running.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download model files directly into the Modal Volume.")
    parser.add_argument("--manifest", required=True, help="TSV file: URL, destination folder, optional filename")
    parser.add_argument("--overwrite", action="store_true", help="Replace files with matching names in the Volume")
    args = parser.parse_args(argv)
    try:
        items = _read_manifest(args.manifest)
        cfg = config.load_config()
    except Exception as exc:
        parser.error(str(exc))
    failures = []
    for index, item in enumerate(items, 1):
        destination = f"/models/{item['model_type']}/{item['filename']}".rstrip("/")
        print(f"[{index}/{len(items)}] {item['url']} → {destination}", flush=True)
        try:
            _download_one(cfg, item, args.overwrite)
        except Exception as exc:
            failures.append((index, str(exc)))
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
    if failures:
        print(f"\nCompleted with {len(failures)} failure(s):", file=sys.stderr)
        for index, reason in failures:
            print(f"  Row {index}: {reason}", file=sys.stderr)
        return 1
    print(f"\nAll {len(items)} model(s) are in the Modal Volume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
