from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urlopen(url, timeout=5) as response:  # noqa: S310 - local/operator smoke target
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check a running Apricot web server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    home_status, home_body = fetch(f"{base_url}/")
    if home_status != 200 or b"Apricot" not in home_body:
        raise SystemExit(f"Home route failed: HTTP {home_status}")

    health_status, health_body = fetch(f"{base_url}/health")
    if health_status != 200:
        raise SystemExit(f"Health route failed: HTTP {health_status}")
    payload = json.loads(health_body.decode("utf-8"))
    required_keys = {"status", "model_exists", "model_path", "version"}
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise SystemExit(f"Health payload missing keys: {', '.join(missing_keys)}")

    print(json.dumps({"home": home_status, "health": payload}, indent=2))


if __name__ == "__main__":
    main()
