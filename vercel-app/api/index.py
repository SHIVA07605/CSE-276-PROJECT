"""
Vercel serverless entry point for Material Match (copied to vercel-app/api/index.py by build_vercel_app.py).

The page itself (index.html) is served statically by Vercel; every /api/* request is rewritten here and handled
by the same Handler as the local web app, using the pure-numpy MiniLM encoder (no PyTorch, fits Vercel's size limit).
"""

import os
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MATERIAL_MATCH_ENCODER", "numpy")
os.environ.setdefault("MATERIAL_MATCH_MINILM_DIR", str(ROOT / "minilm"))
os.environ.setdefault("MATERIAL_MATCH_HOSTED", "1")

import web_app  # noqa: E402

# Loaded once per serverless instance (cold start ~10 s), then reused across requests.
_SERVICE = web_app.MaterialMatchService(verbose=False)


class handler(web_app.Handler):  # noqa: N801 - Vercel looks for a class named `handler`
    service = _SERVICE

    def _normalise_path(self) -> None:
        # The rewrite "/api/(.*)" -> "/api/index?route=$1" may hand us /api/index?route=compare
        url = urllib.parse.urlparse(self.path)
        if url.path.rstrip("/") == "/api/index":
            params = urllib.parse.parse_qs(url.query)
            route = params.pop("route", [""])[0]
            rest = urllib.parse.urlencode({k: v[0] for k, v in params.items()})
            self.path = f"/api/{route}" + (f"?{rest}" if rest else "")

    def do_GET(self) -> None:  # noqa: N802
        self._normalise_path()
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        self._normalise_path()
        super().do_POST()
