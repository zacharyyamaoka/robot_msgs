"""Serve dist/ locally like Cloudflare does: the _headers CORS rule on /assets/viz/ included.

    python3 build/serve.py 4492
"""

import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist"


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        if self.path.startswith("/assets/viz/"):
            self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4492
    server = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(DIST)))
    print(f"http://127.0.0.1:{port}/")
    server.serve_forever()
