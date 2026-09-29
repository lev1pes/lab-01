"""Власне HTTP-дзеркало: лише loopback, тексти вже завантаженого корпусу."""

import argparse
import html
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class Mirror(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path, port=8766, latency=0.08):
        self.latency = latency
        self.counts = Counter()
        self.lock = threading.Lock()
        paths = sorted(root.rglob("*.txt"))[:199]
        if len(paths) < 199:
            raise ValueError("Для досліду потрібно принаймні 199 текстів корпусу")
        self.pages = {
            f"/doc/{i}": (
                f"<title>{html.escape(path.stem)}</title>"
                f"<main>{html.escape(path.read_text(encoding='utf-8-sig'))}</main>"
                '<a href="/">Каталог</a>'
            ).encode()
            for i, path in enumerate(paths)
        }
        self.pages["/"] = (
            "<title>Локальна документація Python</title>"
            "<p>asyncio Python event loop</p>"
            + "".join(
                f'<a href="/doc/{i}">{path.stem}</a>' for i, path in enumerate(paths)
            )
            + '<a href="/private">Не завантажувати</a>'
            '<a href="/doc/0#part">Дублікат</a>'
        ).encode("utf-8")
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        mirror = self.server
        path = urlsplit(self.path).path.rstrip("/") or "/"
        with mirror.lock:
            mirror.counts[path] += 1
            count = mirror.counts[path]
        time.sleep(mirror.latency)  # Затримка сервера, не циклу подій краулера.
        headers = {}
        if path == "/robots.txt":
            status, content = 200, b"User-agent: *\nDisallow: /private\n"
        elif path == "/doc/10" and count == 1:
            status, content = 429, b"controlled failure"
            headers["Retry-After"] = "1"
        else:
            content = mirror.pages.get(path, b"not found")
            status = 200 if path in mirror.pages else 404
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/python-docs"))
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--latency", type=float, default=0.08)
    args = parser.parse_args()
    with Mirror(args.root, args.port, args.latency) as server:
        print(f"Локальне дзеркало: http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
