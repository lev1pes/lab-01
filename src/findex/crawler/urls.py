"""Єдине представлення URL та вилучення видимого тексту."""

from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


def canonicalize(url: str, base: str = "") -> str:
    parts = urlsplit(urljoin(base, url.strip()))
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Потрібна HTTP(S)-адреса")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Облікові дані в URL не підтримуються")
    host = parts.hostname.lower().encode("idna").decode("ascii")
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and port != (443 if scheme == "https" else 80):
        host += f":{port}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag == "a" and not self.hidden:
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)
            if self.in_title:
                self.title_parts.append(data)


def parse_html(text: str) -> tuple[str, str, list[str]]:
    parser = PageParser()
    parser.feed(text)
    return (
        " ".join(" ".join(parser.title_parts).split()),
        " ".join(" ".join(parser.parts).split()),
        parser.links,
    )
