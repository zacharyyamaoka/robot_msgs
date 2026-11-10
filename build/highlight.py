"""A small regex highlighter for the Python and JSON the site shows (no external dependency)."""

from __future__ import annotations

import html
import keyword
import re

PY = re.compile(
    r"""(?P<comment>\#[^\n]*)
      |(?P<string>[rbfRBF]{0,2}(?:'[^'\n]*'|"[^"\n]*"))
      |(?P<number>(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w]))
      |(?P<word>[A-Za-z_]\w*)""",
    re.X,
)
JSON = re.compile(
    r"""(?P<key>"(?:[^"\\]|\\.)*"(?=\s*:))
      |(?P<string>"(?:[^"\\]|\\.)*")
      |(?P<number>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
      |(?P<word>true|false|null)""",
    re.X,
)
KEYWORDS = set(keyword.kwlist)
BUILTINS = {"print", "len", "range", "float", "int", "str", "tuple", "list", "dict", "isinstance"}


def _span(kind: str, text: str) -> str:
    return f'<span class="tok-{kind}">{html.escape(text)}</span>'


def highlight(code: str, language: str = "python") -> str:
    pattern = JSON if language == "json" else PY
    out: list[str] = []
    last = 0
    for match in pattern.finditer(code):
        out.append(html.escape(code[last : match.start()]))
        kind, text = match.lastgroup, match.group()
        if kind == "word":
            if language == "json":
                kind = "keyword"
            elif text in KEYWORDS or text in {"None", "True", "False"}:
                kind = "keyword"
            elif text in BUILTINS:
                kind = "builtin"
            elif text[0].isupper():
                kind = "type"
            else:
                out.append(html.escape(text))
                last = match.end()
                continue
        out.append(_span(kind, text))
        last = match.end()
    out.append(html.escape(code[last:]))
    return "".join(out)
