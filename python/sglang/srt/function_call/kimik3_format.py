import re
from typing import List

THINK_OPEN = "<|open|>think<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"
RESPONSE_OPEN = "<|open|>response<|sep|>"
RESPONSE_CLOSE = "<|close|>response<|sep|>"
TOOLS_OPEN = "<|open|>tools<|sep|>"
TOOLS_CLOSE = "<|close|>tools<|sep|>"
MESSAGE_CLOSE = "<|close|>message<|sep|>"
CALL_OPEN = "<|open|>call"
CALL_CLOSE = "<|close|>call<|sep|>"
ARGUMENT_OPEN = "<|open|>argument"
ARGUMENT_CLOSE = "<|close|>argument<|sep|>"

# max_tokens can stop after an XTML control token or channel name, before <|sep|>.
_PARTIAL_MARKER_SUFFIXES = (
    "<|open|>",
    "<|close|>",
    THINK_CLOSE.removesuffix("<|sep|>"),
    RESPONSE_OPEN.removesuffix("<|sep|>"),
    RESPONSE_CLOSE.removesuffix("<|sep|>"),
    TOOLS_OPEN.removesuffix("<|sep|>"),
    TOOLS_CLOSE.removesuffix("<|sep|>"),
    MESSAGE_CLOSE.removesuffix("<|sep|>"),
)


def partial_suffix_len(text: str, markers: List[str]) -> int:
    best = 0
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:length]):
                best = length
                break
    return best


def strip_partial_marker_suffix(text: str) -> str:
    for suffix in _PARTIAL_MARKER_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def strip_response_wrappers(text: str) -> str:
    open_idx = text.find(RESPONSE_OPEN)
    if open_idx != -1:
        close_idx = text.find(RESPONSE_CLOSE, open_idx + len(RESPONSE_OPEN))
        if close_idx != -1:
            text = text[open_idx + len(RESPONSE_OPEN) : close_idx]
        else:
            text = text[open_idx + len(RESPONSE_OPEN) :]
    else:
        text = text.replace(RESPONSE_CLOSE, "")
    text = text.replace(MESSAGE_CLOSE, "")
    return strip_partial_marker_suffix(text)


# A complete ``call`` or ``argument`` block inside the XTML ``tools`` channel.
# The backreference ties the close to its own kind, so a call cannot be closed
# by an argument tag or vice versa.
_TOOL_BLOCK_RE = re.compile(
    r"<\|open\|>((?:call|argument))(?P<head>[^<]*?)<\|sep\|>.*?"
    r"<\|close\|>\1<\|sep\|>",
    re.DOTALL,
)

# Any bare piece of the tools-channel vocabulary left after complete blocks have
# been removed: an unterminated call/argument head (the stream cut before its
# close tag), or a stray open/close marker whose enclosing block was lost when
# the ``<|open|>tools<|sep|>`` marker was split across a stream boundary.
_BARE_TOOL_MARKER_RE = re.compile(
    r"<\|open\|>tools<\|sep\|>"
    r"|<\|close\|>tools<\|sep\|>"
    r"|<\|open\|>call<\|sep\|>[^<]*"
    r"|<\|open\|>call(?P<head>\s+[^<]*?)<\|sep\|>[^<]*"
    r"|<\|open\|>argument<\|sep\|>[^<]*"
    r"|<\|open\|>argument(?P<ahead>\s+[^<]*?)<\|sep\|>[^<]*"
    r"|<\|close\|>call<\|sep\|>"
    r"|<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)


def strip_tool_markup(text: str) -> str:
    """Remove Kimi-K3 tool-call XTML from visible text.

    A tool call reaching the normal-text path at all is a desync: the wrapper
    `<|open|>tools<|sep|>` marker was split across a stream chunk or lost, so
    the detector never buffered the section as a structured call. The safe
    answer echoes `detect_and_parse`'s own rule -- never ship the XTML tools
    markup to the client -- so complete blocks are dropped and any bare
    leftover markers after them are dropped too.
    """
    text = _TOOL_BLOCK_RE.sub("", text)
    return _BARE_TOOL_MARKER_RE.sub("", text)
