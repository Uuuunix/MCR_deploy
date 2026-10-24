"""Incremental extraction of complete memories from a streamed JSON response."""

from __future__ import annotations

import json


class CandidateStreamAccumulator:
    """Keep complete, distinct objects without imposing a memory count limit."""

    def __init__(self, consecutive_repeat_limit: int = 5) -> None:
        self._chunks: list[str] = []
        self._cursor: int | None = None
        self._decoder = json.JSONDecoder()
        self._seen: set[tuple[str, str]] = set()
        self._memories: list[dict] = []
        self._consecutive_repeats = 0
        self._repeat_limit = consecutive_repeat_limit
        self.repetition_detected = False

    @property
    def complete_count(self) -> int:
        return len(self._memories)

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    def feed(self, chunk: str) -> bool:
        """Return true only after a sustained loop of complete duplicate objects."""
        self._chunks.append(chunk)
        if "}" not in chunk and self._cursor is not None:
            return False
        source = self.text
        if self._cursor is None:
            marker = source.find('"memories"')
            if marker < 0:
                return False
            opening = source.find("[", marker + len('"memories"'))
            if opening < 0:
                return False
            self._cursor = opening + 1
        while self._cursor < len(source):
            while self._cursor < len(source) and source[self._cursor] in " \t\r\n,":
                self._cursor += 1
            if self._cursor >= len(source) or source[self._cursor] == "]":
                break
            try:
                item, end = self._decoder.raw_decode(source, self._cursor)
            except json.JSONDecodeError:
                break
            if not isinstance(item, dict):
                break
            self._cursor = end
            key = (
                str(item.get("kind", "")).casefold().strip(),
                " ".join(str(item.get("content", "")).casefold().split()),
            )
            if key in self._seen:
                self._consecutive_repeats += 1
                if self._consecutive_repeats >= self._repeat_limit:
                    self.repetition_detected = True
                    return True
            else:
                self._seen.add(key)
                self._memories.append(item)
                self._consecutive_repeats = 0
        return False

    def recovered_json(self) -> str | None:
        if not self._memories:
            return None
        return json.dumps({"memories": self._memories}, ensure_ascii=False)
