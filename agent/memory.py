from collections import deque
from typing import Deque, Tuple


class RecentMemory:
    def __init__(self, max_chars: int = 1200):
        self.max_chars = max_chars
        self.buf: Deque[str] = deque()
        self.total = 0

    def add(self, text: str):
        if not text:
            return
        self.buf.append(text)
        self.total += len(text)
        while self.total > self.max_chars and self.buf:
            dropped = self.buf.popleft()
            self.total -= len(dropped)

    def tail(self) -> str:
        return "".join(self.buf)


