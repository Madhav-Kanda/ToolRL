import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Tuple, List
import json


class ToolResult:
    def __init__(self, ok: bool, content: str, meta: Dict[str, Any] | None = None):
        self.ok = ok
        self.content = content
        self.meta = meta or {}


def calculator(expression: str) -> ToolResult:
    # Safe eval: allow numbers and basic operators only
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expression):
        return ToolResult(False, "Invalid expression")
    try:
        val = eval(expression, {"__builtins__": {}}, {"math": math})
        return ToolResult(True, str(val))
    except Exception as e:
        return ToolResult(False, f"Error: {e}")


def load_corpus(corpus_path: Path) -> List[Dict[str, str]]:
    if not corpus_path.exists():
        return []
    docs = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


def retriever(query: str, corpus_path: Path, top_k: int = 3) -> ToolResult:
    # Simple TF-IDF-lite: term overlap score
    docs = load_corpus(corpus_path)
    if not docs:
        return ToolResult(False, "No corpus")
    q_terms = set(re.findall(r"\w+", query.lower()))
    scored = []
    for d in docs:
        text = (d.get("text") or d.get("content") or "").lower()
        t_terms = set(re.findall(r"\w+", text))
        score = len(q_terms & t_terms)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s[1] for s in scored[:top_k]]
    return ToolResult(True, json.dumps(top, ensure_ascii=False))


def calendar(date: str, op: str = "weekday") -> ToolResult:
    # Minimal parser: YYYY-MM-DD
    m = re.fullmatch(r"(\\d{4})-(\\d{2})-(\\d{2})", date)
    if not m:
        return ToolResult(False, "Invalid date")
    import datetime as dt
    y, mo, d = map(int, m.groups())
    try:
        day = dt.date(y, mo, d)
    except Exception:
        return ToolResult(False, "Invalid date")
    if op == "weekday":
        return ToolResult(True, day.strftime("%A"))
    return ToolResult(False, "Unknown op")


def weather(city: str) -> ToolResult:
    # Mocked static table
    table = {"seattle": "rainy", "san francisco": "foggy", "london": "cloudy", "paris": "sunny"}
    cond = table.get(city.lower())
    if cond:
        return ToolResult(True, cond)
    return ToolResult(False, "Unknown city")


def sqldb(query: str, db_path: Path) -> ToolResult:
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        conn.commit()
        conn.close()
        return ToolResult(True, json.dumps(rows))
    except Exception as e:
        return ToolResult(False, f"Error: {e}")


class ToolRegistry:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = set(cfg.get("tools_enabled", []))
        self.corpus_path = Path(cfg.get("retriever", {}).get("corpus_jsonl", "data/tooluse/corpus.jsonl"))
        self.top_k = int(cfg.get("retriever", {}).get("top_k", 3))
        self.db_path = Path("data/tooluse/tool.db")

    def call(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name not in self.enabled:
            return ToolResult(False, f"Tool disabled: {name}")
        if name == "calculator":
            return calculator(str(args.get("expression", "")))
        if name == "retriever":
            return retriever(str(args.get("query", "")), self.corpus_path, self.top_k)
        if name == "calendar":
            return calendar(str(args.get("date", "")), str(args.get("op", "weekday")))
        if name == "weather":
            return weather(str(args.get("city", "")))
        if name == "sqldb":
            return sqldb(str(args.get("query", "")), self.db_path)
        return ToolResult(False, f"Unknown tool: {name}")


