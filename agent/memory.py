"""
记忆模块：管理 FACT.md（持久知识）和 JOURNAL.jsonl（过程日志）
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


class AgentMemory:
    """Agent记忆系统：持久知识 + 过程日志"""

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.fact_path = self.workspace / "memory" / "FACT.md"
        self.journal_path = self.workspace / "memory" / "JOURNAL.jsonl"
        self.fact_path.parent.mkdir(parents=True, exist_ok=True)

    # ── FACT.md 操作 ──────────────────────────────────────────────

    def load_facts(self) -> str:
        """读取 FACT.md 内容"""
        if self.fact_path.exists():
            return self.fact_path.read_text(encoding="utf-8")
        return ""

    def update_facts(self, content: str):
        """覆写 FACT.md"""
        self.fact_path.write_text(content, encoding="utf-8")

    def append_fact(self, section: str, content: str):
        """在 FACT.md 中追加一个章节"""
        existing = self.load_facts()
        marker = f"## {section}"
        if marker in existing:
            # 替换已有章节
            lines = existing.split("\n")
            new_lines = []
            skip = False
            for line in new_lines:
                pass
            # 简化处理：直接在末尾追加
            pass
        with open(self.fact_path, "a", encoding="utf-8") as f:
            f.write(f"\n## {section}\n{content}\n")

    # ── JOURNAL.jsonl 操作 ────────────────────────────────────────

    def append_journal(self, entry: str, tags: Optional[list] = None):
        """追加一条日志"""
        tz = timezone(timedelta(hours=8))
        record = {
            "timestamp": datetime.now(tz).isoformat(),
            "entry": entry,
            "tags": tags or []
        }
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def search_journal(self, query: str = "", tag: str = "", limit: int = 20) -> list:
        """搜索日志"""
        results = []
        if not self.journal_path.exists():
            return results
        with open(self.journal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    match = True
                    if query and query.lower() not in record.get("entry", "").lower():
                        match = False
                    if tag and tag not in record.get("tags", []):
                        match = False
                    if match:
                        results.append(record)
                except json.JSONDecodeError:
                    continue
        return results[-limit:]
