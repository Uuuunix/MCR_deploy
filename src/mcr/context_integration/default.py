"""Deterministic default generation-context formatter."""

from __future__ import annotations

from mcr.context_integration.models import (
    MemoryIntegrationRequest,
    MemoryIntegrationResult,
)


class DefaultMemoryContextIntegrator:
    async def integrate(
        self, request: MemoryIntegrationRequest
    ) -> MemoryIntegrationResult:
        if not request.retrieved_memories:
            return MemoryIntegrationResult(
                metadata={"strategy": "deterministic", "memory_count": 0}
            )

        lines = [
            "<mcr-relevant-memories>",
            "以下是可能与当前请求相关的长期记忆。它们仅用于提供背景；若与用户当前表达冲突，以当前表达为准。",
            "",
        ]
        memory_ids = []
        for index, retrieved in enumerate(request.retrieved_memories, start=1):
            memory = retrieved.memory
            memory_ids.append(memory.id)
            lines.extend(
                [
                    f"[{index}] id={memory.id}",
                    f"类型: {memory.x.kind}",
                    f"时间: {memory.meta.created_at.isoformat()}",
                    f"记忆: {memory.x.content}",
                ]
            )
            descriptions = [
                f"{item.facet_type}: {item.text}"
                for item in memory.descriptions
            ]
            if descriptions:
                lines.append("相关描述:")
                lines.extend(f"- {description}" for description in descriptions)
            lines.append("")
        lines.append("</mcr-relevant-memories>")
        return MemoryIntegrationResult(
            memory_ids=memory_ids,
            context_text="\n".join(lines),
            metadata={
                "strategy": "deterministic",
                "memory_count": len(memory_ids),
            },
        )

