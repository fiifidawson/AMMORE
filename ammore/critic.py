import asyncio
from typing import Literal, Sequence

import dspy
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from .config import config, get_api_key


class Judge(dspy.Signature):
    """Judge if the retrieved chunks address the original question. Be strict: chunks that exist but are off-topic count as no coverage."""

    question: str = dspy.InputField()
    chunks: str = dspy.InputField()
    verdict: Literal["COVERAGE_OK", "NEEDS_MORE"] = dspy.OutputField()
    followups: list[str] = dspy.OutputField(
        desc="up to 3 follow-up queries when NEEDS_MORE, else empty"
    )
    use_web: bool = dspy.OutputField(
        desc="true if chunks are clearly off-topic; the retriever should switch to search_web_tool"
    )


_lm_set = False


def _configure_dspy():
    global _lm_set
    if _lm_set:
        return
    dspy.configure(
        lm=dspy.LM(
            f"openai/{config.mistral_model}",
            api_base=config.mistral_base_url,
            api_key=get_api_key(),
        )
    )
    _lm_set = True


class CriticAgent(BaseChatAgent):
    def __init__(self):
        super().__init__("Critic", "Judges coverage with structured output.")
        _configure_dspy()
        self._judge = dspy.Predict(Judge)

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        question, chunks = _extract(messages)
        out = await asyncio.to_thread(self._judge, question=question, chunks=chunks)
        text = out.verdict
        if out.verdict == "NEEDS_MORE":
            for q in out.followups[:3]:
                text += f"\n- {q}"
            if out.use_web:
                text += "\nUse search_web_tool for these."
        return Response(chat_message=TextMessage(content=text, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken):
        pass


def _extract(messages: Sequence[BaseChatMessage]):
    question = ""
    chunks = []
    for m in messages:
        c = getattr(m, "content", "")
        if not isinstance(c, str):
            continue
        if "Original question:" in c and not question:
            question = c.split("Original question:", 1)[1].split("\n\n", 1)[0].strip()
        if getattr(m, "source", None) == "Retriever":
            chunks.append(c)
    return question, "\n\n".join(chunks[-3:])
