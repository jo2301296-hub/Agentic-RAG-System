"""Agentic RAG: plan -> act (tools) -> verify -> revise, plus the Standard RAG baseline."""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from . import prompts
from .config import MAX_AGENT_STEPS, MAX_PROMPT_TOKENS
from .llm import parse_json
from .retrieval import KnowledgeBase
from .tools import SCHEMAS, Source, Toolbox


@dataclass
class Event:
    """Progress shown live in the UI. type: plan | tool | verify | revise | answer"""
    type: str
    data: dict


@dataclass
class Result:
    answer: str
    sources: list[Source] = field(default_factory=list)  # only the cited ones
    verification: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    tool_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    models: list[str] = field(default_factory=list)


class AgenticRAG:
    def __init__(self, kb: KnowledgeBase, llm, verify: bool = True):
        self.kb, self.llm, self.verify = kb, llm, verify

    def run(self, question: str, history: list[tuple[str, str]] = ()) -> Iterator[Event]:
        started = time.monotonic()
        box = Toolbox(self.kb, self.llm)
        out = Result("")
        calculations: list[str] = []
        docs = "\n".join(f"- {d}: {t}" for d, t in self.kb.docs.items())[:2500]
        tables = self.kb.schema() or "(none)"

        out.plan = self._plan(question, history, docs, tables, out)
        yield Event("plan", out.plan)

        messages = [{"role": "system", "content": prompts.AGENT.format(docs=docs, tables=tables,
                                                                      language=language(question))}]
        for q, a in history[-2:]:
            messages += [{"role": "user", "content": q[:500]}, {"role": "assistant", "content": a[:700]}]
        messages.append({"role": "user", "content": f"Question: {question}\n\nPlan:\n{json.dumps(out.plan, ensure_ascii=False)}"})

        needs_evidence = out.plan["strategy"] != "direct"
        nudged = revised = False
        steps_left = MAX_AGENT_STEPS
        while True:
            final = steps_left <= 0
            if final:
                messages.append({"role": "user", "content": "No more tool calls. Write the final answer now from what you "
                                                            "have, and state what is missing."})
            reply = self.llm.chat(fit(messages), None if final else SCHEMAS)
            self._count(reply, out)
            steps_left -= 1

            if reply.tool_calls and not final:
                messages.append(reply.as_message())
                for call in reply.tool_calls:
                    output = box.run(call.name, call.arguments)
                    out.tool_calls += 1
                    if call.name == "calculate" and not output.startswith("Error"):
                        calculations.append(output)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output[:6000]})
                    yield Event("tool", {"name": call.name, "args": parse_json(call.arguments), "output": output})
                continue

            answer = reply.text.strip()
            if not answer and not final:
                messages.append({"role": "user", "content": "Your reply was empty. Continue: call a tool or answer."})
                continue
            if needs_evidence and not out.tool_calls and not nudged and not final:
                nudged = True
                messages += [reply.as_message(), {"role": "user", "content": "You have not looked at the knowledge base. Use "
                                                                             "the tools before answering."}]
                continue

            answer, cited = finalize_citations(answer, box.registry)
            out.answer, out.sources = answer, cited
            if not (self.verify and needs_evidence and answer):
                break
            out.verification = self._verify(question, answer, box, calculations, out)
            yield Event("verify", out.verification)
            if out.verification.get("ok", True) or revised or final:
                break
            revised = True
            steps_left = max(steps_left, 2)
            yield Event("revise", {"issues": out.verification["issues"]})
            messages += [reply.as_message(), {"role": "user", "content":
                         "A reviewer found problems with that answer:\n- " + "\n- ".join(out.verification["issues"]) +
                         (f"\nSuggested search: {out.verification['suggested_query']}"
                          if out.verification["suggested_query"] else "") +
                         "\nFix them, using tools if needed, then write the full corrected answer for the user. Do not "
                         "mention the review."}]

        if not out.answer:
            out.answer = "I could not produce an answer. Please rephrase the question or try again."
        out.tokens += box.tokens
        out.seconds = time.monotonic() - started
        yield Event("answer", {"result": out})

    def ask(self, question: str, history=()) -> Result:
        return list(self.run(question, history))[-1].data["result"]

    # --- stages --------------------------------------------------------------
    def _plan(self, question, history, docs, tables, out) -> dict:
        context = "".join(f"User: {q[:300]}\nAssistant: {a[:400]}\n" for q, a in history[-2:])
        reply = self.llm.chat([
            {"role": "system", "content": prompts.PLANNER.format(docs=docs, tables=tables)},
            {"role": "user", "content": (f"Conversation:\n{context}\n" if context else "") + f"Question: {question}"}],
            role="fast", as_json=True, max_tokens=700)
        self._count(reply, out)
        raw = parse_json(reply.text)
        strategy = raw.get("strategy")
        return {"strategy": strategy if strategy in ("direct", "single", "multi_step") else "multi_step",
                "reason": str(raw.get("reason", "")), "sub_questions": strings(raw.get("sub_questions"))[:4],
                "steps": strings(raw.get("steps"))[:5]}

    def _verify(self, question, answer, box, calculations, out) -> dict:
        cited = {s.number for s in out.sources}
        ordered = out.sources + [s for s in box.registry.sources if s.number not in cited]
        evidence = "\n\n".join(f"[{s.number}] {s.label}\n{s.text[:1500]}" for s in ordered)[:9000]
        reply = self.llm.chat([
            {"role": "system", "content": prompts.VERIFIER},
            {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence or '(none)'}\n\n"
                                        f"Calculations:\n" + ("\n".join(calculations) or "(none)")}],
            role="fast", as_json=True, max_tokens=800)
        self._count(reply, out)
        raw = parse_json(reply.text)
        if not raw:  # never block an answer because the reviewer output was malformed
            return {"ok": True, "checked": False, "issues": [], "suggested_query": ""}
        issues = strings(raw.get("issues"))
        if box.registry.sources and not cited:
            issues.append("The answer has no source citations.")
        ok = raw.get("grounded") is not False and raw.get("complete") is not False and not (
            box.registry.sources and not cited)
        if not ok and not issues:
            issues = ["The answer is not fully supported or not complete."]
        return {"ok": ok, "checked": True, "issues": issues, "suggested_query": str(raw.get("suggested_query") or "")}

    @staticmethod
    def _count(reply, out) -> None:
        out.tokens += reply.tokens
        if reply.model and reply.model not in out.models:
            out.models.append(reply.model)


class StandardRAG:
    """Baseline: embed the question, take the top passages, answer once."""

    def __init__(self, kb: KnowledgeBase, llm, k: int = 4):
        self.kb, self.llm, self.k = kb, llm, k

    def ask(self, question: str) -> Result:
        started = time.monotonic()
        box = Toolbox(self.kb)
        passages = "\n\n".join(box._cite(c) for c in self.kb.search(question, "semantic", self.k))
        reply = self.llm.chat([
            {"role": "system", "content": prompts.STANDARD.format(language=language(question))},
            {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"}], max_tokens=1000)
        answer, cited = finalize_citations(reply.text.strip(), box.registry)
        return Result(answer, cited, tokens=reply.tokens, seconds=time.monotonic() - started, models=[reply.model])


# --- helpers -----------------------------------------------------------------
def strings(value) -> list[str]:
    return [str(v) for v in value if v] if isinstance(value, list) else []


def language(text: str) -> str:
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "Arabic (keep names and file names in English)" if arabic > latin else "the language of the question"


def approx_tokens(messages: list[dict]) -> int:
    # Arabic text uses about 2 characters per token, English about 4.
    total = 0
    for m in messages:
        text = (m.get("content") or "") + json.dumps(m.get("tool_calls") or "", ensure_ascii=False)
        arabic = len(re.findall(r"[؀-ۿ]", text))
        total += arabic // 2 + (len(text) - arabic) // 4
    return total


def fit(messages: list[dict], limit: int = MAX_PROMPT_TOKENS) -> list[dict]:
    """Shorten the oldest tool outputs until the prompt fits the per-minute token budget."""
    messages = list(messages)
    for cut in (600, 150):
        for i, m in enumerate(messages):
            if approx_tokens(messages) <= limit:
                return messages
            if m["role"] == "tool" and len(m["content"]) > cut:
                messages[i] = {**m, "content": m["content"][:cut] + " ...(shortened)"}
    return messages


DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def finalize_citations(answer: str, registry) -> tuple[str, list[Source]]:
    """Normalize citation styles, drop numbers that do not exist, return the cited sources in order."""
    answer = re.sub(r"【\s*([^】]*?)\s*】", lambda m: "[" + m[1].split("†")[0] + "]", answer)
    answer = re.sub(r"\[[\d٠-٩,،;\s–-]+\]", lambda m: _split_group(m[0].translate(DIGITS)), answer)
    used: list[Source] = []

    def check(match):
        source = registry.get(int(match[1]))
        if not source:
            return ""
        if source not in used:
            used.append(source)
        return match[0]

    answer = re.sub(r"\[(\d+)\]", check, answer)
    return answer.strip(), sorted(used, key=lambda s: s.number)


def _split_group(text: str) -> str:
    """[1, 3] -> [1][3] and [1-3] -> [1][2][3]."""
    numbers: list[int] = []
    for part in re.split(r"[,،;\s]+", text.strip("[] ")):
        if match := re.fullmatch(r"(\d+)[–-](\d+)", part):
            numbers += range(int(match[1]), min(int(match[2]), int(match[1]) + 20) + 1)
        elif part.isdigit():
            numbers.append(int(part))
    return "".join(f"[{n}]" for n in numbers)
