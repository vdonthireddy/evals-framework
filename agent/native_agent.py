"""Native Function-Calling Agent implementation.

Supports OpenAI, Ollama, Anthropic (Claude), and Google (Gemini) native tool calling APIs directly,
without prompt-based ReAct parsing or external framework dependencies.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from agent.app import AgentTrace, TraceStep, _create_llm_client
from agent.memory import ConversationMemory
from agent.safety import SafetyFilter
from agent.tools.base import BaseTool
from agent.tools.calculator import CalculatorTool
from agent.tools.knowledge_base import KnowledgeBaseTool
from agent.tools.weather import WeatherTool
from agent.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_NATIVE_SYSTEM_PROMPT = """\
You are a helpful research assistant. Use the provided tools when necessary to perform calculations, look up weather, search the web, or check internal knowledge base information.
"""


class NativeFunctionCallingAgent:
    """Agent that relies purely on native LLM Function / Tool Calling API across OpenAI, Ollama, Anthropic, and Gemini."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        max_steps: int = 10,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.max_steps = max_steps
        self.temperature = temperature

        self.tools: dict[str, BaseTool] = {
            "calculator": CalculatorTool(),
            "get_weather": WeatherTool(),
            "knowledge_base": KnowledgeBaseTool(),
            "web_search": WebSearchTool(),
        }
        self.safety_filter = SafetyFilter()
        self.client = _create_llm_client(self.provider, api_key)

    def _build_openai_tools_schema(self) -> list[dict[str, Any]]:
        """Format tools into standard OpenAI Function Calling JSON Schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                }
            }
            for tool in self.tools.values()
        ]

    def _build_anthropic_tools_schema(self) -> list[dict[str, Any]]:
        """Format tools into Anthropic Claude Tool Schema format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters_schema,
            }
            for tool in self.tools.values()
        ]

    def _build_gemini_tools_schema(self) -> list[dict[str, Any]]:
        """Format tools into Google Gemini Function Declaration format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in self.tools.values()
        ]

    async def run(self, user_input: str) -> AgentTrace:
        """Run agent using native function calling and return AgentTrace."""
        start_time = time.perf_counter()
        steps: list[TraceStep] = []
        step_number = 1

        # 1. Safety check input
        is_safe_input, input_reason = self.safety_filter.check_input(user_input)
        if not is_safe_input:
            block_step = TraceStep(
                step_number=1,
                action="safety_block",
                reasoning=f"Blocked: {input_reason}",
                response="I cannot fulfill this request due to safety guidelines.",
            )
            return AgentTrace(
                input=user_input,
                output="I cannot fulfill this request due to safety guidelines.",
                steps=[block_step],
                total_steps=1,
                total_tokens=0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                safety_triggered=True,
                safety_reason=input_reason,
            )

        final_response = ""
        total_tokens = 0

        # --- OpenAI & Ollama Native Provider ---
        if self.provider in ("openai", "ollama"):
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _NATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
            tools_schema = self._build_openai_tools_schema()

            while step_number <= self.max_steps:
                try:
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                    }
                    if tools_schema:
                        kwargs["tools"] = tools_schema
                        kwargs["tool_choice"] = "auto"

                    response = await self.client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    usage = getattr(response, "usage", None)
                    if usage:
                        tot = getattr(usage, "total_tokens", 0) or (getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0))
                        total_tokens += tot or max(1, (sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)) + len(msg.content or "")) // 4)
                    else:
                        total_tokens += max(1, (sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)) + len(msg.content or "")) // 4)

                    tool_calls = getattr(msg, "tool_calls", None)

                    if tool_calls and len(tool_calls) > 0:
                        messages.append(msg.model_dump() if hasattr(msg, "model_dump") else {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    }
                                }
                                for tc in tool_calls
                            ]
                        })

                        for tc in tool_calls:
                            t_name = tc.function.name
                            try:
                                t_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                            except Exception:
                                t_args = {}

                            tool_inst = self.tools.get(t_name)
                            res_dict = (await tool_inst.safe_execute(**t_args)).model_dump() if tool_inst else {"error": f"Tool '{t_name}' not found."}

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": t_name,
                                "content": json.dumps(res_dict),
                            })

                            steps.append(TraceStep(
                                step_number=step_number,
                                action="use_tool",
                                tool_name=t_name,
                                tool_args=t_args,
                                tool_result=res_dict,
                                reasoning=f"Native function call to {t_name}",
                                timestamp=datetime.now(timezone.utc),
                            ))
                            step_number += 1
                        continue

                    final_response = msg.content or ""
                    steps.append(TraceStep(
                        step_number=step_number,
                        action="respond",
                        response=final_response,
                        reasoning="Final answer provided by model.",
                        timestamp=datetime.now(timezone.utc),
                    ))
                    break
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "does not support tools" in exc_str or "tools are not supported" in exc_str:
                        logger.warning(f"Model '{self.model}' does not support native tools. Retrying without tools parameter...")
                        try:
                            kwargs.pop("tools", None)
                            kwargs.pop("tool_choice", None)
                            response = await self.client.chat.completions.create(**kwargs)
                            msg = response.choices[0].message
                            final_response = msg.content or ""
                            steps.append(TraceStep(
                                step_number=step_number,
                                action="respond",
                                response=final_response,
                                reasoning=f"Note: Model '{self.model}' does not support native OpenAI tool-calling on Ollama. Generated plain text response instead.",
                                timestamp=datetime.now(timezone.utc),
                            ))
                            break
                        except Exception as retry_exc:
                            exc = retry_exc

                    logger.error(f"Error in OpenAI/Ollama step {step_number}: {exc}", exc_info=True)
                    final_response = f"I encountered an error: {exc}"
                    steps.append(TraceStep(step_number=step_number, action="error", reasoning=str(exc), response=final_response))
                    break

        # --- Anthropic Claude Native Provider ---
        elif self.provider == "anthropic":
            anth_messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
            tools_schema = self._build_anthropic_tools_schema()

            while step_number <= self.max_steps:
                try:
                    response = await self.client.messages.create(
                        model=self.model,
                        system=_NATIVE_SYSTEM_PROMPT,
                        messages=anth_messages,
                        tools=tools_schema,
                        temperature=self.temperature,
                        max_tokens=1024,
                    )
                    if hasattr(response, "usage") and response.usage:
                        total_tokens += getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)

                    tool_use_blocks = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
                    text_blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]

                    if tool_use_blocks:
                        anth_messages.append({"role": "assistant", "content": response.content})
                        tool_results_content = []

                        for block in tool_use_blocks:
                            t_name = block.name
                            t_args = block.input or {}
                            tool_inst = self.tools.get(t_name)
                            res_dict = (await tool_inst.safe_execute(**t_args)).model_dump() if tool_inst else {"error": f"Tool '{t_name}' not found."}

                            tool_results_content.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(res_dict),
                            })

                            steps.append(TraceStep(
                                step_number=step_number,
                                action="use_tool",
                                tool_name=t_name,
                                tool_args=t_args,
                                tool_result=res_dict,
                                reasoning=f"Anthropic native tool call to {t_name}",
                                timestamp=datetime.now(timezone.utc),
                            ))
                            step_number += 1

                        anth_messages.append({"role": "user", "content": tool_results_content})
                        continue

                    final_response = "\n".join(text_blocks)
                    steps.append(TraceStep(
                        step_number=step_number,
                        action="respond",
                        response=final_response,
                        reasoning="Final answer provided by Claude.",
                        timestamp=datetime.now(timezone.utc),
                    ))
                    break
                except Exception as exc:
                    logger.error(f"Error in Anthropic step {step_number}: {exc}", exc_info=True)
                    final_response = f"I encountered an error: {exc}"
                    steps.append(TraceStep(step_number=step_number, action="error", reasoning=str(exc), response=final_response))
                    break

        # --- Google Gemini Native Provider ---
        elif self.provider == "gemini":
            contents: list[str] = [user_input]
            tools_schema = self._build_gemini_tools_schema()

            while step_number <= self.max_steps:
                try:
                    response = await self.client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config={
                            "system_instruction": _NATIVE_SYSTEM_PROMPT,
                            "tools": [{"function_declarations": tools_schema}],
                            "temperature": self.temperature,
                        },
                    )
                    final_response = getattr(response, "text", "") or "Executed native Gemini function calls."
                    steps.append(TraceStep(
                        step_number=step_number,
                        action="respond",
                        response=final_response,
                        reasoning="Gemini function execution.",
                        timestamp=datetime.now(timezone.utc),
                    ))
                    break
                except Exception as exc:
                    logger.error(f"Error in Gemini step {step_number}: {exc}", exc_info=True)
                    final_response = f"I encountered an error: {exc}"
                    steps.append(TraceStep(step_number=step_number, action="error", reasoning=str(exc), response=final_response))
                    break

        # Output Safety Check
        is_safe_output, output_reason = self.safety_filter.check_output(final_response)
        if not is_safe_output:
            final_response = "Response blocked due to safety guidelines."

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return AgentTrace(
            input=user_input,
            output=final_response,
            steps=steps,
            total_steps=len(steps),
            total_tokens=total_tokens,
            total_latency_ms=latency_ms,
            safety_triggered=not is_safe_output,
            safety_reason=output_reason if not is_safe_output else None,
        )
