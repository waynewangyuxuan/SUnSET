"""
LLM Client Module

Provides unified interface for OpenAI and VLLM backends.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from ..config import LLMConfig
from ..utils.logging import StageLogger


class LLMClient:
    """
    Unified LLM client supporting OpenAI and VLLM backends.

    Usage:
        client = LLMClient(config.llm)

        # Sync call
        response = client.generate("Hello")

        # Async call
        response = await client.agenerate("Hello")
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = StageLogger("stage1", "llm_client", log_format="text")

        if config.provider == "openai":
            self.client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
            self.async_client = AsyncOpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
            self.model = config.openai_model
        else:  # vllm
            self.client = OpenAI(
                api_key="dummy",
                base_url=config.vllm_url,
            )
            self.async_client = AsyncOpenAI(
                api_key="dummy",
                base_url=config.vllm_url,
            )
            self.model = config.vllm_model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from prompt (synchronous).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override config temperature
            max_tokens: Override config max_tokens

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        return response.choices[0].message.content

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from prompt (asynchronous).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Generate and parse JSON response.
        """
        response = self.generate(prompt, system_prompt)
        return self._parse_json(response)

    async def agenerate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Generate and parse JSON response (async).
        """
        response = await self.agenerate(prompt, system_prompt)
        return self._parse_json(response)

    def _parse_json(self, text: str) -> Any:
        """
        Parse JSON from LLM response, handling common issues.
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object/array in text
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Give up and return raw text
        self.logger.warning(
            "Failed to parse JSON from LLM response",
            data={"response_preview": text[:200]},
        )
        return text


class BatchLLMClient:
    """
    Batch LLM client with concurrency control.

    Usage:
        batch_client = BatchLLMClient(config.llm, max_concurrent=10)
        results = await batch_client.batch_generate(prompts)
    """

    def __init__(self, config: LLMConfig, max_concurrent: int = 10):
        self.client = LLMClient(config)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = StageLogger("stage1", "batch_llm", log_format="text")

    async def _generate_one(
        self,
        prompt: str,
        index: int,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate for one prompt with semaphore."""
        async with self.semaphore:
            try:
                result = await self.client.agenerate(prompt, system_prompt)
                return {"index": index, "success": True, "result": result}
            except Exception as e:
                self.logger.error(
                    f"LLM generation failed for index {index}",
                    data={"error": str(e)},
                )
                return {"index": index, "success": False, "error": str(e)}

    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate for multiple prompts concurrently.

        Returns:
            List of dicts with {"index", "success", "result"/"error"}
        """
        tasks = [
            self._generate_one(prompt, i, system_prompt)
            for i, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x["index"])

    async def batch_generate_json(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate and parse JSON for multiple prompts.
        """
        results = await self.batch_generate(prompts, system_prompt)

        for r in results:
            if r["success"]:
                r["result"] = self.client._parse_json(r["result"])

        return results
