"""
SET Extractor Module

Implements event and stakeholder extraction using LLM (Paper Section 3.1, Appendix A).
"""

import asyncio
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..config import Config
from ..data import Article, article_to_text
from ..utils.logging import StageLogger
from ..utils.metrics import MetricsCollector
from .llm_client import LLMClient, BatchLLMClient


# Paper Prompt (Appendix A) - Event and Time Generation
EVENT_EXTRACTION_PROMPT = """You are a professional journalist that is tasked to generate date-based event summary of a given article. A single list contains an article and its published time. You should generate a dictionary of the most relevant events of an article, where each key in the dictionary is a string of the expected event start date in terms of Year-Month-Day (e.g. 2011-12-25) and the value will be a summary of the relevant events on that day. Summarize only the most important events found in the article, as succinctly as possible. If you are uncertain of the date of an event, feel free to use the published date. You should only output the dictionary in your answer. Generate a dictionary of events of the following article: {article}"""

# Paper Prompt (Appendix A) - Stakeholder Generation
STAKEHOLDER_EXTRACTION_PROMPT = """You are a professional journalist that is tasked to generate the most relevant stakeholders relevant to a given event summary of an article. A single list contains an article and its published time. You should generate a singular list containing not more than five relevant stakeholders related to only the stipulated event mentioned. These stakeholders should not be general, and must be identifiable named entities that can be matched to a person, organization or role when read on its own. Every single stakeholder generated should also ideally exist in exact wording as mentioned within the original article. You should only output the list of stakeholders in your answer, and all stakeholders should be enclosed in string format. Generate a list of related stakeholders of event: {event}.
Given article: {article}"""


@dataclass
class Event:
    """An extracted event with stakeholders."""
    id: int
    date: str  # YYYY-MM-DD
    summary: str
    stakeholders: List[str] = field(default_factory=list)
    source_article_index: Optional[int] = None
    source_pub_date: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(**d)


class EventExtractor:
    """
    Extracts events from articles using LLM.

    Usage:
        extractor = EventExtractor(config)

        # Extract from single article
        events = await extractor.extract_events(article)

        # Extract from multiple articles
        all_events = await extractor.extract_all(articles)
    """

    def __init__(self, config: Config, max_concurrent: int = 10):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.batch_llm = BatchLLMClient(config.llm, max_concurrent)
        self.logger = StageLogger("stage1", "event_extractor", log_format="text")
        self.metrics = MetricsCollector("stage1_extraction")

    async def extract_events_from_article(
        self,
        article: Article,
        article_index: int = 0,
    ) -> List[Event]:
        """
        Extract events from a single article.

        Returns:
            List of Event objects (without stakeholders yet)
        """
        article_text = article_to_text(article)
        prompt = EVENT_EXTRACTION_PROMPT.format(article=article_text)

        with self.metrics.timer("event_extraction"):
            try:
                response = await self.llm.agenerate(prompt)
                events_dict = self._parse_events_dict(response)
            except Exception as e:
                self.logger.error(
                    f"Event extraction failed for article {article_index}",
                    data={"error": str(e)},
                )
                return []

        events = []
        for date, summary in events_dict.items():
            # Validate date format
            if not self._is_valid_date(date):
                self.logger.warning(
                    f"Invalid date format: {date}, using pub_date",
                    data={"article_index": article_index},
                )
                date = article.pub_date

            events.append(Event(
                id=-1,  # Will be assigned later
                date=date,
                summary=summary,
                source_article_index=article_index,
                source_pub_date=article.pub_date,
            ))

        self.metrics.count("events_per_article", len(events))
        return events

    async def extract_stakeholders_for_event(
        self,
        event: Event,
        article: Article,
    ) -> List[str]:
        """
        Extract stakeholders for a single event.

        Returns:
            List of stakeholder names (max 5)
        """
        article_text = article_to_text(article)
        prompt = STAKEHOLDER_EXTRACTION_PROMPT.format(
            event=event.summary,
            article=article_text,
        )

        with self.metrics.timer("stakeholder_extraction"):
            try:
                response = await self.llm.agenerate(prompt)
                stakeholders = self._parse_stakeholders_list(response)
            except Exception as e:
                self.logger.error(
                    f"Stakeholder extraction failed for event {event.id}",
                    data={"error": str(e)},
                )
                return []

        # Limit to max 5 stakeholders (per paper)
        max_stakeholders = self.config.pipeline.max_stakeholders_per_event
        stakeholders = stakeholders[:max_stakeholders]

        self.metrics.count("stakeholders_per_event", len(stakeholders))
        return stakeholders

    async def extract_all(
        self,
        articles: List[Article],
        include_stakeholders: bool = True,
    ) -> List[Event]:
        """
        Extract events from all articles.

        Args:
            articles: List of articles to process
            include_stakeholders: Whether to also extract stakeholders

        Returns:
            List of all events with unique IDs
        """
        self.metrics.start()
        self.logger.info(f"Starting extraction for {len(articles)} articles")

        # Step 1: Extract events from all articles
        all_events = []
        for i, article in enumerate(articles):
            if i % 50 == 0:
                self.logger.info(f"Processing article {i}/{len(articles)}")

            events = await self.extract_events_from_article(article, i)
            all_events.extend(events)

        # Assign unique IDs
        for i, event in enumerate(all_events):
            event.id = i

        self.logger.info(f"Extracted {len(all_events)} events from {len(articles)} articles")

        # Step 2: Extract stakeholders for each event
        if include_stakeholders:
            self.logger.info("Extracting stakeholders for events...")

            for i, event in enumerate(all_events):
                if i % 100 == 0:
                    self.logger.info(f"Processing stakeholders {i}/{len(all_events)}")

                article = articles[event.source_article_index]
                stakeholders = await self.extract_stakeholders_for_event(event, article)
                event.stakeholders = stakeholders

            # Count events with stakeholders
            events_with_stakeholders = sum(1 for e in all_events if e.stakeholders)
            self.logger.info(
                f"Events with stakeholders: {events_with_stakeholders}/{len(all_events)}"
            )

        self.metrics.end()
        return all_events

    def _parse_events_dict(self, response: str) -> Dict[str, str]:
        """Parse LLM response into events dictionary."""
        # Try direct JSON parse
        try:
            import json
            result = json.loads(response)
            if isinstance(result, dict):
                return {str(k): str(v) for k, v in result.items()}
        except json.JSONDecodeError:
            pass

        # Try to extract from code block
        code_block = re.search(r"```(?:json|python)?\s*([\s\S]*?)```", response)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, dict):
                    return {str(k): str(v) for k, v in result.items()}
            except json.JSONDecodeError:
                pass

        # Try to find dict pattern
        dict_match = re.search(r"\{[\s\S]*\}", response)
        if dict_match:
            try:
                result = json.loads(dict_match.group(0))
                if isinstance(result, dict):
                    return {str(k): str(v) for k, v in result.items()}
            except json.JSONDecodeError:
                pass

            # Try eval (safer with literal_eval)
            try:
                import ast
                result = ast.literal_eval(dict_match.group(0))
                if isinstance(result, dict):
                    return {str(k): str(v) for k, v in result.items()}
            except (ValueError, SyntaxError):
                pass

        self.logger.warning(
            "Failed to parse events dict",
            data={"response_preview": response[:200]},
        )
        return {}

    def _parse_stakeholders_list(self, response: str) -> List[str]:
        """Parse LLM response into stakeholders list."""
        import json

        # Try direct JSON parse
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return [str(s) for s in result if s]
        except json.JSONDecodeError:
            pass

        # Try to extract from code block
        code_block = re.search(r"```(?:json|python)?\s*([\s\S]*?)```", response)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, list):
                    return [str(s) for s in result if s]
            except json.JSONDecodeError:
                pass

        # Try to find list pattern
        list_match = re.search(r"\[[\s\S]*\]", response)
        if list_match:
            try:
                result = json.loads(list_match.group(0))
                if isinstance(result, list):
                    return [str(s) for s in result if s]
            except json.JSONDecodeError:
                pass

            # Try eval
            try:
                import ast
                result = ast.literal_eval(list_match.group(0))
                if isinstance(result, list):
                    return [str(s) for s in result if s]
            except (ValueError, SyntaxError):
                pass

        # Try to extract quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', response)
        if quoted:
            return quoted

        self.logger.warning(
            "Failed to parse stakeholders list",
            data={"response_preview": response[:200]},
        )
        return []

    def _is_valid_date(self, date_str: str) -> bool:
        """Check if date string is in YYYY-MM-DD format."""
        if not date_str:
            return False
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        return bool(re.match(pattern, date_str))

    def get_metrics_report(self) -> dict:
        """Get extraction metrics report."""
        return self.metrics.get_report()
