"""
Coreference Resolution Module

Implements Wikidata-based entity resolution (Paper Appendix B, Algorithm 1).
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import aiohttp

from ..config import WikidataConfig
from ..utils.logging import StageLogger
from ..utils.metrics import MetricsCollector
from .extractor import Event


@dataclass
class EntityMapping:
    """Mapping from raw name to canonical entity."""
    raw_name: str
    canonical_name: str
    wikidata_id: Optional[str] = None
    entity_type: Optional[str] = None  # "person", "organization", "country", "unknown"
    resolution_method: Optional[str] = None  # "direct", "ner", "ampersand", "interface", "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


class EntityCache:
    """
    Persistent cache for entity resolutions.

    Usage:
        cache = EntityCache("entity_cache.json")
        cache.get("Barack Obama")  # Returns EntityMapping or None
        cache.set("Barack Obama", mapping)
        cache.save()
    """

    def __init__(self, cache_file: str = "entity_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, EntityMapping] = {}
        self._load()

    def _load(self):
        """Load cache from file."""
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for name, mapping_dict in data.items():
                    self.cache[name.lower()] = EntityMapping(**mapping_dict)

    def save(self):
        """Save cache to file."""
        data = {name: mapping.to_dict() for name, mapping in self.cache.items()}
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, name: str) -> Optional[EntityMapping]:
        """Get cached mapping."""
        return self.cache.get(name.lower())

    def set(self, name: str, mapping: EntityMapping):
        """Set cached mapping."""
        self.cache[name.lower()] = mapping

    def __contains__(self, name: str) -> bool:
        return name.lower() in self.cache

    def __len__(self) -> int:
        return len(self.cache)


class CoreferenceResolver:
    """
    Resolves entity coreference using Wikidata (Paper Algorithm 1).

    Resolution steps:
    1. Search Wikidata label/alt-label
    2. Remove title using NER, search again
    3. Replace whitespace with &&, search again
    4. Use interface search API

    Usage:
        resolver = CoreferenceResolver(config.wikidata)
        mapping = await resolver.resolve("President Obama")
        # Returns EntityMapping with canonical_name="Barack Obama"
    """

    def __init__(self, config: WikidataConfig):
        self.config = config
        self.cache = EntityCache(config.cache_file)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.logger = StageLogger("stage1", "coreference", log_format="text")
        self.metrics = MetricsCollector("stage1_coreference")

        # Wikidata API endpoints
        self.search_url = "https://www.wikidata.org/w/api.php"
        self.entity_url = "https://www.wikidata.org/wiki/Special:EntityData"

    async def _fetch_with_proxy(self, url: str, session: aiohttp.ClientSession) -> Optional[dict]:
        """Fetch URL through proxy."""
        try:
            # Get proxy
            async with session.get(self.config.proxy_url) as proxy_resp:
                proxy = await proxy_resp.text()
                proxy = proxy.strip()

            # Fetch through proxy
            async with session.get(
                url,
                proxy=f"http://{proxy}",
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            self.logger.warning(f"Proxy fetch failed: {e}", data={"url": url})

        # Fallback: direct fetch
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            self.logger.warning(f"Direct fetch failed: {e}", data={"url": url})

        return None

    async def _search_wikidata(
        self,
        query: str,
        session: aiohttp.ClientSession,
    ) -> Optional[Dict[str, Any]]:
        """Search Wikidata for entity."""
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": "5",
        }
        url = f"{self.search_url}?{'&'.join(f'{k}={quote(str(v))}' for k, v in params.items())}"

        data = await self._fetch_with_proxy(url, session)
        if data and data.get("search"):
            return data["search"][0]  # Return first result
        return None

    async def _get_entity_data(
        self,
        qid: str,
        session: aiohttp.ClientSession,
    ) -> Optional[Dict[str, Any]]:
        """Get full entity data from Wikidata."""
        url = f"{self.entity_url}/{qid}.json"
        data = await self._fetch_with_proxy(url, session)
        if data and "entities" in data:
            return data["entities"].get(qid)
        return None

    def _classify_entity(self, entity_data: Dict[str, Any]) -> str:
        """Classify entity type based on Wikidata claims."""
        if not entity_data:
            return "unknown"

        claims = entity_data.get("claims", {})

        # Check P31 (instance of)
        if "P31" in claims:
            for claim in claims["P31"]:
                try:
                    qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                    # Q5 = human
                    if qid == "Q5":
                        return "person"
                    # Q6256 = country
                    if qid == "Q6256":
                        return "country"
                    # Q43229 = organization, Q4830453 = business
                    if qid in ["Q43229", "Q4830453", "Q783794", "Q891723"]:
                        return "organization"
                except (KeyError, TypeError):
                    continue

        return "unknown"

    def _get_position_holder(self, entity_data: Dict[str, Any]) -> Optional[str]:
        """Check if entity has 'position held by' and return that person."""
        if not entity_data:
            return None

        claims = entity_data.get("claims", {})

        # P1308 = officeholder, P39 = position held
        for prop in ["P1308", "P39"]:
            if prop in claims:
                for claim in claims[prop]:
                    try:
                        return claim["mainsnak"]["datavalue"]["value"]["id"]
                    except (KeyError, TypeError):
                        continue
        return None

    def _remove_title_ner(self, name: str) -> str:
        """Remove title from name using simple rules (fallback if spacy unavailable)."""
        # Common titles to remove
        titles = [
            "president", "prime minister", "minister", "secretary",
            "senator", "governor", "mayor", "king", "queen", "prince",
            "princess", "dr", "mr", "mrs", "ms", "general", "colonel",
            "captain", "chief", "director", "ceo", "chairman",
        ]

        words = name.split()
        filtered = []
        for word in words:
            word_lower = word.lower().rstrip(".")
            if word_lower not in titles:
                filtered.append(word)

        return " ".join(filtered) if filtered else name

    async def resolve(
        self,
        name: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> EntityMapping:
        """
        Resolve a stakeholder name to canonical form.

        Implements Algorithm 1 from Paper Appendix B:
        1. Direct search
        2. Remove title with NER, search
        3. Replace space with &&, search
        4. Interface search
        5. Fallback to original
        """
        # Check cache first
        cached = self.cache.get(name)
        if cached:
            self.metrics.count("cache_hit", 1)
            return cached

        self.metrics.count("cache_miss", 1)

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        try:
            async with self.semaphore:
                with self.metrics.timer("resolution"):
                    mapping = await self._resolve_impl(name, session)
        finally:
            if own_session:
                await session.close()

        self.cache.set(name, mapping)
        return mapping

    async def _resolve_impl(
        self,
        name: str,
        session: aiohttp.ClientSession,
    ) -> EntityMapping:
        """Internal resolution implementation."""

        # Step 1: Direct search
        result = await self._search_wikidata(name, session)
        if result:
            mapping = await self._process_search_result(name, result, session, "direct")
            if mapping:
                return mapping

        # Step 2: Remove title with NER
        name_no_title = self._remove_title_ner(name)
        if name_no_title != name:
            result = await self._search_wikidata(name_no_title, session)
            if result:
                mapping = await self._process_search_result(name, result, session, "ner")
                if mapping:
                    return mapping

        # Step 3: Replace space with &&
        name_ampersand = name_no_title.replace(" ", "&&")
        if name_ampersand != name_no_title:
            result = await self._search_wikidata(name_ampersand, session)
            if result:
                mapping = await self._process_search_result(name, result, session, "ampersand")
                if mapping:
                    return mapping

        # Step 4: Interface search (using haswbstatement)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": name,
            "format": "json",
        }
        url = f"{self.search_url}?{'&'.join(f'{k}={quote(str(v))}' for k, v in params.items())}"
        data = await self._fetch_with_proxy(url, session)
        if data and data.get("query", {}).get("search"):
            title = data["query"]["search"][0].get("title", "")
            if title:
                # Search by title
                result = await self._search_wikidata(title, session)
                if result:
                    mapping = await self._process_search_result(name, result, session, "interface")
                    if mapping:
                        return mapping

        # Step 5: Fallback
        self.metrics.count("resolution_fallback", 1)
        return EntityMapping(
            raw_name=name,
            canonical_name=name,
            resolution_method="fallback",
        )

    async def _process_search_result(
        self,
        raw_name: str,
        result: Dict[str, Any],
        session: aiohttp.ClientSession,
        method: str,
    ) -> Optional[EntityMapping]:
        """Process a Wikidata search result."""
        qid = result.get("id")
        label = result.get("label", raw_name)

        if not qid:
            return None

        # Get entity data
        entity_data = await self._get_entity_data(qid, session)

        # Check for position holder
        position_holder_qid = self._get_position_holder(entity_data)
        if position_holder_qid:
            holder_data = await self._get_entity_data(position_holder_qid, session)
            if holder_data:
                holder_label = holder_data.get("labels", {}).get("en", {}).get("value")
                if holder_label:
                    self.metrics.count(f"resolution_{method}_position", 1)
                    return EntityMapping(
                        raw_name=raw_name,
                        canonical_name=holder_label,
                        wikidata_id=position_holder_qid,
                        entity_type=self._classify_entity(holder_data),
                        resolution_method=f"{method}_position",
                    )

        # Use the label directly
        entity_type = self._classify_entity(entity_data)
        self.metrics.count(f"resolution_{method}", 1)

        return EntityMapping(
            raw_name=raw_name,
            canonical_name=label,
            wikidata_id=qid,
            entity_type=entity_type,
            resolution_method=method,
        )

    async def resolve_all(
        self,
        events: List[Event],
    ) -> tuple[List[Event], Dict[str, EntityMapping]]:
        """
        Resolve all stakeholders in events.

        Returns:
            - Events with normalized stakeholders
            - Mapping from raw names to canonical names
        """
        self.metrics.start()

        # Collect unique stakeholders
        unique_stakeholders: Set[str] = set()
        for event in events:
            unique_stakeholders.update(event.stakeholders)

        self.logger.info(f"Resolving {len(unique_stakeholders)} unique stakeholders")

        # Resolve all
        mappings: Dict[str, EntityMapping] = {}
        async with aiohttp.ClientSession() as session:
            for i, name in enumerate(unique_stakeholders):
                if i % 100 == 0:
                    self.logger.info(f"Resolved {i}/{len(unique_stakeholders)}")

                mapping = await self.resolve(name, session)
                mappings[name] = mapping

        # Apply mappings to events
        for event in events:
            resolved = []
            seen = set()
            for name in event.stakeholders:
                canonical = mappings[name].canonical_name
                if canonical not in seen:
                    resolved.append(canonical)
                    seen.add(canonical)
            event.stakeholders = resolved

        # Save cache
        self.cache.save()
        self.logger.info(f"Cache size: {len(self.cache)}")

        self.metrics.end()
        return events, mappings

    def get_metrics_report(self) -> dict:
        """Get coreference metrics report."""
        return self.metrics.get_report()
