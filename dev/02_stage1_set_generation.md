# Stage 1: SET Generation (Paper Section 3.1)

## Overview

```
Article A → LLM → Multiple Events {(S_e1, e1, t_e1), (S_e2, e2, t_e2), ...}
                         ↓
              Wikidata Coreference Resolution
                         ↓
              Normalized SET Triplets
```

## 1.1 Event & Time Extraction

### Paper Description (Section 3.1)

> "Every article A may contain multiple events: A → e1, e2, e3, ..."
> "After extracting the event and its estimated date (t), we call the same LLM to extract relevant stakeholders"

### Paper Prompt (Appendix A)

**Event and Time Generation:**
```
You are a professional journalist that is tasked to generate date-based
event summary of a given article. A single list contains an article and
its published time. You should generate a dictionary of the most
relevant events of an article, where each key in the dictionary is a
string of the expected event start date in terms of Year-Month-Day
(e.g. 2011-12-25) and the value will be a summary of the relevant events
on that day. Summarize only the most important events found in the
article, as succinctly as possible. If you are uncertain of the date of
an event, feel free to use the published date. You should only output
the dictionary in your answer. Generate a dictionary of events of the
following article: {str(article_x)}.
```

**Output Format:**
```python
{
    "2011-01-25": "Protesters gather in Cairo...",
    "2011-01-26": "Police clash with demonstrators..."
}
```

### Pseudocode

```
FUNCTION extract_events(article) -> Dict[date, event_summary]:
    prompt = EVENT_TIME_PROMPT.format(article=article)

    response = llm.generate(
        prompt = prompt,
        temperature = 0  # deterministic
    )

    events_dict = parse_dict(response)  # {date: summary}
    RETURN events_dict
```

## 1.2 Stakeholder Extraction

### Paper Description (Section 3.1)

> "we call the same LLM to extract relevant stakeholders mentioned within the article with a maximum of 5 stakeholders per event"
> "The term stakeholder used here strictly refers to an entity which is either a person or an organization"

### Paper Prompt (Appendix A)

**Stakeholder Generation:**
```
You are a professional journalist that is tasked to generate the most
relevant stakeholders relevant to a given event summary of an article.
A single list contains an article and its published time. You should
generate a singular list containing not more than five relevant
stakeholders related to only the stipulated event mentioned. These
stakeholders should not be general, and must be identifiable named
entities that can be matched to a person, organization or role when read
on its own. Every single stakeholder generated should also ideally exist
in exact wording as mentioned within the original article. You should
only output the list of stakeholders in your answer, and all
stakeholders should be enclosed in string format. Generate a list of
related stakeholders of event: {dict[key_x]}.
Given article: {str(article_x)}.
```

**Output Format:**
```python
["Barack Obama", "Egypt", "Muslim Brotherhood", "Hosni Mubarak"]
```

### Pseudocode

```
FUNCTION extract_stakeholders(event_summary, article) -> List[str]:
    prompt = STAKEHOLDER_PROMPT.format(
        event = event_summary,
        article = article
    )

    response = llm.generate(prompt)

    stakeholders = parse_list(response)
    stakeholders = stakeholders[:5]  # max 5

    RETURN stakeholders
```

## 1.3 Coreference Resolution (Appendix B)

### Paper Description

> "Coreference Resolution will be done for all of the extracted stakeholders due to difference in naming such as utilizing a title or a position than a name (E.g. President of the United States) or differences in naming (E.g. POTUS v.s. President of America)"

### Paper Algorithm (Algorithm 1 in Appendix B)

```
FUNCTION build_knowledge_graph(all_stakeholders S) -> Dict[raw_name, canonical_id]:
    d = {}

    FOR ς IN S:
        IF ς IN d:
            CONTINUE

        # Step 1: Search Wikidata label/alt-label
        O = wikidata_search(ς)

        IF O not exist:
            # Step 2: Remove title using NER
            ς' = remove_title_with_ner(ς)
            O = wikidata_search(ς')

            IF O not exist:
                # Step 3: Replace whitespace with &&
                ς'' = ς'.replace(" ", "&&")
                O = wikidata_search(ς'')

                IF O not exist:
                    # Step 4: Use request API for interface search
                    O = wikidata_interface_search(ς)

                    IF O not exist:
                        d[ς] = ς  # fallback to original
                        CONTINUE

        # Check "Position Held By" operator
        IF "Position Held By" IN O:
            P = O["Position Held By"]
            d[ς] = P
        ELSE:
            d[ς] = O

    RETURN d
```

### Key APIs (from Appendix B footnote)
- Main API: https://www.wikidata.org/wiki/Wikidata:REST_API
- NER module: spacy/en_core_web_trf
- Request module: Python requests library

### Pseudocode: Apply Coreference

```
FUNCTION resolve_coreference(events_with_stakeholders) -> events_normalized:
    # Collect all unique stakeholders
    all_stakeholders = set()
    FOR event IN events:
        all_stakeholders.update(event.stakeholders)

    # Build KG mapping
    kg_mapping = build_knowledge_graph(all_stakeholders)

    # Apply mapping
    FOR event IN events:
        normalized = []
        FOR ς IN event.stakeholders:
            canonical = kg_mapping[ς]
            normalized.append(canonical)
        event.stakeholders = dedupe(normalized)

    RETURN events
```

## 1.4 Complete Stage 1 Flow

```
FUNCTION stage1_set_generation(articles) -> List[SET]:
    all_sets = []

    FOR article IN articles:
        # Step 1: Extract events with dates
        events_dict = extract_events(article)

        # Step 2: Extract stakeholders for each event
        FOR date, summary IN events_dict.items():
            stakeholders = extract_stakeholders(summary, article)

            set_triplet = SET(
                stakeholders = stakeholders,
                event = summary,
                time = date
            )
            all_sets.append(set_triplet)

    # Step 3: Coreference resolution
    all_sets = resolve_coreference(all_sets)

    RETURN all_sets
```

## Output Data Structure

```
SET:
    stakeholders: List[str]  # normalized entity names, max 5
    event: str               # event summary (1-2 sentences)
    time: str                # YYYY-MM-DD
    source_article: str      # original article reference (optional)
```

## Implementation Notes

1. **LLM calls per article**: 1 (events) + N (stakeholders per event)
2. **Stakeholder constraint**: "person or organization" only
3. **Coreference**: Must handle titles, aliases, positions
4. **Fallback**: If Wikidata fails, use original name
