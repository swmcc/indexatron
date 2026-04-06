"""Image analysis using LLaVA."""

import json
import re
from pathlib import Path

import ollama
from rich.console import Console

from .config import get_settings
from .models import EraEstimate, LocationInfo, PersonInfo, PhotoAnalysis

console = Console()

# Category hierarchy - if we see any key, add its values as parent categories
CATEGORY_HIERARCHY = {
    # Pets
    "puppy": ["pet", "dog", "animal"],
    "dog": ["pet", "animal"],
    "kitten": ["pet", "cat", "animal"],
    "cat": ["pet", "animal"],
    "pet": ["animal"],
    # Events
    "christmas": ["holiday", "celebration"],
    "birthday": ["celebration", "party"],
    "wedding": ["celebration", "event"],
    "graduation": ["celebration", "event"],
    "easter": ["holiday", "celebration"],
    "thanksgiving": ["holiday", "celebration"],
    "halloween": ["holiday", "celebration"],
    # Family
    "baby": ["family", "child"],
    "toddler": ["family", "child"],
    "child": ["family"],
    "kids": ["family", "children"],
}

# Terms to filter out from categories and descriptions (inappropriate for family photos)
BLOCKED_TERMS = {
    "grooming", "beauty", "beautiful", "sexy", "attractive", "hot",
    "gorgeous", "stunning", "pretty girl", "handsome boy",
}

# Maximum number of categories to keep (prevents runaway repetition)
MAX_CATEGORIES = 20

BASE_ANALYSIS_PROMPT = """Analyze this family photo and provide a detailed JSON response:

{
  "description": "A detailed description of what's happening in the photo",
  "location": {
    "setting": "general setting like beach, park, home, restaurant",
    "type": "indoor or outdoor",
    "specific": "specific location if identifiable, or null"
  },
  "people": [
    {
      "name": "person's name if known from context, or null",
      "description": "description of the person",
      "estimated_age": "age or age range like '8 years old' or '30s'",
      "position": "where in the frame: left, center, right, background"
    }
  ],
  "categories": ["hierarchical", "tags", "from specific to general"],
  "era": {
    "decade": "estimated decade like 1990s or 2000s",
    "confidence": "low, medium, or high",
    "reasoning": "why you think this era"
  },
  "mood": "the emotional tone of the photo",
  "colors": ["notable", "colors"],
  "objects": ["visible", "objects"]
}

Focus on:
- Family relationships if apparent
- Activities happening
- Special occasions (birthdays, holidays, christmas, easter, etc.)
- Photo quality and style for era estimation
- Clothing and objects for context

For categories, include BOTH specific and general tags. Examples:
- A puppy photo: ["puppy", "dog", "pet", "animal"]
- A Christmas photo: ["christmas", "holiday", "celebration", "family"]
- A baby photo: ["baby", "infant", "child", "family"]

Respond with ONLY valid JSON, no other text."""


def build_analysis_prompt(metadata: dict | None = None) -> str:
    """Build the analysis prompt, optionally with metadata context.

    Args:
        metadata: Optional dict with title, caption, date_taken

    Returns:
        Complete prompt string
    """
    if not metadata:
        return BASE_ANALYSIS_PROMPT

    # Build context section from available metadata
    context_parts = []

    if metadata.get("title"):
        context_parts.append(f"Title: {metadata['title']}")

    if metadata.get("caption"):
        context_parts.append(f"Caption: {metadata['caption']}")

    if metadata.get("date_taken"):
        context_parts.append(f"Date taken: {metadata['date_taken']}")

    if not context_parts:
        return BASE_ANALYSIS_PROMPT

    context_section = "\n".join(context_parts)

    # Insert context before the JSON schema
    context_prompt = f"""The following context is known about this photo:
{context_section}

Use this context to:
- Identify people by name if mentioned (e.g., "John's wedding" means the groom may be John)
- Confirm or refine the era estimate based on the date
- Extract location hints from the title/caption
- Add relevant categories based on mentioned events or people

{BASE_ANALYSIS_PROMPT}"""

    return context_prompt


class PhotoAnalyzer:
    """Analyzes photos using Llama 3.2 Vision model."""

    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.vision_model

    def _enrich_categories(self, categories: list) -> list[str]:
        """Add parent categories based on hierarchy mapping.

        Also filters blocked terms and limits total count.
        """
        # Flatten any nested lists and convert to strings
        flat = []
        for cat in categories:
            if isinstance(cat, list):
                flat.extend(str(c) for c in cat)
            else:
                flat.append(str(cat))

        # Deduplicate and lowercase
        enriched = set(c.lower().strip() for c in flat if c)

        # Filter out blocked terms
        enriched = {c for c in enriched if c not in BLOCKED_TERMS}

        # Add parent categories from hierarchy
        for cat in list(enriched):
            if cat in CATEGORY_HIERARCHY:
                enriched.update(CATEGORY_HIERARCHY[cat])

        # Limit total count to prevent runaway repetition
        result = sorted(enriched)[:MAX_CATEGORIES]

        return result

    def analyze(self, image_path: Path, metadata: dict | None = None) -> PhotoAnalysis:
        """Analyze a single image and return structured results.

        Args:
            image_path: Path to the image file
            metadata: Optional dict with title, caption, date_taken for context

        Returns:
            PhotoAnalysis with structured results
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        console.print(f"\n[bold blue]🔍 Analyzing:[/bold blue] {image_path.name}")

        if metadata and any(metadata.values()):
            context_parts = []
            if metadata.get("title"):
                context_parts.append(f"Title: {metadata['title']}")
            if metadata.get("caption"):
                context_parts.append(f"Caption: {metadata['caption']}")
            if metadata.get("date_taken"):
                context_parts.append(f"Date: {metadata['date_taken']}")
            console.print(f"[dim]Context: {', '.join(context_parts)}[/dim]")

        # Build prompt with optional metadata context
        prompt = build_analysis_prompt(metadata)

        # Call vision model with the image
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
        )

        raw_response = response.message.content
        console.print(f"[dim]Raw response length: {len(raw_response)} chars[/dim]")

        # Parse the JSON response
        analysis_data = self._parse_response(raw_response)

        # Build the structured result
        return self._build_analysis(image_path.name, analysis_data, raw_response, self.model)

    def _sanitize_text(self, text: str) -> str:
        """Remove inappropriate phrases from text."""
        if not text:
            return text

        result = text
        # Remove phrases like "most beautiful girl" etc.
        inappropriate_phrases = [
            r"most beautiful\s+\w+",
            r"beautiful\s+girl",
            r"beautiful\s+boy",
            r"pretty\s+girl",
            r"handsome\s+boy",
            r"gorgeous\s+\w+",
            r"stunning\s+\w+",
        ]
        for pattern in inappropriate_phrases:
            result = re.sub(pattern, "child", result, flags=re.IGNORECASE)

        return result

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        # Detect repetition loop (model gone haywire)
        if response.count('"grooming"') > 3 or response.count('"beauty"') > 3:
            console.print("[yellow]Warning: Detected repetition loop in model output[/yellow]")
            # Try to extract just the first valid JSON object before the loop
            first_categories = response.find('"categories"')
            if first_categories > 0:
                # Find the closing bracket of the first categories array
                bracket_start = response.find("[", first_categories)
                if bracket_start > 0:
                    depth = 0
                    for i, c in enumerate(response[bracket_start:], bracket_start):
                        if c == "[":
                            depth += 1
                        elif c == "]":
                            depth -= 1
                            if depth == 0:
                                # Truncate after first categories array
                                response = response[:i + 1] + "}"
                                break

        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                console.print(f"[yellow]JSON parse error in code block: {e}[/yellow]")
                # Try to fix common issues - truncate at the error
                try:
                    # Find the last complete object
                    fixed = self._fix_json(json_str)
                    return json.loads(fixed)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try to find JSON object in the response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty dict if parsing fails
        console.print("[yellow]Warning: Could not parse JSON response[/yellow]")
        return {"description": response, "parse_error": True}

    def _fix_json(self, json_str: str) -> str:
        """Attempt to fix malformed JSON."""
        # Count braces and brackets to find where to truncate
        brace_count = 0
        bracket_count = 0
        last_valid = 0

        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_valid = i + 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1

        # If we have unbalanced braces, try to close them
        if brace_count > 0:
            json_str = json_str[:last_valid] if last_valid > 0 else json_str
            json_str = json_str.rstrip(',\n ') + '}' * brace_count

        return json_str

    def _build_analysis(
        self, filename: str, data: dict, raw_response: str, model: str
    ) -> PhotoAnalysis:
        """Build a PhotoAnalysis from parsed data."""
        # Parse location
        location = None
        if loc_data := data.get("location"):
            if isinstance(loc_data, dict):
                location = LocationInfo(
                    setting=loc_data.get("setting", "unknown"),
                    type=loc_data.get("type", "unknown"),
                    specific=loc_data.get("specific"),
                )

        # Parse people (sanitize descriptions)
        people = []
        for person_data in data.get("people", []):
            if isinstance(person_data, dict):
                description = self._sanitize_text(person_data.get("description", "person"))
                people.append(
                    PersonInfo(
                        name=person_data.get("name"),
                        description=description,
                        estimated_age=person_data.get("estimated_age"),
                        position=person_data.get("position"),
                    )
                )

        # Parse era
        era = None
        if era_data := data.get("era"):
            if isinstance(era_data, dict):
                era = EraEstimate(
                    decade=era_data.get("decade", "unknown"),
                    confidence=era_data.get("confidence", "low"),
                    reasoning=era_data.get("reasoning"),
                )

        # Handle objects - might be list of strings, list of dicts, or dict
        raw_objects = data.get("objects", [])
        objects = []
        if isinstance(raw_objects, dict):
            # Flatten dict values into list
            for v in raw_objects.values():
                if isinstance(v, list):
                    objects.extend([str(x) for x in v])
                else:
                    objects.append(str(v))
        elif isinstance(raw_objects, list):
            for obj in raw_objects:
                if isinstance(obj, dict):
                    # Extract description or first string value
                    objects.append(obj.get("description", obj.get("name", str(obj))))
                else:
                    objects.append(str(obj))

        # Handle colors - might be list or dict
        colors = data.get("colors", [])
        if isinstance(colors, dict):
            colors = list(colors.values()) if colors else []

        # Handle categories - might be list or dict
        categories = data.get("categories", [])
        if isinstance(categories, dict):
            categories = list(categories.values()) if categories else []

        # Enrich categories with parent categories from hierarchy
        categories = self._enrich_categories(categories)

        # Sanitize the main description
        description = self._sanitize_text(data.get("description", "No description available"))

        return PhotoAnalysis(
            filename=filename,
            model_used=model,
            description=description,
            location=location,
            people=people,
            categories=categories if isinstance(categories, list) else [],
            era=era,
            mood=self._sanitize_text(data.get("mood")),
            colors=colors if isinstance(colors, list) else [],
            objects=objects if isinstance(objects, list) else [],
            raw_response=raw_response,
        )
