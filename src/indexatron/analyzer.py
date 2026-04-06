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

BASE_ANALYSIS_PROMPT = """Analyze this family photo. Return ONLY valid JSON:

{
  "description": "what is happening in the photo",
  "location": {"setting": "place type", "type": "indoor/outdoor"},
  "people": [{"name": "name or null", "description": "who", "estimated_age": "age"}],
  "categories": ["5-10 relevant tags only"],
  "era": {"decade": "1990s", "confidence": "high/medium/low"},
  "mood": "emotional tone",
  "colors": ["main colors"],
  "objects": ["visible objects"]
}

RULES: Max 10 categories. No repetition. JSON only."""


def _extract_names_from_text(text: str) -> list[str]:
    """Extract potential names from caption/title text."""
    import re

    names = []
    # Common patterns: "John and Mary", "John's birthday", "with Sarah"
    # Look for capitalized words that aren't common words
    common_words = {
        "the", "and", "with", "at", "in", "on", "for", "to", "of", "a", "an",
        "birthday", "wedding", "christmas", "easter", "holiday", "vacation",
        "party", "celebration", "photo", "picture", "day", "night", "morning",
        "home", "house", "beach", "park", "garden", "church", "school",
    }

    # Find capitalized words
    words = re.findall(r"\b([A-Z][a-z]+)\b", text)
    for word in words:
        if word.lower() not in common_words:
            names.append(word)

    return list(set(names))


def _extract_decade_from_date(date_str: str) -> str | None:
    """Extract decade from a date string."""
    import re

    # Try to find a year
    year_match = re.search(r"(19|20)\d{2}", str(date_str))
    if year_match:
        year = int(year_match.group())
        decade = (year // 10) * 10
        return f"{decade}s"
    return None


def build_analysis_prompt(metadata: dict | None = None) -> str:
    """Build the analysis prompt with explicit metadata instructions.

    Args:
        metadata: Optional dict with title, caption, date_taken, gallery_name

    Returns:
        Complete prompt string
    """
    if not metadata or not any(metadata.values()):
        return BASE_ANALYSIS_PROMPT

    # Build directive instructions based on metadata
    directives = []

    # Add gallery context first (provides overall theme)
    if metadata.get("gallery_name"):
        directives.append(f"This photo is from the album: \"{metadata['gallery_name']}\"")

    # Extract names from title/caption/gallery
    all_text = (
        f"{metadata.get('gallery_name', '')} "
        f"{metadata.get('title', '')} "
        f"{metadata.get('caption', '')}"
    )
    names = _extract_names_from_text(all_text)

    if names:
        names_str = ", ".join(names)
        directives.append(
            f"IMPORTANT: This photo includes {names_str}. "
            f"Use these names in the 'people' array where you can identify them."
        )

    # Extract decade from date
    if metadata.get("date_taken"):
        decade = _extract_decade_from_date(str(metadata["date_taken"]))
        if decade:
            directives.append(
                f"IMPORTANT: This photo is from {metadata['date_taken']} ({decade}). "
                f"Use this as the era decade with 'high' confidence."
            )

    # Add caption context
    if metadata.get("caption"):
        directives.append(f"Caption says: \"{metadata['caption']}\"")

    if metadata.get("title"):
        directives.append(f"Title: \"{metadata['title']}\"")

    if not directives:
        return BASE_ANALYSIS_PROMPT

    directive_text = "\n".join(directives)

    return f"""{directive_text}

{BASE_ANALYSIS_PROMPT}"""


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
