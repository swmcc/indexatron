"""Image analysis using LLaVA."""

import json
import re
from pathlib import Path

import ollama
from rich.console import Console

from .models import PhotoAnalysis, LocationInfo, PersonInfo, EraEstimate

console = Console()

ANALYSIS_PROMPT = """Analyze this family photo and provide a detailed JSON response with the following structure:

{
  "description": "A detailed description of what's happening in the photo",
  "location": {
    "setting": "general setting like beach, park, home, restaurant",
    "type": "indoor or outdoor",
    "specific": "specific location if identifiable, or null"
  },
  "people": [
    {
      "description": "description of the person",
      "estimated_age": "age or age range like '8 years old' or '30s'",
      "position": "where in the frame: left, center, right, background"
    }
  ],
  "categories": ["list", "of", "relevant", "tags"],
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
- Special occasions (birthdays, holidays, etc.)
- Photo quality and style for era estimation
- Clothing and objects for context

Respond with ONLY valid JSON, no other text."""


class PhotoAnalyzer:
    """Analyzes photos using LLaVA vision model."""

    MODEL = "llava:7b"

    def analyze(self, image_path: Path) -> PhotoAnalysis:
        """Analyze a single image and return structured results."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        console.print(f"\n[bold blue]🔍 Analyzing:[/bold blue] {image_path.name}")

        # Call LLaVA with the image
        response = ollama.chat(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": ANALYSIS_PROMPT,
                    "images": [str(image_path)],
                }
            ],
        )

        raw_response = response.message.content
        console.print(f"[dim]Raw response length: {len(raw_response)} chars[/dim]")

        # Parse the JSON response
        analysis_data = self._parse_response(raw_response)

        # Build the structured result
        return self._build_analysis(image_path.name, analysis_data, raw_response)

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
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
                except:
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
        self, filename: str, data: dict, raw_response: str
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

        # Parse people
        people = []
        for person_data in data.get("people", []):
            if isinstance(person_data, dict):
                people.append(
                    PersonInfo(
                        description=person_data.get("description", "person"),
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

        return PhotoAnalysis(
            filename=filename,
            description=data.get("description", "No description available"),
            location=location,
            people=people,
            categories=categories if isinstance(categories, list) else [],
            era=era,
            mood=data.get("mood"),
            colors=colors if isinstance(colors, list) else [],
            objects=objects if isinstance(objects, list) else [],
            raw_response=raw_response,
        )
