"""Family nickname/alias mappings for the McCulloughs.

This maps nicknames and informal names to real names so the AI
can properly identify people in photos.
"""

# Nickname -> Real name mappings
# Format: "nickname": "Real Name"
FAMILY_ALIASES: dict[str, str] = {
    # Mum - Isobel McCullough
    "wee mamie": "Isobel McCullough",
    "mamie": "Isobel McCullough",

    # Dad - Edmund McCullough
    "the oul man": "Edmund McCullough",
    "the oul fella": "Edmund McCullough",
    "oul man": "Edmund McCullough",
    "oul fella": "Edmund McCullough",

    # Sister - Christina McCullough
    "the leech": "Christina McCullough",
    "leech": "Christina McCullough",

    # Brother - John McCullough
    "asshole": "John McCullough",
    "the bro": "John McCullough",
}

# Reverse mapping for quick lookup
REAL_NAMES: dict[str, list[str]] = {}
for nickname, real_name in FAMILY_ALIASES.items():
    if real_name not in REAL_NAMES:
        REAL_NAMES[real_name] = []
    REAL_NAMES[real_name].append(nickname)


def resolve_nickname(text: str) -> str:
    """Replace nicknames in text with real names.

    Args:
        text: Text that may contain nicknames

    Returns:
        Text with nicknames replaced by real names
    """
    if not text:
        return text

    result = text
    # Sort by length (longest first) to avoid partial replacements
    for nickname in sorted(FAMILY_ALIASES.keys(), key=len, reverse=True):
        # Case-insensitive replacement
        import re
        pattern = re.compile(re.escape(nickname), re.IGNORECASE)
        result = pattern.sub(FAMILY_ALIASES[nickname], result)

    return result


def get_family_context() -> str:
    """Get a context string about family members for the AI prompt.

    Returns:
        String describing family member nicknames
    """
    lines = ["Family member nicknames to real names:"]
    for real_name, nicknames in REAL_NAMES.items():
        nicks = ", ".join(f'"{n}"' for n in nicknames)
        lines.append(f"  - {nicks} = {real_name}")

    return "\n".join(lines)


def extract_family_names(text: str) -> list[str]:
    """Extract real family member names from text containing nicknames.

    Args:
        text: Text that may contain nicknames

    Returns:
        List of real names found
    """
    if not text:
        return []

    found_names = set()
    text_lower = text.lower()

    for nickname, real_name in FAMILY_ALIASES.items():
        if nickname in text_lower:
            found_names.add(real_name)

    return list(found_names)
