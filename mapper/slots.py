"""Slot extraction — fill the named slots a shape's Cypher template needs.

Each shape in `shapes.CANONICAL_CYPHER` carries `$param` placeholders.
Your `extract_slots(question, shape)` returns a dict whose keys are the
parameter names the template expects, e.g.:

  ShapeId.Q1 → {"ingredient": "ginger"}
  ShapeId.Q5 → {"cuisine": "Sichuan", "ingredient": "ginger"}
  ShapeId.Q9 → {"cuisine": "Italian"}
  ShapeId.Q10 → {"max_minutes": 30}
  ShapeId.Q14 → {"ingredient": "ginger", "exclude_ingredient": "garlic"}

See `data/eval_questions.jsonl` for the gold (question_text, shape, slots)
triples used by the autograder.
"""
import re
import spacy
from .shapes import ShapeId

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

_CUISINE_VOCAB: dict[str, list[str]] = {
    "Italian":       ["italian"],
    "French":        ["french"],
    "Chinese":       ["chinese"],
    "Japanese":      ["japanese"],
    "Indian":        ["indian"],
    "Mexican":       ["mexican"],
    "Thai":          ["thai"],
    "Sichuan":       ["sichuan"],
    "Cantonese":     ["cantonese"],
    "Hunan":         ["hunan"],
    "Tuscan":        ["tuscan"],
    "Sicilian":      ["sicilian"],
    "Asian":         ["asian"],
    "European":      ["european"],
    "World":         ["world"],
    "NorthAmerican": ["northamerican", "north american"],
}

_INGREDIENT_VOCAB: dict[str, list[str]] = {
    "ginger":      ["ginger"],
    "garlic":      ["garlic"],
    "basil":       ["basil"],
    "tomato":      ["tomato", "tomatoes"],
    "olive oil":   ["olive oil"],
    "peppercorn":  ["peppercorn", "peppercorns"],
    "soy sauce":   ["soy sauce"],
    "rice":        ["rice"],
    "noodles":     ["noodles", "noodle"],
    "tofu":        ["tofu"],
    "chicken":     ["chicken"],
    "beef":        ["beef"],
    "pork":        ["pork"],
    "fish":        ["fish"],
    "shrimp":      ["shrimp"],
    "onion":       ["onion", "onions"],
    "pepper":      ["pepper"],
    "salt":        ["salt"],
    "sugar":       ["sugar"],
    "vinegar":     ["vinegar"],
    "sesame oil":  ["sesame oil"],
    "chili":       ["chili", "chilli"],
    "cumin":       ["cumin"],
    "turmeric":    ["turmeric"],
    "coriander":   ["coriander"],
    "lemon":       ["lemon"],
    "lime":        ["lime"],
    "butter":      ["butter"],
    "cream":       ["cream"],
    "flour":       ["flour"],
    "egg":         ["egg", "eggs"],
    "cheese":      ["cheese"],
    "mushroom":    ["mushroom", "mushrooms"],
    "spinach":     ["spinach"],
    "carrot":      ["carrot", "carrots"],
    "potato":      ["potato", "potatoes"],
    "eggplant":    ["eggplant"],
    "zucchini":    ["zucchini"],
    "artichoke":   ["artichoke"],
    "fennel":      ["fennel"],
}

_TECHNIQUE_VOCAB: dict[str, list[str]] = {
    "wok":      ["wok"],
    "grilling": ["grilling", "grill"],
    "steaming": ["steaming", "steam"],
    "frying":   ["frying", "fry"],
    "baking":   ["baking", "bake"],
    "braising": ["braising", "braise"],
}


def extract_slots(question: str, shape: ShapeId) -> dict:
    """Extract slot values for the given shape from the question text.

    Suggested approach:
      - spaCy NER for PERSON entities (q2, q8 author slot).
      - A short hand-authored vocabulary list of the cuisines and
        ingredients in the recipe KG — string-match the question against
        it case-insensitively. The lists are small (16 cuisines, 40
        ingredients) so a literal-match approach is fine.
      - For q10: a regex like `under (\\d+)\\s*minutes` to pull the
        integer threshold.
      - For q14: split the question on "but not" / "without" to get the
        positive and negative ingredient slots.

    Return a dict whose keys EXACTLY match the `$param` names in
    shapes.CANONICAL_CYPHER[shape]. Returning a slot dict missing a
    required parameter will surface as a Neo4j ParameterMissing error
    at query time — that is fail-loud and desired.

    Values must be the canonical form the KG uses (e.g., 'Italian' not
    'italian'; 'ginger' not 'Ginger'). Match against the schema vocabulary
    rather than echoing the surface form of the question.
    """
    # (slot extraction):
    # 1. For the given shape, list the parameter names you need to fill.
    # 2. For each parameter, use a vocabulary list or a regex over the
    #    question text to extract the value in canonical form.
    # 3. Return the dict.

    q_lower = question.lower()

    if shape == ShapeId.Q1:
        return {
            "ingredient": _match_ingredient(q_lower),
        }

    elif shape == ShapeId.Q2:
        return {
            "author": _extract_author(question),
        }

    elif shape == ShapeId.Q3:
        return {
            "cuisine": _match_cuisine(q_lower),
        }

    elif shape == ShapeId.Q4:
        return {
            "cuisine": _match_cuisine(q_lower),
        }

    elif shape == ShapeId.Q5:
        return {
            "cuisine":    _match_cuisine(q_lower),
            "ingredient": _match_ingredient(q_lower),
        }

    elif shape == ShapeId.Q6:
        return {
            "cuisine":    _match_cuisine(q_lower),
            "ingredient": _match_ingredient(q_lower),
        }

    elif shape == ShapeId.Q7:
        return {
            "technique": _match_technique(q_lower),
        }

    elif shape == ShapeId.Q8:
        return {
            "author":     _extract_author(question),
            "ingredient": _match_ingredient(q_lower),
        }

    elif shape == ShapeId.Q9:
        return {
            "cuisine": _match_cuisine(q_lower),
        }

    elif shape == ShapeId.Q10:
        return {
            "max_minutes": _extract_minutes(q_lower),
        }

    elif shape == ShapeId.Q11:
        return {
            "cuisine": _match_cuisine(q_lower),
        }

    elif shape == ShapeId.Q12:
        return {
            "cuisine": _match_cuisine(q_lower),
        }

    elif shape == ShapeId.Q13:
        return {
            "ingredient": _match_ingredient(q_lower),
        }

    elif shape == ShapeId.Q14:
        return _extract_negation_slots(q_lower)

    elif shape == ShapeId.Q15:
        return {
            "technique": _match_technique(q_lower),
        }

    return {}

def _match_cuisine(q_lower: str) -> str:
    """Return the canonical cuisine name found in the lowercased question.

    Scans _CUISINE_VOCAB variations; returns the canonical key on the first
    match (e.g., 'italian' in question → returns 'Italian').
    Returns an empty string if nothing matches.
    """
    for canonical, variations in _CUISINE_VOCAB.items():
        for v in variations:
            if v in q_lower:
                return canonical
    return ""


def _match_ingredient(q_lower: str) -> str:
    """Return the canonical ingredient name found in the lowercased question.

    Handles plurals via the variations list (e.g., 'tomatoes' → 'tomato').
    Returns an empty string if nothing matches.
    """
    for canonical, variations in _INGREDIENT_VOCAB.items():
        for v in variations:
            if v in q_lower:
                return canonical
    return ""


def _match_technique(q_lower: str) -> str:
    """Return the canonical technique name found in the lowercased question.

    Returns an empty string if nothing matches.
    """
    for canonical, variations in _TECHNIQUE_VOCAB.items():
        for v in variations:
            if v in q_lower:
                return canonical
    return ""


def _extract_author(question: str) -> str:
    """Return the person name found in the question using spaCy NER.

    Uses the original (non-lowercased) question because spaCy's NER model
    relies on capitalisation to identify PERSON entities correctly.

    Falls back to a regex pattern (two or more capitalised words) if NER
    does not find a PERSON entity.
    Returns an empty string if neither strategy finds a name.
    """
    # Primary: spaCy named-entity recognition
    doc = _nlp(question)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    # Fallback: two or more Title-Cased words joined by space or hyphen
    match = re.search(
        r'\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+)\b',
        question
    )
    if match:
        return match.group(1)

    return ""


def _extract_minutes(q_lower: str) -> int:
    """Extract the integer threshold from 'under N minutes'.

    Example: 'under 30 minutes' → 30
    Returns 30 as a safe default if the pattern is not found.
    """
    match = re.search(r"under\s+(\d+)\s*minutes", q_lower)
    if match:
        return int(match.group(1))
    return 30


def _extract_negation_slots(q_lower: str) -> dict:
    """Split the question on 'but not' or 'without' and extract both slots.

    Example:
      'find recipes that use ginger but not garlic'
      → {"ingredient": "ginger", "exclude_ingredient": "garlic"}

    Strategy:
      1. Split on the negation marker to get a positive and negative part.
      2. Search each part independently using _match_ingredient.
      3. Return both under the exact param names the Q14 template expects.
    """
    if "but not" in q_lower:
        parts = q_lower.split("but not", 1)
    elif "without" in q_lower:
        parts = q_lower.split("without", 1)
    else:
        parts = [q_lower, ""]

    positive_part = parts[0]
    negative_part = parts[1] if len(parts) > 1 else ""

    # Search the positive half for the ingredient to include
    pos = ""
    for canonical, variations in _INGREDIENT_VOCAB.items():
        for v in variations:
            if v in positive_part:
                pos = canonical
                break
        if pos:
            break

    # Search the negative half for the ingredient to exclude
    neg = ""
    for canonical, variations in _INGREDIENT_VOCAB.items():
        for v in variations:
            if v in negative_part:
                neg = canonical
                break
        if neg:
            break

    return {
        "ingredient":         pos,
        "exclude_ingredient": neg,
    }





