"""Intent classifier — map NL question to a canonical ShapeId.

This file is your responsibility. Read the 15 supported shapes in
`shapes.ShapeId` and `shapes.CANONICAL_CYPHER`, then implement
`detect_shape` so that each of the 15 canonical eval questions in
`data/eval_questions.jsonl` is classified to the gold shape, and
adversarial / off-template questions return None.

The deterministic mapper is the production-discipline arm of M9B; a
classifier that returns the wrong shape on a supported question is a
real bug, and a classifier that returns a confident answer on an
off-template question is the silent-failure mode the Reading warns
against. Prefer None over a false positive.
"""

import re
from .shapes import ShapeId

_CUISINE_NAMES = {
    "italian", "french", "chinese", "japanese", "indian",
    "mexican", "thai", "sichuan", "cantonese", "hunan",
    "tuscan", "sicilian", "asian", "european", "world",
    "northamerican", "north american",
}

_HIER_CUISINES = {"asian", "european", "world", "chinese"}

_INGREDIENT_NAMES = {
    "ginger", "garlic", "basil", "tomato", "tomatoes",
    "olive oil", "peppercorn", "peppercorns", "soy sauce",
    "rice", "noodles", "noodle", "tofu", "chicken", "beef",
    "pork", "fish", "shrimp", "onion", "onions", "pepper",
    "salt", "sugar", "vinegar", "sesame oil", "chili", "chilli",
    "cumin", "turmeric", "coriander", "lemon", "lime", "butter",
    "cream", "flour", "egg", "eggs", "cheese", "mushroom",
    "mushrooms", "spinach", "carrot", "carrots", "potato",
    "potatoes", "eggplant", "zucchini", "artichoke", "fennel",
}

_TECHNIQUE_NAMES = {
    "wok", "grilling", "grill", "steaming", "steam",
    "frying", "fry", "baking", "bake", "braising", "braise",
}


def detect_shape(question: str) -> ShapeId | None:
    """Classify the question into one of the 15 ShapeId values, or None."""
    original = question.strip()
    q = original.lower()

    # q14: negation
    if ("but not" in q or "without" in q) and _has_ingredient(q):
        return ShapeId.Q14

    # q13: ingredient hierarchy
    if "or any subtype" in q or "or any kind" in q:
        return ShapeId.Q13

    # q15: optional technique
    if "optionally tagged" in q:
        return ShapeId.Q15

    # q12: authors of cuisine recipes
    if "authors of" in q and _has_cuisine(q):
        return ShapeId.Q12

    # q11: ingredients used in cuisine recipes
    if "ingredients used in" in q and _has_cuisine(q):
        return ShapeId.Q11

    # q10: prep time
    if re.search(r"\bunder\s+\d+\s*minutes\b", q):
        return ShapeId.Q10

    # q9: popularity ranking
    if "ranked by popularity" in q or "most popular" in q:
        return ShapeId.Q9

    # q7: technique requirement
    if re.search(r"\brequires?\b", q) and _has_technique(q):
        return ShapeId.Q7

    # q8: author + ingredient
    if _has_author_cue(original) and _has_ingredient_cue(q):
        return ShapeId.Q8

    # q6: hierarchical cuisine + ingredient
    if _has_hier_cuisine(q) and _has_ingredient_cue(q):
        return ShapeId.Q6

    # q5: direct cuisine + ingredient
    if _has_cuisine(q) and _has_ingredient_cue(q):
        return ShapeId.Q5

    # q2: author only
    if _has_author_cue(original):
        return ShapeId.Q2

    # q4: hierarchical cuisine only
    if _has_hier_cuisine(q):
        return ShapeId.Q4

    # q3: direct cuisine only
    if _has_cuisine(q):
        return ShapeId.Q3

    # q1: ingredient only
    if _has_ingredient_cue(q):
        return ShapeId.Q1

    return None


def _has_cuisine(q: str) -> bool:
    return any(c in q for c in _CUISINE_NAMES)


def _has_hier_cuisine(q: str) -> bool:
    return any(c in q for c in _HIER_CUISINES)


def _has_ingredient(q: str) -> bool:
    return any(i in q for i in _INGREDIENT_NAMES)


def _has_ingredient_cue(q: str) -> bool:
    has_cue = bool(re.search(r"\b(use|uses|using|with)\b", q))
    return has_cue and _has_ingredient(q)


def _has_technique(q: str) -> bool:
    return any(t in q for t in _TECHNIQUE_NAMES)


def _has_author_cue(question_original: str) -> bool:
    """Detect author questions conservatively.

    Accept only:
      - 'by author Maria Rossi'
      - 'by Maria Rossi'
    """
    q = question_original.lower()

    if "by author" in q:
        return True

    return bool(
        re.search(
            r"\bby\s+[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+\b",
            question_original
        )
    )