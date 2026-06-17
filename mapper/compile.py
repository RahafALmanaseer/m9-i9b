"""Compile a (shape, slots) pair to (cypher_string, params_dict).

The cypher MUST come from `shapes.CANONICAL_CYPHER[shape]` — do not
construct a Cypher string from the slot values via string formatting.
This is the load-bearing safety beat from the Reading:

  - The query body is a static template, identical across all calls.
  - All slot values flow through the params dict only.
  - The Neo4j driver binds params at query time, never at string-build
    time, so an attacker-controlled slot value cannot inject Cypher.

The autograder's AST check inspects this file's source code: it must
NOT contain an f-string that interpolates a slot value into a string
that is passed to `session.run` as Cypher. Use the template-from-dict
pattern only.
"""

from .shapes import CANONICAL_CYPHER, ShapeId


def compile_to_cypher(shape: ShapeId, slots: dict) -> tuple[str, dict]:
    """Return (cypher_string, params_dict).

    Required behaviour:
      - Look up the static template string from CANONICAL_CYPHER[shape].
      - Return that string unchanged (it already contains the $param
        placeholders).
      - Return the slots dict unchanged (or a shallow copy) as the
        params bound by the driver.
      - Do NOT format, f-string, or .format() the cypher string with
        the slot values. The driver does the binding.

    Example:
      compile_to_cypher(ShapeId.Q1, {"ingredient": "ginger"})
      -> ("MATCH (r:Recipe)-[:USES_INGREDIENT]->(:Ingredient {name: $ingredient}) ...",
          {"ingredient": "ginger"})
    """
    # (compile):
    # 1. Look up the template in CANONICAL_CYPHER[shape]. Raise KeyError
    #    if missing — that's an upstream bug in detect_shape, not a
    #    compile-time mistake.
    # 2. Return (template, slots). Do not string-format the template
    #    with the slot values; the driver binds $params at query time.
    
    template = CANONICAL_CYPHER[shape]
    return template, dict(slots)
