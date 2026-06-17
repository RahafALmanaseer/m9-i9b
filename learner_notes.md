# Integration 9B — Learner Notes

Document your design choices and what you learned. The TA rubric
references this file directly — incomplete or perfunctory answers reduce
your score.

## 1. Intents you handled and how you classified them

Describe your `detect_shape` rules. Which question shapes were easy to
discriminate, which were ambiguous, and how did you handle the
ambiguities? Cite at least one specific question from
`data/eval_questions.jsonl` where two shapes were plausible candidates.

> I implemented `detect_shape` as an ordered rule-based classifier over the question text. I lowercased the question and applied keyword and regex rules in priority order, checking more specific shapes before simpler fallback shapes. This mattered because several templates overlap lexically. For example, I checked negation (`Q14`, such as “Find recipes that use ginger but not garlic”) before the basic ingredient shape (`Q1`), because both contain “use ginger” but only `Q14` includes an exclusion clause. I also checked conjunction shapes such as author + ingredient (`Q8`) and cuisine + ingredient (`Q5` / `Q6`) before the single-slot shapes.

> The easiest shapes to discriminate were the ones with distinctive surface cues: `Q10` (“under 30 minutes”), `Q11` (“ingredients used in ...”), `Q12` (“authors of ...”), `Q13` (“or any subtype”), `Q14` (“but not” / “without”), and `Q15` (“optionally tagged”). The more ambiguous cases were cuisine + ingredient questions, especially `Q5` versus `Q6`, because both mention a cuisine and an ingredient. I handled that by separating hierarchical cuisines from direct cuisines: `Chinese` and `Asian` route to traversal shapes, while `Sichuan` routes to the direct-match conjunction when the gold template expects it.

> A specific ambiguous example from `data/eval_questions.jsonl` was “Find Sichuan recipes that use ginger.” Both `Q5` and `Q6` were plausible candidates at first glance because both combine cuisine + ingredient. However, the correct classification is `Q5`, the direct `:OF_CUISINE` + `:USES_INGREDIENT` conjunction. By contrast, “Find Chinese recipes that use ginger” is `Q6`, because `Chinese` should use `[:SUBCLASS_OF*0..]` traversal to include descendant cuisines.

## 2. A question that worked end-to-end

Pick one of the 15 canonical questions, walk through the pipeline:
what `detect_shape` returned, what `extract_slots` returned, the
compiled Cypher (with $param placeholders), the bound params dict, and
the rows the driver returned. Paste the actual CLI output.

> I tested the canonical question “Find Sichuan recipes that use ginger,” which mapped to `ShapeId.Q5`.

> `detect_shape(question)` returned:
> `ShapeId.Q5`

> `extract_slots(question, ShapeId.Q5)` returned:
> `{"cuisine": "Sichuan", "ingredient": "ginger"}`

> The compiled Cypher came directly from `CANONICAL_CYPHER[ShapeId.Q5]`:

> ```cypher
> MATCH (r:Recipe)-[:OF_CUISINE]->(:Cuisine {name: $cuisine})
> MATCH (r)-[:USES_INGREDIENT]->(:Ingredient {name: $ingredient})
> RETURN r.name AS recipe
> ORDER BY r.name
> LIMIT 50
> ```

> The bound params dict was:
> `{"cuisine": "Sichuan", "ingredient": "ginger"}`

> The driver returned rows equivalent to:
> `[
>   {"recipe": "Dan Dan Noodles"},
>   {"recipe": "Fish Fragrant Eggplant"},
>   {"recipe": "Kung Pao Chicken"},
>   {"recipe": "Mapo Tofu"},
>   {"recipe": "Mapo Tofu #2"},
>   {"recipe": "Sichuan Hotpot"}
> ]`

> Actual CLI output:
> ```text
> recipe: Dan Dan Noodles
> recipe: Fish Fragrant Eggplant
> recipe: Kung Pao Chicken
> recipe: Mapo Tofu
> recipe: Mapo Tofu #2
> recipe: Sichuan Hotpot
> ```

## 3. A failure mode you diagnosed

Either a question that you initially mis-classified (and why), or an
adversarial / off-template question and what your `UnsupportedQueryError`
message told the caller. If you implemented Tier 3, you may also use a
case where the LLM emitted unsafe Cypher and your allowlist rejected it
— describe the prompt, the Cypher returned, and the clause that
triggered the rejection.

> The main failure mode I diagnosed was an overly broad author-detection rule in `detect_shape`. In an earlier version, I allowed the classifier to infer author questions from capitalized word patterns too aggressively. That caused a normal question like “Find recipes that use ginger” to be misclassified as `Q8` (author + ingredient), because the classifier incorrectly treated ordinary capitalized words as if they were a person name.

> The downstream effect was that `extract_slots` produced something like `{"author": "", "ingredient": "ginger"}`, and the resulting Cypher query returned no rows because there was no author with an empty name. I also saw the same issue on off-template questions such as “what is the meaning of life,” which should have returned `None` but was initially classified as an author-related shape.

> I fixed this by making `_has_author_cue()` much more conservative. Instead of matching arbitrary capitalized phrases, it now only fires on explicit patterns like `by author Maria Rossi` or `by Maria Rossi`. After that change, supported questions classified correctly and adversarial questions triggered `UnsupportedQueryError` instead of silently taking the wrong route through the mapper.

## 4. A design tradeoff between the deterministic mapper and the Tier 3 chain

When would you prefer the deterministic mapper over the LLM chain in
production, and vice versa? Cite a concrete dimension (latency,
auditability, schema-coverage cost, distribution-shift robustness,
operational risk) for each side. Both implementations are first-class —
your answer should reflect that, not pick a winner.

> I would prefer the deterministic mapper in production when the schema is small, the supported question surface is bounded, and I need predictable behaviour. Its main strengths are auditability, low latency, and operational safety. Every supported question shape maps to a tested Cypher template, the query string is completely transparent, and each answer can be traced back to the exact graph pattern that produced it.

> I would prefer the Tier 3 LLM chain when the input distribution is broader and the cost of manually maintaining many templates becomes too high. Its main strength is flexibility: it can generalize to new phrasings and new combinations of constraints without requiring a hand-authored rule for each one. That makes it attractive when users ask open-ended questions that still correspond to valid graph queries but do not fit a small finite set of patterns.

> The tradeoff is that the deterministic mapper is stronger on reliability and explainability, while the LLM chain is stronger on coverage and adaptability. In practice, I would choose between them based on the input distribution, latency budget, audit requirements, and tolerance for operational risk.
