"""Labeled prompt pairs: the behavioral ground truth for semcache.

This is data, not a test. It answers one question: given two prompts, should a
semantic cache serve the same answer for both?

Runnable standalone, before semcache exists, to calibrate the threshold:

    python -m tests.golden_pairs

Adding pairs: every wrong answer served in production becomes a MUST_MISS entry
here. That is how this file earns its keep over time.
"""

from collections import namedtuple

Pair = namedtuple("Pair", "a b why")

# ---------------------------------------------------------------------------
# MUST HIT - same intent, different surface. A miss here costs one API call.
# ---------------------------------------------------------------------------

MUST_HIT = [
    # -- casing, punctuation, whitespace -----------------------------------
    Pair("what is github?", "What is GitHub", "casing + punctuation"),
    Pair("how do i reset my password", "How do I reset my password?", "casing"),
    Pair("explain  docker   volumes", "explain docker volumes", "whitespace"),
    Pair("WHAT IS A VPN", "what is a vpn", "shouting"),

    # -- abbreviation / expansion ------------------------------------------
    Pair("what is a CDN", "what is a content delivery network", "acronym expanded"),
    Pair("explain CI/CD", "explain continuous integration and deployment", "acronym"),
    Pair("how does DNS work", "how does the domain name system work", "acronym"),
    Pair("what's the diff between TCP and UDP",
         "what is the difference between TCP and UDP", "informal contraction"),

    # -- paraphrase ---------------------------------------------------------
    Pair("how do I center a div",
         "what's the best way to horizontally center a div", "paraphrase"),
    Pair("why is my docker build slow",
         "what makes docker builds take so long", "paraphrase"),
    Pair("explain python decorators",
         "can you describe how decorators work in python", "paraphrase"),
    Pair("what does the yield keyword do",
         "explain what yield does in python", "paraphrase"),
    Pair("how much does GPT-4 cost",
         "what is the pricing for GPT-4", "paraphrase"),

    # -- question form vs imperative ---------------------------------------
    Pair("what is kubernetes", "explain kubernetes", "question vs imperative"),
    Pair("how do I install postgres", "show me how to install postgres", "form"),
    Pair("what are python generators", "tell me about python generators", "form"),

    # -- politeness / filler padding ---------------------------------------
    Pair("what is rust",
         "hi! could you please explain to me what rust is? thanks!", "padding"),
    Pair("summarize the CAP theorem",
         "I'd really appreciate it if you could summarize the CAP theorem",
         "padding"),
    Pair("how do I exit vim", "ok this is embarrassing but how do I exit vim",
         "conversational padding"),

    # -- typos --------------------------------------------------------------
    Pair("how to instal numpy", "how to install numpy", "typo"),
    Pair("what is kubernets", "what is kubernetes", "typo in entity"),
    Pair("explain recursin", "explain recursion", "typo"),

    # -- word order ---------------------------------------------------------
    Pair("in python, how do I read a file",
         "how do I read a file in python", "clause order"),
    Pair("for a beginner, what is machine learning",
         "what is machine learning for a beginner", "clause order"),

    # -- synonyms -----------------------------------------------------------
    Pair("how do I remove a git commit",
         "how do I delete a git commit", "remove/delete"),
    Pair("what is a fast sorting algorithm",
         "what is a quick sorting algorithm", "fast/quick"),
    Pair("how to fix a memory leak", "how to solve a memory leak", "fix/solve"),

    # -- determiners and plurality -----------------------------------------
    Pair("what is a design pattern", "what are design patterns", "plurality"),
    Pair("explain the singleton pattern", "explain singleton pattern", "article"),
]

# ---------------------------------------------------------------------------
# MUST MISS - different intent, deceptively similar surface.
# A false hit here serves a WRONG ANSWER to a user. These are the expensive ones.
# ---------------------------------------------------------------------------

MUST_MISS = [
    # -- entity swap (highest surface similarity, totally different answer) --
    Pair("what is the capital of France", "what is the capital of Germany",
         "entity swap"),
    Pair("how do I install postgres", "how do I install mysql", "entity swap"),
    Pair("explain the react useEffect hook", "explain the react useState hook",
         "entity swap"),
    Pair("what is the population of Tokyo", "what is the population of Osaka",
         "entity swap"),
    Pair("who wrote Hamlet", "who wrote Macbeth", "entity swap"),

    # -- negation (one token flips the answer) ------------------------------
    Pair("how do I enable swap on linux", "how do I disable swap on linux",
         "negation"),
    Pair("why should I use microservices", "why should I not use microservices",
         "negation"),
    Pair("is python garbage collected", "is python not garbage collected",
         "negation"),
    Pair("how to make a container writable",
         "how to make a container read-only", "antonym"),

    # -- numeric / version (models treat digits as near-noise) --------------
    Pair("python 2 print syntax", "python 3 print syntax", "version"),
    Pair("what changed in postgres 14", "what changed in postgres 15", "version"),
    Pair("how do I upgrade to node 18", "how do I upgrade to node 20", "version"),
    Pair("convert 100 USD to EUR", "convert 1000 USD to EUR", "magnitude"),
    Pair("give me 3 examples", "give me 30 examples", "magnitude"),

    # -- opaque identifiers (the classic semantic-cache failure) ------------
    Pair("what does error code 401 mean", "what does error code 403 mean",
         "status code"),
    Pair("what is a P0 incident", "what is a P1 incident", "internal severity"),
    Pair("status of ticket ENG-4471", "status of ticket ENG-4472", "ticket id"),
    Pair("look up SKU AX-9920", "look up SKU AX-9921", "sku"),
    Pair("what is CVE-2021-44228", "what is CVE-2021-45046", "cve id"),

    # -- direction / asymmetry ----------------------------------------------
    Pair("convert celsius to fahrenheit", "convert fahrenheit to celsius",
         "direction"),
    Pair("how to migrate from mysql to postgres",
         "how to migrate from postgres to mysql", "direction"),
    Pair("translate this to french", "translate this from french", "direction"),

    # -- temporal ------------------------------------------------------------
    Pair("what is the weather today", "what is the weather tomorrow", "temporal"),
    Pair("Q1 2024 revenue", "Q2 2024 revenue", "period"),
    Pair("what happened in the 2020 election",
         "what happened in the 2024 election", "year"),

    # -- scope / granularity -------------------------------------------------
    Pair("how do I delete a file in bash",
         "how do I delete a directory in bash", "scope"),
    Pair("what is git", "what is git rebase", "general vs specific"),
    Pair("explain HTTP", "explain HTTPS", "protocol"),
    Pair("summarize this document", "translate this document", "task verb"),

    # -- unit swap ------------------------------------------------------------
    Pair("how many miles in a marathon", "how many kilometers in a marathon",
         "unit"),
    Pair("set the timeout to 30 seconds", "set the timeout to 30 minutes",
         "unit"),
]

# ---------------------------------------------------------------------------
# BORDERLINE - reasonable people disagree. Do not assert; record the decision
# your product makes and watch these when you move the threshold.
# ---------------------------------------------------------------------------

BORDERLINE = [
    Pair("how do I sort a list in python", "how do I sort an array in javascript",
         "same task, different language - almost always MISS"),
    Pair("what is a good python web framework",
         "what is the best python web framework", "good/best - usually HIT"),
    Pair("explain docker", "explain containers",
         "hypernym - depends how general your answers are"),
    Pair("how do I write a unit test", "how do I write an integration test",
         "sibling concepts - usually MISS"),
    Pair("what is machine learning", "what is deep learning",
         "subset relation - usually MISS"),
    Pair("fix this bug", "debug this",
         "same intent, no content - both depend on context you are not caching"),
]


# ---------------------------------------------------------------------------
# Calibration harness. Runnable against any embedder, before semcache exists.
# ---------------------------------------------------------------------------

def default_embedder():
    """sentence-transformers all-MiniLM-L6-v2, normalized."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text):
        import numpy as np  # noqa: PLC0415

        v = model.encode(text).astype("float32")
        return v / np.linalg.norm(v)

    return embed


def sweep(embed, thresholds=(0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85, 0.90)):
    """Score every pair once, then report hit/miss rates per threshold."""
    import numpy as np  # noqa: PLC0415

    def score(p):
        return float(np.dot(embed(p.a), embed(p.b)))

    hits = [(p, score(p)) for p in MUST_HIT]
    misses = [(p, score(p)) for p in MUST_MISS]

    print(f"{len(hits)} must-hit, {len(misses)} must-miss pairs\n")
    print(f"{'thresh':>7} {'recall':>8} {'false hits':>11}  {'verdict'}")
    print("-" * 46)

    best = None
    for t in thresholds:
        recall = sum(1 for _, s in hits if s >= t) / len(hits)
        false = sum(1 for _, s in misses if s >= t)
        verdict = "SAFE" if false == 0 else f"{false} wrong answers"
        print(f"{t:>7.2f} {recall:>7.0%} {false:>11}  {verdict}")
        if false == 0 and (best is None or recall > best[1]):
            best = (t, recall)

    print()
    if best:
        print(f"Lowest safe threshold: {best[0]:.2f} (recall {best[1]:.0%})")
    else:
        print("No threshold separates the sets. Your embedder is wrong for this "
              "domain, or a MUST_MISS pair is mislabeled.")

    print("\nWorst must-hit pairs (raise recall by fixing these or lowering t):")
    for p, s in sorted(hits, key=lambda x: x[1])[:5]:
        print(f"  {s:.3f}  [{p.why}] {p.a!r} / {p.b!r}")

    print("\nMost dangerous must-miss pairs (these serve wrong answers first):")
    for p, s in sorted(misses, key=lambda x: -x[1])[:5]:
        print(f"  {s:.3f}  [{p.why}] {p.a!r} / {p.b!r}")

    return hits, misses


def _self_check():
    """No model needed: catch data rot in the dataset itself."""
    seen = set()
    for group, name in ((MUST_HIT, "MUST_HIT"), (MUST_MISS, "MUST_MISS"),
                        (BORDERLINE, "BORDERLINE")):
        for p in group:
            assert p.a and p.b, f"{name}: empty prompt"
            assert p.a != p.b, f"{name}: identical pair {p.a!r}"
            assert p.why, f"{name}: pair {p.a!r} has no stated reason"
            key = frozenset((p.a.lower(), p.b.lower()))
            assert key not in seen, f"duplicate pair across groups: {p.a!r}"
            seen.add(key)
    print(f"dataset ok: {len(MUST_HIT)} hit / {len(MUST_MISS)} miss / "
          f"{len(BORDERLINE)} borderline, no duplicates")


if __name__ == "__main__":
    _self_check()
    try:
        sweep(default_embedder())
    except ImportError:
        print("\nInstall sentence-transformers to run the threshold sweep.")
