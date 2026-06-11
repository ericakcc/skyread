"""End-to-end spike: sounding -> MetPy indices -> dual-layer interpretation.

Run:
    uv run python -m skyread.spike
"""

from __future__ import annotations

from skyread.indices import compute_indices
from skyread.interpret import build_grandma_prompt, interpret_rule_based
from skyread.sounding import load_sample


def main() -> None:
    """Run the full SkyRead chain on a bundled sample sounding and print it."""
    snd = load_sample("may4_sounding.txt")
    print(f"# Sounding: {snd.name}  ({len(snd.pressure)} levels)\n")

    indices = compute_indices(snd)
    print("## Step 1 — MetPy computed indices (deterministic):")
    for key, value in indices.items():
        print(f"   {key:>14}: {value}")

    cards = interpret_rule_based(indices, snd.name)
    print("\n## Step 2 — dual-layer cards (rule-based draft / fallback):")
    print("   " + cards["pro"])
    print("   " + cards["grandma"])

    print("\n## Step 3 — the rewrite prompt that goes to the small LLM:")
    print("   " + build_grandma_prompt(indices, snd.name).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
