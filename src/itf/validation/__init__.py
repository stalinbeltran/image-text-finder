"""Can this network train on this dataset? (contracts ① and ②)

**① and ② are the same question, so they share one validator.** That is the whole
insight: `patch_size == input_size`, `border_features` over a dataset without
`border`, and `in_channels` vs `patch_shape` all ask *"can this C train on this
B?"* -- same moment (before training), same place (where B and C meet), same kind
of answer (a 400 with a reason). No separate mechanism for `border` was needed;
it came free in the validator ① already required.

**A pure function of two dicts.** It never sees an object: B's manifest against
C's config. No torch, no dataset, milliseconds.

That purity is not a detail, it is the proof it sits in the right layer: **if
validating required training, the validation would be in the wrong place**
(tests.md §3). It is also what allows the two call sites, which are deliberate
and not redundant:

  - `POST /runs` -> a **400 before the job is even created**;
  - inside `train()` -> the safety net, because **`itf-train` does not go through
    the API** and without it the CLI walks straight past the gate.

The old code had neither. A mismatch did not fail on the way in: it failed inside
the job thread, half an hour later, mid-batch, with `mat1 and mat2 shapes cannot
be multiplied` -- a message that says nothing about the actual problem.

Imports nothing from `itf`, and must not (contract ⑦).
"""

from __future__ import annotations

from typing import Any, TypedDict


class Problem(TypedDict):
    """R4's shape: a stable slug, a message for people, and how to fix it."""

    code: str
    message: str
    hint: str


def check_compatible(manifest: dict[str, Any], network: dict[str, Any]) -> list[Problem]:
    """Every reason this network cannot train on this dataset. Empty == it can.

    Returns ALL of them, not just the first: a mismatch of both `patch_size` and
    `border_features` should be two lines in one 400, not two round trips.
    """
    problems: list[Problem] = []

    # ① The critical one. Declared in two independent places -- the manifest's
    #    config and the network's config -- with nothing but goodwill joining them.
    patch_size = (manifest.get("config") or {}).get("patch_size")
    input_size = network.get("input_size")
    if patch_size is not None and input_size is not None and int(patch_size) != int(input_size):
        problems.append(
            {
                "code": "patch_size_mismatch",
                "message": (
                    f"la red espera patches de {input_size}x{input_size} y el dataset "
                    f"los tiene de {patch_size}x{patch_size}"
                ),
                "hint": (
                    f"elige un dataset con patch_size {input_size}, "
                    f"o una red con input_size {patch_size}"
                ),
            }
        )

    # ② Ausente ≠ cero (formatos.md §2). Filling an absent field is legal only if
    #    the consumer does not read it. Zero means "touches no border", not
    #    "unknown": a network trained on fabricated zeros learns to ignore those
    #    4 inputs, and then inference hands it the REAL flags -- so it meets a
    #    distribution it never trained on, exactly at the image borders, which is
    #    where the flag was supposed to help. Nothing raises. It just gets worse.
    #
    #    `has_border` is a DECLARATION, so absent means "I don't have them": the
    #    safe default, which refuses too much and never too little. That is why
    #    no manifest needs migrating.
    if network.get("border_features") and not manifest.get("has_border", False):
        problems.append(
            {
                "code": "border_features_unavailable",
                "message": (
                    "el dataset se construyó sin los flags de borde y la red los pide "
                    "(border_features: true)"
                ),
                "hint": "reconstruye el dataset, o entrena con border_features: false",
            }
        )

    # (③ of the same box) in_channels ↔ patch_shape [n, n, c].
    patch_shape = manifest.get("patch_shape")
    in_channels = network.get("in_channels")
    if patch_shape and in_channels is not None and int(patch_shape[-1]) != int(in_channels):
        problems.append(
            {
                "code": "in_channels_mismatch",
                "message": (
                    f"la red espera {in_channels} canal(es) y los patches tienen "
                    f"{patch_shape[-1]}"
                ),
                "hint": f"pon in_channels: {patch_shape[-1]} en la red",
            }
        )

    return problems
