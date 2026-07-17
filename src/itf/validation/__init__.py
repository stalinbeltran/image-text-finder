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


def check_measurable(manifest: dict[str, Any]) -> list[Problem]:
    """Can anything be MEASURED on this dataset? (protocolo.md §1.3)

    Separate from `check_compatible` because it asks a different question: not
    "does this C fit this B?" but "is this B a measuring tool at all?". No
    network is involved, and the answer does not change with one.

    A dataset with no val split is not a measuring tool. The old
    `monitor = val_metrics.get("loss", train_loss)` fell back to the TRAIN loss
    with no warning, so `best.pt` quietly became the most overfitted checkpoint
    -- and it happened to the README's own example. Building such a dataset
    warns (fase 2, `manifest.warnings`); starting a run on it refuses, because
    this is where the damage is done.

    Absent is not zero (formatos.md §2): a manifest that does not declare its
    splits is not declaring an empty val, so it is not refused here. `train()`
    still checks the real split it loaded, which is the stronger check and the
    reason this one can afford to be lenient.
    """
    per_split = manifest.get("patches_per_split")
    if isinstance(per_split, dict) and per_split.get("val") == 0:
        return [
            {
                "code": "no_validation_split",
                "message": (
                    "el dataset no tiene patches de val, así que no hay con qué elegir "
                    "best.pt ni con qué medir"
                ),
                "hint": (
                    "reconstruye el dataset con una fracción de val > 0: sin val, elegir "
                    "checkpoint cae en la pérdida de entrenamiento y se queda el más "
                    "sobreajustado, en silencio"
                ),
            }
        ]
    return []


def check_run(manifest: dict[str, Any], network: dict[str, Any]) -> list[Problem]:
    """Everything that can be refused **before a run starts**. Empty == go ahead.

    One function because there are two doors -- `POST /runs` and `itf-train` --
    and they must refuse the same things. Asking the two checks separately in each
    of them is how the doors drift apart, and the door that ends up more lenient
    is the one a sweep script comes through.

    It is also what lets both refuse **before reserving the name**. Reserving
    first and validating after leaves a corpse for every mistake: you fix the
    dataset, retry with the same name, and get "that run already exists" for a
    run that never saw a single batch.
    """
    return check_compatible(manifest, network) + check_measurable(manifest)
