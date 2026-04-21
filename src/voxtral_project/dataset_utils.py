from __future__ import annotations


def load_fleurs_streaming(*, lang_code: str, split: str = "test"):
    import datasets
    from datasets import load_dataset

    major_version = int(str(datasets.__version__).split(".", 1)[0])
    if major_version >= 4:
        raise RuntimeError(
            "FLEURS loading in this repo requires `datasets<4` because the current "
            "`google/fleurs` packaging still depends on a dataset script. "
            f"Installed version: {datasets.__version__}. "
            "Reinstall the pinned workspace requirements before running FLEURS-based scripts."
        )

    return load_dataset(
        "google/fleurs",
        lang_code,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
