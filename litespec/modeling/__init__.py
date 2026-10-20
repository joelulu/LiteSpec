"""DLite model implementations."""

from .draft import (
    DFlash2DraftModel,
    DFlashDraftModel,
    DLiteDraftModel,
    DLiteSequentialHead,
    DSparkDraftModel,
)


def get_training_target_model(*args, **kwargs):
    # HF VLM prototypes must not import SGLang/sequence-parallel dependencies.
    from .target.training_target_model import get_training_target_model as factory

    return factory(*args, **kwargs)

__all__ = [
    "DLiteDraftModel",
    "DLiteSequentialHead",
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DSparkDraftModel",
    "get_training_target_model",
]
