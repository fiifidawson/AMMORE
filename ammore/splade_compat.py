"""Keep pymilvus' SPLADE sparse embedding working on transformers >= 5.

transformers 5 removed the long-deprecated `tokenizer.batch_encode_plus`, but
pymilvus 2.6.6 and milvus-model 0.2.12 still call it, so every sparse embedding
dies with:

    AttributeError: BertTokenizer has no attribute batch_encode_plus

Pinning transformers < 5 is not an option: marker-pdf and surya-ocr (mmore's
PDF/OCR path) require transformers >= 5.12. There is no fixed release upstream
either -- milvus-model 0.2.12 is the newest published version.

`tokenizer(...)` accepts exactly the arguments SPLADE passes, so the fix is to
re-expose it under the old name rather than to patch each call site (which would
mean editing site-packages, and losing the change on the next reinstall).

mmore runs as a subprocess (`python -m mmore ...`), so patching in this process
does not help the child. `bootstrap_cmd()` builds a command line that applies
the patch inside the child before handing control to mmore's CLI.
"""

import sys
from typing import List


def apply() -> None:
    """Re-expose `batch_encode_plus` on tokenizers that no longer provide it."""
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:  # transformers not installed -> nothing to patch
        return
    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__


# Applied in the child before mmore's CLI takes over. click reads sys.argv[1:],
# which under `python -c` is exactly the mmore arguments that follow.
_BOOTSTRAP = (
    "import ammore.splade_compat as _c; _c.apply(); "
    "from mmore.cli import main; main()"
)


def bootstrap_cmd(*args: str) -> List[str]:
    """`python -m mmore <args>`, with the SPLADE compat patch applied first."""
    return [sys.executable, "-c", _BOOTSTRAP, *args]
