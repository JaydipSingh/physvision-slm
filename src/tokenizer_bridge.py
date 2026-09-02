"""
Tokenizer Bridge for PhysVision-SLM
===================================
Provides a single tokenize(text) -> List[int] function that matches the
vocabulary TinyLMv3 was trained with (32K Rust BPE), so the pretrained
language-model weights are actually usable in the multimodal model.

This module calls the compiled Rust BPE tokenizer binary DIRECTLY. It does not
import paper 1's training script (which pulls in arxiv/requests), so it is
robust to different folder layouts.

Resolution order for the Rust tokenizer project directory:
  1. $PHYSVISION_TOKENIZER_DIR (explicit override)
  2. slm_v0/subword_tokenizer               (sibling-of-parent layout)
  3. slm_v0/subword_tokenizer/slm_v2/../..  (checkpoint-nested layout)
  4. A recursive search upward for a "subword_tokenizer" folder containing
     target/release/bpe-tokenizer

If the binary or model is not found, falls back to a character-level encoder
(WARNING printed). The fallback is only for Windows smoke tests — its token IDs
will NOT match the pretrained LM vocabulary.

Usage:
  from src.tokenizer_bridge import get_tokenizer
  tok = get_tokenizer(vocab_size=32000)
  ids = tok("Question: Is there a tumor? Answer: Yes")
"""
import ast
import os
import shutil
import subprocess
from pathlib import Path


def _candidate_tokenizer_dirs():
    """Yield plausible subword_tokenizer directories, most specific first."""
    # 1. Explicit override
    env = os.environ.get("PHYSVISION_TOKENIZER_DIR")
    if env:
        yield Path(env)

    here = Path(__file__).resolve()
    # slm_v0 is two levels up from src/ (physvision-slm/src/ -> physvision-slm -> slm_v0)
    slm_v0 = here.parents[2]

    # 2. Standard sibling location
    yield slm_v0 / "subword_tokenizer"

    # 3. Search a few levels up for any subword_tokenizer folder
    search_root = slm_v0
    for base in [search_root, search_root.parent]:
        if base and base.exists():
            try:
                for p in base.rglob("subword_tokenizer"):
                    if p.is_dir():
                        yield p
            except Exception:
                pass


def _find_tokenizer_project():
    """Return (project_dir, binary_path, model_path) or (None, None, None)."""
    seen = set()
    for d in _candidate_tokenizer_dirs():
        try:
            d = d.resolve()
        except Exception:
            continue
        if d in seen or not d.exists():
            continue
        seen.add(d)

        binary = d / "target" / "release" / "bpe-tokenizer"
        model = d / "model_32k.json"
        if binary.exists() and model.exists():
            return d, binary, model
    return None, None, None


def _make_rust_tokenizer(project_dir: Path, binary: Path, model: Path):
    """Build a tokenize(text)->list[int] that shells out to the Rust binary."""
    active_model = project_dir / "model.json"

    # Activate the 32K model (the binary reads model.json from the project dir).
    try:
        if model.exists():
            shutil.copy2(model, active_model)
    except Exception:
        pass

    def _tokenize(text: str):
        result = subprocess.run(
            [str(binary), "tokenize", text],
            capture_output=True, text=True, cwd=str(project_dir),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tokenizer failed: {result.stderr.strip()}")
        for line in result.stdout.splitlines():
            if line.startswith("IDs:"):
                return ast.literal_eval(line.replace("IDs:", "").strip())
        return []

    return _tokenize


def get_tokenizer(vocab_size: int = 32000):
    """
    Return a tokenize(text) -> list[int] callable.

    Prefers the real Rust BPE tokenizer (correct vocabulary). Falls back to a
    character-level encoder if the Rust tokenizer is unavailable.
    """
    project_dir, binary, model = _find_tokenizer_project()

    if binary is not None:
        try:
            tok = _make_rust_tokenizer(project_dir, binary, model)
            # Smoke check that it actually runs
            _ = tok("test")
            print(f"  [tokenizer] Using real Rust BPE tokenizer (32K vocab) at "
                  f"{project_dir}")
            return tok
        except Exception as e:
            print(f"  [tokenizer] Rust tokenizer found but failed to run: {e}")

    print("  [tokenizer] WARNING: Rust tokenizer unavailable — using "
          "character-level fallback. Token IDs will NOT match the pretrained "
          "LM vocabulary. Use only for smoke tests, not real training.\n"
          "  [tokenizer] Set PHYSVISION_TOKENIZER_DIR to your subword_tokenizer "
          "folder if it was not auto-detected.")

    def _char_tokenize(text: str):
        return [ord(c) % vocab_size for c in text]

    return _char_tokenize
