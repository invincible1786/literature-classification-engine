"""Setup script to ensure required NLTK tokenizers are installed."""
import sys
import nltk

def setup():
    packages = ["punkt", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
            print(f"[OK] NLTK package '{pkg}' ready.")
        except Exception as e:
            print(f"[ERROR] Failed to download NLTK package '{pkg}': {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    setup()

