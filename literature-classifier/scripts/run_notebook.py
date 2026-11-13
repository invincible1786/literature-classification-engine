"""Execute notebooks/evaluation.ipynb cleanly using nbclient and persist executed cell outputs."""
import os
import nbformat
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(BASE_DIR, "notebooks", "evaluation.ipynb")


def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": os.path.join(BASE_DIR, "notebooks")}})
    client.execute()

    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved outputs into {NB_PATH}")


if __name__ == "__main__":
    main()

