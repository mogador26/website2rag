"""Pose une question au RAG.

Usage :
    python ask.py "Quels sont les tarifs ?"
    python ask.py                      # mode interactif
    python ask.py "..." --rerank       # avec reranker (plus précis, plus lent)
Le LLM se configure via LLM_PROVIDER / LLM_MODEL / LLM_API_KEY (voir llm.py et .env.example).
"""
import argparse

from common import search
from llm import complete

SYSTEM = (
    "Tu es l'assistant documentaire d'un site web. Réponds en français, uniquement à partir "
    "des extraits fournis. Cite tes sources avec [n] après chaque affirmation. "
    "Si l'information n'est pas dans les extraits, dis-le clairement sans inventer."
)


def answer(question, k=6, rerank=False):
    hits = search(question, k=k, rerank=rerank)
    context = "\n\n".join(
        f"[{n}] ({h['meta']['type']}) {h['meta']['url']}\n{h['text']}" for n, h in enumerate(hits, 1)
    )
    text = complete(f"Extraits :\n\n{context}\n\nQuestion : {question}", system=SYSTEM)
    sources = "\n".join(f"[{n}] {h['meta']['url']}" for n, h in enumerate(hits, 1))
    return text, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    def run(q):
        text, sources = answer(q, args.k, args.rerank)
        print(f"\n{text}\n\nSources :\n{sources}\n")

    if args.question:
        run(args.question)
    else:
        while (q := input("Question (vide pour quitter) > ").strip()):
            run(q)


if __name__ == "__main__":
    main()
