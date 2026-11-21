"""Évalue la recherche à partir de la feuille « Évaluation » du classeur Excel.

Colonne A = question, colonne B = URL (ou fragment d'URL) de la page attendue.
Le script écrit le rang obtenu en colonne C (0 = absente des 20 premiers résultats)
et enregistre un nouveau classeur ; le recall@k se recalcule à l'ouverture dans Excel.

Usage :
    python eval.py rag_site_statique.xlsx
    python eval.py rag_site_statique.xlsx --rerank --out resultats.xlsx
"""
import argparse

from openpyxl import load_workbook

from common import search

FIRST, LAST = 5, 34   # lignes de données de la feuille


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default="evaluation_resultats.xlsx")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    wb = load_workbook(args.xlsx)
    ws = wb["Évaluation"]
    ws["G2"] = args.k
    done = hits = 0
    for r in range(FIRST, LAST + 1):
        q, expected = ws.cell(r, 1).value, ws.cell(r, 2).value
        if not q or not expected:
            continue
        results = search(q, k=20, rerank=args.rerank)
        rank = next((i for i, h in enumerate(results, 1) if expected in h["meta"]["url"]), 0)
        ws.cell(r, 3, rank)
        done += 1
        hits += 1 if 0 < rank <= args.k else 0
        print(f"{'OK ' if 0 < rank <= args.k else 'KO '} rang={rank:<3} {q}")
    wb.save(args.out)
    if done:
        print(f"\nrecall@{args.k} = {hits}/{done} = {hits / done:.0%}  ->  {args.out}")


if __name__ == "__main__":
    main()
