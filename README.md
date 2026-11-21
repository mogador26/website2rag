# RAG pour site web statique {HTML,PDF,images}

Indexe le site construit (ex: dossier `public/`), puis répond aux questions en français
avec citations des pages et des PDF (`#page=N`).

## Installation

```bash
python3 -m venv venv
source venv/bin/activate            # Windows : venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env                # puis choisir et renseigner le fournisseur de LLM
```

Python 3.11 ou 3.12 recommandé. Le premier lancement télécharge le modèle
d'embeddings `BAAI/bge-m3` (~2 Go).

> `pymupdf4llm` doit rester en version `<2` : la 2.0 est un paquet vide.

## Choix du LLM

Seule la génération des réponses (et la description des images avec `--images`) utilise un LLM ;
les embeddings et le reranker tournent en local. Le LLM se règle dans `.env` :

| `LLM_PROVIDER` | Services couverts | Variables |
|---|---|---|
| `openai` (défaut) | OpenAI, **Ollama / LM Studio / vLLM (local)**, Mistral, Groq, OpenRouter, Azure... | `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` |
| `anthropic` | API Anthropic (installer `pip install anthropic`) | `LLM_MODEL`, `LLM_API_KEY` |

`LLM_MODEL` est obligatoire : aucun modèle n'est imposé par défaut. Exemples dans `.env.example`.
Pour un autre fournisseur, ajouter une fonction dans `llm.py` et l'enregistrer dans `PROVIDERS`.

## Utilisation

```bash
# 1. Indexer (HTML + PDF). Relancer après chaque build : seuls les fichiers modifiés sont retraités.
python index.py public --base-url https://monsite.fr

#    Avec description des images par le LLM configuré (multimodal requis, résultats mis en cache) :
python index.py public --base-url https://monsite.fr --images

#    Exclure des contenus (voir « Exclusions » plus bas) :
python index.py public --base-url https://monsite.fr --no-pdf
python index.py public --base-url https://monsite.fr --exclude 'docs/*' --exclude 'archives/*'

#    Extraction PDF rapide (texte brut, sans analyse de mise en page ni OCR) :
python index.py public --base-url https://monsite.fr --fast-pdf

#    Tout réindexer :
python index.py public --base-url https://monsite.fr --full

# 2. Poser une question (mode interactif si aucune question)
python ask.py "Quels sont les tarifs ?"
python ask.py "..." --rerank        # reranker : plus précis, plus lent

# 3. Évaluer la recherche avec le classeur Excel (feuille « Évaluation »)
python eval.py rag_site_statique.xlsx
```

## Options d'indexation

| Option | Effet |
|---|---|
| `--no-pdf` | ignore tous les PDF |
| `--images` | décrit les images avec le LLM configuré (désactivé par défaut) |
| `--no-images` | n'indexe pas les images (l'emporte sur `--images`) |
| `--exclude MOTIF` | ignore les fichiers dont le chemin relatif correspond au motif (ex. `'docs/*'`, `'*.pdf'`) ; répétable |
| `--fast-pdf` | extraction PDF rapide, sans mise en page ni OCR |
| `--full` | réindexe tout depuis zéro |

**Exclusions** : un fichier exclu n'est pas traité, mais s'il était déjà indexé il reste dans l'index.
Pour retirer des contenus déjà indexés, relancer avec `--full` et les mêmes exclusions.

## Fichiers

| Fichier | Rôle |
|---|---|
| `index.py` | Extraction HTML/PDF/images, découpage par section, embeddings, stockage Chroma (`db/`) |
| `common.py` | Embeddings, base vectorielle, recherche hybride (vecteurs + BM25, fusion RRF), reranker |
| `llm.py` | Accès au LLM indépendant du fournisseur (API compatible OpenAI ou Anthropic) |
| `ask.py` | Question → recherche → réponse du LLM configuré, avec sources |
| `eval.py` | Calcule le rang de la bonne page pour chaque question du classeur et le recall@k |
| `rag_site_statique.xlsx` | Architecture, plan de travail, inventaire, jeu d'évaluation, pièges |

## Limites connues

- Pas d'OCR : les pages PDF sans texte (scans) sont signalées mais non indexées.
- Les liens cités dépendent de `--base-url` : à adapter si le site est servi dans un sous-dossier (`baseurl` Jekyll).
- Pour tester la recherche seule, sans clé API :
  `python -c "from common import search; [print(h['meta']['url'], h['text'][:200]) for h in search('ma question', k=5)]"`
