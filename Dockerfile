# RAG pour site statique : indexation + questions-réponses
FROM python:3.12-slim

ARG UID=1000
ARG GID=1000
# Paquets pip supplémentaires, ex. : --build-arg EXTRA_PIP="anthropic" si LLM_PROVIDER=anthropic
ARG EXTRA_PIP=""
# 1 = télécharge le modèle d'embeddings pendant le build (image ~2 Go plus lourde, démarrage plus rapide)
ARG PRELOAD_MODEL=0
ARG EMBED_MODEL=BAAI/bge-m3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    RAG_DB=/data/db \
    RAG_EMBED_MODEL=${EMBED_MODEL}

# Utilisateur non-root ; /data (index + cache des modèles) lui appartient
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app \
    && mkdir -p /data/db /data/hf /site \
    && chown -R app:app /data

WORKDIR /app

# PyTorch en version CPU uniquement (évite ~3 Go de bibliothèques CUDA) ;
# pour un GPU NVIDIA, retirer cette ligne et utiliser l'image CUDA de PyTorch.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt ${EXTRA_PIP}

COPY *.py ./

USER app

# Téléchargement optionnel du modèle d'embeddings dans l'image
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"; \
    fi

VOLUME ["/data"]

# Usage : docker run ... <image> index.py /site --base-url https://monsite.fr
#         docker run -it ... <image> ask.py "ma question"
ENTRYPOINT ["python"]
CMD ["ask.py"]
