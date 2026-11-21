"""Couche d'accès au LLM, indépendante du fournisseur.

Configuration par variables d'environnement (ou fichier .env) :

    LLM_PROVIDER   "openai" (défaut) : API compatible OpenAI -> OpenAI, Ollama, LM Studio,
                                        vLLM, Mistral, Groq, OpenRouter, Azure, etc.
                   "anthropic"      : API Anthropic
    LLM_MODEL      nom du modèle (obligatoire), ex. "gpt-4o-mini", "llama3.1", "mistral-small-latest"
    LLM_API_KEY    clé API (sinon OPENAI_API_KEY / ANTHROPIC_API_KEY selon le fournisseur)
    LLM_BASE_URL   URL de l'API compatible OpenAI (ex. http://localhost:11434/v1 pour Ollama)

Pour ajouter un autre fournisseur, ajouter une fonction _complete_xxx et l'enregistrer dans PROVIDERS.
"""
import base64
import os


def _require_model():
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit(
            "LLM_MODEL n'est pas défini. Renseignez LLM_PROVIDER, LLM_MODEL (et LLM_API_KEY / "
            "LLM_BASE_URL selon le fournisseur) dans l'environnement ou le fichier .env."
        )
    return model


def _complete_openai(prompt, system, image_jpeg, max_tokens):
    from openai import OpenAI

    base_url = os.environ.get("LLM_BASE_URL") or None
    api_key = os.environ.get("LLM_API_KEY") or None
    if api_key is None and base_url and not os.environ.get("OPENAI_API_KEY"):
        api_key = "not-needed"  # serveurs locaux (Ollama, LM Studio...) sans clé
    client = OpenAI(base_url=base_url, api_key=api_key)

    content = [{"type": "text", "text": prompt}]
    if image_jpeg:
        b64 = base64.standard_b64encode(image_jpeg).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content if image_jpeg else prompt}
    ]
    # max_tokens volontairement omis : son nom diffère selon les serveurs et les modèles.
    resp = client.chat.completions.create(model=_require_model(), messages=messages)
    return (resp.choices[0].message.content or "").strip()


def _complete_anthropic(prompt, system, image_jpeg, max_tokens):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("LLM_API_KEY") or None)
    content = [{"type": "text", "text": prompt}]
    if image_jpeg:
        b64 = base64.standard_b64encode(image_jpeg).decode()
        content.insert(0, {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": b64}})
    kwargs = {"system": system} if system else {}
    msg = client.messages.create(
        model=_require_model(), max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}], **kwargs,
    )
    return msg.content[0].text.strip()


PROVIDERS = {"openai": _complete_openai, "anthropic": _complete_anthropic}


def complete(prompt, system=None, image_jpeg=None, max_tokens=1000):
    """Envoie un prompt (texte, + image JPEG optionnelle) et renvoie la réponse en texte."""
    name = os.environ.get("LLM_PROVIDER", "openai").lower()
    if name not in PROVIDERS:
        raise SystemExit(f"LLM_PROVIDER inconnu : {name!r} (valeurs possibles : {', '.join(PROVIDERS)})")
    return PROVIDERS[name](prompt, system, image_jpeg, max_tokens)
