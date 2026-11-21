"""Indexation d'un site Jekyll construit (HTML + PDF + images) dans Chroma.

Usage :
    python index.py public --base-url https://monsite.fr
    python index.py public --base-url https://monsite.fr --images   # + description des images
    python index.py public --full                                    # réindexe tout
"""
import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from common import DB_DIR, collection, embed
from llm import complete

MAX_CHARS = 1800        # ~450 tokens en français
MIN_CHARS = 30
MIN_IMAGE_BYTES = 5 * 1024
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
NOISE_TAGS = ["script", "style", "nav", "aside", "form", "noscript", "svg", "iframe"]


# ---------------------------------------------------------------- HTML -----
def html_to_md(html):
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("main") or soup.find("article")
    noise = list(NOISE_TAGS)
    if root is None:  # pas de <main> : on retire aussi bandeau et pied de page
        root = soup.body or soup
        noise += ["header", "footer"]
    for t in root.find_all(noise):
        t.decompose()
    images = [(img.get("src", ""), img.get("alt", "").strip()) for img in root.find_all("img")]
    text = markdownify(str(root), heading_style="ATX")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text, images


# ------------------------------------------------------------ Découpage -----
def split_long(text):
    """Découpe un texte trop long par paragraphes, avec petit chevauchement."""
    if len(text) <= MAX_CHARS:
        return [text]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > MAX_CHARS:  # paragraphe géant : coupe brute
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:MAX_CHARS])
            p = p[MAX_CHARS:]
        if cur and len(cur) + len(p) + 2 > MAX_CHARS:
            chunks.append(cur)
            cur = cur.split("\n\n")[-1] if len(cur.split("\n\n")[-1]) < 300 else ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur)
    return chunks


def markdown_chunks(md, title):
    """Un chunk par section (titres #), préfixé de 'titre > section'."""
    stack, buf, sections = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join([title] + [h for _, h in stack if h]), body))
        buf.clear()

    in_code = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = [(l, h) for l, h in stack if l < level] + [(level, m.group(2).strip())]
        else:
            buf.append(line)
    flush()

    out = []
    for path, body in sections:
        for part in split_long(body):
            if len(part.strip()) >= MIN_CHARS:
                out.append(f"{path}\n\n{part.strip()}")
    return out


# ------------------------------------------------------------------ URLs -----
def page_url(root, path, base):
    rel = path.relative_to(root).as_posix()
    if rel.endswith("index.html"):
        rel = rel[: -len("index.html")]
    return f"{base.rstrip('/')}/{rel}" if base else f"/{rel}"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- Images -----
def describe_image(path, cache):
    key = file_hash(path)
    if key in cache:
        return cache[key]
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    cache[key] = complete(
        "Décris cette image en français pour un moteur de recherche documentaire : "
        "sujet, éléments visibles, et transcris fidèlement tout texte visible. "
        "Si l'image est purement décorative (logo, icône, fond, séparateur), "
        "réponds exactement DECORATIF.",
        image_jpeg=buf.getvalue(), max_tokens=500,
    )
    return cache[key]


# ------------------------------------------------------------------ Main -----
def add_chunks(col, source, chunks, metas):
    for i in range(0, len(chunks), 64):
        part = chunks[i:i + 64]
        col.add(
            ids=[f"{source}#{i + j}" for j in range(len(part))],
            documents=part,
            embeddings=embed(part),
            metadatas=metas[i:i + 64],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site_dir")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--images", action="store_true", help="décrire les images avec le LLM configuré (doit être multimodal)")
    ap.add_argument("--full", action="store_true", help="ignore le manifeste et réindexe tout")
    ap.add_argument("--no-pdf", action="store_true", help="ignorer les fichiers PDF")
    ap.add_argument("--no-images", action="store_true",
                    help="ne pas décrire les images (déjà le cas par défaut ; l'emporte sur --images)")
    ap.add_argument("--exclude", action="append", default=[], metavar="MOTIF",
                    help="ignorer les fichiers dont le chemin relatif correspond au motif "
                         "(ex. --exclude 'docs/*' --exclude '*.pdf' ; option répétable)")
    ap.add_argument("--fast-pdf", action="store_true",
                    help="extraction PDF rapide (texte brut, sans analyse de mise en page ni OCR)")
    args = ap.parse_args()
    args.images = args.images and not args.no_images

    root = Path(args.site_dir).resolve()
    DB_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DB_DIR / "manifest.json"
    manifest = {} if args.full or not manifest_path.exists() else json.loads(manifest_path.read_text())
    cache_path = DB_DIR / "images_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    col = collection()
    if args.full:
        for key in set(m["source"] for m in col.get(include=["metadatas"])["metadatas"]):
            col.delete(where={"source": key})

    if not root.is_dir():
        sys.exit(f"Dossier introuvable : {root}\n"
                 "Indiquez le dossier du site construit (celui qui contient les .html), "
                 "par ex. : python index.py public --base-url https://monsite.fr")
    def excluded(path):
        rel = path.relative_to(root).as_posix()
        if args.no_pdf and path.suffix.lower() == ".pdf":
            return True
        return any(fnmatch.fnmatch(rel, pat) for pat in args.exclude)

    files, kept = [], {}   # kept : fichiers exclus, laissés tels quels dans l'index
    for dirpath, _dirs, names in os.walk(root, followlinks=True):
        for n in names:
            f = Path(dirpath) / n
            if f.suffix.lower() not in {".html", ".htm", ".pdf"}:
                continue
            if excluded(f):
                key = f.relative_to(root).as_posix()
                if key in manifest:
                    kept[key] = manifest[key]
            else:
                files.append(f)
    if args.no_pdf or args.exclude or args.no_images:
        print(f"Exclusions actives : {'PDF ' if args.no_pdf else ''}"
              f"{'images ' if args.no_images else ''}{args.exclude or ''}")
    n_html = sum(f.suffix.lower() in {".html", ".htm"} for f in files)
    print(f"Dossier : {root}\nFichiers trouvés : {n_html} HTML, {len(files) - n_html} PDF")
    if not files:
        sys.exit("Aucun fichier .html/.htm/.pdf trouvé dans ce dossier. Vérifiez qu'il s'agit du site "
                 "construit (jekyll build -> _site/ par défaut), et non des sources Markdown.")
    current = {}
    referenced_images = {}   # chemin image -> (url de la page, alt)
    stats = {"pages": 0, "pdf": 0, "images": 0, "chunks": 0, "pdf_sans_texte": 0}

    t0 = time.time()
    for idx, p in enumerate(sorted(files), 1):
        key = p.relative_to(root).as_posix()
        h = file_hash(p)
        current[key] = h
        is_html = p.suffix.lower() in {".html", ".htm"}

        # On relit toujours le HTML pour repérer les images, mais on n'indexe que s'il a changé.
        if is_html:
            title, md, imgs = html_to_md(p.read_text(encoding="utf-8", errors="ignore"))
            url = page_url(root, p, args.base_url)
            for src, alt in imgs:
                if not src or src.startswith(("http://", "https://", "data:")):
                    continue
                src = src.split("?")[0].split("#")[0]
                ip = (root / src.lstrip("/")) if src.startswith("/") else (p.parent / src)
                ip = ip.resolve()
                if ip.exists() and ip.suffix.lower() in IMAGE_EXT:
                    referenced_images.setdefault(ip, (url, alt))

        if manifest.get(key) == h:
            continue
        col.delete(where={"source": key})
        print(f"[{idx}/{len(files)}] {key}  ({time.time() - t0:.0f}s)", flush=True)

        if is_html:
            chunks = markdown_chunks(md, title or key)
            metas = [{"source": key, "url": url, "title": title, "type": "html", "page": 0}] * len(chunks)
            stats["pages"] += 1
        else:
            url = page_url(root, p, args.base_url)
            chunks, metas = [], []
            if args.fast_pdf:
                import pymupdf
                with pymupdf.open(str(p)) as doc:
                    pages = [{"text": pg.get_text("text")} for pg in doc]
            else:
                import pymupdf4llm
                pages = pymupdf4llm.to_markdown(str(p), page_chunks=True)
            print(f"    {len(pages)} pages extraites", flush=True)
            for n, page in enumerate(pages, 1):
                cs = markdown_chunks(page["text"], p.stem)
                if not cs:
                    stats["pdf_sans_texte"] += 1
                chunks += cs
                metas += [{"source": key, "url": f"{url}#page={n}", "title": p.stem,
                           "type": "pdf", "page": n}] * len(cs)
            stats["pdf"] += 1
        if chunks:
            print(f"    {len(chunks)} chunks -> calcul des embeddings...", flush=True)
            add_chunks(col, key, chunks, metas)
            stats["chunks"] += len(chunks)

    # Images (une seule fois chacune, rattachées à la première page qui les affiche)
    if args.images:
        for ip, (url, alt) in referenced_images.items():
            key = "img:" + ip.relative_to(root).as_posix()
            if excluded(ip):
                if key in manifest:
                    current[key] = manifest[key]
                continue
            if ip.stat().st_size < MIN_IMAGE_BYTES:
                continue
            h = file_hash(ip)
            current[key] = h
            if manifest.get(key) == h:
                continue
            col.delete(where={"source": key})
            desc = describe_image(ip, cache)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            if desc.strip().upper().startswith("DECORATIF"):
                continue
            text = f"[Image] {alt}\n\n{desc}".strip()
            add_chunks(col, key, [text], [{"source": key, "url": url, "title": alt or ip.name,
                                           "type": "image", "page": 0}])
            stats["images"] += 1
            stats["chunks"] += 1
    else:  # sans --images, on conserve l'état des images déjà indexées
        current.update({k: v for k, v in manifest.items() if k.startswith("img:")})

    current.update(kept)

    # Fichiers supprimés du site
    for key in set(manifest) - set(current):
        col.delete(where={"source": key})

    manifest_path.write_text(json.dumps(current, indent=1))
    print(f"Terminé : {stats}  |  total en base : {col.count()} chunks")
    if stats["pdf_sans_texte"]:
        print(f"Attention : {stats['pdf_sans_texte']} page(s) PDF sans texte (scan ? OCR nécessaire).")


if __name__ == "__main__":
    sys.exit(main())
