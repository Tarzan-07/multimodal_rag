import uuid, pdfplumber
from pathlib import Path

import fitz
from psycopg2.extras import Json
from .db import get_conn
from .embeddings import embed_text
from .chunking import split_text_into_chunks
from .caption_vlm import caption_img_png
import sys

PUBLIC_THUMBS = Path('samples/images')
PUBLIC_THUMBS.mkdir(parents=True, exist_ok=True)

def insert(sql, params):
    with get_conn() as  conn, conn.cursor() as cur:
        cur.execute(sql,params)

def save_thumb(pix: fitz.Pixmap, name: str) -> str:
    out = PUBLIC_THUMBS / f"{name}.png"
    pix.save(str(out))
    return f"/thumbs/{name}.png"

def ingest_pdf(path: str):
    doc_id = uuid.uuid4()
    insert("INSERT INTO documents(doc_id, title, source_uri) VALUES (%s,%s,%s)",
           (str(doc_id), str(Path).name, str(Path(path).resolve())))
    print(f"Ingested {path}")

    with pdfplumber.open(path) as pdf:
        for pnum, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for chunk in split_text_into_chunks(text):
                emb = embed_text(chunk)
                insert("INSERT INTO chunks(chunk_id, doc_id, page, section_path, text, embedding, meta)"\
                       "Values (%s,%s,%s,%s,%s,%s,%s)",
                       (str(uuid.uuid4()), str(doc_id), pnum, None, chunk, emb, None)
                    )
                
    pdf_doc = fitz.open(path)
    for pnum, page in enumerate(pdf_doc, start=1):
        images = page.get_images(full=True)
        if not images:
            continue
        for idx, img in enumerate(images, start=1):
            xref = img[0]
            try:
                base = pdf_doc.extract_image(xref)
                image_bytes = base['image']
                image_ext = base.get("ext", "png")

                doc = fitz.open(stream=image_bytes, filetype=image_ext)
                page = doc[0]
                pix = page.get_pixmap()
                thumb_uri = save_thumb(pix, f"{doc_id}_{pnum}_{idx}")

                cap = {}
                try:
                    cap = caption_img_png(image_bytes)
                except Exception as e:
                    print(e)
                    cap = {
                        "global_caption_short": "(caption failed)",
                        "global_caption_dense": "",
                        "text_in_image": [],
                        "entites": [],
                        "tags": ["uncaptioned"],
                    }

                caption_dense = cap.get("global_caption_dense", "")
                ocr_text = "".join(cap.get("text_in_image", [])or [])
                emb = embed_text(caption_dense or cap.get("global_caption_short", ""))

                insert(
                    """
                    INSERT INTO figures(figure_id, doc_id, page, bbox, caption_short, caption_dense, ocr_text, thumb_uri, embedding, meta)\
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        str(uuid.uuid4()),
                        str(doc_id),
                        pnum,
                        None,
                        cap.get("global_caption_short"),
                        caption_dense,
                        ocr_text,
                        thumb_uri,
                        emb,
                        Json(cap),
                    ),
                )
            except Exception as e:
                print(e)
                continue

    pdf_doc.close()            

if __name__ == "__main__":
    print(sys.argv[1])
    ingest_pdf(sys.argv[1])