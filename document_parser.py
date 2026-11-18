"""
document_parser.py
==================
Universal Multi-Format Document & OCR Extraction Engine for JobShield AI.

Supports:
  - PDF Documents (.pdf): Digital text + table extraction via pdfplumber & pypdf,
    with automatic high-DPI page rendering (pypdfium2) + Tesseract OCR fallback for scanned PDFs.
  - Images (.png, .jpg, .jpeg, .webp, .bmp, .tiff, .tif, .gif): OpenCV + Pillow
    adaptive contrast enhancement, upscaling, denoising, and Tesseract OCR extraction.
  - Word & Rich Text Documents (.docx, .doc, .rtf, .odt): Full paragraph, header/footer,
    and table extraction via python-docx, striprtf, XML parsing, and LibreOffice fallback.
  - Text, Web & Email Files (.txt, .md, .html, .htm, .eml, .json, .xml, .rst, .log):
    Multi-encoding decoding, BeautifulSoup/html2text cleanup, and RFC-822 email parsing.
  - Spreadsheets & Tabular Data (.csv, .tsv, .xlsx, .xls): Single-posting synthesis and
    multi-row batch job posting extraction via pandas & openpyxl.
  - Universal Any-Document Fallback: ZIP/XML container inspection (e.g., .pptx, .epub)
    and binary printable string recovery.
"""

import io
import os
import re
import json
import email
import zipfile
import tempfile
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

try:
    import docx
except ImportError:
    docx = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
WORD_EXTENSIONS = {".docx", ".doc", ".rtf", ".odt"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".yaml", ".yml"}
WEB_EXTENSIONS = {".html", ".htm", ".xml"}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
EMAIL_EXTENSIONS = {".eml"}


def detect_title_and_company(text: str, filename: str = "") -> tuple[str, str]:
    """Heuristically extracts job title and company name from extracted document text."""
    if not text or not text.strip():
        stem = Path(filename).stem.replace("_", " ").replace("-", " ").title() if filename else "Untitled Posting"
        return stem, "Unknown Organization"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    job_title = ""
    company_name = ""

    # Look for explicit labels first
    for ln in lines[:25]:
        m_title = re.match(r"^(?:job\s*title|position|role|title|designation|opening)\s*[:\-–]\s*(.+)$", ln, re.I)
        if m_title and not job_title:
            job_title = m_title.group(1).strip()[:90]
        m_comp = re.match(r"^(?:company|organization|employer|firm|company\s*name|hiring\s*organization)\s*[:\-–]\s*(.+)$", ln, re.I)
        if m_comp and not company_name:
            company_name = m_comp.group(1).strip()[:90]

    if not job_title:
        # Use first non-trivial short line as title
        for ln in lines[:8]:
            clean_ln = re.sub(r"\s+", " ", ln).strip()
            if 4 <= len(clean_ln) <= 85 and not clean_ln.lower().startswith(("http", "dear ", "date:", "from:")):
                job_title = clean_ln
                break
        if not job_title:
            job_title = " ".join(text.split()[:7]) + ("..." if len(text.split()) > 7 else "")

    if not company_name:
        # Check for "at <Company>" or "<Company> is hiring"
        m_at = re.search(r"\b(?:at|with|join)\s+([A-Z][A-Za-z0-9&\.\-\s]{2,35}?)(?:\s+is\s+|\s+in\s+|[\.,\n\|])", text)
        if m_at:
            company_name = m_at.group(1).strip()
        else:
            company_name = "Not Explicitly Listed"

    return job_title[:90], company_name[:90]


def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Enhances image contrast, resolution, and sharpness for high-accuracy OCR."""
    img = pil_img.convert("RGB")
    w, h = img.size
    # Upscale small screenshots for better Tesseract recognition
    if max(w, h) < 1200:
        scale = 2.0
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)

    if cv2 is not None:
        arr = np.array(gray)
        # Apply gentle bilateral filter to keep edges sharp while removing noise
        denoised = cv2.bilateralFilter(arr, 5, 40, 40)
        return Image.fromarray(denoised)
    else:
        return gray.filter(ImageFilter.SHARPEN)


def extract_text_from_image_bytes(file_bytes: bytes, filename: str = "image.png") -> dict:
    """Runs OCR on an image file and returns extracted text plus metadata."""
    pil_img = Image.open(io.BytesIO(file_bytes))
    width, height = pil_img.size
    img_format = pil_img.format or Path(filename).suffix.upper().lstrip(".")

    ocr_text = ""
    method = "Image OCR (Tesseract)"
    if pytesseract is not None:
        try:
            proc_img = preprocess_image_for_ocr(pil_img)
            ocr_text = pytesseract.image_to_string(proc_img, config="--oem 3 --psm 6")
            if len(ocr_text.strip()) < 15:
                # Fallback to sparse text page segmentation mode
                ocr_text = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 11")
        except Exception as e:
            method = f"Image OCR Fallback ({e})"

    ocr_text = ocr_text.strip()
    return {
        "text": ocr_text,
        "source_type": "Image (OCR)",
        "extraction_method": method,
        "metadata": {
            "dimensions": f"{width}x{height} px",
            "format": img_format,
            "ocr_characters": len(ocr_text),
        },
    }


def extract_text_from_pdf_bytes(file_bytes: bytes, filename: str = "document.pdf") -> dict:
    """
    Extracts text and tables from a PDF file.
    Automatically falls back to page rendering + Tesseract OCR for scanned pages.
    """
    pages_text = []
    ocr_pages = 0
    total_pages = 0
    tables_extracted = 0

    # 1. Try pdfplumber first for rich layout + table extraction
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                total_pages = len(pdf.pages)
                for page in pdf.pages:
                    p_txt = (page.extract_text() or "").strip()
                    # Also extract tables if present
                    try:
                        tables = page.extract_tables() or []
                        for tbl in tables:
                            tables_extracted += 1
                            for row in tbl:
                                row_str = " | ".join(str(cell).strip() for cell in row if cell)
                                if row_str and row_str not in p_txt:
                                    p_txt += "\n" + row_str
                    except Exception:
                        pass
                    pages_text.append(p_txt)
        except Exception:
            pages_text = []

    # 2. Fallback to pypdf if pdfplumber didn't populate pages
    if not pages_text and pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            for page in reader.pages:
                pages_text.append((page.extract_text() or "").strip())
        except Exception:
            pass

    # 3. Check for scanned / image-only pages and run OCR via pypdfium2 + pytesseract
    if pdfium is not None and pytesseract is not None:
        try:
            pdf_doc = pdfium.PdfDocument(file_bytes)
            if total_pages == 0:
                total_pages = len(pdf_doc)
                pages_text = [""] * total_pages
            for i in range(min(len(pdf_doc), 15)):
                if i >= len(pages_text) or len(pages_text[i].strip()) < 25:
                    page = pdf_doc[i]
                    bitmap = page.render(scale=2.2)
                    pil_page = bitmap.to_pil()
                    proc_page = preprocess_image_for_ocr(pil_page)
                    ocr_txt = pytesseract.image_to_string(proc_page, config="--oem 3 --psm 6").strip()
                    if ocr_txt:
                        if i < len(pages_text):
                            pages_text[i] = (pages_text[i] + "\n" + ocr_txt).strip()
                        else:
                            pages_text.append(ocr_txt)
                        ocr_pages += 1
        except Exception:
            pass

    combined_text = "\n\n".join(p for p in pages_text if p)
    method = "PDF Digital Text + Table Parser"
    if ocr_pages > 0:
        method = f"PDF Hybrid Parser ({ocr_pages}/{total_pages} page(s) via OCR)"

    return {
        "text": combined_text.strip(),
        "source_type": "PDF Document",
        "extraction_method": method,
        "metadata": {
            "total_pages": total_pages,
            "ocr_pages": ocr_pages,
            "tables_extracted": tables_extracted,
        },
    }


def extract_text_from_word_bytes(file_bytes: bytes, filename: str = "document.docx") -> dict:
    """Extracts text, headers, and tables from .docx, .doc, .rtf, and .odt files."""
    ext = Path(filename).suffix.lower()
    paragraphs = []
    tables_count = 0
    method = "Word Document Parser"

    if ext == ".docx" and docx is not None:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            method = "Microsoft Word (.docx) Parser"
            # Headers & footers
            for section in doc.sections:
                for hp in section.header.paragraphs:
                    if hp.text.strip():
                        paragraphs.append(hp.text.strip())
            # Main body paragraphs
            for p in doc.paragraphs:
                if p.text.strip():
                    paragraphs.append(p.text.strip())
            # Tables
            for tbl in doc.tables:
                tables_count += 1
                for row in tbl.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    # Deduplicate merged adjacent cells
                    dedup_cells = []
                    for c in cells:
                        if not dedup_cells or dedup_cells[-1] != c:
                            dedup_cells.append(c)
                    if dedup_cells:
                        paragraphs.append(" | ".join(dedup_cells))
        except Exception as e:
            method = f"DOCX XML Fallback ({e})"

    elif ext == ".rtf" and rtf_to_text is not None:
        method = "Rich Text Format (.rtf) Parser"
        raw_str = file_bytes.decode("utf-8", errors="ignore")
        paragraphs.append(rtf_to_text(raw_str))

    elif ext == ".odt":
        method = "OpenDocument (.odt) XML Parser"
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if "content.xml" in zf.namelist():
                    xml_bytes = zf.read("content.xml")
                    root = ET.fromstring(xml_bytes)
                    paragraphs.append(" ".join(root.itertext()))
        except Exception:
            pass

    # Fallback for .doc or failed containers using LibreOffice or ZIP XML
    if not paragraphs:
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                for name in zf.namelist():
                    if name.endswith(".xml"):
                        root = ET.fromstring(zf.read(name))
                        txt = " ".join(root.itertext()).strip()
                        if txt:
                            paragraphs.append(txt)
                method = "Office XML Container Parser"
        except Exception:
            pass

    if not paragraphs:
        try:
            with tempfile.TemporaryDirectory(prefix="js_doc_") as tmpdir:
                in_path = os.path.join(tmpdir, filename)
                with open(in_path, "wb") as f:
                    f.write(file_bytes)
                lo_prof = os.path.join(tmpdir, "lo_profile")
                subprocess.run(
                    [
                        "libreoffice",
                        f"-env:UserInstallation=file://{lo_prof}",
                        "--headless",
                        "--convert-to",
                        "txt:Text",
                        "--outdir",
                        tmpdir,
                        in_path,
                    ],
                    env={**os.environ, "HOME": "/tmp", "TMPDIR": "/tmp"},
                    timeout=30,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                txt_files = list(Path(tmpdir).glob("*.txt"))
                if txt_files:
                    paragraphs.append(txt_files[0].read_text(encoding="utf-8", errors="ignore"))
                    method = "LibreOffice Headless Document Converter"
        except Exception:
            pass

    if not paragraphs:
        # Recover printable strings
        decoded = file_bytes.decode("utf-8", errors="ignore")
        clean_runs = re.findall(r"[A-Za-z0-9\s\.,;:\-\(\)/\$@!'\"]{8,}", decoded)
        paragraphs.append("\n".join(clean_runs))

    combined = "\n".join(paragraphs).strip()
    return {
        "text": combined,
        "source_type": f"Word/Document ({ext or '.docx'})",
        "extraction_method": method,
        "metadata": {"tables_extracted": tables_count, "paragraphs": len(paragraphs)},
    }


def decode_text_bytes(file_bytes: bytes) -> tuple[str, str]:
    """Decodes text bytes trying common encodings."""
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace"), "utf-8-replace"


def parse_uploaded_document(file_bytes: bytes, filename: str) -> dict:
    """
    Master router that extracts clean text and metadata from ANY uploaded document:
    PDF, Image (PNG/JPG/WEBP/BMP/TIFF), Word (.docx/.doc/.rtf/.odt), TXT/MD,
    HTML/XML, Email (.eml), JSON, Excel/CSV (.xlsx/.csv), PowerPoint (.pptx), etc.
    """
    ext = Path(filename).suffix.lower()
    file_size_kb = round(len(file_bytes) / 1024.0, 2)

    if ext == ".pdf" or file_bytes[:4] == b"%PDF":
        res = extract_text_from_pdf_bytes(file_bytes, filename)
    elif ext in IMAGE_EXTENSIONS:
        res = extract_text_from_image_bytes(file_bytes, filename)
    elif ext in WORD_EXTENSIONS:
        res = extract_text_from_word_bytes(file_bytes, filename)
    elif ext in EMAIL_EXTENSIONS:
        msg = email.message_from_bytes(file_bytes)
        subj = str(msg.get("Subject", ""))
        sender = str(msg.get("From", ""))
        body_parts = [f"Subject: {subj}", f"From: {sender}"]
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(decode_text_bytes(payload)[0])
                elif ctype == "text/html" and BeautifulSoup is not None:
                    payload = part.get_payload(decode=True)
                    if payload:
                        soup = BeautifulSoup(decode_text_bytes(payload)[0], "lxml")
                        body_parts.append(soup.get_text(separator="\n"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(decode_text_bytes(payload)[0])
        res = {
            "text": "\n\n".join(body_parts).strip(),
            "source_type": "Email (.eml)",
            "extraction_method": "RFC-822 Email Parser",
            "metadata": {"sender": sender, "subject": subj},
        }
    elif ext in WEB_EXTENSIONS:
        raw_html, enc = decode_text_bytes(file_bytes)
        if BeautifulSoup is not None:
            soup = BeautifulSoup(raw_html, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            clean_txt = soup.get_text(separator="\n")
        else:
            clean_txt = re.sub(r"<[^>]+>", " ", raw_html)
        clean_txt = re.sub(r"\n{3,}", "\n\n", clean_txt).strip()
        res = {
            "text": clean_txt,
            "source_type": f"Web/Markup ({ext})",
            "extraction_method": f"HTML/DOM Parser ({enc})",
            "metadata": {"encoding": enc},
        }
    elif ext == ".json":
        raw_json, enc = decode_text_bytes(file_bytes)
        try:
            obj = json.loads(raw_json)
            if isinstance(obj, dict):
                lines = [f"{k}: {v}" for k, v in obj.items() if v is not None]
                clean_txt = "\n".join(lines)
            else:
                clean_txt = json.dumps(obj, indent=2)
        except Exception:
            clean_txt = raw_json
        res = {
            "text": clean_txt.strip(),
            "source_type": "JSON Document",
            "extraction_method": "Structured JSON Parser",
            "metadata": {"encoding": enc},
        }
    elif ext in TABULAR_EXTENSIONS:
        try:
            if ext in {".xlsx", ".xls"}:
                df_tab = pd.read_excel(io.BytesIO(file_bytes))
            elif ext == ".tsv":
                df_tab = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
            else:
                df_tab = pd.read_csv(io.BytesIO(file_bytes))
            rows_preview = []
            for idx, row in df_tab.head(25).iterrows():
                row_items = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip() != ""]
                rows_preview.append(" | ".join(row_items))
            clean_txt = "\n\n".join(rows_preview)
            res = {
                "text": clean_txt.strip(),
                "source_type": f"Spreadsheet ({ext})",
                "extraction_method": f"Tabular Data Parser ({len(df_tab)} rows, {len(df_tab.columns)} cols)",
                "metadata": {"rows": int(len(df_tab)), "columns": list(df_tab.columns)},
                "dataframe": df_tab,
            }
        except Exception as e:
            raw_txt, enc = decode_text_bytes(file_bytes)
            res = {
                "text": raw_txt.strip(),
                "source_type": f"Tabular Text ({ext})",
                "extraction_method": f"Fallback Text Decoder ({e})",
                "metadata": {},
            }
    elif ext in TEXT_EXTENSIONS:
        raw_txt, enc = decode_text_bytes(file_bytes)
        res = {
            "text": raw_txt.strip(),
            "source_type": f"Text File ({ext})",
            "extraction_method": f"UTF/Text Decoder ({enc})",
            "metadata": {"encoding": enc},
        }
    else:
        # Universal fallback: check if ZIP container (e.g., .pptx, .epub, etc.) or text/binary
        extracted_chunks = []
        method = "Universal Document Inspector"
        try:
            if zipfile.is_zipfile(io.BytesIO(file_bytes)):
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    for name in zf.namelist():
                        if name.endswith((".xml", ".txt", ".html", ".xhtml", ".md")):
                            data = zf.read(name)
                            try:
                                root = ET.fromstring(data)
                                t = " ".join(root.itertext()).strip()
                                if t:
                                    extracted_chunks.append(t)
                            except Exception:
                                extracted_chunks.append(decode_text_bytes(data)[0])
                method = f"Archive/XML Container Extractor ({ext or 'binary'})"
        except Exception:
            pass

        if not extracted_chunks:
            raw_txt, enc = decode_text_bytes(file_bytes)
            extracted_chunks.append(raw_txt)
            method = f"Universal Stream Decoder ({enc})"

        res = {
            "text": "\n".join(extracted_chunks).strip(),
            "source_type": f"Document ({ext or 'raw'})",
            "extraction_method": method,
            "metadata": {},
        }

    text_out = res.get("text", "")
    title_det, comp_det = detect_title_and_company(text_out, filename)
    res["file_name"] = filename
    res["file_size_kb"] = file_size_kb
    res["job_title_detected"] = title_det
    res["company_detected"] = comp_det
    res["word_count"] = len(text_out.split())
    res["char_count"] = len(text_out)
    return res
