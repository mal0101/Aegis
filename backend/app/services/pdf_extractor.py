import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _sanitize_id(filename: str) -> str:
    """Convert filename to a clean, short chunk ID prefix."""
    stem = Path(filename).stem
    clean = re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')
    if len(clean) > 50:
        clean = clean[:50].rstrip('_')
    return clean


@dataclass
class PolicyChunk:
    """A single chunk of policy text with full metadata."""
    chunk_id: str
    text: str                           # The actual text content
    enriched_text: str                  # Context-prefixed text for embedding
    source_file: str                    # PDF filename
    country: str
    policy_name: str
    enacted_date: str
    policy_type: str
    language: str
    legal_system: str
    section_header: str = ""
    tags: list = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0


def extract_pdf_to_markdown(pdf_path: str) -> str:
    """Extract a PDF to structured markdown using pymupdf4llm."""
    import pymupdf4llm
    logger.info(f"Extracting PDF: {pdf_path}")
    markdown = pymupdf4llm.to_markdown(pdf_path)

    if len(markdown.strip()) < 100:
        logger.warning(f"Extraction produced very little text ({len(markdown)} chars) from {Path(pdf_path).name} — may be scanned/image PDF")

    logger.info(f"Extracted {len(markdown)} chars from {Path(pdf_path).name}")
    return markdown


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ~ 4 chars for English, 3 for French."""
    return len(text) // 4


def chunk_policy_document(
    markdown: str,
    manifest_entry: dict,
    min_tokens: int = 128,
    max_tokens: int = 512
) -> list[PolicyChunk]:
    """
    Structure-aware chunking for policy documents.

    Strategy:
    1. Split on markdown headers (articles, sections, chapters)
    2. If a section exceeds max_tokens, split at paragraph boundaries
    3. Merge tiny sections with their neighbors
    4. Prepend jurisdictional context to each chunk for embedding
    5. Never split mid-paragraph
    """
    filename = manifest_entry["filename"]
    country = manifest_entry["country"]
    policy_name = manifest_entry["policy_name"]
    enacted_date = manifest_entry.get("enacted_date", "unknown")
    policy_type = manifest_entry.get("policy_type", "unknown")
    language = manifest_entry.get("language", "en")
    legal_system = manifest_entry.get("legal_system", "unknown")
    tags = manifest_entry.get("tags", [])

    # Step 1: Split on markdown headers (##, ###, ####) or numbered articles
    sections = re.split(r'(?=\n#{1,4}\s+)', markdown)
    refined_sections = []
    for section in sections:
        if estimate_tokens(section) > max_tokens * 2:
            subsections = re.split(
                r'(?=\n(?:Article|Section|Chapter|Part|Clause|Title|ARTICLE|SECTION|CHAPTER|PART|CLAUSE|TITLE)\s+\d+)',
                section
            )
            refined_sections.extend(subsections)
        else:
            refined_sections.append(section)

    # Step 2: Process each section into chunks
    raw_chunks = []
    for section in refined_sections:
        section = section.strip()
        if not section or estimate_tokens(section) < 20:
            continue

        header_match = re.match(r'^(#{1,4}\s+.+?)$', section, re.MULTILINE)
        section_header = header_match.group(1).strip('# ').strip() if header_match else ""

        if estimate_tokens(section) <= max_tokens:
            raw_chunks.append((section, section_header))
        else:
            paragraphs = section.split('\n\n')
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if estimate_tokens(current + "\n\n" + para) > max_tokens and current:
                    raw_chunks.append((current.strip(), section_header))
                    current = para
                else:
                    current = (current + "\n\n" + para).strip()
            if current.strip():
                raw_chunks.append((current.strip(), section_header))

    # Step 3: Merge tiny chunks with neighbors
    merged_chunks = []
    buffer = ""
    buffer_header = ""
    for text, header in raw_chunks:
        if estimate_tokens(buffer + "\n\n" + text) <= max_tokens:
            buffer = (buffer + "\n\n" + text).strip()
            buffer_header = buffer_header or header
        else:
            if buffer:
                merged_chunks.append((buffer, buffer_header))
            buffer = text
            buffer_header = header
    if buffer:
        merged_chunks.append((buffer, buffer_header))

    # Step 4: Build PolicyChunk objects with enriched text
    context_prefix = f"[Country: {country}] [Policy: {policy_name}] [Enacted: {enacted_date}] [Type: {policy_type}] [Legal System: {legal_system}]"

    chunks = []
    for i, (text, header) in enumerate(merged_chunks):
        enriched = f"{context_prefix}\n[Section: {header}]\n\n{text}" if header else f"{context_prefix}\n\n{text}"

        chunks.append(PolicyChunk(
            chunk_id=f"{_sanitize_id(filename)}_chunk_{i:04d}",
            text=text,
            enriched_text=enriched,
            source_file=filename,
            country=country,
            policy_name=policy_name,
            enacted_date=enacted_date,
            policy_type=policy_type,
            language=language,
            legal_system=legal_system,
            section_header=header,
            tags=tags,
            chunk_index=i,
            total_chunks=len(merged_chunks)
        ))

    for c in chunks:
        c.total_chunks = len(chunks)

    logger.info(f"Chunked {filename}: {len(chunks)} chunks from {len(markdown)} chars")
    return chunks


def process_all_pdfs(pdf_dir: str, manifest: dict) -> list[PolicyChunk]:
    """Process all PDFs listed in the manifest."""
    from app.config import CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS

    all_chunks = []
    pdf_path = Path(pdf_dir)

    for entry in manifest.get("documents", []):
        filepath = pdf_path / entry["filename"]
        if not filepath.exists():
            logger.warning(f"PDF not found: {filepath} — skipping")
            continue

        try:
            markdown = extract_pdf_to_markdown(str(filepath))
            chunks = chunk_policy_document(
                markdown, entry,
                min_tokens=CHUNK_MIN_TOKENS,
                max_tokens=CHUNK_MAX_TOKENS
            )
            all_chunks.extend(chunks)
            logger.info(f"Processed {entry['filename']}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process {entry['filename']}: {e}")

    logger.info(f"Total PDF chunks: {len(all_chunks)}")
    return all_chunks
