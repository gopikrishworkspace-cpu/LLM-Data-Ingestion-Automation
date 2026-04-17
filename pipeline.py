"""
pipeline.py — Three-step LLM extraction pipeline orchestrator.

Steps:
  Step 1: Extraction     — raw entities from each chunk (parallel)
  Step 2: Normalization  — link, normalize, assign domains (size-based batches)
  Step 3: Refinement     — deduplicate, set canonical status (hierarchical batching)

CRITICAL: No intermediate outputs are written to disk.
Steps 1–3 run entirely in memory.  ONLY Step 4 persists final
canonical entities via the storage module.

Key design choices:
  - Step 1 runs chunks in parallel via ThreadPoolExecutor.
  - Step 2 batches by serialized JSON *size* (not count) to stay
    within the LLM prompt budget.
  - Step 3 uses hierarchical batching: first refine within batches,
    then run a merge pass across batch results.
  - Every entity gets a deterministic entity_id derived from
    (entity_type, name, key attributes) so IDs are stable across runs.
  - LLM health check runs before any processing begins.

Usage::

    from pipeline import PipelineConfig, run_pipeline

    result = run_pipeline("/path/to/doc.pdf", PipelineConfig())
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chunking import Chunk, ChunkingConfig, chunk_file, chunk_pdf
from llm_client import LLMResponse, PromptTooLargeError, OpenAIClient
from storage import KnowledgeBase
from validation import (
    apply_type_corrections,
    apply_type_alias_corrections,
    check_chunk_quality,
    cluster_similar_entities,
    enrich_relationships,
    validate_step1,
    validate_step2,
    validate_step3,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_JSON_ENFORCEMENT = "You MUST output strictly valid JSON with no extra text."


def _load_prompt(path: str) -> str:
    """Load a prompt from a file, prepending the JSON enforcement line if absent."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _JSON_ENFORCEMENT not in text:
        text = _JSON_ENFORCEMENT + "\n\n" + text
    return text


_PROMPT_DIR = Path(__file__).parent / "prompt"

SYSTEM_STEP1_EXTRACTION  = _load_prompt(_PROMPT_DIR / "step1.txt")
SYSTEM_STEP2_NORMALIZATION = _load_prompt(_PROMPT_DIR / "step2.txt")
SYSTEM_STEP3_REFINEMENT  = _load_prompt(_PROMPT_DIR / "step3.txt")
SYSTEM_STEP3_MERGE       = _load_prompt(_PROMPT_DIR / "step4_merge.txt")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    knowledge_base_dir: str = "knowledge_base"
    # Size-based batching for Step 2: max chars of serialized JSON per batch
    step2_batch_max_chars: int = 12_000
    # Step 3 hierarchical batch count (entities per intra-batch refine)
    step3_batch_size: int = 20
    # Step 1 parallelism
    step1_max_workers: int = 4
    # If True, halt pipeline on first LLM failure; otherwise skip and continue
    fail_fast: bool = False
    # Validation retry budget per LLM call
    max_validation_retries: int = 2
    # Jaccard threshold for pre-dedup similarity clustering
    dedup_similarity_threshold: float = 0.5
    # LLM client — must be set before calling run_pipeline.
    llm_client: object | None = None


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Summary of a full pipeline run."""

    pdf_path: str
    total_chunks: int = 0
    step1_entities: int = 0
    step2_entities: int = 0
    step3_entities: int = 0
    persisted: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "pdf_path": self.pdf_path,
            "total_chunks": self.total_chunks,
            "step1_entities": self.step1_entities,
            "step2_entities": self.step2_entities,
            "step3_entities": self.step3_entities,
            "persisted": self.persisted,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Deterministic entity ID generation
# ---------------------------------------------------------------------------


def generate_entity_id(entity: dict) -> str:
    """Produce a stable, deterministic entity_id from identity fields.

    Hash inputs: entity_type + lowercased name + sorted key attributes.
    This ensures the same conceptual entity always receives the same ID
    regardless of extraction ordering or run timestamp.
    """
    etype = str(entity.get("entity_type", "unknown")).lower().strip()
    name = str(entity.get("name", "")).lower().strip()

    # Include sorted top-level attribute keys+values for disambiguation
    attrs = entity.get("attributes", {})
    attr_sig = ""
    if isinstance(attrs, dict):
        parts = sorted(f"{k}={v}" for k, v in attrs.items() if v is not None)
        attr_sig = "|".join(parts)

    payload = f"{etype}::{name}::{attr_sig}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    # Human-readable prefix
    prefix = etype.upper().replace(" ", "-")[:8]
    return f"{prefix}-{digest}"


def _stamp_deterministic_ids(entities: list[dict]) -> list[dict]:
    """Overwrite entity_id on every entity with a deterministic hash-based ID.

    Called after Step 1 so that all downstream steps operate on stable IDs.
    """
    for ent in entities:
        ent["entity_id"] = generate_entity_id(ent)
    return entities


# ---------------------------------------------------------------------------
# Size-based batch splitter (Step 2)
# ---------------------------------------------------------------------------


def _split_by_size(
    entities: list[dict],
    max_chars: int,
) -> list[list[dict]]:
    """Split entity list into batches where each batch's serialized
    JSON stays under ``max_chars``.

    Falls back to one-entity-per-batch if a single entity exceeds the limit.
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_size = 0

    for ent in entities:
        ent_size = len(json.dumps(ent, default=str))
        # If adding this entity would bust the budget, flush current batch
        if current_batch and (current_size + ent_size) > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(ent)
        current_size += ent_size

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Step 1: Parallel extraction
# ---------------------------------------------------------------------------


def _extract_one_chunk(
    chunk: Chunk,
    client: OpenAIClient,
    max_validation_retries: int = 2,
) -> tuple[list[dict], str | None]:
    """Extract entities from a single chunk. Returns (entities, error_or_None)."""
    # Guard: skip chunks that are too short or URL-only
    usable, reason = check_chunk_quality(chunk.text)
    if not usable:
        logger.warning(
            "Skipping chunk %d — %s", chunk.index, reason,
        )
        return [], f"Step1 chunk {chunk.index} skipped: {reason}"

    prompt = (
        f"Extract all entities from this document chunk "
        f"(chunk {chunk.index}, pages {chunk.page_start}–{chunk.page_end}):\n\n"
        f"{chunk.text}"
    )
    try:
        resp = client.generate_with_validation(
            prompt=prompt,
            system=SYSTEM_STEP1_EXTRACTION,
            entity_extractor=_safe_entity_list,
            validator=validate_step1,
            max_validation_retries=max_validation_retries,
            expect_json=True,
        )
    except PromptTooLargeError as e:
        return [], f"Step1 chunk {chunk.index}: {e}"

    if not resp.success or resp.parsed_json is None:
        return [], (
            f"Step1 chunk {chunk.index} failed: "
            f"{resp.error or 'no JSON returned'}"
        )

    entities = _safe_entity_list(resp.parsed_json)
    for ent in entities:
        ent["_source_chunk"] = chunk.index
        ent["_source_pages"] = f"{chunk.page_start}-{chunk.page_end}"
    return entities, None


def _run_step1(
    chunks: list[Chunk],
    client: OpenAIClient,
    config: PipelineConfig,
) -> tuple[list[dict], list[str]]:
    """Step 1: Extract raw entities from each chunk in parallel.

    Returns (flat entity list, error list). All in memory.
    """
    all_entities: list[dict] = []
    errors: list[str] = []

    max_workers = min(config.step1_max_workers, len(chunks))

    if max_workers <= 1:
        # Sequential fallback — simpler for debugging / single-chunk PDFs
        for chunk in chunks:
            ents, err = _extract_one_chunk(
                chunk, client, config.max_validation_retries,
            )
            if err:
                logger.warning(err)
                errors.append(err)
                if config.fail_fast:
                    raise RuntimeError(err)
            all_entities.extend(ents)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _extract_one_chunk, chunk, client,
                    config.max_validation_retries,
                ): chunk.index
                for chunk in chunks
            }
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    ents, err = future.result()
                except Exception as exc:
                    err = f"Step1 chunk {chunk_idx} exception: {exc}"
                    ents = []

                if err:
                    logger.warning(err)
                    errors.append(err)
                    if config.fail_fast:
                        raise RuntimeError(err)
                all_entities.extend(ents)

    logger.info(
        "Step 1 complete: %d entities from %d chunks (%d errors)",
        len(all_entities), len(chunks), len(errors),
    )
    return all_entities, errors


# ---------------------------------------------------------------------------
# Step 2: Size-based normalization batches
# ---------------------------------------------------------------------------


def _run_step2(
    raw_entities: list[dict],
    client: OpenAIClient,
    config: PipelineConfig,
) -> tuple[list[dict], list[str]]:
    """Step 2: Normalize and link entities in size-based batches.

    Returns (flat normalized list, error list). All in memory.
    """
    if not raw_entities:
        return [], []

    all_normalized: list[dict] = []
    errors: list[str] = []
    batches = _split_by_size(raw_entities, config.step2_batch_max_chars)

    for batch_num, batch in enumerate(batches, 1):
        prompt = (
            f"Normalize and link this batch of {len(batch)} raw entities "
            f"(batch {batch_num}/{len(batches)}):\n\n"
            f"{json.dumps(batch, indent=2, default=str)}"
        )
        try:
            resp = client.generate_with_validation(
                prompt=prompt,
                system=SYSTEM_STEP2_NORMALIZATION,
                entity_extractor=_safe_entity_list,
                validator=validate_step2,
                max_validation_retries=config.max_validation_retries,
                expect_json=True,
            )
        except PromptTooLargeError as e:
            msg = f"Step2 batch {batch_num}: {e}"
            logger.error(msg)
            errors.append(msg)
            if config.fail_fast:
                raise
            continue

        if not resp.success or resp.parsed_json is None:
            msg = (
                f"Step2 batch {batch_num} failed: "
                f"{resp.error or 'no JSON returned'}"
            )
            logger.warning(msg)
            errors.append(msg)
            if config.fail_fast:
                raise RuntimeError(msg)
            continue

        entities = _safe_entity_list(resp.parsed_json)
        all_normalized.extend(entities)
        logger.debug(
            "Step2 batch %d/%d → %d entities",
            batch_num, len(batches), len(entities),
        )

    logger.info(
        "Step 2 complete: %d normalized entities in %d batches (%d errors)",
        len(all_normalized), len(batches), len(errors),
    )
    return all_normalized, errors


# ---------------------------------------------------------------------------
# Step 3: Hierarchical batched refinement
# ---------------------------------------------------------------------------


def _refine_batch(
    entities: list[dict],
    client: OpenAIClient,
    system: str,
    label: str,
    config: PipelineConfig,
    validator=None,
) -> tuple[list[dict], str | None]:
    """Send a single refinement batch to the LLM. Returns (entities, error)."""
    prompt = (
        f"{label} — refine these {len(entities)} entities:\n\n"
        f"{json.dumps(entities, indent=2, default=str)}"
    )
    try:
        if validator is not None:
            resp = client.generate_with_validation(
                prompt=prompt,
                system=system,
                entity_extractor=_safe_entity_list,
                validator=validator,
                max_validation_retries=config.max_validation_retries,
                expect_json=True,
            )
        else:
            resp = client.generate(
                prompt=prompt,
                system=system,
                expect_json=True,
            )
    except PromptTooLargeError as e:
        return [], str(e)

    if not resp.success or resp.parsed_json is None:
        return [], resp.error or "no JSON returned"

    return _safe_entity_list(resp.parsed_json), None


def _run_step3(
    normalized_entities: list[dict],
    client: OpenAIClient,
    config: PipelineConfig,
) -> tuple[list[dict], list[str]]:
    """Step 3: Hybrid deduplication and refinement.

    Phase A — Cluster-based: use deterministic similarity clustering to
              group likely duplicates. Send multi-entity clusters to LLM
              for merge. Singletons pass through without an LLM call.
    Phase B — Cross-batch merge: take canonical entities from Phase A
              and run a merge pass to catch anything clustering missed.
              Skipped if Phase A produced one batch or fewer.

    Returns (flat refined list, error list). All in memory.
    """
    if not normalized_entities:
        return [], []

    errors: list[str] = []
    batch_size = config.step3_batch_size

    # --- Phase A: cluster-based deduplication ---
    clusters = cluster_similar_entities(
        normalized_entities, threshold=config.dedup_similarity_threshold,
    )

    phase_a_canonical: list[dict] = []
    cluster_batches_sent = 0

    for cluster_num, cluster in enumerate(clusters, 1):
        if len(cluster) == 1:
            # Singleton — no dedup needed, pass through
            ent = cluster[0]
            if not ent.get("status"):
                ent["status"] = "canonical"
            phase_a_canonical.append(ent)
            continue

        # Multi-entity cluster — send to LLM for merge/dedup
        label = f"Step3 Phase-A cluster {cluster_num}/{len(clusters)} ({len(cluster)} entities)"
        refined, err = _refine_batch(
            cluster, client, SYSTEM_STEP3_REFINEMENT, label, config,
            validator=validate_step3,
        )
        cluster_batches_sent += 1
        if err:
            msg = f"{label} failed: {err}"
            logger.warning(msg)
            errors.append(msg)
            if config.fail_fast:
                raise RuntimeError(msg)
            # Fallback: keep originals as canonical
            for ent in cluster:
                if not ent.get("status"):
                    ent["status"] = "canonical"
                phase_a_canonical.append(ent)
            continue

        canonical = [e for e in refined if e.get("status") == "canonical"]
        phase_a_canonical.extend(canonical)
        logger.debug(
            "%s → %d refined (%d canonical)",
            label, len(refined), len(canonical),
        )

    logger.info(
        "Step 3 Phase A: %d canonical entities from %d clusters "
        "(%d sent to LLM, %d singletons)",
        len(phase_a_canonical), len(clusters), cluster_batches_sent,
        len(clusters) - cluster_batches_sent,
    )

    if not phase_a_canonical:
        return [], errors

    # --- Phase B: cross-batch merge (skip if few clusters) ---
    if cluster_batches_sent <= 1:
        logger.info("Step 3 Phase B: skipped (<=1 cluster batch)")
        return phase_a_canonical, errors

    # Phase B may itself need batching for very large sets
    merge_batches = [
        phase_a_canonical[i : i + batch_size]
        for i in range(0, len(phase_a_canonical), batch_size)
    ]

    if len(merge_batches) == 1:
        label = "Step3 Phase-B merge"
        merged, err = _refine_batch(
            phase_a_canonical, client, SYSTEM_STEP3_MERGE, label, config,
            validator=validate_step3,
        )
        if err:
            msg = f"{label} failed: {err}"
            logger.warning(msg)
            errors.append(msg)
            return phase_a_canonical, errors
        final = [e for e in merged if e.get("status") == "canonical"]
    else:
        intermediate: list[dict] = []
        for mb_num, mb in enumerate(merge_batches, 1):
            label = f"Step3 Phase-B merge batch {mb_num}/{len(merge_batches)}"
            merged, err = _refine_batch(
                mb, client, SYSTEM_STEP3_MERGE, label, config,
                validator=validate_step3,
            )
            if err:
                msg = f"{label} failed: {err}"
                logger.warning(msg)
                errors.append(msg)
                intermediate.extend(mb)
                continue
            intermediate.extend(
                e for e in merged if e.get("status") == "canonical"
            )

        if len(intermediate) > batch_size:
            label = "Step3 Phase-B final merge"
            merged, err = _refine_batch(
                intermediate, client, SYSTEM_STEP3_MERGE, label, config,
                validator=validate_step3,
            )
            if err:
                logger.warning("Final merge failed (%s), using intermediates", err)
                errors.append(f"Step3 final merge: {err}")
                final = intermediate
            else:
                final = [e for e in merged if e.get("status") == "canonical"]
        else:
            final = intermediate

    logger.info(
        "Step 3 Phase B: %d canonical entities after cross-batch merge",
        len(final),
    )
    return final, errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_entity_list(data: Any) -> list[dict]:
    """Robustly extract a list of entity dicts from LLM JSON output."""
    _WRAPPER_KEYS = ("entities", "results", "data")
    # Fields that indicate a dict is a real entity (not a wrapper)
    _ENTITY_MARKERS = ("entity_id", "entity_type", "name")

    def _is_entity(d: dict) -> bool:
        return any(k in d for k in _ENTITY_MARKERS)

    if isinstance(data, dict):
        for key in _WRAPPER_KEYS:
            val = data.get(key)
            if isinstance(val, list):
                return [e for e in val if isinstance(e, dict)]
        if _is_entity(data):
            return [data]
        return []

    if isinstance(data, list):
        result: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if _is_entity(item):
                result.append(item)
            else:
                # Item might be a wrapper object (e.g. {"entities": [...]})
                for key in _WRAPPER_KEYS:
                    val = item.get(key)
                    if isinstance(val, list):
                        result.extend(e for e in val if isinstance(e, dict))
                        break
        return result

    return []


class LLMHealthCheckError(RuntimeError):
    """Raised when the LLM server is unreachable at pipeline start."""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    pdf_path: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full extraction pipeline on a PDF or plain-text file.

    Steps:
      Pre. LLM health check
      0.   Chunk file (PDF or .txt/.md/etc.)
      1.   LLM extraction (parallel, per chunk)
      1b.  Stamp deterministic entity IDs
      2.   LLM normalization & linking (size-based batches)
      3.   LLM deduplication & refinement (hierarchical batches)
      4.   Persist canonical entities to knowledge base

    All intermediate data (steps 1–3) lives in memory only.

    Raises:
        LLMHealthCheckError: if the LLM server is not reachable.
    """
    if config is None:
        config = PipelineConfig()

    pdf_path = Path(pdf_path)
    result = PipelineResult(pdf_path=str(pdf_path))
    start = time.time()

    if config.llm_client is None:
        raise LLMHealthCheckError("No llm_client set in PipelineConfig. Provide an OpenAIClient.")
    client = config.llm_client

    # --- Pre: health check ---
    base_url = getattr(getattr(client, 'config', None), 'base_url', '?')
    logger.info("Pre-flight: checking LLM server at %s", base_url)
    if not client.ping():
        msg = f"LLM server unreachable at {base_url}. Check credentials or base_url."
        logger.error(msg)
        raise LLMHealthCheckError(msg)
    logger.info("LLM server OK")

    # --- Step 0: chunk ---
    logger.info("Step 0: Chunking %s", pdf_path.name)
    try:
        chunks = chunk_file(pdf_path, config.chunking)
    except Exception as e:
        result.errors.append(f"Chunking failed: {e}")
        result.duration_ms = int((time.time() - start) * 1000)
        return result

    result.total_chunks = len(chunks)
    if not chunks:
        logger.warning("No chunks produced from %s", pdf_path.name)
        result.duration_ms = int((time.time() - start) * 1000)
        return result

    # --- Step 1: extraction (in memory, parallel) ---
    logger.info("Step 1: Extraction (%d chunks, %d workers)",
                len(chunks), config.step1_max_workers)
    raw_entities, step1_errors = _run_step1(chunks, client, config)
    result.errors.extend(step1_errors)

    # Stamp deterministic IDs
    _stamp_deterministic_ids(raw_entities)

    # Post-Step 1: rule-based entity type correction (metric overuse + alias)
    apply_type_corrections(raw_entities)
    apply_type_alias_corrections(raw_entities)

    result.step1_entities = len(raw_entities)

    if not raw_entities:
        logger.warning("Step 1 produced no entities; aborting pipeline")
        result.duration_ms = int((time.time() - start) * 1000)
        return result

    # --- Step 2: normalization (in memory, size-based batches) ---
    logger.info("Step 2: Normalization (%d raw entities)", len(raw_entities))
    normalized, step2_errors = _run_step2(raw_entities, client, config)
    result.errors.extend(step2_errors)
    result.step2_entities = len(normalized)

    del raw_entities  # discard intermediate

    if not normalized:
        logger.warning("Step 2 produced no entities; aborting pipeline")
        result.duration_ms = int((time.time() - start) * 1000)
        return result

    # Post-Step 2: type correction + alias fix + relationship enrichment
    apply_type_corrections(normalized)
    apply_type_alias_corrections(normalized)
    enrich_relationships(normalized)

    # --- Step 3: refinement (in memory, hybrid clustering) ---
    logger.info("Step 3: Refinement (%d normalized entities)", len(normalized))
    refined, step3_errors = _run_step3(normalized, client, config)
    result.errors.extend(step3_errors)
    result.step3_entities = len(refined)

    del normalized  # discard intermediate

    if not refined:
        logger.warning("Step 3 produced no entities; nothing to persist")
        result.duration_ms = int((time.time() - start) * 1000)
        return result

    # --- Step 4: persist ONLY canonical entities ---
    logger.info("Step 4: Persisting canonical entities")
    kb = KnowledgeBase(config.knowledge_base_dir)
    persist_stats = kb.persist_entities(refined)
    result.persisted = persist_stats

    del refined  # discard intermediate

    result.duration_ms = int((time.time() - start) * 1000)
    logger.info(
        "Pipeline complete for %s in %dms — %s",
        pdf_path.name, result.duration_ms, persist_stats,
    )
    return result
