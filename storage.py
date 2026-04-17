"""
storage.py — Persistent knowledge base storage engine.

Responsibilities:
  - Atomic state management (registry + counter in one file)
  - Confidence-based field merging
  - Canonical-only persistence gate
  - Entity type / domain conflict detection
  - Schema validation before persistence
  - Global sequential file numbering across runs

Path layout:  knowledge_base/{entity_type}/{primary_domain}/{NNN}_{entity_id}.json
Registry key: {primary_domain}::{entity_type}::{entity_id}
Domain field: list[str] — an entity can belong to multiple domains.
              The *primary* domain (first element) determines the storage path.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-platform file locking
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    _WIN_LOCK = threading.Lock()

    def _acquire_file_lock(fd: int) -> None:  # noqa: ARG001
        _WIN_LOCK.acquire()

    def _release_file_lock(fd: int) -> None:  # noqa: ARG001
        _WIN_LOCK.release()
else:
    import fcntl

    def _acquire_file_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release_file_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    # original types
    "pattern", "ingestion", "analytics", "metric", "architecture", "code",
    # extended Orion types (from step1 prompt)
    "product_theory", "product", "implementation_ops", "comparison_example",
    "negative_boundary", "expert_heuristic", "failure_case", "evolution_record",
    "non_goal", "data_quality_exception", "field_feedback",
    "security_interpretation", "acceptance_criteria", "terminology",
    # truncated variants produced by Mistral
    "implementation", "comparison",
})

STATE_FILE = "index/state.json"
CONFLICTS_FILE = "index/conflicts.json"
LOCK_FILE = ".lock"

# Fields that every entity MUST carry (non-empty).
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "entity_id": str,
    "entity_type": str,
    "domain": list,
    "status": str,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _primary_domain(domains: list[str]) -> str:
    """First element is the canonical storage domain."""
    return domains[0]


def _registry_key(domain: str, entity_type: str, entity_id: str) -> str:
    """Three-part composite key: domain::entity_type::entity_id."""
    return f"{domain}::{entity_type}::{entity_id}"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when an entity fails pre-persistence validation."""


def validate_entity(entity: dict) -> list[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: list[str] = []

    for field, expected_type in REQUIRED_FIELDS.items():
        val = entity.get(field)
        if val is None:
            errors.append(f"Missing required field '{field}'")
        elif not isinstance(val, expected_type):
            errors.append(
                f"Field '{field}' must be {expected_type.__name__}, "
                f"got {type(val).__name__}"
            )

    # domain must be a non-empty list of non-empty strings
    domain = entity.get("domain")
    if isinstance(domain, list):
        if len(domain) == 0:
            errors.append("Field 'domain' must be a non-empty list")
        for i, d in enumerate(domain):
            if not isinstance(d, str) or not d.strip():
                errors.append(f"domain[{i}] must be a non-empty string")

    # entity_type must be in the allowed set
    etype = entity.get("entity_type")
    if isinstance(etype, str) and etype not in VALID_ENTITY_TYPES:
        errors.append(
            f"entity_type '{etype}' not in {sorted(VALID_ENTITY_TYPES)}"
        )

    # entity_id must be a non-empty string with no path separators
    eid = entity.get("entity_id")
    if isinstance(eid, str):
        if not eid.strip():
            errors.append("entity_id must be non-empty")
        if "/" in eid or "\\" in eid:
            errors.append("entity_id must not contain path separators")

    # status must be a non-empty string
    status = entity.get("status")
    if isinstance(status, str) and not status.strip():
        errors.append("status must be non-empty")

    return errors


# ---------------------------------------------------------------------------
# Confidence-aware merge engine
# ---------------------------------------------------------------------------


class MergeEngine:
    """Merges two entity dicts using confidence-based rules.

    Rules:
      - Scalars: higher confidence wins; ties -> newer (incoming) wins.
      - Lists: union, deduplicated, order-preserving.
      - Nested dicts: recursive merge.
      - Special field ``confidence_scores`` maps field names -> float.
      - ``domain`` list: always union-merged (domain is additive).
    """

    @staticmethod
    def merge(existing: dict, incoming: dict) -> dict:
        merged = copy.deepcopy(existing)
        MergeEngine._merge_recursive(merged, incoming)
        # domain is always a union
        merged["domain"] = MergeEngine._merge_lists(
            existing.get("domain", []), incoming.get("domain", [])
        )
        merged["last_updated"] = _utcnow_iso()
        return merged

    @staticmethod
    def _merge_recursive(target: dict, source: dict) -> None:
        target_conf: dict = target.get("confidence_scores", {})
        source_conf: dict = source.get("confidence_scores", {})

        for key, src_val in source.items():
            if key in ("confidence_scores", "domain"):
                continue  # domain handled specially in merge()

            tgt_val = target.get(key)

            if src_val is None:
                continue  # never overwrite with null

            if tgt_val is None:
                target[key] = copy.deepcopy(src_val)
                if key in source_conf:
                    target_conf[key] = source_conf[key]
                continue

            # Both non-null — type-specific merge
            if isinstance(src_val, list) and isinstance(tgt_val, list):
                target[key] = MergeEngine._merge_lists(tgt_val, src_val)
            elif isinstance(src_val, dict) and isinstance(tgt_val, dict):
                MergeEngine._merge_recursive(tgt_val, src_val)
            else:
                # Scalar: confidence arbitration
                s_conf = source_conf.get(key, 0.0)
                t_conf = target_conf.get(key, 0.0)
                if s_conf >= t_conf:  # ties -> incoming wins
                    target[key] = copy.deepcopy(src_val)
                    target_conf[key] = s_conf

        if target_conf:
            target["confidence_scores"] = target_conf

    @staticmethod
    def _merge_lists(existing: list, incoming: list) -> list:
        """Union of two lists, deduplicated, order-preserving."""
        seen: set[int] = set()
        merged: list = []
        for item in existing + incoming:
            h = MergeEngine._hashable(item)
            if h not in seen:
                seen.add(h)
                merged.append(item)
        return merged

    @staticmethod
    def _hashable(obj: Any) -> int:
        """Best-effort content hash for dedup."""
        if isinstance(obj, dict):
            return hash(json.dumps(obj, sort_keys=True))
        try:
            return hash(obj)
        except TypeError:
            return hash(json.dumps(obj, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Conflict detector
# ---------------------------------------------------------------------------


class ConflictDetector:
    """Detects irreconcilable differences between existing and incoming."""

    @staticmethod
    def check(
        existing: dict, incoming: dict, rkey: str
    ) -> str | None:
        """Return a human-readable conflict string, or None if clean."""

        # Entity type mismatch (hard conflict)
        e_type = existing.get("entity_type")
        i_type = incoming.get("entity_type")
        if e_type and i_type and e_type != i_type:
            return (
                f"entity_type conflict for {rkey}: "
                f"existing={e_type}, incoming={i_type}"
            )

        # Confidence tie on critical identity fields
        e_conf = existing.get("confidence_scores", {})
        i_conf = incoming.get("confidence_scores", {})
        for field in ("name", "entity_type"):
            ev = existing.get(field)
            iv = incoming.get(field)
            if ev and iv and ev != iv:
                ec = e_conf.get(field, 0.0)
                ic = i_conf.get(field, 0.0)
                if ec == ic:
                    return (
                        f"Confidence tie on critical field '{field}' for "
                        f"{rkey}: existing='{ev}' vs incoming='{iv}' "
                        f"(both conf={ec})"
                    )
        return None


# ---------------------------------------------------------------------------
# Atomic file I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via temp file + os.replace for crash safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# KnowledgeBase — main public interface
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Manages the persistent knowledge base directory.

    Storage path: {base}/{entity_type}/{primary_domain}/{NNN}_{entity_id}.json
    Registry key: {primary_domain}::{entity_type}::{entity_id}

    Thread-safety: uses advisory file lock so concurrent processes
    (not just threads) are safe.
    """

    def __init__(self, base_dir: str | Path = "knowledge_base") -> None:
        self.base = Path(base_dir)
        self._state_path = self.base / STATE_FILE
        self._conflicts_path = self.base / CONFLICTS_FILE
        self._lock_path = self.base / LOCK_FILE
        self._lock_fd: int | None = None
        self._ensure_dirs()

    # -- directory scaffolding --

    def _ensure_dirs(self) -> None:
        """Create top-level entity_type dirs. Domain subdirs are
        created on-demand when entities arrive."""
        for etype in VALID_ENTITY_TYPES:
            (self.base / etype).mkdir(parents=True, exist_ok=True)
        (self.base / "index").mkdir(parents=True, exist_ok=True)

    # -- advisory locking --

    def _acquire_lock(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(
            str(self._lock_path), os.O_CREAT | os.O_RDWR
        )
        _acquire_file_lock(self._lock_fd)

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            _release_file_lock(self._lock_fd)
            os.close(self._lock_fd)
            self._lock_fd = None

    # -- state load / save --

    def _load_state(self) -> dict:
        """Returns {"current_count": int, "registry": {key: {...}}}."""
        data = _read_json(self._state_path)
        if data is None:
            return {"current_count": 0, "registry": {}}
        return data

    def _save_state(self, state: dict) -> None:
        _atomic_write_json(self._state_path, state)

    # -- conflict log --

    def _load_conflicts(self) -> list[dict]:
        data = _read_json(self._conflicts_path)
        return data if isinstance(data, list) else []

    def _append_conflict(self, conflict: dict) -> None:
        conflicts = self._load_conflicts()
        conflicts.append(conflict)
        _atomic_write_json(self._conflicts_path, conflicts)

    # -- public API --

    def persist_entities(self, entities: list[dict]) -> dict:
        """Persist a batch of canonical entities.

        Args:
            entities: list of entity dicts from Step 3. Each must have:
                - entity_id   (str)
                - entity_type (str, one of VALID_ENTITY_TYPES)
                - domain      (list[str], non-empty; first = primary)
                - status      (str, must be 'canonical' to persist)

        Returns:
            Summary dict: saved, merged, skipped, conflicts, invalid.
        """
        stats = {
            "saved": 0, "merged": 0, "skipped": 0,
            "conflicts": 0, "invalid": 0,
        }

        # Canonical gate
        canonical = [e for e in entities if e.get("status") == "canonical"]
        stats["skipped"] += len(entities) - len(canonical)

        if not canonical:
            return stats

        self._acquire_lock()
        try:
            state = self._load_state()
            for entity in canonical:
                result = self._persist_one(entity, state)
                stats[result] += 1
            self._save_state(state)
        finally:
            self._release_lock()

        logger.info(
            "Persist complete — saved=%d merged=%d skipped=%d "
            "conflicts=%d invalid=%d",
            stats["saved"], stats["merged"], stats["skipped"],
            stats["conflicts"], stats["invalid"],
        )
        return stats

    def _persist_one(self, entity: dict, state: dict) -> str:
        """Persist or merge a single entity. Mutates state in-place.

        Returns one of: 'saved', 'merged', 'skipped', 'conflicts', 'invalid'.
        """
        # --- type alias correction (before schema validation) ---
        from validation import correct_invalid_type
        correct_invalid_type(entity)

        # --- schema validation ---
        errors = validate_entity(entity)
        if errors:
            logger.warning(
                "Validation failed for entity %s: %s",
                entity.get("entity_id", "<unknown>"),
                "; ".join(errors),
            )
            return "invalid"

        # --- quality scoring gate ---
        from validation import score_entity
        es = score_entity(entity)
        entity["_quality_score"] = es.total
        if es.verdict == "reject":
            logger.warning(
                "Quality score rejection for %s: %.3f (%s)",
                entity.get("entity_id", "<unknown>"),
                es.total, es.verdict,
            )
            return "invalid"
        if es.verdict == "warn":
            logger.info(
                "Quality score warning for %s: %.3f",
                entity.get("entity_id", "<unknown>"), es.total,
            )

        entity_id: str = entity["entity_id"]
        entity_type: str = entity["entity_type"]
        domains: list[str] = entity["domain"]
        primary = _primary_domain(domains)

        rkey = _registry_key(primary, entity_type, entity_id)
        registry: dict = state["registry"]

        if rkey in registry:
            return self._merge_existing(entity, rkey, registry)
        else:
            return self._save_new(entity, rkey, state)

    # -- merge path --

    def _merge_existing(
        self, entity: dict, rkey: str, registry: dict
    ) -> str:
        entry = registry[rkey]
        file_path = self.base / entry["path"]

        existing = _read_json(file_path)
        if existing is None:
            logger.error(
                "Registry points to missing file %s — re-saving", file_path
            )
            entity["last_updated"] = _utcnow_iso()
            _atomic_write_json(file_path, entity)
            entry["last_updated"] = entity["last_updated"]
            entry["domain"] = entity["domain"]
            return "saved"

        # Conflict detection
        conflict = ConflictDetector.check(existing, entity, rkey)
        if conflict:
            logger.warning("CONFLICT: %s", conflict)
            self._append_conflict(
                {
                    "registry_key": rkey,
                    "message": conflict,
                    "timestamp": _utcnow_iso(),
                    "incoming_entity": entity,
                }
            )
            return "conflicts"

        merged = MergeEngine.merge(existing, entity)
        _atomic_write_json(file_path, merged)
        entry["last_updated"] = merged["last_updated"]
        entry["domain"] = merged["domain"]
        logger.debug("Merged entity %s", rkey)
        return "merged"

    # -- new-entity path --

    def _save_new(self, entity: dict, rkey: str, state: dict) -> str:
        state["current_count"] += 1
        number = state["current_count"]

        entity_id = entity["entity_id"]
        entity_type = entity["entity_type"]
        primary = _primary_domain(entity["domain"])

        file_name = f"{number:03d}_{entity_id}.json"
        # Path: entity_type / primary_domain / file
        rel_path = f"{entity_type}/{primary}/{file_name}"
        full_path = self.base / rel_path

        entity["last_updated"] = _utcnow_iso()
        _atomic_write_json(full_path, entity)

        state["registry"][rkey] = {
            "file_name": file_name,
            "entity_type": entity_type,
            "domain": entity["domain"],
            "path": rel_path,
            "last_updated": entity["last_updated"],
        }
        logger.debug("Saved new entity %s -> %s", rkey, rel_path)
        return "saved"
