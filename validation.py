"""
validation.py — Deterministic validation, scoring, type correction,
relationship enrichment, and similarity clustering for the Orion pipeline.

This module enforces quality gates that do NOT depend on LLM compliance.
Every function is pure-Python with zero external dependencies.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

from storage import VALID_ENTITY_TYPES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    entity: dict
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auto_fixes: dict = field(default_factory=dict)


@dataclass
class EntityScore:
    total: float
    description_depth: float
    relationship_quality: float
    type_confidence: float
    uniqueness: float
    verdict: str  # "accept" | "warn" | "reject"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_RELATIONSHIP_TYPES = frozenset({
    "depends_on", "enables", "constrains", "contrasts_with",
    "causes", "fails_under", "refines", "implements",
})

BANNED_DESCRIPTION_OPENERS = [
    "a record of",
    "an implementation of",
    "a type of",
    "a metric used to",
    "a pattern of",
    "a feature that",
    "a product theory",
    "a non-goal",
    "an evolution",
]

# ---------------------------------------------------------------------------
# Invalid type auto-correction map
# ---------------------------------------------------------------------------

# Maps LLM-invented type names to the closest valid ORION type.
# Only add entries where the mapping is unambiguous.
TYPE_ALIASES: dict[str, str] = {
    "procedure":            "implementation_ops",
    "process":              "implementation_ops",
    "workflow":             "implementation_ops",
    "operation":            "implementation_ops",
    "steps":                "implementation_ops",
    "component":            "architecture",
    "structure":            "architecture",
    "system":               "architecture",
    "design":               "architecture",
    "feature":              "product_theory",
    "concept":              "product_theory",
    "theory":               "product_theory",
    "principle":            "product_theory",
    "benchmark":            "metric",
    "measurement":          "metric",
    "kpi":                  "metric",
    "indicator":            "metric",
    "use_case":             "acceptance_criteria",
    "use case":             "acceptance_criteria",
    "requirement":          "acceptance_criteria",
    "constraint":           "negative_boundary",
    "limitation":           "negative_boundary",
    "exception":            "data_quality_exception",
    "anomaly":              "data_quality_exception",
    "observation":          "field_feedback",
    "feedback":             "field_feedback",
    "heuristic":            "expert_heuristic",
    "rule":                 "expert_heuristic",
    "definition":           "terminology",
    "term":                 "terminology",
    "glossary":             "terminology",
    "security":             "security_interpretation",
    "threat":               "security_interpretation",
    "evolution":            "evolution_record",
    "history":              "evolution_record",
    "anti_pattern":         "negative_boundary",
    "anti-pattern":         "negative_boundary",
    "regulatory_status":    "acceptance_criteria",
    "regulatory":           "acceptance_criteria",
    "compliance":           "acceptance_criteria",
    "certification":        "acceptance_criteria",
    "approval":             "acceptance_criteria",
    "standard":             "acceptance_criteria",
    "specification":        "architecture",
    "spec":                 "architecture",
    "mode":                 "product_theory",
    "mechanism":            "product_theory",
    "behavior":             "product_theory",
    "behaviour":            "product_theory",
    "characteristic":       "product_theory",
    "property":             "product_theory",
    "capability":           "product_theory",
    "function":             "product_theory",
    "parameter":            "metric",
    "threshold":            "metric",
    "guideline":            "expert_heuristic",
    "best_practice":        "expert_heuristic",
    "best practice":        "expert_heuristic",
    "risk":                 "failure_case",
    "issue":                "failure_case",
    "defect":               "failure_case",
    "error":                "failure_case",
    "change":               "evolution_record",
    "update":               "evolution_record",
    "revision":             "evolution_record",
}


def correct_invalid_type(entity: dict) -> tuple[dict, bool]:
    """If entity_type is not in VALID_ENTITY_TYPES, try to remap it
    via TYPE_ALIASES. Returns (entity, was_corrected)."""
    etype = entity.get("entity_type", "")
    if etype in VALID_ENTITY_TYPES:
        return entity, False
    # Exact match in alias map (case-insensitive)
    mapped = TYPE_ALIASES.get(etype.lower())
    if mapped:
        logger.info(
            "Type alias correction: '%s' -> '%s' for entity '%s'",
            etype, mapped, entity.get("name", "<unknown>"),
        )
        entity["entity_type"] = mapped
        return entity, True
    # PascalCase / CamelCase → snake_case (e.g. NegativeBoundary → negative_boundary)
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", etype).lower()
    if snake in VALID_ENTITY_TYPES:
        logger.info(
            "Type PascalCase correction: '%s' -> '%s' for entity '%s'",
            etype, snake, entity.get("name", "<unknown>"),
        )
        entity["entity_type"] = snake
        return entity, True
    # Partial match — check if any alias key is a substring of the type
    etype_lower = etype.lower()
    for alias, valid in TYPE_ALIASES.items():
        if alias in etype_lower or etype_lower in alias:
            logger.info(
                "Type partial-match correction: '%s' -> '%s' for entity '%s'",
                etype, valid, entity.get("name", "<unknown>"),
            )
            entity["entity_type"] = valid
            return entity, True
    # Cannot map — leave unchanged (will fail schema validation)
    logger.warning(
        "Unknown entity_type '%s' for entity '%s' — no alias found",
        etype, entity.get("name", "<unknown>"),
    )
    return entity, False


def apply_type_alias_corrections(entities: list[dict]) -> list[dict]:
    """Apply correct_invalid_type to every entity with an unknown type."""
    corrected = 0
    for ent in entities:
        _, was = correct_invalid_type(ent)
        if was:
            corrected += 1
    if corrected:
        logger.info("Type alias corrections applied: %d entities fixed", corrected)
    return entities


# ---------------------------------------------------------------------------
# Short-chunk / low-signal guard
# ---------------------------------------------------------------------------

_MIN_CHUNK_CHARS = 200  # below this, extraction is likely hallucination
_URL_ONLY_RE = re.compile(r'^\s*https?://\S+\s*$')


def check_chunk_quality(chunk_text: str) -> tuple[bool, str]:
    """Check whether a chunk has enough signal for extraction.

    Returns (is_usable, reason_if_not).
    """
    if _URL_ONLY_RE.match(chunk_text):
        return False, "chunk contains only a URL — fetch the page content first"
    if len(chunk_text.strip()) < _MIN_CHUNK_CHARS:
        return False, (
            f"chunk too short ({len(chunk_text.strip())} chars, "
            f"min {_MIN_CHUNK_CHARS}) — likely to produce hallucinated entities"
        )
    return True, ""


# ---------------------------------------------------------------------------


METRIC_KEYWORDS = frozenset({
    "latency", "throughput", "accuracy", "precision", "recall",
    "rate", "percentage", "ratio", "count", "duration", "time",
    "frequency", "bandwidth", "capacity", "volume", "score",
    "milliseconds", "seconds", "ms", "gb", "mb", "mhz", "years",
    "months", "%", "bpm", "ohms", "watts", "impedance",
})

NON_METRIC_SIGNALS = frozenset({
    "feature", "designed to", "captures", "enables",
    "provides", "supports", "manages", "implements",
    "algorithm", "protocol", "system", "device", "component",
})


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------


def _fix_name_field(entity: dict) -> dict | None:
    """If name is a list, take first element. Returns fix description or None."""
    name = entity.get("name")
    if isinstance(name, list):
        entity["name"] = name[0] if name else ""
        return {"name": f"list -> str: '{entity['name']}'"}
    return None


def _clamp_confidence_scores(entity: dict) -> dict | None:
    """If every confidence score is 1.0, clamp to 0.85."""
    cs = entity.get("confidence_scores")
    if not isinstance(cs, dict) or not cs:
        return None
    values = [v for v in cs.values() if isinstance(v, (int, float))]
    if values and all(v == 1.0 for v in values):
        for k in cs:
            if isinstance(cs[k], (int, float)):
                cs[k] = 0.85
        return {"confidence_scores": "all 1.0 clamped to 0.85"}
    return None


def _count_sentences(text: str) -> int:
    """Rough sentence count via terminal punctuation."""
    return len(re.findall(r'[.!?](?:\s|$)', text))


def _has_banned_opener(text: str) -> bool:
    lower = text.strip().lower()
    return any(lower.startswith(b) for b in BANNED_DESCRIPTION_OPENERS)


# ---------------------------------------------------------------------------
# Step 1 validator
# ---------------------------------------------------------------------------


def validate_step1(entities: list[dict]) -> list[ValidationResult]:
    """Validate raw entities after Step 1 extraction."""
    results = []
    for ent in entities:
        errors: list[str] = []
        warnings: list[str] = []
        auto_fixes: dict = {}

        # Auto-fix name list -> string
        fix = _fix_name_field(ent)
        if fix:
            auto_fixes.update(fix)

        # Auto-fix entity_type via alias map (before validity check)
        original_type = ent.get("entity_type", "")
        correct_invalid_type(ent)  # mutates ent in place if alias found
        etype = ent.get("entity_type", "")
        if etype != original_type:
            auto_fixes["entity_type"] = f"alias: '{original_type}' -> '{etype}'"

        # entity_type
        if not etype or etype not in VALID_ENTITY_TYPES:
            errors.append(
                f"Invalid entity_type '{etype}'. Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"
            )

        # name
        name = ent.get("name", "")
        if not isinstance(name, str) or not name.strip():
            errors.append("name must be a non-empty string")

        # domain
        domain = ent.get("domain")
        if not isinstance(domain, list) or len(domain) == 0:
            errors.append("domain must be a non-empty list")

        # description depth
        desc = ent.get("description", "")
        if not isinstance(desc, str):
            desc = ""
        if len(desc) < 80:
            errors.append(
                f"Description too short ({len(desc)} chars, min 80)"
            )
        elif _count_sentences(desc) < 2:
            errors.append(
                f"Description needs >= 2 sentences (found {_count_sentences(desc)})"
            )

        if desc and _has_banned_opener(desc):
            errors.append(
                "Description uses a banned generic opener"
            )

        # metric plausibility
        if etype == "metric" and desc:
            desc_lower = desc.lower()
            has_number = bool(re.search(r'\d', desc))
            has_kw = any(kw in desc_lower for kw in METRIC_KEYWORDS)
            if not has_number and not has_kw:
                warnings.append(
                    "entity_type is 'metric' but description lacks "
                    "numbers or measurement keywords"
                )

        passed = len(errors) == 0
        results.append(ValidationResult(ent, passed, errors, warnings, auto_fixes))
    return results


# ---------------------------------------------------------------------------
# Step 2 validator
# ---------------------------------------------------------------------------


def validate_step2(entities: list[dict]) -> list[ValidationResult]:
    """Validate normalized entities after Step 2."""
    # Run step1 checks first, then add step2-specific
    step1_results = validate_step1(entities)
    results = []
    for vr in step1_results:
        ent = vr.entity
        errors = list(vr.errors)
        warnings = list(vr.warnings)
        auto_fixes = dict(vr.auto_fixes)

        # Auto-fix confidence clamping
        fix = _clamp_confidence_scores(ent)
        if fix:
            auto_fixes.update(fix)

        # entity_id
        eid = ent.get("entity_id", "")
        if not isinstance(eid, str) or not eid.strip():
            errors.append("entity_id must be a non-empty string")

        # relationships
        rels = ent.get("relationships")
        if not isinstance(rels, list) or len(rels) < 1:
            errors.append(
                "Must have at least 1 relationship"
            )
        elif isinstance(rels, list):
            for i, rel in enumerate(rels):
                if not isinstance(rel, dict):
                    errors.append(f"relationship[{i}] is not a dict")
                    continue
                rtype = rel.get("type", "")
                if rtype not in ALLOWED_RELATIONSHIP_TYPES:
                    errors.append(
                        f"relationship[{i}].type '{rtype}' not in allowed set"
                    )
                target = rel.get("target", "")
                if not target:
                    errors.append(f"relationship[{i}].target is empty")
                if target and target == eid:
                    errors.append(
                        f"relationship[{i}] is a self-reference"
                    )

        # confidence_scores
        cs = ent.get("confidence_scores")
        if not isinstance(cs, dict):
            warnings.append("Missing confidence_scores dict")
        else:
            for key in ("classification", "description", "relationships"):
                val = cs.get(key)
                if val is None:
                    warnings.append(f"confidence_scores.{key} missing")
                elif not isinstance(val, (int, float)):
                    warnings.append(f"confidence_scores.{key} not a number")

        passed = len(errors) == 0
        results.append(ValidationResult(ent, passed, errors, warnings, auto_fixes))
    return results


# ---------------------------------------------------------------------------
# Step 3 validator
# ---------------------------------------------------------------------------


def validate_step3(entities: list[dict]) -> list[ValidationResult]:
    """Validate deduplicated entities after Step 3."""
    step2_results = validate_step2(entities)
    results = []

    # Track names for cross-entity uniqueness
    seen_names: dict[str, str] = {}  # lower_name -> entity_id

    for vr in step2_results:
        ent = vr.entity
        errors = list(vr.errors)
        warnings = list(vr.warnings)
        auto_fixes = dict(vr.auto_fixes)

        # status
        status = ent.get("status", "")
        if status not in ("canonical", "duplicate"):
            # Auto-fix: default to canonical if everything else is OK
            if not errors:
                ent["status"] = "canonical"
                auto_fixes["status"] = "defaulted to 'canonical'"
            else:
                errors.append(
                    f"status must be 'canonical' or 'duplicate', got '{status}'"
                )

        # Stricter relationship count for canonical
        if ent.get("status") == "canonical":
            rels = ent.get("relationships", [])
            if isinstance(rels, list) and len(rels) < 2:
                warnings.append(
                    f"Canonical entity should have 2-4 relationships "
                    f"(has {len(rels)})"
                )

        # Stricter description length
        desc = ent.get("description", "")
        if isinstance(desc, str) and len(desc) < 120:
            warnings.append(
                f"Canonical description ideally >= 120 chars "
                f"(has {len(desc)})"
            )

        # Cross-entity name uniqueness
        name = ent.get("name", "")
        eid = ent.get("entity_id", "")
        if isinstance(name, str) and name.strip():
            lower = name.strip().lower()
            if lower in seen_names and seen_names[lower] != eid:
                warnings.append(
                    f"Duplicate canonical name '{name}' "
                    f"(also used by {seen_names[lower]})"
                )
            else:
                seen_names[lower] = eid

        passed = len(errors) == 0
        results.append(ValidationResult(ent, passed, errors, warnings, auto_fixes))
    return results


# ---------------------------------------------------------------------------
# Entity scoring
# ---------------------------------------------------------------------------


def score_entity(
    entity: dict,
    seen_names: set[str] | None = None,
) -> EntityScore:
    """Compute a quality score for an entity. Returns EntityScore."""
    desc = entity.get("description", "") or ""
    rels = entity.get("relationships", []) or []
    cs = entity.get("confidence_scores", {}) or {}
    name = str(entity.get("name", "")).strip().lower()

    # --- description_depth ---
    sentence_count = _count_sentences(desc)
    char_count = len(desc)
    has_banned = _has_banned_opener(desc)
    desc_depth = min(1.0, (
        min(1.0, sentence_count / 3) * 0.4 +
        min(1.0, char_count / 200) * 0.3 +
        (0.0 if has_banned else 0.3)
    ))

    # --- relationship_quality ---
    rel_count = len(rels) if isinstance(rels, list) else 0
    rel_quality = min(1.0, rel_count / 3)

    # --- type_confidence ---
    type_conf = float(cs.get("classification", cs.get("entity_type", 0.5)))
    type_conf = max(0.0, min(1.0, type_conf))

    # --- uniqueness ---
    if seen_names is not None:
        uniq = 0.3 if name in seen_names else 1.0
    else:
        uniq = 0.8  # neutral when no context

    total = (
        desc_depth * 0.35 +
        rel_quality * 0.25 +
        type_conf * 0.25 +
        uniq * 0.15
    )

    if total >= 0.6:
        verdict = "accept"
    elif total >= 0.4:
        verdict = "warn"
    else:
        verdict = "reject"

    return EntityScore(
        total=round(total, 3),
        description_depth=round(desc_depth, 3),
        relationship_quality=round(rel_quality, 3),
        type_confidence=round(type_conf, 3),
        uniqueness=round(uniq, 3),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Metric overuse — rule-based type correction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r'\d+')


def correct_entity_type(entity: dict) -> tuple[dict, bool]:
    """If entity_type is 'metric' but content is non-measurable,
    reclassify deterministically. Returns (entity, was_corrected)."""
    if entity.get("entity_type") != "metric":
        return entity, False

    desc = str(entity.get("description", "")).lower()
    name = str(entity.get("name", "")).lower()
    combined = desc + " " + name

    has_metric_signal = any(kw in combined for kw in METRIC_KEYWORDS)
    has_number = bool(_NUMBER_RE.search(desc))
    has_non_metric = any(kw in desc for kw in NON_METRIC_SIGNALS)

    if has_metric_signal and has_number:
        return entity, False  # legitimate metric

    if has_non_metric and not (has_metric_signal and has_number):
        # Reclassify based on strongest signal
        if any(kw in desc for kw in ("feature", "captures", "detects")):
            entity["entity_type"] = "product_theory"
        elif any(kw in desc for kw in ("designed to", "system", "device", "component")):
            entity["entity_type"] = "architecture"
        elif any(kw in desc for kw in ("step", "procedure", "implements")):
            entity["entity_type"] = "implementation_ops"
        else:
            entity["entity_type"] = "product_theory"

        logger.info(
            "Type correction: '%s' reclassified from metric -> %s",
            entity.get("name"), entity["entity_type"],
        )
        return entity, True

    # Ambiguous but no number and no metric keyword — downgrade
    if not has_number and not has_metric_signal:
        entity["entity_type"] = "product_theory"
        logger.info(
            "Type correction: '%s' reclassified from metric -> product_theory "
            "(no numeric/measurement evidence)",
            entity.get("name"),
        )
        return entity, True

    return entity, False


def apply_type_corrections(entities: list[dict]) -> list[dict]:
    """Apply correct_entity_type to every entity. Returns the list (mutated)."""
    corrected = 0
    for ent in entities:
        _, was = correct_entity_type(ent)
        if was:
            corrected += 1
    if corrected:
        logger.info("Type corrections applied: %d entities reclassified", corrected)
    return entities


# ---------------------------------------------------------------------------
# Relationship enrichment
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens from text."""
    return set(re.findall(r'[a-z][a-z0-9_]+', text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _infer_relationship_type(source: dict, target: dict) -> str:
    """Heuristic relationship type based on entity types."""
    st = source.get("entity_type", "")
    tt = target.get("entity_type", "")

    if tt == "metric" and st in ("product", "product_theory", "architecture"):
        return "constrains"
    if st == "metric" and tt in ("product", "product_theory", "architecture"):
        return "constrains"
    if tt == "architecture" and st == "implementation_ops":
        return "implements"
    if st == "architecture" and tt == "implementation_ops":
        return "implements"
    if st == tt:
        return "refines"
    return "depends_on"


def enrich_relationships(entities: list[dict]) -> list[dict]:
    """For entities with 0 relationships, auto-generate candidates
    using word-overlap similarity within the same domain."""
    # Build index
    id_to_ent = {}
    domain_index: dict[str, list[str]] = {}
    signatures: dict[str, set[str]] = {}

    for ent in entities:
        eid = ent.get("entity_id", "")
        if not eid:
            continue
        id_to_ent[eid] = ent
        name = str(ent.get("name", ""))
        desc = str(ent.get("description", ""))
        signatures[eid] = _tokenize(name + " " + desc)
        for d in ent.get("domain", []):
            domain_index.setdefault(d.lower(), []).append(eid)

    enriched_count = 0
    for ent in entities:
        eid = ent.get("entity_id", "")
        rels = ent.get("relationships")
        if not isinstance(rels, list):
            ent["relationships"] = []
            rels = ent["relationships"]

        if len(rels) > 0:
            continue  # already has relationships

        # Find candidates in same domain
        candidates: list[tuple[float, str]] = []
        ent_domains = {d.lower() for d in ent.get("domain", [])}
        for d in ent_domains:
            for cid in domain_index.get(d, []):
                if cid == eid:
                    continue
                sim = _jaccard(signatures.get(eid, set()), signatures.get(cid, set()))
                if sim > 0.05:
                    candidates.append((sim, cid))

        # Fall back to all entities if no same-domain candidates
        if not candidates:
            for cid, csig in signatures.items():
                if cid == eid:
                    continue
                sim = _jaccard(signatures.get(eid, set()), csig)
                if sim > 0.05:
                    candidates.append((sim, cid))

        # Top 2
        candidates.sort(key=lambda x: x[0], reverse=True)
        added = 0
        for _, cid in candidates[:2]:
            target_ent = id_to_ent.get(cid)
            if target_ent is None:
                continue
            rtype = _infer_relationship_type(ent, target_ent)
            rels.append({
                "type": rtype,
                "target": cid,
                "_auto_generated": True,
            })
            added += 1

        if added:
            # Mark confidence as low for auto-generated
            cs = ent.setdefault("confidence_scores", {})
            cs["relationships"] = 0.5
            enriched_count += 1

    if enriched_count:
        logger.info(
            "Relationship enrichment: %d entities received auto-generated relationships",
            enriched_count,
        )
    return entities


# ---------------------------------------------------------------------------
# Similarity clustering for deduplication
# ---------------------------------------------------------------------------


def cluster_similar_entities(
    entities: list[dict],
    threshold: float = 0.5,
) -> list[list[dict]]:
    """Group entities into clusters of likely duplicates using Jaccard
    similarity on name+description tokens.

    Returns list of clusters. Singletons are clusters of size 1.
    """
    n = len(entities)
    if n == 0:
        return []

    # Build signatures
    sigs: list[set[str]] = []
    for ent in entities:
        name = str(ent.get("name", ""))
        desc = str(ent.get("description", ""))
        sigs.append(_tokenize(name + " " + desc))

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairwise comparison
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(sigs[i], sigs[j]) >= threshold:
                union(i, j)

    # Collect clusters
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    clusters = [[entities[i] for i in idxs] for idxs in groups.values()]

    multi = sum(1 for c in clusters if len(c) > 1)
    if multi:
        logger.info(
            "Clustering: %d entities -> %d clusters (%d multi-entity)",
            n, len(clusters), multi,
        )
    return clusters


# ---------------------------------------------------------------------------
# Feedback prompt builder (for retry mechanism)
# ---------------------------------------------------------------------------

_FEEDBACK_TEMPLATE = """The following entities FAILED validation. Fix them and return the COMPLETE corrected JSON array (including entities that already passed).

FAILED ENTITIES:
{failed_json}

VALIDATION ERRORS:
{error_list}

RULES:
- Fix ONLY the listed errors
- Do NOT remove entities — improve them
- Descriptions must be >= 2 sentences, >= 80 chars, explain HOW it works
- Every entity must have at least 1 relationship
- Valid entity_type values are EXACTLY: {valid_types}
- Return the full corrected list as a JSON array
"""


def build_feedback_prompt(
    validation_results: list[ValidationResult],
    all_entities: list[dict],
) -> str:
    """Build a feedback prompt from validation failures."""
    import json

    failed = [vr for vr in validation_results if not vr.passed]
    failed_ents = [vr.entity for vr in failed]

    error_lines = []
    for vr in failed:
        name = vr.entity.get("name", "<unknown>")
        for err in vr.errors:
            error_lines.append(f"- {name}: {err}")

    return _FEEDBACK_TEMPLATE.format(
        failed_json=json.dumps(failed_ents, indent=2, default=str),
        error_list="\n".join(error_lines),
        valid_types=", ".join(sorted(VALID_ENTITY_TYPES)),
    )
