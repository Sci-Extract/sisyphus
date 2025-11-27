"""
Entity resolution utilities for materials extraction.

Contains:
- normalization helpers
- strict partitioning (exact normalized-match)
- fuzzy partitioning (subset/overlap merging with confidence)
- Pydantic schemas for structured outputs
- example runner and prompt strings

Drop into your notebook and call the partition_* functions.
"""
from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

# -----------------
# Pydantic schemas
# -----------------
class CanonicalMetadata(BaseModel):
    composition: str
    label: str
    processing_kw: List[str]

class Partition(BaseModel):
    canonical_metadata: CanonicalMetadata
    members: List[int]
    records: List[Dict[str, Any]]
    confidence: Optional[str] = None

class PartitionsOutput(BaseModel):
    partitions: List[Partition]


# -----------------
# LLM integration helpers
# -----------------

class LLMPartitionOutput(BaseModel):
    # partitions is a list of clusters; each cluster is a list of integer indices
    reason: str = Field(description="Reasoning behind the partitioning decisions made by the LLM.")
    partitions: List[List[int]]


def flatten_record_groups(record_groups: List[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Flatten a list of record lists (e.g., per-property groups) into a single list.

    Returns (flat_records, group_indices) where group_indices[i] is the group id
    (0..G-1) the i-th flattened record came from.
    """
    flat = []
    group_indices: List[int] = []
    for gid, group in enumerate(record_groups):
        for rec in group:
            flat.append(rec)
            group_indices.append(gid)
    return flat, group_indices


def build_normalized_metadata(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a list of normalized metadata dicts for each record.

    Each dict contains: composition_norm, label_norm, processing_kw_norm (list)
    """
    out = []
    for rec in records:
        out.append(normalized(rec))
    return out

def normalized(rec):
    comp = normalize_composition(rec.get('composition', '') )
    label = normalize_label(rec.get('label', '') )
    pkw = normalize_processing_kw(rec.get('processing_kw', []) or [])
    return {'composition_norm': comp, 'label_norm': label, 'processing_kw_norm': pkw}


LLM_PROMPT_IDS = \
"""You are given a list of extracted material records. Each record has been normalized and is identified by a zero-based integer index.
Each record belongs to a source group (for example, “phase extraction” or “strength extraction” or "grain size extraction"). A separate list provides the group index for every record.
Your task is to cluster records that refer to the same underlying experimental material entity. These identities originate from different extraction sources because the original paper describes material information in scattered paragraphs.

Important rules (strict):
Do NOT cluster two records that come from the same source group unless their normalized metadata
(composition, processing_kw) are identical.
When clustering records from different source groups, determine whether they describe the same experimental entity.
Processing_kw may be incomplete or partially captured in some records. Use the provided synthesis paragraphs as contextual evidence to resolve cases where partial or mismatched processing_kw might still refer to the same material.
Synthesis paragraphs describe the actual experimental procedure of the paper. Use them to infer whether incomplete processing_kw in different records is still consistent with a single underlying material identity.
Sometimes noise may be present in metadata fields, which was introduced during previous extraction stages, such as property-relevant parameters mistakenly included in processing_kw or label fields. Focus on the synthsis relevant information to resolve such discrepancies.

Output Format:
Return ONLY valid JSON with the following structure:
{{
"reason": "",
"partitions": [
[list of integer indices forming one cluster],
...
]
}}
Example:
{{
"reason": "",
"partitions": [[0, 3], [1], [2, 4]]
}}
Do not output anything else besides the JSON object.
"""


def finalize_merge_from_partitions(flat_records: List[Dict[str, Any]], partitions_indices: List[List[int]]) -> PartitionsOutput:
    """Given flat records and partitions (list of index lists) returned by an LLM,
    compute canonical metadata per partition using normalized metadata and then
    merge partitions that have identical canonical metadata (this performs the final entity merge).

    Returns a PartitionsOutput with canonical metadata and member indices and original records.
    """
    norm = build_normalized_metadata(flat_records)

    # compute canonical metadata per incoming partition
    temp_parts: List[Dict[str, Any]] = []
    for members in partitions_indices:
        if not members:
            continue
        # choose canonical metadata from the first member
        first = members[0]
        cm = CanonicalMetadata(
            composition=norm[first]['composition_norm'],
            label=norm[first]['label_norm'],
            processing_kw=norm[first]['processing_kw_norm'],
        )
        temp_parts.append({'canonical': cm, 'members': sorted(members)})

    # merge partitions with identical canonical metadata
    merged: List[Dict[str, Any]] = []
    seen_keys = {}
    for part in temp_parts:
        key = (part['canonical'].composition, part['canonical'].label, tuple(part['canonical'].processing_kw))
        if key in seen_keys:
            # merge members
            idx = seen_keys[key]
            merged[idx]['members'].extend(part['members'])
            merged[idx]['members'] = sorted(set(merged[idx]['members']))
        else:
            seen_keys[key] = len(merged)
            merged.append({'canonical': part['canonical'], 'members': list(part['members'])})

    # build final PartitionsOutput
    out_parts: List[Partition] = []
    for m in merged:
        members = sorted(m['members'])
        records = [flat_records[i] for i in members]
        part = Partition(
            canonical_metadata=m['canonical'],
            members=members,
            records=records,
            confidence='high',
        )
        out_parts.append(part)

    return PartitionsOutput(partitions=out_parts)
# -----------------
# Normalizers
# -----------------
_re_spaces = re.compile(r"\s+")
_re_degree = re.compile(r"°")
_re_temp = re.compile(r"(?i)(\b(\d+)\s*°?\s*C\b)")
_re_time_h = re.compile(r"(?i)(\b(\d+)\s*h\b)")
_re_time_min = re.compile(r"(?i)(\b(\d+)\s*min\b)")


def normalize_composition(comp: str) -> str:
    if comp is None:
        return ""
    s = comp.strip()
    # remove stray spaces between element/count groups
    s = re.sub(r"\s+", "", s)
    # ensure uppercase letters for element symbols and keep digits as-is
    # do not reorder elements
    return s


def normalize_label(label: str) -> str:
    if label is None:
        return ""
    s = label.strip()
    # remove trailing commas
    s = s.rstrip(',')
    s = _re_spaces.sub(' ', s)
    return s


def _normalize_time_and_temp(phrase: str) -> str:
    s = phrase
    # replace degree symbol
    s = _re_degree.sub('', s)
    # replace "900 °C" or "900 ° C" -> "900C"
    s = re.sub(r"(?i)(\b(\d+)\s*°?\s*C\b)", lambda m: m.group(2) + 'C', s)
    # normalize hours "6 h" -> "6h"
    s = re.sub(r"(?i)(\b(\d+)\s*h\b)", lambda m: m.group(2) + 'h', s)
    # normalize minutes "10 min" -> "10min"
    s = re.sub(r"(?i)(\b(\d+)\s*min\b)", lambda m: m.group(2) + 'min', s)
    # collapse whitespace
    s = _re_spaces.sub(' ', s).strip()
    return s


def normalize_processing_kw(phrases: List[str]) -> List[str]:
    if not phrases:
        return []
    seen = set()
    out = []
    for p in phrases:
        if p is None:
            continue
        s = p.strip()
        s = _normalize_time_and_temp(s)
        # preserve original capitalization except normalized tokens
        # but for matching, we keep case-insensitive equality by lowercasing the dedup key
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# -----------------
# Partitioners
# -----------------

def _record_key_strict(record: Dict[str, Any]) -> Tuple[str, str, Tuple[str, ...]]:
    comp = normalize_composition(record.get('composition', ''))
    label = normalize_label(record.get('label', ''))
    pkw = normalize_processing_kw(record.get('processing_kw', []) or [])
    return (comp, label, tuple(pkw))


def partition_strict(records: List[Dict[str, Any]]) -> PartitionsOutput:
    """Partition records using strict normalized equality on (composition, label, processing_kw).

    canonical processing_kw is taken from the first member (after normalization).
    """
    index_map: Dict[Tuple[str, str, Tuple[str, ...]], List[int]] = {}
    keys_map: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[str, str, List[str]]] = {}

    for i, rec in enumerate(records):
        key = _record_key_strict(rec)
        index_map.setdefault(key, []).append(i)
        if key not in keys_map:
            keys_map[key] = (key[0], key[1], list(key[2]))

    partitions: List[Partition] = []
    for key, members in index_map.items():
        comp, label, pkw_list = keys_map[key]
        canonical = CanonicalMetadata(
            composition=comp,
            label=label,
            processing_kw=pkw_list,
        )
        part = Partition(
            canonical_metadata=canonical,
            members=sorted(members),
            records=[records[i] for i in sorted(members)],
            confidence='high',
        )
        partitions.append(part)

    return PartitionsOutput(partitions=partitions)


def _is_subset(a: List[str], b: List[str]) -> bool:
    # compare case-insensitive keys
    aset = set(x.lower() for x in a)
    bset = set(x.lower() for x in b)
    return aset.issubset(bset)


def _overlap_fraction(a: List[str], b: List[str]) -> float:
    aset = set(x.lower() for x in a)
    bset = set(x.lower() for x in b)
    if not aset and not bset:
        return 1.0
    if not aset:
        return 0.0
    inter = aset.intersection(bset)
    return len(inter) / len(aset)


def _merge_processing_union(order_preserve_lists: List[List[str]]) -> List[str]:
    seen = set()
    out = []
    for lst in order_preserve_lists:
        for p in lst:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def partition_fuzzy(records: List[Dict[str, Any]]) -> PartitionsOutput:
    """Partition records using fuzzy merging rules.

    Rules implemented:
    - If composition and label equal (after normalization)
      AND one processing_kw list is a subset of another -> merge (confidence=medium)
    - If composition and label equal and processing_kw overlap >= 0.5 -> merge (confidence=low)
    - Exact match -> confidence=high

    Merging is greedy: iterate records and try to attach to first compatible cluster.
    """
    # prepare normalized representations
    norm_recs = []
    for i, rec in enumerate(records):
        comp = normalize_composition(rec.get('composition', ''))
        label = normalize_label(rec.get('label', ''))
        pkw = normalize_processing_kw(rec.get('processing_kw', []) or [])
        norm_recs.append({'index': i, 'rec': rec, 'comp': comp, 'label': label, 'pkw': pkw})

    clusters: List[Dict[str, Any]] = []

    for item in norm_recs:
        placed = False
        for cluster in clusters:
            if item['comp'] != cluster['comp'] or item['label'] != cluster['label']:
                continue
            # check exact
            if tuple(item['pkw']) == tuple(cluster['pkw']):
                cluster['members'].append(item['index'])
                cluster['records'].append(item['rec'])
                cluster['confidence'] = 'high'
                placed = True
                break
            # subset
            if _is_subset(item['pkw'], cluster['pkw']) or _is_subset(cluster['pkw'], item['pkw']):
                # merge: extend pkw union preserving order
                cluster['members'].append(item['index'])
                cluster['records'].append(item['rec'])
                cluster['pkw'] = _merge_processing_union([cluster['pkw'], item['pkw']])
                cluster['confidence'] = 'medium'
                placed = True
                break
            # overlap
            frac = _overlap_fraction(item['pkw'], cluster['pkw'])
            frac2 = _overlap_fraction(cluster['pkw'], item['pkw'])
            if (frac >= 0.5) or (frac2 >= 0.5):
                cluster['members'].append(item['index'])
                cluster['records'].append(item['rec'])
                cluster['pkw'] = _merge_processing_union([cluster['pkw'], item['pkw']])
                cluster['confidence'] = 'low'
                placed = True
                break
        if not placed:
            clusters.append({
                'comp': item['comp'],
                'label': item['label'],
                'pkw': list(item['pkw']),
                'members': [item['index']],
                'records': [item['rec']],
                'confidence': 'high',
            })

    partitions: List[Partition] = []
    for cl in clusters:
        canonical = CanonicalMetadata(
            composition=cl['comp'],
            label=cl['label'],
            processing_kw=cl['pkw'],
        )
        part = Partition(
            canonical_metadata=canonical,
            members=sorted(cl['members']),
            records=cl['records'],
            confidence=cl.get('confidence'),
        )
        partitions.append(part)

    return PartitionsOutput(partitions=partitions)

# -----------------
# Example runner and prompts
# -----------------

STRICT_PROMPT = r"""
You are a JSON-only assistant. Partition the provided list of records into entity clusters.
Records are JSON objects with keys: composition, label, processing_kw (array).
Apply normalization: composition -> remove spaces; label -> lower and trim; processing_kw -> normalize temperatures (°C->C) and times (6 h -> 6h), remove duplicates.
Strict rule: two records are same only if normalized composition, label, and processing_kw are identical.
Return exact JSON following this schema: {"partitions":[{"canonical_metadata":{"composition":...,"label":...,"processing_kw":[...]},"members":[idx,...],"records":[...]}]}
Only output JSON.
"""

FUZZY_PROMPT = r"""
As above, but allow fuzzy merging when composition and label match and processing_kw lists are subset or have >=50% overlap.
When fuzzy merged, include a "confidence" field (high/medium/low) on each partition.
"""


def example_runner(records: List[Dict[str, Any]], mode: str = 'strict') -> Dict[str, Any]:
    if mode == 'strict':
        out = partition_strict(records)
    elif mode == 'fuzzy':
        out = partition_fuzzy(records)
    else:
        raise ValueError('mode must be "strict" or "fuzzy"')
    return json.loads(out.json())

# Example records convenient for quick local testing
EXAMPLE_RECORDS = [
    {'composition': 'V10Cr15Mn5Fe35Co10Ni25', 'label': 'FG, annealed', 'processing_kw': ['homogenized at 1100C 6h Ar', 'cold rolled 79%', 'annealed 900C 10min']},
    {'composition': 'V10Cr15Mn5Fe35Co10Ni25', 'label': 'CG, annealed', 'processing_kw': ['homogenized at 1100C 6h Ar', 'cold rolled 79%', 'annealed 1100C 60min']},
    {'composition': 'V10Cr15Mn5Fe35Co10Ni25', 'label': 'FG, HPT 1/4 turn', 'processing_kw': ['homogenized at 1100C 6h Ar', 'cold rolled 79%', 'annealed 900C 10min', 'HPT 1/4 turn 6GPa 1rpm']},
    {'composition': 'V10Cr15Mn5Fe35Co10Ni25', 'label': 'FG annealed 900C 10min', 'processing_kw': ['vacuum induction melting', 'homogenized 1100C 6h Ar', 'water quenched', 'cold rolled 79%', 'EDM disk', 'annealed 900C 10min']},
    {'composition': 'V10Cr15Mn5Fe35Co10Ni25', 'label': 'CG annealed 1100C 60min', 'processing_kw': ['vacuum induction melting', 'homogenized 1100C 6h Ar', 'water quenched', 'cold rolled 79%', 'EDM disk', 'annealed 1100C 60min']},
]

if __name__ == '__main__':
    print('Strict partitions:')
    print(json.dumps(example_runner(EXAMPLE_RECORDS, mode='strict'), indent=2))
    print('\nFuzzy partitions:')
    print(json.dumps(example_runner(EXAMPLE_RECORDS, mode='fuzzy'), indent=2))
