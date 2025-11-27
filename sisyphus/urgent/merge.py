from sisyphus.urgent.entity_resolution_utils import normalized
from typing import TypedDict
from pydantic import BaseModel


class MetaData(TypedDict):
    composition: str
    label: str | None = None
    processing_kw: list[str] | None = None

class MetaDataNorm(TypedDict):
    composition_norm: str
    label_norm: str | None = None
    processing_kw_norm: list[str] | None = None

class Record(TypedDict):
    metadata: MetaDataNorm
    properties: list[dict[str, dict]]
    extract_syn: bool | None = None


REFERRED = 'referred'
METADATA = 'metadata'

def merge(partitions: list[list[MetaData]], records: list[dict]) -> list[dict]:
    """Merge records based on partitions from entity resolution.

    Args:
        partitions: List of partitions, each partition is a list of MetaData dicts.
        records: Original list of record dicts.
    """
    
    merged_records = []
    grouped_records =[]
    # dump records if they are pydantic models
    records_ = records.copy()
    for i, record in enumerate(records_):
        if isinstance(record, BaseModel):
            records_[i] = record.model_dump()
    for partition in partitions:
        merged = []
        for record in records_:
            # TODO: fix this
            # if record in grouped_records:
                # continue
            if record[METADATA] in partition:
                merged.append(record)
                grouped_records.append(record)
        if not merged:
            raise RuntimeError("No records found to merge in partition.")
        # Simple merge strategy: take the first record as the base
        base_record = merged[0]
        normalized_metadata = normalized(base_record[METADATA])
        properties = []
        for rec in merged:
            properties.append(
                {k: v for k, v in rec.items() if k not in (METADATA, REFERRED)}
            )
       # whether needed extract
        refers = [rec.get(REFERRED, None) for rec in merged] 
        extract_syn = False
        for ref in refers:
            if ref is False:
                extract_syn = True
                break
        merged_record = Record(
            metadata=normalized_metadata,
            properties=properties,
            extract_syn=extract_syn
        )

        merged_records.append(merged_record)

    return merged_records
        
