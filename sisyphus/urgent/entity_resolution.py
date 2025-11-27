from typing import TypedDict

from langchain.prompts import ChatPromptTemplate

from sisyphus.urgent import entity_resolution_utils as er

class MetaData(TypedDict):
    composition: str
    label: str
    processing_kw: str


def entity_resolution_llms(record_groups: list[list[MetaData]], chat_model, syn_text) -> list[list[MetaData]]:
    """Perform entity resolution on records from different source groups using LLMs.
    Assumption: Records from the same source group should not be clustered together (unless they have identical normalized metadata).
    """
    # 1) Remove duplicates
    normalized_groups = [er.build_normalized_metadata(group) for group in record_groups]
    normalized_origin_map = {}
    for group in record_groups:
        for record in group:
            normalized = er.normalized(record)
            key = str(normalized)
            if key not in normalized_origin_map:
                normalized_origin_map[key] = [record]
            else:
                normalized_origin_map[key].append(record)

    deduped_groups = []
    seen = []
    for group in normalized_groups:
        deduped = []
        for record in group:
            if record not in seen:
                seen.append(record)
                deduped.append(record)
        if deduped:
            deduped_groups.append(deduped)

    # 2) Flatten deduped groups
    flat_records, group_indices = er.flatten_record_groups(deduped_groups)
    # 3) Prepare payload for LLM
    payload = []
    for idx, record in enumerate(flat_records):
        entry = {
            "index": idx,
            "group_index": group_indices[idx],
            **record
        }
        payload.append(entry)
    entities_markdown = dict_list_to_markdown_table(payload)

    # 4) Build prompt, runs entity resolution process
    user_text = er.LLM_PROMPT_IDS + '\nSynthesis paragraphs\n' + syn_text + "\n\nHere is the normalized records list (index+group_index+normalized metadata):\n\n"
    user_text += entities_markdown
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system', 'You are a JSON-only assitant'
            ),
            (
                'user', user_text
            )
        ]
    )
    chain = prompt | chat_model.with_structured_output(er.LLMPartitionOutput, method='json_schema')
    result = chain.invoke({})
    partitions_indices = result.partitions # e.g., [[0, 3], [1, 2], [4]]
    partitions_indices = [partition for partition in partitions_indices if partition]  # filter out empty partitions
    # validate length matches
    total_indices = sum(len(partition) for partition in partitions_indices)
    if total_indices != len(flat_records):
        raise RuntimeError("Entity resolution LLM output partitions do not match the number of records.")
    # debug
    # debug_info = f"prompt: {user_text}\nLLM output partitions: {partitions_indices}\nreason: {result.reason}" 
    # with open("debug", "a+", encoding="utf-8") as f:
    #     f.write(debug_info + "\n")
    #     f.write("----\n\n\n")

    # 5) return partitions (non-normalized metadata)
    normalized_partitions = [[flat_records[i] for i in indices] for indices in partitions_indices]
    partitions = []
    for partition in normalized_partitions:
        group =[]
        for normalized in partition:
            group.extend(normalized_origin_map[str(normalized)])
        partitions.append(group)
    
    return partitions

def entity_resolution_rule(records: list[MetaData], keys: list[str]) -> list[list[MetaData]]:
    """group records by keys"""
    if not records:
        return []
    if not keys: # use default
        keys = ['composition', 'label']

    groups: dict[tuple, list[MetaData]] = {}
    for rec in records:
        key_parts = []
        for k in keys:
            val = rec.get(k) if isinstance(rec, dict) else getattr(rec, k, None)
            if isinstance(val, list):
                val = tuple(val)
            elif isinstance(val, dict):
                val = tuple(sorted(val.items()))
            key_parts.append(val)
        key_tuple = tuple(key_parts)
        groups.setdefault(key_tuple, []).append(rec)

    return list(groups.values())

# --Utils--        

def dict_list_to_markdown_table(entry):
    """
    Convert a list of dictionaries to a markdown table.
    Groups rows by group_index and only shows the source_group value once per group.
    
    Args:
        entry: List of dictionaries with 'index', 'group_index', and other fields
    
    Returns:
        str: Markdown formatted table
    """
    if not entry:
        return ""
    
    # Get all unique keys, ensuring group_index and index are first
    all_keys = []
    if 'group_index' in entry[0]:
        all_keys.append('group_index')
    if 'index' in entry[0]:
        all_keys.append('index')
    
    # Add remaining keys
    for key in entry[0].keys():
        if key not in ['group_index', 'index']:
            all_keys.append(key)
    
    # Sort entries by group_index then by index
    sorted_entry = sorted(entry, key=lambda x: (x.get('group_index', 0), x.get('index', 0)))
    
    # Build markdown table
    md_lines = []
    
    # Header row
    header = "| " + " | ".join(all_keys) + " |"
    md_lines.append(header)
    
    # Separator row
    separator = "| " + " | ".join(["---"] * len(all_keys)) + " |"
    md_lines.append(separator)
    
    # Data rows
    prev_group_index = None
    for record in sorted_entry:
        row_values = []
        for key in all_keys:
            if key == 'group_index':
                # Only show group_index if it's different from previous row
                current_group = record.get(key, '')
                if current_group != prev_group_index:
                    row_values.append(str(current_group))
                    prev_group_index = current_group
                else:
                    row_values.append("")  # Empty cell for grouped rows
            else:
                row_values.append(str(record.get(key, '')))
        
        row = "| " + " | ".join(row_values) + " |"
        md_lines.append(row)
    
    return "\n".join(md_lines)
