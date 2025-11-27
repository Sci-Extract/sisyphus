import json
from sisyphus.urgent import entity_resolution_utils as er


def test_strict_partition_example():
    recs = er.EXAMPLE_RECORDS
    out = er.partition_strict(recs)
    data = json.loads(out.json())
    # each input record should be in exactly one partition
    all_members = [m for p in data['partitions'] for m in p['members']]
    assert sorted(all_members) == list(range(len(recs)))
    # strict expected: no merges in this example -> number of partitions == len(recs)
    assert len(data['partitions']) == len(recs)


def test_fuzzy_partition_example():
    recs = er.EXAMPLE_RECORDS
    out = er.partition_fuzzy(recs)
    data = json.loads(out.json())
    all_members = [m for p in data['partitions'] for m in p['members']]
    assert sorted(all_members) == list(range(len(recs)))
    # fuzzy mode may merge records 0 and 3 because processing_kw of 0 is subset of 3
    # ensure at least 1 partition <= len(recs)
    assert 1 <= len(data['partitions']) <= len(recs)


def test_flatten_and_finalize_merge():
    # simulate two groups: first three from group 0, next two from group 1
    group0 = er.EXAMPLE_RECORDS[:3]
    group1 = er.EXAMPLE_RECORDS[3:]
    flat, _groups = er.flatten_record_groups([group0, group1])
    # simulate an LLM output that clustered index 0 (flat idx 0) with index 3 (flat idx 3)
    # and clustered index 1 with 4, leaving index 2 alone
    simulated_partitions = [[0, 3], [1, 4], [2]]
    final = er.finalize_merge_from_partitions(flat, simulated_partitions)
    data = json.loads(final.json())
    # ensure coverage and no duplicates
    all_members = [m for p in data['partitions'] for m in p['members']]
    assert sorted(all_members) == list(range(len(flat)))
    # canonical metadata keys should match normalized forms
    norms = er.build_normalized_metadata(flat)
    for p in data['partitions']:
        first_idx = p['members'][0]
        assert p['canonical_metadata']['composition'] == norms[first_idx]['composition_norm']
