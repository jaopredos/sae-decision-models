"""Role/identity/offset token alignment, independent of token repetition."""


def token_alignment(snapshot, sep_id):
    prefix = snapshot["question_tokens"]
    spans = snapshot["option_spans"]
    tail_start = len(prefix) + sum(map(len, spans))
    indices = list(range(len(prefix)))
    marker_indices = []
    for identity, span in enumerate(spans):
        marker_indices.append(len(indices))
        start = snapshot["positions"][identity]
        indices.extend(range(start, start+len(span)))
    indices.extend(range(tail_start, snapshot["seq_len"]))
    tokens = prefix + [t for span in spans for t in span] + [sep_id] + snapshot["state_tokens"]
    if sorted(indices) != list(range(snapshot["seq_len"])) or len(tokens) != len(indices):
        raise ValueError("token mapping must cover every valid token exactly once")
    marker_set = set(marker_indices)
    return {"indices": indices, "tokens": tokens,
            "groups": {"markers": marker_indices,
                       "nonmarkers": [i for i in range(len(indices)) if i not in marker_set],
                       "all_tokens": list(range(len(indices)))}}
