from run_position_causal import lab, span_of


def test_layer_labels_cover_backbone_norm_and_head():
    assert [lab(l) for l in (1, 22, 23, 24, 25)] == ["enc_01", "enc_22", "enc_final", "head_1", "head_2"]


def test_span_of_each_mode():
    assert span_of(14, "single") == [14] and span_of("14", "single") == [14]
    assert span_of(20, "onward") == [20, 21, 22]
    assert span_of(22, "tail") == [22, 23, 24, 25] and span_of(24, "tail") == [24, 25]
    assert span_of("14+22", "set") == [14, 22] and span_of("14+17+22", "set") == [14, 17, 22]
