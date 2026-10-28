"""Provenance-aware map parsing refuses fabricated or unscored evidence."""
import pytest

from workflows.query_workflow import QueryWorkflow


def test_map_point_rejects_fabricated_ids_and_preserves_exact_valid_sources():
    raw = (
        '{"points":['
        '{"description":"valid","score":72,'
        '"source_reports":["L0_C0","invented","L0_C0"]},'
        '{"description":"fabricated","score":99,'
        '"source_reports":["invented"]},'
        '{"description":"invalid numeric","score":"nan",'
        '"source_reports":["L0_C0"]}'
        ']}'
    )
    points = QueryWorkflow._parse_global_map_points(raw, ["L0_C0", "L0_C1"])
    assert points == [{
        "description": "valid",
        "score": 72.0,
        "source_reports": ["L0_C0"],
    }]


def test_map_invalid_json_fails_instead_of_fabricating_low_score_point():
    with pytest.raises(ValueError, match="JSON points list"):
        QueryWorkflow._parse_global_map_points(
            "The model gave a freeform answer without structured evidence.",
            ["L0_C0"],
        )


def test_map_empty_valid_points_are_legitimate():
    assert QueryWorkflow._parse_global_map_points('{"points":[]}', ["L0_C0"]) == []
