import pytest

from workflows.query_workflow import QueryWorkflow


def test_global_map_parser_clamps_scores_and_keeps_provenance():
    points = QueryWorkflow._parse_global_map_points(
        '{"points":[{"description":"fact","score":120}]}',
        ["L1_C0"],
    )
    assert points == [
        {
            "description": "fact",
            "score": 100.0,
            "source_reports": ["L1_C0"],
        }
    ]


@pytest.mark.asyncio
async def test_global_map_reduce_runs_with_offline_mock():
    workflow = QueryWorkflow()
    result = await workflow.generate_global_answer(
        {
            "query": "What are the main themes?",
            "relevant_context": {
                "community_summaries": [
                    {
                        "id": "L0_C0",
                        "summary": "Graph retrieval and indexing.",
                        "level": 0,
                        "size": 3,
                    }
                ]
            },
        }
    )
    assert result["current_step"] == "completed"
    assert result["answer"]
    assert result["relevant_context"]["map_points"]
