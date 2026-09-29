# graphRagX V2.2 Architecture

## Goal

V2.2 adds an iterative DRIFT-style query engine on top of the V2.1 hierarchy and retrieval stack.

The implementation follows the core DRIFT shape:

~~~text
User Query
   |
   v
Primer from relevant community reports
   |
   +--> broad answer
   +--> ranked follow-up questions
                 |
                 v
          Local graph retrieval
                 |
                 v
        Intermediate evidence
                 |
                 +--> confidence
                 +--> provenance
                 +--> next follow-ups
                 |
                 v
          depth/action limits
                 |
                 v
           Final reduce
~~~

## Primer

The primer retrieves semantically relevant community reports at a configured hierarchy level.

Reports are split across DRIFT_PRIMER_FOLDS. Each fold produces:

- a broad evidence-grounded answer
- ranked follow-up questions

Follow-ups are deduplicated against the original query and against one another.

## Follow-up exploration

Each follow-up runs the same local retrieval primitives used by Local Search:

- entity embedding retrieval
- graph-neighbor expansion
- relation collection
- source text-unit lookup
- query reranking
- tokenizer-aware context packing

The follow-up LLM returns:

- an intermediate answer
- confidence in [0, 1]
- optional next follow-up questions

A branch only expands when confidence is at least DRIFT_EXPANSION_MIN_CONFIDENCE.

## Query state

DRIFT is represented explicitly by DriftTrace.

The trace contains:

- primer_answer
- primer_reports
- all accepted follow-up questions
- evidence ledger
- visited questions
- actions_executed
- max_depth_reached
- termination_reason

Each DriftEvidence record contains:

- question
- intermediate answer
- confidence
- depth
- parent_id
- source_ids

This trace is returned through the query state and is also used to shape API source metadata.

## Termination

The engine is bounded by multiple independent controls:

- DRIFT_N_DEPTH
- DRIFT_MAX_ACTIONS
- DRIFT_K_FOLLOWUPS per depth
- confidence-gated branch expansion
- follow-up deduplication

Typical termination reasons are:

- primer_only
- no_followups
- max_depth
- max_actions
- completed

## Final reduce

Evidence is ranked by confidence and packed under DRIFT_REDUCE_DATA_TOKENS.

The final reducer receives:

- the original user query
- a token-limited primer answer
- selected intermediate evidence
- source provenance

The reducer is instructed to reconcile conflicts and prefer better-supported, higher-confidence evidence.

## Configuration

~~~env
DRIFT_K_FOLLOWUPS=3
DRIFT_PRIMER_FOLDS=3
DRIFT_N_DEPTH=2
DRIFT_MAX_ACTIONS=12
DRIFT_COMMUNITY_LEVEL=1
DRIFT_PRIMER_DATA_TOKENS=5000
DRIFT_LOCAL_MAX_DATA_TOKENS=2500
DRIFT_REDUCE_DATA_TOKENS=5000
DRIFT_LOCAL_TOP_K_ENTITIES=5
DRIFT_EXPANSION_MIN_CONFIDENCE=0.35
~~~

## Relationship to V2.1

DRIFT does not maintain a separate retrieval implementation.

V2.2 refactors Local Search and Global community retrieval into reusable internal primitives so Local, Global, and DRIFT share:

- embedding compatibility checks
- entity ranking
- chunk ranking
- graph expansion
- community-level selection
- tokenizer-aware context packing

This prevents query modes from drifting into inconsistent retrieval semantics.

## External references

- Microsoft GraphRAG DRIFT Search:
  https://microsoft.github.io/graphrag/query/drift_search/
- GraphRAG query overview:
  https://microsoft.github.io/graphrag/query/overview/
- GraphRAG configuration:
  https://microsoft.github.io/graphrag/config/yaml/
