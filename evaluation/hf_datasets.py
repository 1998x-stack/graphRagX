"""Versioned Hugging Face benchmark adapters for GraphRAG.

No network access occurs when importing this module. Downloading data is an
explicit CLI step; datasets is an optional dependency.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator


DATASETS = {
    "hotpotqa": {
        "repo": "hotpotqa/hotpot_qa", "config": "distractor",
        "split": "validation", "task": "multihop_qa",
        "license": "cc-by-sa-4.0",
    },
    "musique": {
        "repo": "awinml/musique", "config": None,
        "split": "validation", "task": "multihop_qa",
        "license": "see_dataset_card",
    },
    "multi_news": {
        "repo": "alexfabbri/multi_news", "config": None,
        "split": "validation", "task": "multi_document_summary",
        "license": "see_dataset_card",
    },
    "qasper": {
        "repo": "allenai/qasper", "config": None,
        "split": "validation", "task": "paper_qa",
        "license": "see_dataset_card",
    },
}
DatasetName = Literal["hotpotqa", "musique", "multi_news", "qasper"]


class SourceDocument(BaseModel):
    """An input document. Its position is stable within this case."""
    id: str
    title: str
    text: str = Field(min_length=1)


class BenchmarkCase(BaseModel):
    id: str
    dataset: DatasetName
    revision: str = Field(min_length=7)
    split: str
    task_type: str
    question: str = Field(min_length=1)
    documents: list[SourceDocument] = Field(min_length=1)
    reference_answers: list[str] = Field(default_factory=list)
    # SourceDocument.id; empty means *unlabeled*, not zero relevant docs.
    gold_document_ids: list[str] = Field(default_factory=list)
    gold_sentence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_documents(self) -> "BenchmarkCase":
        doc_ids = [doc.id for doc in self.documents]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Duplicate source document IDs")
        if not set(self.gold_document_ids).issubset(doc_ids):
            raise ValueError("Gold evidence points to absent source document")
        if len(set(self.gold_document_ids)) != len(self.gold_document_ids):
            raise ValueError("Duplicate gold document IDs")
        return self


def _case_id(dataset: str, original_id: Any) -> str:
    return dataset + "_" + hashlib.sha256(
        str(original_id).encode("utf-8")
    ).hexdigest()[:20]


def _column_records(value: Any, names: tuple[str, ...]) -> list[dict]:
    """Accept HF's dict-of-columns or list-of-dicts arrow representations."""
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        columns = [value.get(name, []) for name in names]
        if not all(isinstance(col, list) for col in columns):
            raise ValueError("Expected list-valued nested dataset fields")
        if len({len(col) for col in columns}) != 1:
            raise ValueError("Inconsistent nested dataset column lengths")
        return [
            dict(zip(names, values))
            for values in zip(*columns)
        ]
    raise ValueError("Unexpected nested Hugging Face dataset structure")


def hotpotqa_cases(row: dict, revision: str, split: str) -> list[BenchmarkCase]:
    context = _column_records(row["context"], ("title", "sentences"))
    facts = _column_records(row["supporting_facts"], ("title", "sent_id"))
    docs: list[SourceDocument] = []
    title_to_id: dict[str, str] = {}
    for index, paragraph in enumerate(context):
        title = str(paragraph["title"])
        sentences = paragraph["sentences"]
        if not isinstance(sentences, list):
            raise ValueError("HotpotQA context sentences must be a list")
        doc_id = f"d{index}"
        docs.append(SourceDocument(id=doc_id, title=title, text=" ".join(sentences)))
        title_to_id[title] = doc_id
    # The distractor configuration supplies the candidate context. This is an
    # *oracle candidate set*, not an open-world full Wikipedia retrieval test.
    if any(str(fact["title"]) not in title_to_id for fact in facts):
        raise ValueError("HotpotQA supporting title absent from context")
    return [BenchmarkCase(
        id=_case_id("hotpotqa", row["id"]),
        dataset="hotpotqa", revision=revision, split=split,
        task_type="multihop_qa",
        question=str(row["question"]),
        documents=docs,
        reference_answers=[str(row["answer"])],
        gold_document_ids=list(dict.fromkeys(
            title_to_id[str(fact["title"])] for fact in facts
        )),
        gold_sentence_refs=[
            f'{title_to_id[str(fact["title"])]}:s{fact["sent_id"]}' for fact in facts
        ],
        metadata={"original_id": str(row["id"]),
                  "question_type": row.get("type"),
                  "difficulty": row.get("level"),
                  "oracle_context": True},
    )]


def musique_cases(row: dict, revision: str, split: str) -> list[BenchmarkCase]:
    paragraphs = _column_records(
        row["paragraphs"], ("idx", "is_supporting", "paragraph_text", "title")
    )
    docs = [
        SourceDocument(
            id=f'd{paragraph["idx"]}', title=str(paragraph["title"]),
            text=str(paragraph["paragraph_text"]),
        ) for paragraph in paragraphs
    ]
    gold = [
        f'd{paragraph["idx"]}' for paragraph in paragraphs
        if paragraph["is_supporting"]
    ]
    steps = _column_records(
        row.get("question_decomposition", []),
        ("id", "question", "answer", "paragraph_support_idx"),
    )
    return [BenchmarkCase(
        id=_case_id("musique", row["id"]),
        dataset="musique", revision=revision, split=split,
        task_type="multihop_qa",
        question=str(row["question"]),
        documents=docs,
        reference_answers=[str(row["answer"])],
        gold_document_ids=gold,
        metadata={"original_id": str(row["id"]),
                  "answerable": bool(row.get("answerable", True)),
                  "hop_count": len(steps),
                  "answer_aliases": row.get("answer_aliases", []),
                  "oracle_context": True},
    )]


def multi_news_cases(row: dict, revision: str, split: str) -> list[BenchmarkCase]:
    articles = [
        article.strip() for article in str(row["document"]).split("|||||")
        if article.strip()
    ]
    docs = [
        SourceDocument(id=f"d{index}", title=f"Article {index + 1}", text=article)
        for index, article in enumerate(articles)
    ]
    summary = str(row["summary"]).strip()
    digest = hashlib.sha256(
        (str(row["document"]) + "\n" + summary).encode("utf-8")
    ).hexdigest()
    return [BenchmarkCase(
        id=_case_id("multi_news", digest),
        dataset="multi_news", revision=revision, split=split,
        task_type="multi_document_summary",
        question=("Summarize the main developments and relationships across "
                  "all supplied news articles, distinguishing major viewpoints."),
        documents=docs, reference_answers=[summary],
        # Multi-News does not annotate gold source relevance by document.
        gold_document_ids=[],
        metadata={"source_document_count": len(docs),
                  "oracle_context": True,
                  "gold_evidence_unlabeled": True},
    )]


def qasper_cases(row: dict, revision: str, split: str) -> list[BenchmarkCase]:
    sections = _column_records(
        row["full_text"], ("section_name", "paragraphs")
    )
    docs = [SourceDocument(
        id="abstract", title=str(row.get("title", "Abstract")),
        text=str(row.get("abstract", "")) or "Abstract unavailable.",
    )]
    for section_idx, section in enumerate(sections):
        for paragraph_idx, text in enumerate(section["paragraphs"]):
            if str(text).strip():
                docs.append(SourceDocument(
                    id=f"s{section_idx}_p{paragraph_idx}",
                    title=str(section["section_name"]),
                    text=str(text),
                ))
    qas = _column_records(row["qas"], ("question", "question_id", "answers"))
    cases = []
    for qa in qas:
        # Answers can contain multiple annotator records with nested answers.
        answers = qa.get("answers", [])
        annotations = []
        if isinstance(answers, dict):
            annotations = _column_records(
                answers, ("annotation_id", "answer", "worker_id")
            )
        elif isinstance(answers, list):
            annotations = answers
        reference_answers = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            raw = annotation.get("answer", annotation)
            if not isinstance(raw, dict):
                continue
            if raw.get("unanswerable"):
                continue
            text = str(raw.get("free_form_answer") or "").strip()
            if not text:
                text = " ".join(raw.get("extractive_spans") or [])
            if not text and raw.get("yes_no") is not None:
                text = "Yes" if raw["yes_no"] else "No"
            if text:
                reference_answers.append(text)
        # QASPER paper evidence is textual, not stable paragraph IDs;
        # do NOT invent gold doc labels from approximate string matching.
        cases.append(BenchmarkCase(
            id=_case_id("qasper", f'{row["id"]}/{qa["question_id"]}'),
            dataset="qasper", revision=revision, split=split,
            task_type="paper_qa", question=str(qa["question"]),
            documents=docs, reference_answers=list(dict.fromkeys(reference_answers)),
            gold_document_ids=[],
            metadata={"paper_id": row["id"],
                      "question_id": qa["question_id"],
                      "gold_evidence_unmapped": True,
                      "oracle_context": True},
        ))
    return cases


ADAPTERS = {
    "hotpotqa": hotpotqa_cases,
    "musique": musique_cases,
    "multi_news": multi_news_cases,
    "qasper": qasper_cases,
}


def adapt_row(dataset: DatasetName, row: dict, revision: str, split: str) -> list[BenchmarkCase]:
    if dataset not in ADAPTERS:
        raise ValueError(f"Unrecognized benchmark dataset: {dataset}")
    cases = ADAPTERS[dataset](row, revision, split)
    if not cases or any(not case.documents for case in cases):
        raise ValueError("Dataset adapter produced an empty case")
    return cases


def deterministic_sample(
    cases: Iterable[BenchmarkCase], count: int, seed: int
) -> list[BenchmarkCase]:
    """Order-independent bounded top-N hash sampling, not first-N bias."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    heap: list[tuple[int, str, BenchmarkCase]] = []
    for case in cases:
        score = int.from_bytes(hashlib.sha256(
            f"{seed}:{case.id}".encode("utf-8")
        ).digest()[:8], "big")
        # Keep the *smallest* hash values.
        entry = (-score, case.id, case)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def download_cases(
    dataset: DatasetName, count: int = 20, seed: int = 42,
    revision: str | None = None, split: str | None = None,
    max_source_documents: int = 50,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    """Load a dataset streaming at an immutable HF commit SHA.

    Requires pip install -r requirements-eval.txt. The Hub lookup and stream
    are network operations and are never run from unit tests.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install optional Hugging Face dependencies: "
            "pip install -r requirements-eval.txt"
        ) from exc
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    spec = DATASETS[dataset]
    repo = spec["repo"]
    resolved = HfApi().dataset_info(repo_id=repo, revision=revision or "main").sha
    if not resolved:
        raise RuntimeError(f"Unable to pin dataset revision for {repo}")
    split = split or spec["split"]
    stream = load_dataset(
        repo, spec["config"], split=split, streaming=True, revision=resolved
    )

    def iter_cases() -> Iterable[BenchmarkCase]:
        seen: set[str] = set()
        for row in stream:
            for case in adapt_row(dataset, row, resolved, split):
                if case.id in seen:
                    raise ValueError(f"Duplicate benchmark case: {case.id}")
                seen.add(case.id)
                if len(case.documents) <= max_source_documents:
                    yield case

    cases = deterministic_sample(iter_cases(), count, seed)
    if not cases:
        raise ValueError("No benchmark cases after document-count filtering")
    meta = {
        "dataset": dataset, "repo": repo, "revision": resolved,
        "config": spec["config"], "split": split,
        "seed": seed, "requested_cases": count,
        "selected_cases": len(cases),
        "max_source_documents": max_source_documents,
        "license_reference": spec["license"],
    }
    return cases, meta


def write_fixture(cases: list[BenchmarkCase], meta: dict, output: Path) -> None:
    """Store immutable content and checksum; no automatic dataset redistribution."""
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [json.dumps(case.model_dump(mode="json"),
                          ensure_ascii=False, sort_keys=True)
               for case in cases]
    output.write_text("\n".join(records) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = dict(meta, sha256=checksum, output_file=output.name)
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_fixture(path: Path, verify_manifest: bool = True) -> list[BenchmarkCase]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if verify_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest.get("sha256"):
            raise ValueError("Benchmark fixture SHA-256 does not match manifest")
    cases = [
        BenchmarkCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Repeated case IDs in fixture")
    return cases
