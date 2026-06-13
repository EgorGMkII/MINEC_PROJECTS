from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from rag_support import DEFAULT_ANALYSIS_TASK, DEFAULT_SYSTEM_PROMPT
from simple_rag import (
    DEFAULT_GROUP_LABELS as SIMPLE_GROUP_LABELS,
    SourceText,
    build_context_documents,
    collect_context_sources,
    collect_context_sources_from_uploads,
    create_bm25_retriever,
    format_documents_for_context,
    get_langchain_openai_chat_model,
    load_project_text_from_paths,
    load_project_text_from_uploads,
    load_sources_from_paths,
    load_uploaded_sources,
)
from summary_prompts import (
    PROJECT_CHANGES_SUMMARY_HUMAN_TEMPLATE,
    PROJECT_CHANGES_SUMMARY_SYSTEM_PROMPT,
    PROJECT_PACKAGE_SUMMARY_HUMAN_TEMPLATE,
    PROJECT_PACKAGE_SUMMARY_SYSTEM_PROMPT,
    SUPPORTING_DOCUMENT_SUMMARY_HUMAN_TEMPLATE,
    SUPPORTING_DOCUMENT_SUMMARY_SYSTEM_PROMPT,
)
from ufas_prompts import UFAS_PRECHECK_HUMAN_TEMPLATE, UFAS_PRECHECK_SYSTEM_PROMPT
from rank_bm25 import BM25Okapi


DEFAULT_GROUP_LABELS = {
    **SIMPLE_GROUP_LABELS,
    "extra_summary": "КРАТКОЕ СОДЕРЖАНИЕ ДОПОЛНИТЕЛЬНОГО ДОКУМЕНТА",
}

DEFAULT_SUMMARY_OUTPUT_DIR = Path(__file__).parent / "test_cases" / "Сх" / "summaries"
DEFAULT_SUMMARY_CHUNK_SIZE = 5000
DEFAULT_SUMMARY_CHUNK_OVERLAP = 400
UFAS_RULES_PATH = Path(__file__).parent / "УФАС" / "UFAS.txt"
DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_MAX_CHUNKS_PER_SOURCE = 8


@dataclass(slots=True)
class SummaryArtifact:
    source_name: str
    summary_kind: str
    summary_text: str
    source_hash: str
    text_path: Optional[Path] = None
    meta_path: Optional[Path] = None


def load_ufas_rules() -> str:
    if not UFAS_RULES_PATH.exists():
        return ""
    return UFAS_RULES_PATH.read_text(encoding="utf-8").strip()


def build_retrieval_query(project_package_summary: str) -> str:
    return project_package_summary.strip()


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def select_documents_with_coverage(
    documents: Sequence[Document],
    query: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
) -> list[Document]:
    if not documents:
        return []

    tokenized_corpus = [_tokenize_for_bm25(document.page_content) for document in documents]
    tokenized_query = _tokenize_for_bm25(query)
    if not tokenized_query:
        return list(documents[:top_k])

    scorer = BM25Okapi(tokenized_corpus)
    scores = scorer.get_scores(tokenized_query)
    ranked_indexes = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)

    selected_indexes: list[int] = []
    selected_sources: set[str] = set()
    chunks_per_source: dict[str, int] = defaultdict(int)

    for index in ranked_indexes:
        source = str(documents[index].metadata.get("source", "unknown"))
        if chunks_per_source[source] >= max_chunks_per_source:
            continue
        selected_indexes.append(index)
        selected_sources.add(source)
        chunks_per_source[source] += 1
        if len(selected_indexes) >= top_k:
            break

    best_index_by_source: dict[str, int] = {}
    for index in ranked_indexes:
        source = str(documents[index].metadata.get("source", "unknown"))
        if source not in best_index_by_source:
            best_index_by_source[source] = index

    for source, index in best_index_by_source.items():
        if source in selected_sources:
            continue
        if chunks_per_source[source] >= max_chunks_per_source:
            continue
        selected_indexes.append(index)
        selected_sources.add(source)
        chunks_per_source[source] += 1

    selected_documents = [documents[index] for index in selected_indexes]
    return selected_documents


def build_summary_prompt(summary_kind: str) -> ChatPromptTemplate:
    if summary_kind == "project_changes":
        return ChatPromptTemplate.from_messages(
            [
                ("system", PROJECT_CHANGES_SUMMARY_SYSTEM_PROMPT),
                ("human", PROJECT_CHANGES_SUMMARY_HUMAN_TEMPLATE),
            ]
        )
    if summary_kind == "project_package":
        return ChatPromptTemplate.from_messages(
            [
                ("system", PROJECT_PACKAGE_SUMMARY_SYSTEM_PROMPT),
                ("human", PROJECT_PACKAGE_SUMMARY_HUMAN_TEMPLATE),
            ]
        )
    if summary_kind == "supporting_document":
        return ChatPromptTemplate.from_messages(
            [
                ("system", SUPPORTING_DOCUMENT_SUMMARY_SYSTEM_PROMPT),
                ("human", SUPPORTING_DOCUMENT_SUMMARY_HUMAN_TEMPLATE),
            ]
        )
    raise ValueError(f"Unsupported summary kind: {summary_kind}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_stem(name: str) -> str:
    stem = Path(name).stem if name else "document"
    normalized = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return normalized or "document"


def _summary_cache_paths(output_dir: Path, source_name: str, summary_kind: str) -> tuple[Path, Path]:
    base_name = f"{_safe_stem(source_name)}__{summary_kind}"
    return output_dir / f"{base_name}.txt", output_dir / f"{base_name}.meta.json"


def load_cached_summary(
    output_dir: Optional[Path],
    source_name: str,
    summary_kind: str,
    source_hash: str,
) -> Optional[SummaryArtifact]:
    if output_dir is None:
        return None
    text_path, meta_path = _summary_cache_paths(output_dir, source_name, summary_kind)
    if not text_path.exists() or not meta_path.exists():
        return None

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if metadata.get("source_hash") != source_hash or metadata.get("summary_kind") != summary_kind:
        return None

    return SummaryArtifact(
        source_name=source_name,
        summary_kind=summary_kind,
        summary_text=text_path.read_text(encoding="utf-8"),
        source_hash=source_hash,
        text_path=text_path,
        meta_path=meta_path,
    )


def save_summary_to_cache(
    output_dir: Optional[Path],
    source_name: str,
    summary_kind: str,
    source_hash: str,
    summary_text: str,
) -> SummaryArtifact:
    text_path: Optional[Path] = None
    meta_path: Optional[Path] = None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        text_path, meta_path = _summary_cache_paths(output_dir, source_name, summary_kind)
        text_path.write_text(summary_text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "source_name": source_name,
                    "summary_kind": summary_kind,
                    "source_hash": source_hash,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return SummaryArtifact(
        source_name=source_name,
        summary_kind=summary_kind,
        summary_text=summary_text,
        source_hash=source_hash,
        text_path=text_path,
        meta_path=meta_path,
    )


def summarize_text(
    document_text: str,
    document_title: str,
    summary_kind: str,
    llm: ChatOpenAI,
    output_dir: Optional[Path] = None,
    source_name: Optional[str] = None,
    force_resummarize: bool = False,
) -> SummaryArtifact:
    cleaned_text = document_text.strip()
    if not cleaned_text:
        return SummaryArtifact(
            source_name=source_name or document_title,
            summary_kind=summary_kind,
            summary_text="",
            source_hash="",
        )

    resolved_source_name = source_name or document_title
    source_hash = _hash_text(cleaned_text)

    if not force_resummarize:
        cached = load_cached_summary(output_dir, resolved_source_name, summary_kind, source_hash)
        if cached is not None:
            return cached

    prompt = build_summary_prompt(summary_kind)
    summary_text = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "document_title": document_title,
            "document_text": cleaned_text,
        }
    ).strip()

    return save_summary_to_cache(
        output_dir=output_dir,
        source_name=resolved_source_name,
        summary_kind=summary_kind,
        source_hash=source_hash,
        summary_text=summary_text,
    )


def summarize_project_text(
    project_text: str,
    llm: ChatOpenAI,
    output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
    force_resummarize: bool = False,
    source_name: str = "project_changes",
) -> SummaryArtifact:
    return summarize_text(
        document_text=project_text,
        document_title="Пакет проекта изменений",
        summary_kind="project_changes",
        llm=llm,
        output_dir=output_dir,
        source_name=source_name,
        force_resummarize=force_resummarize,
    )


def summarize_project_sources(
    project_sources: Sequence[SourceText],
    llm: ChatOpenAI,
    output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
    force_resummarize: bool = False,
) -> list[SummaryArtifact]:
    artifacts: list[SummaryArtifact] = []
    for source in project_sources:
        artifact = summarize_text(
            document_text=source.text,
            document_title=source.title or source.source,
            summary_kind="project_changes",
            llm=llm,
            output_dir=output_dir,
            source_name=source.source,
            force_resummarize=force_resummarize,
        )
        artifacts.append(artifact)
    return artifacts


def format_summary_artifacts_for_prompt(
    summary_artifacts: Sequence[SummaryArtifact],
    heading_prefix: str = "Документ",
) -> str:
    blocks: list[str] = []
    for index, artifact in enumerate(summary_artifacts, start=1):
        summary_text = artifact.summary_text.strip()
        if not summary_text:
            continue
        blocks.append(
            "\n".join(
                [
                    f"### {heading_prefix} {index}: {artifact.source_name}",
                    summary_text,
                ]
            )
        )
    return "\n\n".join(blocks)


def summarize_project_package(
    project_summary_artifacts: Sequence[SummaryArtifact],
    llm: ChatOpenAI,
    output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
    force_resummarize: bool = False,
    source_name: str = "project_changes_package",
) -> SummaryArtifact:
    combined_summary_text = format_summary_artifacts_for_prompt(
        project_summary_artifacts,
        heading_prefix="Документ пакета изменений",
    ).strip()
    if not combined_summary_text:
        return SummaryArtifact(
            source_name=source_name,
            summary_kind="project_package",
            summary_text="",
            source_hash="",
        )

    return summarize_text(
        document_text=combined_summary_text,
        document_title="Пакет проекта изменений",
        summary_kind="project_package",
        llm=llm,
        output_dir=output_dir,
        source_name=source_name,
        force_resummarize=force_resummarize,
    )


def summarize_extra_sources(
    extra_sources: Sequence[SourceText],
    llm: ChatOpenAI,
    output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
    force_resummarize: bool = False,
) -> list[SummaryArtifact]:
    artifacts: list[SummaryArtifact] = []
    for source in extra_sources:
        artifact = summarize_text(
            document_text=source.text,
            document_title=source.title or source.source,
            summary_kind="supporting_document",
            llm=llm,
            output_dir=output_dir,
            source_name=source.source,
            force_resummarize=force_resummarize,
        )
        artifacts.append(artifact)
    return artifacts


def build_ufas_precheck_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", UFAS_PRECHECK_SYSTEM_PROMPT),
            ("human", UFAS_PRECHECK_HUMAN_TEMPLATE),
        ]
    )


def run_ufas_precheck(
    project_text: str,
    llm: ChatOpenAI,
    output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
    force_resummarize: bool = False,
    source_name: str = "project_changes_ufas_precheck",
) -> SummaryArtifact:
    cleaned_text = project_text.strip()
    if not cleaned_text:
        return SummaryArtifact(
            source_name=source_name,
            summary_kind="ufas_precheck",
            summary_text="",
            source_hash="",
        )

    rules_text = load_ufas_rules()
    combined_hash = _hash_text(f"{cleaned_text}\n\n===UFAS===\n\n{rules_text}")

    if not force_resummarize:
        cached = load_cached_summary(output_dir, source_name, "ufas_precheck", combined_hash)
        if cached is not None:
            return cached

    prompt = build_ufas_precheck_prompt()
    precheck_text = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "ufas_rules": rules_text,
            "project_text": cleaned_text,
        }
    ).strip()

    return save_summary_to_cache(
        output_dir=output_dir,
        source_name=source_name,
        summary_kind="ufas_precheck",
        source_hash=combined_hash,
        summary_text=precheck_text,
    )


def compose_final_answer(main_answer: str, ufas_precheck_text: str) -> str:
    main_answer = main_answer.strip()
    ufas_precheck_text = ufas_precheck_text.strip()
    if not ufas_precheck_text:
        return main_answer
    if not main_answer:
        return ufas_precheck_text
    return f"{ufas_precheck_text}\n\n______________________________\n\n{main_answer}"


def build_retrieval_sources_with_extra_summaries(
    context_sources: Sequence[SourceText],
    extra_summary_artifacts: Sequence[SummaryArtifact],
) -> list[SourceText]:
    retrieval_sources = [source for source in context_sources if source.group != "extra"]
    extra_summary_sources = [
        SourceText(
            source=f"{artifact.source_name}::summary",
            title=f"Summary: {artifact.source_name}",
            group="extra_summary",
            text=artifact.summary_text,
        )
        for artifact in extra_summary_artifacts
        if artifact.summary_text.strip()
    ]
    retrieval_sources.extend(extra_summary_sources)
    return retrieval_sources


def build_mixed_context_documents(
    source_texts: Sequence[SourceText],
    chunk_size: int = 1400,
    chunk_overlap: int = 250,
    summary_chunk_size: int = DEFAULT_SUMMARY_CHUNK_SIZE,
    summary_chunk_overlap: int = DEFAULT_SUMMARY_CHUNK_OVERLAP,
) -> list[Document]:
    regular_sources = [source for source in source_texts if source.group != "extra_summary"]
    summary_sources = [source for source in source_texts if source.group == "extra_summary"]

    documents: list[Document] = []
    if regular_sources:
        documents.extend(
            build_context_documents(
                source_texts=regular_sources,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    if summary_sources:
        documents.extend(
            build_context_documents(
                source_texts=summary_sources,
                chunk_size=summary_chunk_size,
                chunk_overlap=summary_chunk_overlap,
            )
        )
    return documents


def build_prompt_template(base_system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{base_system_prompt}\n\n"
                    "============================================================\n"
                    "ЗАДАЧА АНАЛИЗА\n\n"
                    "{analysis_task}\n\n"
                    "============================================================\n"
                    "ЕДИНОЕ SUMMARY ПАКЕТА ИЗМЕНЕНИЙ\n\n"
                    "{project_package_summary}\n\n"
                    "============================================================\n"
                    "SUMMARY ДОКУМЕНТОВ ИЗ СЛОТА ИЗМЕНЕНИЙ\n\n"
                    "{project_document_summaries}\n\n"
                    "============================================================\n"
                    "SUMMARY ДОПОЛНИТЕЛЬНЫХ ДОКУМЕНТОВ\n\n"
                    "{extra_document_summaries}\n\n"
                    "============================================================\n"
                    "РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ КОНТЕКСТА\n\n"
                    "{context}\n\n"
                    "Используй единое summary пакета изменений как сжатое описание объекта анализа. "
                    "Summary отдельных документов из слота изменений и summary дополнительных документов используй "
                    "как уточняющий материал. Релевантный контекст используй как основание для проверки. "
                    "Если в материалах есть противоречия или не хватает оснований, укажи это явно."
                ),
            ),
        ]
    )


def build_rag_chain(
    documents: Sequence[Document],
    project_package_summary: str,
    project_document_summaries: str,
    extra_document_summaries: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
    llm: Optional[ChatOpenAI] = None,
    base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    prompt = build_prompt_template(base_system_prompt=base_system_prompt)
    llm = llm or get_langchain_openai_chat_model()

    chain = (
        RunnableLambda(
            lambda analysis_task: {
                "analysis_task": analysis_task,
                "project_package_summary": project_package_summary,
                "project_document_summaries": project_document_summaries,
                "extra_document_summaries": extra_document_summaries,
                "context": format_documents_for_context(
                    select_documents_with_coverage(
                        documents=documents,
                        query=build_retrieval_query(project_package_summary),
                        top_k=top_k,
                        max_chunks_per_source=max_chunks_per_source,
                    )
                ),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_debug_rag_chain(
    documents: Sequence[Document],
    project_text: str,
    project_package_summary: str,
    project_document_summary_artifacts: Sequence[SummaryArtifact],
    extra_summary_artifacts: Sequence[SummaryArtifact],
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
    llm: Optional[ChatOpenAI] = None,
    base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    prompt = build_prompt_template(base_system_prompt=base_system_prompt)
    llm = llm or get_langchain_openai_chat_model()

    project_document_summaries = format_summary_artifacts_for_prompt(
        project_document_summary_artifacts,
        heading_prefix="Документ из пакета изменений",
    )
    extra_document_summaries = format_summary_artifacts_for_prompt(
        extra_summary_artifacts,
        heading_prefix="Дополнительный документ",
    )

    def invoke_model(inputs: dict[str, str]) -> dict[str, Any]:
        message = (prompt | llm).invoke(inputs)
        usage = getattr(message, "usage_metadata", None) or message.response_metadata.get("token_usage", {})
        return {
            "text": message.content,
            "usage": usage,
            "response_metadata": getattr(message, "response_metadata", {}),
        }

    def serialize_artifact(artifact: SummaryArtifact) -> dict[str, Any]:
        return {
            "source_name": artifact.source_name,
            "summary_kind": artifact.summary_kind,
            "summary_text": artifact.summary_text,
            "text_path": str(artifact.text_path) if artifact.text_path else None,
            "meta_path": str(artifact.meta_path) if artifact.meta_path else None,
        }

    def build_payload(analysis_task: str) -> dict[str, Any]:
        retrieval_query = build_retrieval_query(project_package_summary)
        docs = select_documents_with_coverage(
            documents=documents,
            query=retrieval_query,
            top_k=top_k,
            max_chunks_per_source=max_chunks_per_source,
        )
        context = format_documents_for_context(docs)
        return {
            "analysis_task": analysis_task,
            "project_text": project_text,
            "project_package_summary": project_package_summary,
            "project_document_summaries": project_document_summaries,
            "extra_document_summaries": extra_document_summaries,
            "retrieval_query": retrieval_query,
            "documents": docs,
            "context": context,
            "project_summary_artifacts": [
                serialize_artifact(artifact) for artifact in project_document_summary_artifacts
            ],
            "extra_summary_artifacts": [
                serialize_artifact(artifact) for artifact in extra_summary_artifacts
            ],
        }

    payload = RunnableLambda(build_payload)

    chain = payload | RunnableLambda(
        lambda data: {
            "retrieval_query": data["retrieval_query"],
            "documents": data["documents"],
            "project_text": data["project_text"],
            "project_package_summary": data["project_package_summary"],
            "project_document_summaries": data["project_document_summaries"],
            "extra_document_summaries": data["extra_document_summaries"],
            "project_summary_artifacts": data["project_summary_artifacts"],
            "extra_summary_artifacts": data["extra_summary_artifacts"],
            "context": data["context"],
            "prompt_value": prompt.invoke(
                {
                    "analysis_task": data["analysis_task"],
                    "project_package_summary": data["project_package_summary"],
                    "project_document_summaries": data["project_document_summaries"],
                    "extra_document_summaries": data["extra_document_summaries"],
                    "context": data["context"],
                }
            ),
            "model_output": invoke_model(
                {
                    "analysis_task": data["analysis_task"],
                    "project_package_summary": data["project_package_summary"],
                    "project_document_summaries": data["project_document_summaries"],
                    "extra_document_summaries": data["extra_document_summaries"],
                    "context": data["context"],
                }
            ),
        }
    )
    return chain


class SummaryRAGPipeline:
    def __init__(
        self,
        project_text: str,
        project_sources: Sequence[SourceText],
        context_sources: Sequence[SourceText],
        top_k: int = 20,
        chunk_size: int = 1400,
        chunk_overlap: int = 250,
        summary_chunk_size: int = DEFAULT_SUMMARY_CHUNK_SIZE,
        summary_chunk_overlap: int = DEFAULT_SUMMARY_CHUNK_OVERLAP,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[ChatOpenAI] = None,
        summary_output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
        force_resummarize: bool = False,
    ):
        self.project_text = project_text.strip()
        self.project_sources = list(project_sources)
        self.raw_context_sources = list(context_sources)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summary_chunk_size = summary_chunk_size
        self.summary_chunk_overlap = summary_chunk_overlap
        self.retrieval_top_k = retrieval_top_k
        self.max_chunks_per_source = max_chunks_per_source
        self.base_system_prompt = base_system_prompt
        self.summary_output_dir = Path(summary_output_dir) if summary_output_dir is not None else None
        self.llm = llm or get_langchain_openai_chat_model()
        self.ufas_precheck_artifact = run_ufas_precheck(
            project_text=self.project_text,
            llm=self.llm,
            output_dir=self.summary_output_dir,
            force_resummarize=force_resummarize,
        )
        self.ufas_precheck_text = self.ufas_precheck_artifact.summary_text

        self.project_document_summary_artifacts = summarize_project_sources(
            project_sources=self.project_sources,
            llm=self.llm,
            output_dir=self.summary_output_dir,
            force_resummarize=force_resummarize,
        )
        self.project_package_summary_artifact = summarize_project_package(
            project_summary_artifacts=self.project_document_summary_artifacts,
            llm=self.llm,
            output_dir=self.summary_output_dir,
            force_resummarize=force_resummarize,
        )
        self.project_summary_artifact = self.project_package_summary_artifact
        self.project_summary = self.project_package_summary_artifact.summary_text
        self.project_document_summaries = format_summary_artifacts_for_prompt(
            self.project_document_summary_artifacts,
            heading_prefix="Документ из пакета изменений",
        )

        self.extra_sources = [source for source in self.raw_context_sources if source.group == "extra"]
        self.extra_summary_artifacts = summarize_extra_sources(
            extra_sources=self.extra_sources,
            llm=self.llm,
            output_dir=self.summary_output_dir,
            force_resummarize=force_resummarize,
        )
        self.extra_document_summaries = format_summary_artifacts_for_prompt(
            self.extra_summary_artifacts,
            heading_prefix="Дополнительный документ",
        )

        self.summary_artifacts = [
            self.ufas_precheck_artifact,
            *self.project_document_summary_artifacts,
            self.project_package_summary_artifact,
            *self.extra_summary_artifacts,
        ]

        self.context_sources = build_retrieval_sources_with_extra_summaries(
            context_sources=self.raw_context_sources,
            extra_summary_artifacts=self.extra_summary_artifacts,
        )
        self.context_documents = build_mixed_context_documents(
            source_texts=self.context_sources,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            summary_chunk_size=summary_chunk_size,
            summary_chunk_overlap=summary_chunk_overlap,
        )
        self.retriever = create_bm25_retriever(self.context_documents, top_k=top_k)
        self.prompt = build_prompt_template(base_system_prompt=base_system_prompt)
        self.chain = build_rag_chain(
            documents=self.context_documents,
            project_package_summary=self.project_summary,
            project_document_summaries=self.project_document_summaries,
            extra_document_summaries=self.extra_document_summaries,
            top_k=retrieval_top_k,
            max_chunks_per_source=max_chunks_per_source,
            llm=self.llm,
            base_system_prompt=base_system_prompt,
        )
        self.debug_chain = build_debug_rag_chain(
            documents=self.context_documents,
            project_text=self.project_text,
            project_package_summary=self.project_summary,
            project_document_summary_artifacts=self.project_document_summary_artifacts,
            extra_summary_artifacts=self.extra_summary_artifacts,
            top_k=retrieval_top_k,
            max_chunks_per_source=max_chunks_per_source,
            llm=self.llm,
            base_system_prompt=base_system_prompt,
        )

    @classmethod
    def from_project_paths(
        cls,
        project_paths: Sequence[str | Path],
        base_program_paths: Sequence[str | Path],
        npa_files: Optional[Sequence[str]] = None,
        regional_paths: Optional[Sequence[str | Path]] = None,
        federal_paths: Optional[Sequence[str | Path]] = None,
        extra_paths: Optional[Sequence[str | Path]] = None,
        top_k: int = 20,
        chunk_size: int = 1400,
        chunk_overlap: int = 250,
        summary_chunk_size: int = DEFAULT_SUMMARY_CHUNK_SIZE,
        summary_chunk_overlap: int = DEFAULT_SUMMARY_CHUNK_OVERLAP,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[ChatOpenAI] = None,
        summary_output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
        force_resummarize: bool = False,
    ) -> "SummaryRAGPipeline":
        project_text = load_project_text_from_paths(project_paths)
        project_sources = load_sources_from_paths(project_paths, group="project")
        context_sources = collect_context_sources(
            npa_files=npa_files,
            base_program_paths=base_program_paths,
            regional_paths=regional_paths,
            federal_paths=federal_paths,
            extra_paths=extra_paths,
        )
        return cls(
            project_text=project_text,
            project_sources=project_sources,
            context_sources=context_sources,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            summary_chunk_size=summary_chunk_size,
            summary_chunk_overlap=summary_chunk_overlap,
            retrieval_top_k=retrieval_top_k,
            max_chunks_per_source=max_chunks_per_source,
            base_system_prompt=base_system_prompt,
            llm=llm,
            summary_output_dir=summary_output_dir,
            force_resummarize=force_resummarize,
        )

    @classmethod
    def from_uploads(
        cls,
        project_files: Iterable[tuple[str, bytes]],
        base_program_files: Iterable[tuple[str, bytes]],
        npa_files: Optional[Sequence[str]] = None,
        regional_files: Optional[Iterable[tuple[str, bytes]]] = None,
        federal_files: Optional[Iterable[tuple[str, bytes]]] = None,
        extra_files: Optional[Iterable[tuple[str, bytes]]] = None,
        top_k: int = 20,
        chunk_size: int = 1400,
        chunk_overlap: int = 250,
        summary_chunk_size: int = DEFAULT_SUMMARY_CHUNK_SIZE,
        summary_chunk_overlap: int = DEFAULT_SUMMARY_CHUNK_OVERLAP,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[ChatOpenAI] = None,
        summary_output_dir: Optional[Path] = DEFAULT_SUMMARY_OUTPUT_DIR,
        force_resummarize: bool = False,
    ) -> "SummaryRAGPipeline":
        project_files = list(project_files)
        base_program_files = list(base_program_files)
        project_text = load_project_text_from_uploads(project_files)
        project_sources = load_uploaded_sources(project_files, group="project")
        context_sources = collect_context_sources_from_uploads(
            base_program_files=base_program_files,
            npa_files=npa_files,
            regional_files=regional_files,
            federal_files=federal_files,
            extra_files=extra_files,
        )
        return cls(
            project_text=project_text,
            project_sources=project_sources,
            context_sources=context_sources,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            summary_chunk_size=summary_chunk_size,
            summary_chunk_overlap=summary_chunk_overlap,
            retrieval_top_k=retrieval_top_k,
            max_chunks_per_source=max_chunks_per_source,
            base_system_prompt=base_system_prompt,
            llm=llm,
            summary_output_dir=summary_output_dir,
            force_resummarize=force_resummarize,
        )

    def retrieve(self) -> list[Document]:
        retrieval_query = build_retrieval_query(self.project_summary)
        return select_documents_with_coverage(
            documents=self.context_documents,
            query=retrieval_query,
            top_k=self.retrieval_top_k,
            max_chunks_per_source=self.max_chunks_per_source,
        )

    def build_context(self) -> str:
        return format_documents_for_context(self.retrieve())

    def invoke(self, analysis_task: str = DEFAULT_ANALYSIS_TASK) -> str:
        main_answer = self.chain.invoke(analysis_task)
        return compose_final_answer(main_answer, self.ufas_precheck_text)

    def invoke_with_usage(self, analysis_task: str = DEFAULT_ANALYSIS_TASK) -> dict[str, Any]:
        debug_data = self.debug(analysis_task)
        return {
            "answer": debug_data["answer"],
            "usage": debug_data["usage"],
            "response_metadata": debug_data["response_metadata"],
            "retrieval_query": debug_data["retrieval_query"],
            "context": debug_data["context"],
            "project_package_summary": debug_data["project_package_summary"],
            "project_document_summaries": debug_data["project_document_summaries"],
            "extra_document_summaries": debug_data["extra_document_summaries"],
            "ufas_precheck": debug_data["ufas_precheck"],
        }

    def debug(self, analysis_task: str = DEFAULT_ANALYSIS_TASK) -> dict[str, Any]:
        debug_data = self.debug_chain.invoke(analysis_task)
        model_output = debug_data.pop("model_output")
        debug_data["ufas_precheck"] = self.ufas_precheck_text
        debug_data["answer"] = compose_final_answer(model_output["text"], self.ufas_precheck_text)
        debug_data["usage"] = model_output["usage"]
        debug_data["response_metadata"] = model_output["response_metadata"]
        return debug_data


SimpleRAGPipeline = SummaryRAGPipeline
