from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from rag_support import (
    DEFAULT_ANALYSIS_TASK,
    DEFAULT_SYSTEM_PROMPT,
    NPA_LABELS,
    load_npa_text,
    parse_upload_bytes,
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.3-chat-latest")

DEFAULT_GROUP_LABELS = {
    "base_program": "ИСХОДНАЯ ГОСУДАРСТВЕННАЯ ПРОГРАММА",
    "regional": "РЕГИОНАЛЬНАЯ ГОСУДАРСТВЕННАЯ ПРОГРАММА",
    "federal": "ФЕДЕРАЛЬНЫЕ ДОКУМЕНТЫ",
    "extra": "ДОПОЛНИТЕЛЬНЫЕ МАТЕРИАЛЫ",
    "npa": "НОРМАТИВНАЯ БАЗА",
}


@dataclass(slots=True)
class SourceText:
    source: str
    text: str
    group: str = "extra"
    title: Optional[str] = None


def get_langchain_openai_chat_model(
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
) -> ChatOpenAI:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is empty. Set it in the environment before running the pipeline.")
    return ChatOpenAI(
        api_key=SecretStr(api_key),
        base_url=base_url,
        model=model,
    )


def load_npa_sources(npa_files: Sequence[str]) -> list[SourceText]:
    sources: list[SourceText] = []
    for filename in npa_files:
        try:
            text = load_npa_text(filename)
        except Exception:  # noqa: BLE001
            continue
        if not text.strip():
            continue
        sources.append(
            SourceText(
                source=filename,
                title=NPA_LABELS.get(filename, filename),
                group="npa",
                text=text,
            )
        )
    return sources


def load_uploaded_sources(files: Iterable[tuple[str, bytes]], group: str = "extra") -> list[SourceText]:
    sources: list[SourceText] = []
    for filename, data in files:
        try:
            text = parse_upload_bytes(filename, data)
        except Exception:  # noqa: BLE001
            continue
        if not text.strip():
            continue
        sources.append(SourceText(source=filename, title=filename, group=group, text=text))
    return sources


def load_sources_from_paths(paths: Sequence[str | Path], group: str = "extra") -> list[SourceText]:
    files: list[tuple[str, bytes]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            files.append((path.name, path.read_bytes()))
        except Exception:  # noqa: BLE001
            continue
    return load_uploaded_sources(files, group=group)


def load_joined_text_from_uploads(files: Iterable[tuple[str, bytes]], label: str) -> str:
    parts: list[str] = []
    for filename, data in files:
        try:
            text = parse_upload_bytes(filename, data).strip()
        except Exception:  # noqa: BLE001
            continue
        if text:
            parts.append(f"### {label}: {filename}\n{text}")
    return "\n\n---\n\n".join(parts)


def load_joined_text_from_paths(paths: Sequence[str | Path], label: str) -> str:
    files: list[tuple[str, bytes]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            files.append((path.name, path.read_bytes()))
        except Exception:  # noqa: BLE001
            continue
    return load_joined_text_from_uploads(files, label=label)


def load_project_text_from_paths(project_paths: Sequence[str | Path]) -> str:
    return load_joined_text_from_paths(project_paths, label="Файл проекта изменений")


def load_project_text_from_uploads(project_files: Iterable[tuple[str, bytes]]) -> str:
    return load_joined_text_from_uploads(project_files, label="Файл проекта изменений")


def collect_context_sources(
    npa_files: Optional[Sequence[str]] = None,
    base_program_paths: Optional[Sequence[str | Path]] = None,
    regional_paths: Optional[Sequence[str | Path]] = None,
    federal_paths: Optional[Sequence[str | Path]] = None,
    extra_paths: Optional[Sequence[str | Path]] = None,
) -> list[SourceText]:
    source_texts: list[SourceText] = []
    if npa_files:
        source_texts.extend(load_npa_sources(npa_files))
    if base_program_paths:
        source_texts.extend(load_sources_from_paths(base_program_paths, group="base_program"))
    if regional_paths:
        source_texts.extend(load_sources_from_paths(regional_paths, group="regional"))
    if federal_paths:
        source_texts.extend(load_sources_from_paths(federal_paths, group="federal"))
    if extra_paths:
        source_texts.extend(load_sources_from_paths(extra_paths, group="extra"))
    return source_texts


def collect_context_sources_from_uploads(
    base_program_files: Optional[Iterable[tuple[str, bytes]]] = None,
    npa_files: Optional[Sequence[str]] = None,
    regional_files: Optional[Iterable[tuple[str, bytes]]] = None,
    federal_files: Optional[Iterable[tuple[str, bytes]]] = None,
    extra_files: Optional[Iterable[tuple[str, bytes]]] = None,
) -> list[SourceText]:
    source_texts: list[SourceText] = []
    if npa_files:
        source_texts.extend(load_npa_sources(npa_files))
    if base_program_files:
        source_texts.extend(load_uploaded_sources(base_program_files, group="base_program"))
    if regional_files:
        source_texts.extend(load_uploaded_sources(regional_files, group="regional"))
    if federal_files:
        source_texts.extend(load_uploaded_sources(federal_files, group="federal"))
    if extra_files:
        source_texts.extend(load_uploaded_sources(extra_files, group="extra"))
    return source_texts


def build_context_documents(
    source_texts: Sequence[SourceText],
    chunk_size: int = 1400,
    chunk_overlap: int = 250,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for source in source_texts:
        title = source.title or source.source
        base_document = Document(
            page_content=source.text,
            metadata={
                "source": source.source,
                "title": title,
                "group": source.group,
                "group_label": DEFAULT_GROUP_LABELS.get(source.group, source.group.upper()),
            },
        )
        chunks = splitter.split_documents([base_document])
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = f"{source.source}::chunk_{index}"
            chunk.metadata["chunk_order"] = index
            documents.append(chunk)
    return documents


def create_bm25_retriever(documents: Sequence[Document], top_k: int = 8):
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(list(documents))
    retriever.k = top_k
    return retriever


class BM25TextRetriever:
    def create_retriever(
        self,
        texts: list[str],
        n: int = 5,
        sources: Optional[list[str]] = None,
    ):
        source_texts = [
            SourceText(
                source=sources[index] if sources and index < len(sources) else f"source_{index + 1}",
                title=sources[index] if sources and index < len(sources) else f"source_{index + 1}",
                text=text,
            )
            for index, text in enumerate(texts)
        ]
        documents = build_context_documents(source_texts)
        return create_bm25_retriever(documents, top_k=n)


def format_documents_for_context(documents: Sequence[Document]) -> str:
    blocks: list[str] = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            "\n".join(
                [
                    f"### {metadata.get('group_label', metadata.get('group', 'UNKNOWN'))}",
                    f"Источник: {metadata.get('title', metadata.get('source', 'unknown'))}",
                    f"Чанк: {metadata.get('chunk_order', '?')}",
                    document.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_retrieval_query(project_text: str) -> str:
    return project_text


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
                    "ПРОЕКТ ИЗМЕНЕНИЙ ДЛЯ АНАЛИЗА\n\n"
                    "{project_text}\n\n"
                    "============================================================\n"
                    "РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ КОНТЕКСТА\n\n"
                    "{context}\n\n"
                    "Используй задачу анализа, проект изменений и релевантный контекст вместе. "
                    "Если в материалах есть противоречия или не хватает оснований, укажи это явно."
                ),
            ),
        ]
    )


def build_rag_chain(
    retriever,
    project_text: str,
    llm: Optional[ChatOpenAI] = None,
    base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    prompt = build_prompt_template(base_system_prompt=base_system_prompt)
    llm = llm or get_langchain_openai_chat_model()

    chain = (
        RunnableLambda(
            lambda analysis_task: {
                "analysis_task": analysis_task,
                "project_text": project_text,
                "context": format_documents_for_context(
                    retriever.invoke(build_retrieval_query(project_text))
                ),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_debug_rag_chain(
    retriever,
    project_text: str,
    llm: Optional[ChatOpenAI] = None,
    base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    prompt = build_prompt_template(base_system_prompt=base_system_prompt)
    llm = llm or get_langchain_openai_chat_model()

    def build_payload(analysis_task: str) -> dict[str, Any]:
        retrieval_query = build_retrieval_query(project_text)
        docs = retriever.invoke(retrieval_query)
        context = format_documents_for_context(docs)
        return {
            "analysis_task": analysis_task,
            "project_text": project_text,
            "retrieval_query": retrieval_query,
            "documents": docs,
            "context": context,
        }

    payload = RunnableLambda(build_payload)

    chain = payload | RunnableLambda(
        lambda data: {
            "retrieval_query": data["retrieval_query"],
            "documents": data["documents"],
            "project_text": data["project_text"],
            "context": data["context"],
            "prompt_value": prompt.invoke(
                {
                    "analysis_task": data["analysis_task"],
                    "project_text": data["project_text"],
                    "context": data["context"],
                }
            ),
            "answer": (prompt | llm | StrOutputParser()).invoke(
                {
                    "analysis_task": data["analysis_task"],
                    "project_text": data["project_text"],
                    "context": data["context"],
                }
            ),
        }
    )
    return chain


class SimpleRAGPipeline:
    def __init__(
        self,
        project_text: str,
        context_sources: Sequence[SourceText],
        top_k: int = 8,
        chunk_size: int = 1400,
        chunk_overlap: int = 250,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[ChatOpenAI] = None,
    ):
        self.project_text = project_text.strip()
        self.context_sources = list(context_sources)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_system_prompt = base_system_prompt
        self.context_documents = build_context_documents(
            source_texts=self.context_sources,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = create_bm25_retriever(self.context_documents, top_k=top_k)
        self.prompt = build_prompt_template(base_system_prompt=base_system_prompt)
        self.llm = llm or get_langchain_openai_chat_model()
        self.chain = build_rag_chain(
            retriever=self.retriever,
            project_text=self.project_text,
            llm=self.llm,
            base_system_prompt=base_system_prompt,
        )
        self.debug_chain = build_debug_rag_chain(
            retriever=self.retriever,
            project_text=self.project_text,
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
        top_k: int = 8,
        chunk_size: int = 1400,
        chunk_overlap: int = 250,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[ChatOpenAI] = None,
    ) -> "SimpleRAGPipeline":
        project_text = load_project_text_from_paths(project_paths)
        context_sources = collect_context_sources(
            npa_files=npa_files,
            base_program_paths=base_program_paths,
            regional_paths=regional_paths,
            federal_paths=federal_paths,
            extra_paths=extra_paths,
        )
        return cls(
            project_text=project_text,
            context_sources=context_sources,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_system_prompt=base_system_prompt,
            llm=llm,
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
        top_k: int = 8,
        chunk_size: int = 1400,
        chunk_overlap: int = 250,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[ChatOpenAI] = None,
    ) -> "SimpleRAGPipeline":
        project_files = list(project_files)
        base_program_files = list(base_program_files)
        project_text = load_project_text_from_uploads(project_files)
        context_sources = collect_context_sources_from_uploads(
            base_program_files=base_program_files,
            npa_files=npa_files,
            regional_files=regional_files,
            federal_files=federal_files,
            extra_files=extra_files,
        )
        return cls(
            project_text=project_text,
            context_sources=context_sources,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_system_prompt=base_system_prompt,
            llm=llm,
        )

    def retrieve(self) -> list[Document]:
        retrieval_query = build_retrieval_query(self.project_text)
        return self.retriever.invoke(retrieval_query)

    def build_context(self) -> str:
        return format_documents_for_context(self.retrieve())

    def invoke(self, analysis_task: str = DEFAULT_ANALYSIS_TASK) -> str:
        return self.chain.invoke(analysis_task)

    def debug(self, analysis_task: str = DEFAULT_ANALYSIS_TASK) -> dict[str, Any]:
        return self.debug_chain.invoke(analysis_task)
