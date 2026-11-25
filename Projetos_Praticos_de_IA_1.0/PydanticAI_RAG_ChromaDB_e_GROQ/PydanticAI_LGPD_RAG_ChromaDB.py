#!/usr/bin/env python3
"""Sistema RAG da LGPD: recuperação híbrida ChromaDB + BM25 + PydanticAI.

Baseado no código original fornecido pelo usuário. Versão revisada para APIs atuais.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PDF_PATH = Path(os.getenv("PDF_PATH", str(BASE_DIR / "LGPD.pdf"))).expanduser()
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))).expanduser()
if not PDF_PATH.is_absolute():
    PDF_PATH = BASE_DIR / PDF_PATH
if not CHROMA_DB_DIR.is_absolute():
    CHROMA_DB_DIR = BASE_DIR / CHROMA_DB_DIR
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "groq:openai/gpt-oss-120b")
ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LGPD_RAG_CHROMA_BM25")


class SearchResult(BaseModel):
    """Contextos recuperados, seus IDs e pontuações normalizadas."""

    context: str = Field(description="Trechos de texto recuperados do documento")
    source_ids: list[int] = Field(description="IDs dos trechos do documento")
    similarity_scores: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Pontuações de recuperação híbrida entre 0 e 1"
    )


class LGPDResponse(BaseModel):
    """Resposta estruturada a uma pergunta sobre a LGPD."""

    answer: str = Field(description="Resposta em português baseada no documento")
    sources: list[str] = Field(description="Identificadores dos trechos usados")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiança estimada pelo modelo")


class DocumentProcessor:
    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
        logger.info("Extraindo texto de %s", pdf_path)
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("O PDF não contém texto extraível; pode ser digitalizado.")
        logger.info("Extração concluída: %d caracteres", len(text))
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
        if chunk_size <= 0 or not (0 <= overlap < chunk_size):
            raise ValueError("CHUNK_SIZE deve ser positivo e CHUNK_OVERLAP < CHUNK_SIZE.")
        clean = re.sub(r"\s+", " ", text).strip()
        chunks: list[dict] = []
        start = 0
        n = len(clean)
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                boundary = clean.rfind(" ", start + max(1, chunk_size // 2), end)
                if boundary != -1:
                    end = boundary + 1
            part = clean[start:end].strip()
            if part:
                chunks.append({"id": len(chunks), "text": part})
            if end == n:
                break
            start = max(start + 1, end - overlap)
        logger.info("Gerados %d trechos", len(chunks))
        return chunks


class ChromaStore:
    def __init__(self, persist_dir: Path, embed_model_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_model_name = embed_model_name
        self.collection = None

    def index(self, chunks: list[dict], embedder: SentenceTransformer) -> None:
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model": self.embed_model_name, "chunks": [c["text"] for c in chunks]},
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        collection_name = f"lgpd_{fingerprint}"
        self.collection = self.client.get_or_create_collection(name=collection_name)

        if self.collection.count() == len(chunks):
            logger.info("Índice vetorial já presente: %s", collection_name)
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.collection.upsert(
            documents=texts,
            ids=[str(chunk["id"]) for chunk in chunks],
            embeddings=embeddings.tolist(),
        )
        logger.info("Indexados %d trechos no ChromaDB (%s)", len(chunks), collection_name)

    def search(self, query: str, count: int, embedder: SentenceTransformer) -> dict:
        if self.collection is None:
            raise RuntimeError("ChromaDB ainda não foi inicializado.")
        count = min(count, self.collection.count())
        if count <= 0:
            return {"ids": [[]], "distances": [[]]}
        vector = embedder.encode([query], normalize_embeddings=True).tolist()
        return self.collection.query(
            query_embeddings=vector,
            n_results=count,
            include=["distances", "documents"],
        )


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


@dataclass
class RagDependencies:
    chroma: ChromaStore
    embedder: SentenceTransformer
    bm25: BM25Okapi
    chunk_texts: list[str]


SYSTEM_PROMPT = """
Você é um assistente especializado na Lei Geral de Proteção de Dados (LGPD) do Brasil.
Responda exclusivamente em português brasileiro formal, com terminologia jurídica brasileira.
Utilize APENAS informações recuperadas pela ferramenta retrieve_context.
Sempre chame retrieve_context antes de responder à pergunta do usuário.
Se os trechos forem insuficientes, informe que não é possível responder com base no PDF.
Cite os identificadores dos trechos consultados (ex.: Trecho 3) na resposta e em sources.
Não invente artigos, citações, decisões jurídicas nem conteúdo não apresentado no documento.
Evite termos estrangeiros; prefira "responsabilização", "conformidade", "proteção de dados",
"privacidade desde a concepção", "controlador de dados", "titular dos dados",
"perfilamento", "interesse legítimo", "violação de dados" e "consentimento".
Atribua uma estimativa de confiança entre 0 e 1, considerando a relevância dos trechos.
"""


class LGPDRagSystem:
    def __init__(self) -> None:
        self.processor = DocumentProcessor()
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma = ChromaStore(CHROMA_DB_DIR, EMBEDDING_MODEL)
        self.deps: RagDependencies | None = None
        self.agent = None

    async def initialize(self) -> None:
        if not (0 <= ALPHA <= 1):
            raise ValueError("HYBRID_ALPHA deve estar entre 0 e 1.")
        if TOP_K < 1:
            raise ValueError("TOP_K deve ser >= 1.")

        text = self.processor.extract_text(PDF_PATH)
        chunks = self.processor.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            raise ValueError("Nenhum trecho foi produzido a partir do PDF.")

        texts = [chunk["text"] for chunk in chunks]
        self.chroma.index(chunks, self.embedder)
        self.deps = RagDependencies(
            chroma=self.chroma,
            embedder=self.embedder,
            bm25=BM25Okapi([tokenize(text) for text in texts]),
            chunk_texts=texts,
        )
        logger.info("Índice BM25 construído com %d trechos", len(texts))

    async def setup_agent(self) -> None:
        if self.deps is None:
            await self.initialize()

        self.agent = Agent(
            LLM_MODEL_NAME,
            deps_type=RagDependencies,
            output_type=LGPDResponse,
            instructions=SYSTEM_PROMPT,
            retries=2,
        )

        @self.agent.tool
        def retrieve_context(ctx: RunContext[RagDependencies], query: str) -> SearchResult:
            """Recupera trechos relevantes da LGPD por busca híbrida vetorial e BM25."""
            started = time.perf_counter()
            deps = ctx.deps
            total = len(deps.chunk_texts)
            size = min(total, TOP_K * 3)

            emb_res = deps.chroma.search(query, size, deps.embedder)
            emb_ids = [int(i) for i in (emb_res.get("ids") or [[]])[0]]
            distances = [float(d) for d in (emb_res.get("distances") or [[]])[0]]
            # 1 / (1 + distância) é limitada a [0, 1] para distâncias não negativas.
            vec_scores = {
                i: 1.0 / (1.0 + max(0.0, d))
                for i, d in zip(emb_ids, distances)
                if 0 <= i < total and math.isfinite(d)
            }

            raw_bm25 = [float(s) for s in deps.bm25.get_scores(tokenize(query))]
            finite = [s for s in raw_bm25 if math.isfinite(s)]
            min_s, max_s = (min(finite), max(finite)) if finite else (0.0, 0.0)
            spread = max_s - min_s
            bm25_scores = [
                (s - min_s) / spread if spread > 1e-12 and math.isfinite(s) else 0.0
                for s in raw_bm25
            ]
            candidates = set(vec_scores)
            candidates.update(sorted(range(total), key=bm25_scores.__getitem__, reverse=True)[:size])

            ranked = sorted(
                (
                    (idx, ALPHA * vec_scores.get(idx, 0.0) + (1.0 - ALPHA) * bm25_scores[idx])
                    for idx in candidates
                    if 0 <= idx < total
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:TOP_K]

            contexts = [
                f"[Trecho {idx + 1} | relevância {score:.2f}]\n{deps.chunk_texts[idx]}"
                for idx, score in ranked
            ]
            logger.info("Recuperação: %d trechos em %.2fs", len(ranked), time.perf_counter() - started)
            return SearchResult(
                context="\n\n---\n\n".join(contexts),
                source_ids=[idx + 1 for idx, _ in ranked],
                similarity_scores=[min(1.0, max(0.0, float(score))) for _, score in ranked],
            )

    async def ask(self, query: str) -> LGPDResponse:
        if self.agent is None:
            await self.setup_agent()
        if self.deps is None:
            raise RuntimeError("Sistema RAG não inicializado.")
        result = await self.agent.run(query, deps=self.deps)
        return result.output


async def interactive_session() -> None:
    print("\n" + "=" * 60)
    print(" Sistema RAG LGPD — ChromaDB + BM25 ".center(60))
    print("=" * 60)

    try:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY não configurada no .env ou no ambiente.")
        system = LGPDRagSystem()
        start = time.perf_counter()
        await system.initialize()
        await system.setup_agent()
        print(f"Sistema inicializado em {time.perf_counter() - start:.1f}s. Digite 'sair' para encerrar.")

        while True:
            try:
                question = input("\nPergunta sobre a LGPD (ou 'sair'):\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEntrada encerrada.")
                break
            if question.casefold() in {"sair", "exit", "quit"}:
                break
            if not question:
                continue
            start = time.perf_counter()
            try:
                response = await system.ask(question)
                print("\n" + "=" * 60)
                print(f"Resposta (confiança estimada: {response.confidence:.2f}):")
                print(response.answer)
                print("Fontes:", ", ".join(response.sources) or "Nenhuma")
                print(f"Tempo de resposta: {time.perf_counter() - start:.2f}s")
            except Exception:
                logger.exception("Falha ao responder pergunta")
                print("Não foi possível processar essa pergunta. Verifique os logs.")

    except Exception:
        logger.exception("Falha na inicialização do sistema")
        raise
    finally:
        print("\nSessão encerrada.")


if __name__ == "__main__":
    asyncio.run(interactive_session())
