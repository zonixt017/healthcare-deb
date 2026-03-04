import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DATA_PATH = os.getenv("PDF_DATA_PATH", "data")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "7000"))
APP_PORT = int(os.getenv("PORT", "7860"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")

# LLM provider settings
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "huggingface") or "huggingface").strip().lower()
HF_INFERENCE_API = os.getenv("HF_INFERENCE_API", "mistralai/Mistral-7B-Instruct-v0.2").strip()
HF_INFERENCE_FALLBACKS = [
    m.strip() for m in os.getenv("HF_INFERENCE_FALLBACKS", "mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-7B-Instruct").split(",") if m.strip()
]
HF_API_TIMEOUT = float(os.getenv("HF_API_TIMEOUT", "45"))
HF_TASK = (os.getenv("HF_INFERENCE_TASK", "conversational") or "conversational").strip()

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "45"))

SYSTEM_PROMPT = (
    "You are a careful healthcare information assistant. Use ONLY the supplied context. "
    "If context is insufficient, say that clearly. Keep response concise, medically responsible, "
    "and end with a short disclaimer that this is informational and not a diagnosis."
)


class RouterLLM(LLM):
    """Simple LangChain-compatible LLM wrapper for HF Inference or OpenRouter."""

    provider: str
    model_id: str
    token: str
    timeout: float = 45.0
    task: str = "conversational"
    max_new_tokens: int = 450
    temperature: float = 0.2
    base_url: str = ""

    @property
    def _llm_type(self) -> str:
        return f"router-{self.provider}"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        if self.provider == "openrouter":
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        else:
            client = InferenceClient(model=self.model_id, token=self.token, timeout=self.timeout)
            if self.task == "conversational":
                output = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                text = output.choices[0].message.content if output.choices else ""
            else:
                text = client.text_generation(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)

        if stop:
            for token in stop:
                if token and token in text:
                    text = text.split(token)[0]
                    break
        return text.strip()


def _file_signature(pdf_paths: List[Path]) -> str:
    cwd = Path.cwd().resolve()

    def _stable_path(p: Path) -> str:
        resolved = p.resolve()
        try:
            return str(resolved.relative_to(cwd))
        except ValueError:
            return str(p)

    payload = [
        {
            "path": _stable_path(p),
            "size": p.stat().st_size,
            "mtime": int(p.stat().st_mtime),
        }
        for p in sorted(pdf_paths)
    ]
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _discover_pdfs() -> List[Path]:
    root = Path(PDF_DATA_PATH)
    if not root.exists():
        return []
    return [p for p in root.rglob("*.pdf") if p.is_file()]


def _load_or_build_vectorstore() -> Tuple[FAISS, Dict[str, str]]:
    start = time.time()
    meta = {"status": "cached", "details": ""}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    store_dir = Path(VECTOR_STORE_PATH)
    manifest_file = store_dir / "manifest.json"

    pdf_paths = _discover_pdfs()
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found under `{PDF_DATA_PATH}`.")

    current_signature = _file_signature(pdf_paths)

    # Reuse existing index if signature matches
    if store_dir.exists() and (store_dir / "index.faiss").exists() and (store_dir / "index.pkl").exists() and manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("signature") == current_signature:
                vs = FAISS.load_local(str(store_dir), embeddings, allow_dangerous_deserialization=True)
                meta["status"] = "cached"
                meta["details"] = f"Loaded cached index for {len(pdf_paths)} PDF(s) in {time.time() - start:.2f}s"
                return vs, meta
        except Exception:
            pass

    # Build/rebuild index
    all_docs = []
    skipped = []
    for path in pdf_paths:
        try:
            docs = PyPDFLoader(str(path)).load()
            for d in docs:
                d.metadata["source"] = str(path)
                d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
            all_docs.extend(docs)
        except Exception as exc:
            skipped.append(f"{path.name}: {type(exc).__name__}")

    if not all_docs:
        raise RuntimeError(f"Could not load any PDFs. Skipped: {', '.join(skipped) if skipped else 'unknown error'}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    vs = FAISS.from_documents(chunks, embedding=embeddings)
    store_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(store_dir))
    manifest_file.write_text(
        json.dumps(
            {
                "signature": current_signature,
                "pdf_count": len(pdf_paths),
                "pages": len(all_docs),
                "chunks": len(chunks),
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "built_at": int(time.time()),
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta["status"] = "rebuilt"
    meta["details"] = f"Built index: {len(pdf_paths)} PDF(s), {len(all_docs)} pages, {len(chunks)} chunks in {time.time() - start:.2f}s"
    return vs, meta


def _select_llm() -> Tuple[RouterLLM, str]:
    hf_token = (os.getenv("HUGGINGFACEHUB_API_TOKEN", "") or os.getenv("HF_TOKEN", "") or "").strip()
    or_token = os.getenv("OPENROUTER_API_KEY", "").strip()

    # Provider priority: explicit provider, then fallback.
    candidates = [LLM_PROVIDER, "openrouter", "huggingface"]
    tried = []
    for provider in candidates:
        if provider == "openrouter" and or_token:
            llm = RouterLLM(
                provider="openrouter",
                model_id=OPENROUTER_MODEL,
                token=or_token,
                timeout=OPENROUTER_TIMEOUT,
                base_url=OPENROUTER_BASE_URL,
            )
            try:
                llm.invoke("Reply with exactly: ok")
                return llm, f"openrouter:{OPENROUTER_MODEL}"
            except Exception as exc:
                tried.append(f"openrouter ({type(exc).__name__})")

        if provider == "huggingface" and hf_token:
            for model in [HF_INFERENCE_API, *HF_INFERENCE_FALLBACKS]:
                llm = RouterLLM(
                    provider="huggingface",
                    model_id=model,
                    token=hf_token,
                    timeout=HF_API_TIMEOUT,
                    task=HF_TASK,
                )
                try:
                    llm.invoke("Reply with exactly: ok")
                    return llm, f"huggingface:{model}"
                except Exception as exc:
                    tried.append(f"{model} ({type(exc).__name__})")

    raise RuntimeError(
        "No working cloud LLM found. Set OPENROUTER_API_KEY for free OpenRouter models or "
        "HUGGINGFACEHUB_API_TOKEN for HF Inference API. Attempts: " + "; ".join(tried)
    )


def _format_history(history_pairs: List[Tuple[str, str]]) -> List:
    msgs = []
    for user_msg, bot_msg in history_pairs[-4:]:
        if user_msg:
            msgs.append(HumanMessage(content=user_msg))
        if bot_msg:
            msgs.append(AIMessage(content=bot_msg))
    return msgs


class RAGService:
    def __init__(self):
        self.vectorstore, self.vs_meta = _load_or_build_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": max(RETRIEVER_K * 4, 12)},
        )
        self.llm, self.llm_label = _select_llm()

    def answer(self, question: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
        q = (question or "").strip()
        if not q:
            return "Please enter a question.", ""

        docs = self.retriever.invoke(q)
        docs = docs[:RETRIEVER_K]

        context_parts = []
        sources = []
        total = 0
        for d in docs:
            page = d.metadata.get("page", "?")
            src = os.path.basename(str(d.metadata.get("source", "unknown")))
            chunk = d.page_content.strip()
            if not chunk:
                continue
            if total + len(chunk) > MAX_CONTEXT_CHARS:
                break
            total += len(chunk)
            context_parts.append(f"[{src} p.{page}] {chunk}")
            sources.append(f"- {src} (page {page})")

        context = "\n\n".join(context_parts)
        history_text = "\n".join(
            [f"User: {u}\nAssistant: {a}" for u, a in history[-3:] if u or a]
        )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Recent conversation:\n{history_text or 'N/A'}\n\n"
            f"Question:\n{q}\n\n"
            f"Retrieved context:\n{context or 'No relevant context found.'}\n\n"
            "Answer:"
        )

        answer = self.llm.invoke(prompt).strip()
        if not answer:
            answer = "I couldn't generate a grounded answer from the current knowledge base."

        source_text = "\n".join(dict.fromkeys(sources)) if sources else "No sources retrieved."
        return answer, source_text


service = RAGService()


def chat_fn(message: str, history: List[Dict]) -> str:
    pairs = []
    for item in history or []:
        pairs.append((item.get("role") == "user" and item.get("content") or "", ""))
    # Better pair extraction for type='messages'
    compact_pairs = []
    user_buffer = None
    for item in history or []:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            user_buffer = content
        elif role == "assistant":
            compact_pairs.append((user_buffer or "", content))
            user_buffer = None
    if user_buffer:
        compact_pairs.append((user_buffer, ""))

    answer, sources = service.answer(message, compact_pairs)
    return f"{answer}\n\n---\n**Sources**\n{sources}"


with gr.Blocks(title="Healthcare RAG Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🩺 Healthcare Assistant (RAG)
Ask questions grounded in your medical PDFs. Supports multi-PDF ingestion, cached indexing, and cloud LLM routing.
""")

    with gr.Accordion("System status", open=False):
        gr.Markdown(
            f"""
- **LLM**: `{service.llm_label}`
- **Vectorstore**: `{service.vs_meta['status']}`
- **Index details**: {service.vs_meta['details']}
- **Embedding model**: `{EMBEDDING_MODEL}` on `{EMBED_DEVICE}`
- **Retriever**: MMR, k={RETRIEVER_K}
"""
        )

    gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        chatbot=gr.Chatbot(height=520, type="messages"),
        textbox=gr.Textbox(placeholder="Ask a health question based on your PDFs...", lines=2),
        title="",
        description="⚠️ Informational use only. Not a substitute for professional medical advice.",
    )


if __name__ == "__main__":
    demo.launch(server_name=APP_HOST, server_port=APP_PORT)
