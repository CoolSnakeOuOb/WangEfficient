from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "wangefficient_rules"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_markdown_files():
    documents = []

    for file_path in Path(DATA_DIR).rglob("*.md"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file_path.name
            doc.metadata["file_path"] = str(file_path)

        documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def build_vectorstore(chunks):
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


def load_vectorstore():
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


def retrieve_test(query, top_k=5):
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n[Top {rank}] Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source_file')}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print("-" * 80)
        print(doc.page_content[:1000])


def main():
    print("Loading markdown files...")
    documents = load_markdown_files()
    print(f"Loaded markdown documents: {len(documents)}")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Generated chunks: {len(chunks)}")

    print("Building ChromaDB...")
    build_vectorstore(chunks)
    print("ChromaDB built successfully.")

    test_queries = [
        "加班費怎麼計算？",
        "天然災害停止上班期間出勤薪資怎麼算？",
        "輪班人員更換班次需要休息多久？",
        "員工特休如何計算？",
    ]

    for query in test_queries:
        retrieve_test(query, top_k=3)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

