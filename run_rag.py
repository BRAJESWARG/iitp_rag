import argparse
import os

from pdf_rag.qa import RAGPDFQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a PDF RAG knowledge base.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--index-path", default="./store/pdf_rag_index", help="Path prefix for saved FAISS index and metadata")
    parser.add_argument("--query", help="Question to ask the PDF knowledge base")
    parser.add_argument("--build", action="store_true", help="Build the index from the PDF")
    parser.add_argument("--load", action="store_true", help="Load an existing index from disk")
    parser.add_argument("--top-k", type=int, default=8, help="Number of documents to return in initial retrieval")
    parser.add_argument("--rerank-k", type=int, default=5, help="Number of top documents to rerank and feed to the final answer generator")
    args = parser.parse_args()

    qa = RAGPDFQA(
        pdf_path=args.pdf_path,
        index_path=args.index_path,
    )

    if args.build:
        print(f"Building knowledge base from {args.pdf_path}...")
        qa.build_knowledge_base()
        print("Knowledge base built and saved.")

    if args.load:
        print(f"Loading index from {args.index_path}...")
        qa.load_knowledge_base()
        print("Index loaded.")

    if args.query:
        if qa.index is None:
            qa.load_knowledge_base()
        print("Query:", args.query)
        answer = qa.query(args.query, search_top_k=args.top_k, rerank_top_k=args.rerank_k)
        print("\nAnswer:\n", answer)
    elif not (args.build or args.load):
        parser.print_help()


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Please set OPENAI_API_KEY in your environment.")
    main()
