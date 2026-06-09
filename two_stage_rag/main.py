"""
main.py — CLI Interface for the Two-Stage RAG Pipeline
=======================================================
Interactive command-line interface that:
  1. Accepts document paths to ingest (or uses existing ChromaDB)
  2. Initializes the full Two-Stage RAG pipeline
  3. Runs an interactive query loop
  4. Displays formatted answers with stage-by-stage progress

Usage:
    # First time: ingest documents
    python main.py --ingest path/to/file.pdf path/to/file.txt

    # Subsequent queries (reuse existing ChromaDB):
    python main.py

    # Single query (non-interactive):
    python main.py --query "What is a transformer model?"
"""

import sys
import os
import argparse
import time

# Load .env FIRST before any other imports that need environment variables
from dotenv import load_dotenv

# Look for .env in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path)

# Now import pipeline (which imports llm which needs GOOGLE_API_KEY)
from pipeline import create_pipeline, rag_pipeline


# ------------------------------------------------------------------
# ANSI Color Codes for prettier terminal output
# ------------------------------------------------------------------
class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"


def print_banner():
    """Print a styled banner for the CLI."""
    banner = f"""
{Color.CYAN}{Color.BOLD}
╔══════════════════════════════════════════════════════════════╗
║          TWO-STAGE RAG PIPELINE  •  Powered by Gemini        ║
╠══════════════════════════════════════════════════════════════╣
║  Stage 1: Hybrid Search  (BM25 + Vector)  →  100 candidates  ║
║  Stage 2: Cross Encoder  Reranker         →  Top 5 docs      ║
║  Final:   Google Gemini  LLM              →  Final Answer    ║
╚══════════════════════════════════════════════════════════════╝
{Color.RESET}"""
    print(banner)


def print_separator(char="─", width=62):
    """Print a separator line."""
    print(Color.DIM + char * width + Color.RESET)


def format_answer(answer: str) -> str:
    """Format the final answer with color and indentation."""
    lines = answer.strip().split("\n")
    formatted = "\n".join(f"  {line}" for line in lines)
    return f"{Color.GREEN}{formatted}{Color.RESET}"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-Stage RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest PDF and start interactive session:
  python main.py --ingest documents/paper.pdf

  # Ingest multiple files:
  python main.py --ingest doc1.pdf doc2.txt

  # Use existing ChromaDB (already ingested):
  python main.py

  # Ask a single question and exit:
  python main.py --query "What is attention mechanism?"

  # Ingest AND ask a question:
  python main.py --ingest paper.pdf --query "Summarize the paper"
        """,
    )

    parser.add_argument(
        "--ingest",
        nargs="+",
        metavar="FILE",
        help="One or more document files to ingest (PDF or TXT)",
        default=None,
    )

    parser.add_argument(
        "--query",
        type=str,
        metavar="QUESTION",
        help="Single query to run (non-interactive mode)",
        default=None,
    )

    return parser.parse_args()


def run_single_query(pipeline, query: str):
    """
    Run a single query through the pipeline and display results.

    Args:
        pipeline: Initialized TwoStageRAGPipeline.
        query: User's question string.
    """
    print(f"\n{Color.YELLOW}{'─' * 62}{Color.RESET}")
    print(f"{Color.BOLD}Query:{Color.RESET} {query}")
    print(f"{Color.YELLOW}{'─' * 62}{Color.RESET}")

    start_time = time.time()

    # Run the full pipeline
    answer = rag_pipeline(query, pipeline)

    elapsed = time.time() - start_time

    # Display final answer
    print(f"\n{Color.BOLD}{Color.CYAN}Answer:{Color.RESET}")
    print(format_answer(answer))
    print(f"\n{Color.DIM}(Pipeline completed in {elapsed:.2f}s){Color.RESET}")
    print(f"{Color.YELLOW}{'─' * 62}{Color.RESET}\n")


def interactive_loop(pipeline):
    """
    Run an interactive question-answering loop.

    Continues prompting the user for queries until they type
    'exit', 'quit', or press Ctrl+C.

    Args:
        pipeline: Initialized TwoStageRAGPipeline.
    """
    print(f"\n{Color.CYAN}Interactive Mode — Type your questions below.{Color.RESET}")
    print(f"{Color.DIM}Commands: 'exit' or 'quit' to stop, Ctrl+C to force quit{Color.RESET}\n")

    while True:
        try:
            # Prompt user for input
            print(f"{Color.BOLD}{Color.BLUE}> Enter your query:{Color.RESET} ", end="")
            user_query = input().strip()

            # Handle exit commands
            if user_query.lower() in {"exit", "quit", "q", "bye", ":q"}:
                print(f"\n{Color.CYAN}Goodbye! Exiting the RAG pipeline.{Color.RESET}\n")
                break

            # Handle empty input
            if not user_query:
                print(f"{Color.YELLOW}  Please enter a question (not empty).{Color.RESET}")
                continue

            # Run the pipeline for this query
            run_single_query(pipeline, user_query)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print(f"\n\n{Color.CYAN}Interrupted. Exiting the RAG pipeline.{Color.RESET}\n")
            break
        except EOFError:
            # Handle piped input ending
            break


def main():
    """
    Main entry point for the Two-Stage RAG CLI.
    
    Flow:
      1. Print banner
      2. Parse CLI arguments
      3. Initialize pipeline (with or without ingestion)
      4. Run single query OR interactive loop
    """
    print_banner()
    args = parse_arguments()

    # ----------------------------------------------------------
    # Initialize the pipeline
    # ----------------------------------------------------------
    try:
        if args.ingest:
            # Validate that all provided files exist
            missing = [f for f in args.ingest if not os.path.exists(f)]
            if missing:
                print(f"{Color.RED}Error: The following files were not found:{Color.RESET}")
                for f in missing:
                    print(f"  ✗ {f}")
                sys.exit(1)

            print(f"{Color.CYAN}Ingesting {len(args.ingest)} document(s)...{Color.RESET}")
            for f in args.ingest:
                print(f"  • {f}")

            # Create pipeline with document ingestion
            pipeline = create_pipeline(file_paths=args.ingest)

        else:
            # Load existing ChromaDB (skip ingestion)
            print(f"{Color.CYAN}Loading existing ChromaDB...{Color.RESET}")
            print(f"{Color.DIM}(Tip: Use --ingest <file.pdf> to add new documents){Color.RESET}")
            pipeline = create_pipeline(file_paths=None)

    except FileNotFoundError as e:
        # ChromaDB doesn't exist yet
        print(f"\n{Color.RED}Error: No documents ingested yet!{Color.RESET}")
        print(f"{Color.YELLOW}Please ingest documents first:{Color.RESET}")
        print(f"  python main.py --ingest your_document.pdf\n")
        sys.exit(1)

    except ValueError as e:
        # API key not configured
        print(f"\n{Color.RED}{str(e)}{Color.RESET}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{Color.RED}Initialization failed: {str(e)}{Color.RESET}")
        sys.exit(1)

    # ----------------------------------------------------------
    # Run query or enter interactive loop
    # ----------------------------------------------------------
    if args.query:
        # Single query mode (non-interactive)
        run_single_query(pipeline, args.query)
    else:
        # Interactive mode — keep asking until user exits
        interactive_loop(pipeline)


# Entry point
if __name__ == "__main__":
    main()
