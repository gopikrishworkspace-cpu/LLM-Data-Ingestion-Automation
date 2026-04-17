import logging
import os
import sys

from fetcher import resolve_input
from pipeline import run_pipeline, PipelineConfig
from chunking import ChunkingConfig
from llm_client import OpenAIConfig, OpenAIClient


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Accept optional path argument:  python run.py input/myfile.txt
    raw_input = sys.argv[1] if len(sys.argv) > 1 else "input/input2.txt"

    # Resolve input: if it's a URL file, fetch the page and use real content
    input_path = resolve_input(raw_input)

    # --- LLM credentials ----------------------------------------------------
    # Get from DevTools on https://gpt.medtronic.com/apps/api-management/getting-started
    # Network tab → POST /api/tokens → Response tab:
    #   apiToken                  → ORION_API_TOKEN
    #   subscription.primaryKey   → ORION_SUBSCRIPTION_KEY
    _sub_key = os.environ.get("ORION_SUBSCRIPTION_KEY", "")
    _api_token = os.environ.get("ORION_API_TOKEN", "")

    if not _sub_key or not _api_token:
        print(
            "ERROR: Set credentials from https://gpt.medtronic.com DevTools:\n"
            "  Network → POST /api/tokens → Response tab\n"
            "  $env:ORION_API_TOKEN        = '<apiToken field>'\n"
            "  $env:ORION_SUBSCRIPTION_KEY = '<subscription.primaryKey field>'"
        )
        sys.exit(1)

    llm_client = OpenAIClient(OpenAIConfig(
        model="gpt-41",
        subscription_key=_sub_key,
        api_token=_api_token,
        timeout_seconds=60,
        temperature=0.1,
    ))

    config = PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=1500,
            chunk_overlap=150,
        ),
        llm_client=llm_client,
        step2_batch_max_chars=8_000,
        max_validation_retries=2,
        step1_max_workers=1,
    )

    result = run_pipeline(
        pdf_path=input_path,
        config=config,
    )

    print(result.to_dict())