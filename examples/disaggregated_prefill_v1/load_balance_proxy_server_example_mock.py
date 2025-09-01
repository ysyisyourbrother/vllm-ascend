import argparse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.logger import init_logger


logger = init_logger(__name__)


class ProxyState:
    def __init__(self, decoder_host, decoder_port, tokenizer_path):
        self.decoder_host = decoder_host
        self.decoder_port = decoder_port
        self.decoder_url = f'http://{decoder_host}:{decoder_port}'
        # Increase connection pool size for better concurrency
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_connections=1000,  # Maximum total connections
                max_keepalive_connections=200  # Maximum keep-alive connections
            )
        )
        self.req_id_counter = 0

        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    async def next_req_id(self):
        self.req_id_counter += 1
        return str(self.req_id_counter)

    def generate_kv_transfer_params(self, req_data, prompt: str, request_id: str) -> tuple:
        """Generate KV transfer parameters based on prompt tokenization."""

        # Get token count
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_count = len(tokens)

        # Calculate number of blocks (token_count // 128, rounded down)
        num_blocks = token_count // 128

        # Generate random block IDs
        remote_block_ids = list(range(1, num_blocks + 1)) if num_blocks > 0 else []

        # Truncate prompt to (num_blocks * 128) tokens
        truncated_token_length = num_blocks * 128
        if truncated_token_length > 0:
            truncated_tokens = tokens[:truncated_token_length]
            truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        else:
            truncated_prompt = prompt

        print(f"Request {request_id}: token_count={token_count}, num_blocks={num_blocks}, max_output_tokens={req_data.get('max_tokens', 0)}")

        # Verify encode-decode consistency for debugging
        if truncated_token_length > 0:
            print(f"Original prompt length: {len(prompt)}, Truncated prompt length: {len(truncated_prompt)}")

        return {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": "0",
            "remote_host": self.decoder_host,
            "remote_port": self.decoder_port,  # Use the port from your curl example
            "remote_tp_size": 1,                # llmdata_dist强制检查
        }, truncated_prompt


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--decoder-host", type=str, default="localhost")
    parser.add_argument("--decoder-port", type=int, default=8002)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer (HuggingFace model path or local path)")

    parser.add_argument("--max-retries",
                        type=int,
                        default=3,
                        help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.001,
        help="Base delay (seconds) for exponential backoff retries")
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.decoder_host, global_args.decoder_port, global_args.tokenizer)
    print(f"Initialized decoder client: {proxy_state.decoder_url} (mock PD separation mode).")
    yield
    await proxy_state.client.aclose()


app = FastAPI(lifespan=lifespan)


async def _handle_completions(api: str, request: Request):
    """Handle completion requests by processing prompt and sending to decoder."""
    req_data = await request.json()
    request_id = await proxy_state.next_req_id()

    # Extract prompt from request
    prompt = req_data.get("prompt", "")
    # TODO(Brandon): output_length 写死
    req_data["max_tokens"] = 200

    # Generate KV transfer parameters and get truncated prompt
    kv_transfer_params, truncated_prompt = proxy_state.generate_kv_transfer_params(req_data, prompt, request_id)

    # Start with the original request data from benchmark_serving
    completions_data = req_data.copy()

    completions_data["prompt"] = truncated_prompt
    completions_data["kv_transfer_params"] = kv_transfer_params

    # Always send to /v1/completions endpoint
    target_url = f"{proxy_state.decoder_url}/v1/completions"

    # Stream response from decoder
    async def generate_stream():
        headers = {"Content-Type": "application/json"}
        try:
            async with proxy_state.client.stream("POST", target_url,
                                               json=completions_data, headers=headers) as response:
                response.raise_for_status()  # Check for HTTP errors

                chunk_count = 0
                async for chunk in response.aiter_bytes():
                    if chunk:  # Only yield non-empty chunks
                        chunk_count += 1
                        if chunk_count == 1:
                            print(f"First chunk received for request {request_id}", flush=True)
                        yield chunk

                print(f"Completed streaming {chunk_count} chunks for request {request_id}", flush=True)
        except Exception as e:
            logger.error(f"Error streaming from decoder: {e}")
            raise

    return StreamingResponse(generate_stream(), media_type="application/json")


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)



@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "mode": "mock_pd_separation",
        "decoder_url": proxy_state.decoder_url,
        "prefill_instances": 0  # Always 0 in mock mode
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
