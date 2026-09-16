from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, List

import torch
from freetoken.env import ENV
from freetoken.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.mm.config import MultimodalConfig
from freetoken.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
)


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def _prompt_admitted_reply(msg: PromptAdmittedMsg) -> UserReply:
    """Translate the scheduler's admission signal onto the existing frontend usage path."""
    return UserReply(
        uid=msg.uid,
        incremental_output="",
        finished=False,
        prompt_tokens_delta=msg.prompt_tokens,
        cached_tokens=msg.cached_tokens,
    )


def _error_reply(msg: ErrorReplyMsg) -> UserReply:
    return UserReply(
        uid=msg.uid, incremental_output="", finished=True, error=msg.error, error_code=msg.code,
    )


def _put_user_replies(send_frontend: Any, replies: List[UserReply]) -> None:
    if replies:
        send_frontend.put(
            replies[0] if len(replies) == 1 else BatchFrontendMsg(data=replies)
        )


def _send_generation_replies(
    send_frontend: Any,
    admitted: List[UserReply],
    sampled: List[UserReply],
    terminal_errors: List[UserReply],
) -> None:
    """Preserve the accounting barrier within one tokenizer queue drain.

    Scheduler abort acknowledgements are terminal: every already-sampled DetokenizeMsg
    drained alongside them must reach FrontendManager first. Admission messages remain
    first so per-request usage precedes that request's sampled completion.
    """
    _put_user_replies(send_frontend, admitted)
    _put_user_replies(send_frontend, sampled)
    _put_user_replies(send_frontend, terminal_errors)


def _public_prefix_len(tokenize_manager, msg: TokenizeMsg, tokens: torch.Tensor) -> int:
    """How many leading tokens are the same for every session, and therefore shareable.

    The system section -- system message plus tool definitions -- is byte-identical across
    sessions by construction, so a cache hit on it tells nobody anything. Everything from the
    first user turn on is that session's own. Splitting there is what keeps isolation
    affordable: re-prefilling a system prompt carrying 81 MCP tools per session does not fit
    in the KV budget, and without the split isolation would cost exactly that.

    Rendered through the same tokenizer as the real prompt, so the boundary lands on a token
    edge rather than near one. A prompt with no user turn, or one whose rendering does not
    prefix the full one, yields 0 -- nothing shared, which is the safe direction: the request
    still works, it only reuses less.
    """
    if msg.cache_ns is None or not isinstance(msg.text, list):
        return 0
    testa = []
    for m in msg.text:
        if isinstance(m, dict) and m.get("role") == "user":
            break
        testa.append(m)
    if not testa:
        return 0
    try:
        # Without the generation prompt, or it would not be a prefix: the full render ends with
        # the assistant's header, and a slice rendered the ordinary way ends with one too, in
        # the middle. Measured before this was fixed: the check below rejected every rendering
        # and the shared section was never shared at all -- two namespaces sending the same
        # 1502-token system prompt each prefilled it in full.
        pubblici = tokenize_manager.tokenize_prefix(msg, testa)
    except Exception:  # noqa: BLE001 -- a template that will not render half a chat is not an error
        return 0
    n = int(pubblici.numel())
    if n == 0 or n > int(tokens.numel()) or not torch.equal(tokens[:n], pubblici):
        return 0
    return n


def _tokenize_requests(
    tokenize_manager: Any,
    messages: List[TokenizeMsg],
    logger: Any,
) -> tuple[List[UserMsg], List[UserReply]]:
    """Tokenize independently, returning backend work plus terminal frontend errors.

    Successful tokenization deliberately emits no prompt-token reply: accounting starts
    only when the scheduler later confirms first-prefill admission.
    """
    backend: List[UserMsg] = []
    errors: List[UserReply] = []
    for msg in messages:
        try:
            user_msg = tokenize_manager.tokenize([msg])[0]
        except Exception as exc:  # noqa: BLE001 — isolate, never crash the worker
            logger.warning(f"tokenization failed for request {msg.uid}: {exc!r}")
            errors.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output="",
                    finished=True,
                    error=f"could not encode request: {exc}",
                )
            )
            continue
        # A zero-token prompt would trip the scheduler's input_len > 0 invariant and
        # crash the worker; reject it here as a terminal error instead.
        if user_msg.input_ids.numel() == 0:
            errors.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output="",
                    finished=True,
                    error="prompt must contain at least one token",
                )
            )
            continue
        backend.append(user_msg)
    return backend, errors


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
    mm: MultimodalConfig | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from freetoken.mm.processor import get_mm_processor

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer, get_mm_processor(tokenizer_path, mm))
    detokenize_manager = DetokenizeManager(
        tokenizer, load_eos_token_ids(tokenizer_path, tokenizer)
    )

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            t_received = time.monotonic() if ENV.TTFT_MARKS else 0.0
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            prompt_admitted_msg = [m for m in pending_msg if isinstance(m, PromptAdmittedMsg)]
            error_reply_msg = [m for m in pending_msg if isinstance(m, ErrorReplyMsg)]
            # Cache-rebuild control messages are pure passthrough (no tokenization):
            # CacheRebuildMsg (api -> scheduler) and CacheRebuildResultMsg (scheduler -> api).
            for m in pending_msg:
                if isinstance(m, CacheRebuildMsg):
                    send_backend.put(
                        CacheRebuildBackendMsg(
                            request_id=m.request_id,
                            moe_cache_size=m.moe_cache_size,
                            num_pages=m.num_pages,
                            num_mamba_slots=m.num_mamba_slots,
                            num_swa_pages=m.num_swa_pages,
                            mode=m.mode,
                        )
                    )
                elif isinstance(m, CacheRebuildResultMsg):
                    send_frontend.put(
                        CacheRebuildReply(
                            request_id=m.request_id,
                            status=m.status,
                            moe_cache_size=m.moe_cache_size,
                            num_pages=m.num_pages,
                            mamba_slots=m.mamba_slots,
                            num_swa_pages=m.num_swa_pages,
                            error=m.error,
                        )
                    )
            n_control = sum(
                isinstance(
                    m,
                    (CacheRebuildMsg, CacheRebuildResultMsg, ErrorReplyMsg, PromptAdmittedMsg),
                )
                for m in pending_msg
            )
            assert (
                len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) + n_control
                == len(pending_msg)
            )
            sampled_replies: List[UserReply] = []
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                sampled_replies = [
                    UserReply(
                        uid=msg.uid,
                        incremental_output=reply,
                        finished=msg.finished,
                        finish_reason=msg.finish_reason,
                        matched_stop=msg.matched_stop,
                        completion_tokens_delta=1,
                        kv_used_pages=msg.kv_used_pages,
                        kv_total_pages=msg.kv_total_pages,
                        mamba_used_slots=msg.mamba_used_slots,
                        mamba_total_slots=msg.mamba_total_slots,
                        swa_used_tokens=msg.swa_used_tokens,
                        swa_total_tokens=msg.swa_total_tokens,
                        gpu_mem_bytes=msg.gpu_mem_bytes,
                    )
                    for msg, reply in zip(detokenize_msg, replies, strict=True)
                ]

            # An error reply and a client abort are both terminal for their uid, and neither
            # produces the finished DetokenizeMsg that would release the decode state.
            for msg in error_reply_msg:
                detokenize_manager.discard(msg.uid)
            for msg in abort_msg:
                detokenize_manager.discard(msg.uid)

            _send_generation_replies(
                send_frontend,
                [_prompt_admitted_reply(msg) for msg in prompt_admitted_msg],
                sampled_replies,
                [_error_reply(msg) for msg in error_reply_msg],
            )

            if len(tokenize_msg) > 0:
                # Tokenize per-message so a single un-renderable request (e.g. a chat template
                # that rejects the message layout) becomes a terminal error reply for THAT uid
                # instead of an uncaught exception that kills the worker and bricks the server.
                t_before = time.monotonic() if ENV.TTFT_MARKS else 0.0
                backend, errors = _tokenize_requests(tokenize_manager, tokenize_msg, logger)
                if ENV.TTFT_MARKS:
                    # The chat template and the tokenizer sit on the critical path of the
                    # first token: without this number there is no telling whether the
                    # milliseconds before the scheduler sees the request are spent here or
                    # in transport.
                    logger.info(
                        "tokenize trace: %d messages, wait+sort=%.1f ms, tokenize=%.1f ms, "
                        "t_out=%.3f",
                        len(tokenize_msg), (t_before - t_received) * 1000,
                        (time.monotonic() - t_before) * 1000, time.monotonic(),
                    )
                # Prefix-cache tenancy travels with the REQUEST, not with the tokenizer, and
                # _tokenize_requests now builds the UserMsg itself: the two fields are attached
                # here instead. Paired by uid, never by position -- a request that failed to
                # encode is dropped in there, so the two lists are not the same length.
                per_uid = {m.uid: m for m in tokenize_msg}
                for um in backend:
                    src = per_uid.get(um.uid)
                    if src is None:
                        continue
                    um.cache_ns = src.cache_ns
                    um.cache_public_len = _public_prefix_len(tokenize_manager, src, um.input_ids)
                if errors:
                    send_frontend.put(
                        errors[0] if len(errors) == 1 else BatchFrontendMsg(data=errors)
                    )
                if backend:
                    send_backend.put(backend[0] if len(backend) == 1 else BatchBackendMsg(data=backend))
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
