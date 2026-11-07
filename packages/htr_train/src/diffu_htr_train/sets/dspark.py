"""A DSpark draft's training data: the target model's own answers (a draft learns what its target says, so the data is
on-policy). Each prompt of a set (a crop and its ``Text Recognition:``) is sent to the target served by vLLM
(OpenAI's chat API, greedy, thinking off; served by its path under that name, as ``htr-train serve-target`` renders
it) and its answer written as a speculators conversation, the crop as an image part; ``<stem>.target.json`` records
the target that answered, which a DSpark run checks against its own (:func:`diffu_htr_train.jobs.check_responses`). A
prompt the server fails on is left out and recorded and the rest go on; a set of which more than ``max_failed`` of the
prompts fail (or every one) is refused.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from diffu_htr_train.jobs import ResponsesMeta
from diffu_htr_train.records import Sample, meta_path, read_jsonl, read_meta, skipped_path, target_record_path
from diffu_htr_train.sets import SetRefused

PLACEHOLDER = "<image>"


class Prompted(BaseModel):
    """A prompt to send: its sample's place in the set, its image (absolute) and its text."""

    model_config = ConfigDict(frozen=True)

    number: int
    image: Path
    text: str


class ResponsesSettings(BaseModel):
    """How a response set is made: the served target (its folder, served under its path), the server (an http or https
    URL, kept without a trailing slash, so the API's paths join it), the answers' length, how long a request may take,
    how many prompts to send (None: all; the set's first ones, or a draw of them by ``seed``), how many are asked at
    once and the share that may fail."""

    model_config = ConfigDict(frozen=True)

    target: Path
    endpoint: str
    max_tokens: int = Field(default=512, gt=0)
    timeout: float = Field(default=120.0, gt=0)
    limit: int | None = Field(default=None, gt=0)
    seed: int | None = None
    concurrency: int = Field(default=1, gt=0)
    max_failed: float = Field(default=0.01, ge=0, lt=1)

    @field_validator("endpoint")
    @classmethod
    def _server_url(cls, endpoint: str) -> str:
        parts = urlsplit(endpoint)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"{endpoint!r} is no server's URL: give it as http://<host>:<port>")
        return endpoint.rstrip("/")


class ImageURL(BaseModel):
    """An image part's image: a local crop as a ``file://`` URL."""

    model_config = ConfigDict(frozen=True)

    url: str


class ImagePart(BaseModel):
    """A chat turn's image, in OpenAI's chat API."""

    model_config = ConfigDict(frozen=True)

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class TextPart(BaseModel):
    """A turn's text, in OpenAI's chat API and in a speculators conversation."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ChatTurn(BaseModel):
    """The user's turn of a chat request: the crop and the prompt."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    content: list[ImagePart | TextPart]


class TemplateArgs(BaseModel):
    """What a chat request asks of the target's chat template: thinking off."""

    model_config = ConfigDict(frozen=True)

    enable_thinking: bool = False


class ChatRequest(BaseModel):
    """A chat completion request to the served target: one user turn of the crop and the prompt, greedy."""

    model_config = ConfigDict(frozen=True)

    model: str
    messages: list[ChatTurn]
    temperature: float = 0
    max_tokens: int
    chat_template_kwargs: TemplateArgs = TemplateArgs()


class ImagePath(BaseModel):
    """A speculators conversation's image: the crop by its path."""

    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    path: str


class UserTurn(BaseModel):
    """A speculators conversation's prompt: the crop and its text."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    content: list[ImagePath | TextPart]


class AssistantTurn(BaseModel):
    """A speculators conversation's answer: the target's, as one text part (speculators loads the responses with
    ``datasets``' JSON loader, whose ``content`` column reads a bare-number string such as ``"897"`` as a number, and
    that conversation then fails; a list of parts, as the prompt's, keeps it text, the same token ids and loss mask)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] = "assistant"
    content: list[TextPart]


class Conversation(BaseModel):
    """A speculators conversation: the prompt (the crop by its path, the text) and the target's answer."""

    model_config = ConfigDict(frozen=True)

    conversations: tuple[UserTurn, AssistantTurn]


class Failed(BaseModel):
    """A prompt left out: its place in the set and the error."""

    model_config = ConfigDict(frozen=True)

    number: int
    error: str


class Responses(BaseModel):
    """What making a response set came to."""

    model_config = ConfigDict(frozen=True)

    answered: int
    failed: int


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message


class _Completion(BaseModel):
    choices: list[_Choice] = Field(min_length=1)


def prompts(path: Path, limit: int | None, seed: int | None = None) -> list[Prompted]:
    """The prompts of a set: each sample's first image and its user turn less its image placeholders; of ``limit``
    samples only (where given), the first ones or, with a ``seed``, a draw of them in the set's order.

    Raises:
        SetRefused: If the set or its meta cannot be read, they differ in length, a sample is a real one not of the
            GT's training side (a draft's responses are training data too), or a sample shows no single image.
    """
    try:
        samples, metas = read_jsonl(path, Sample), read_meta(meta_path(path))
    except (OSError, ValueError) as error:
        raise SetRefused(f"{path} and its meta are not a set: {error}") from error
    if len(samples) != len(metas):
        raise SetRefused(f"{meta_path(path)} has {len(metas)} lines for {len(samples)} samples in {path.name}")
    if leaked := next((meta for meta in metas if not meta.is_of("train")), None):
        raise SetRefused(f"{path} holds {leaked.id}, a real sample not of the GT's training side: prompt training sets")
    chosen = range(len(samples)) if limit is None or limit >= len(samples) else range(limit)
    if seed is not None and len(chosen) < len(samples):
        chosen = sorted(random.Random(seed).sample(range(len(samples)), len(chosen)))
    made: list[Prompted] = []
    for number in (index + 1 for index in chosen):
        sample = samples[number - 1]
        if len(sample.images) != 1:
            raise SetRefused(f"{path}:{number} shows {len(sample.images)} images: a draft's prompt shows one crop")
        text = sample.messages[0].content.replace(PLACEHOLDER, "").strip()
        made.append(Prompted(number=number, image=Path(sample.images[0]).resolve(), text=text))
    return made


def request(prompt: Prompted, settings: ResponsesSettings) -> ChatRequest:
    """The chat request for a prompt: the target by its served name (its path), the crop as a local file."""
    content: list[ImagePart | TextPart] = [
        ImagePart(image_url=ImageURL(url=prompt.image.as_uri())),
        TextPart(text=prompt.text),
    ]
    return ChatRequest(
        model=str(settings.target.resolve()), messages=[ChatTurn(content=content)], max_tokens=settings.max_tokens
    )


def conversation(prompt: Prompted, answer: str) -> Conversation:
    """A prompt and its answer as a speculators conversation."""
    return Conversation(
        conversations=(
            UserTurn(content=[ImagePath(path=str(prompt.image)), TextPart(text=prompt.text)]),
            AssistantTurn(content=[TextPart(text=answer)]),
        )
    )


def answer(client: httpx.Client, prompt: Prompted, settings: ResponsesSettings) -> str:
    """The target's answer to a prompt.

    Raises:
        httpx.HTTPError: If the request fails or the server answers with an error.
        ValueError: If the server's reply is not a chat completion.
    """
    reply = client.post(f"{settings.endpoint}/v1/chat/completions", json=request(prompt, settings).model_dump())
    reply.raise_for_status()
    try:
        return _Completion.model_validate(reply.json()).choices[0].message.content
    except ValidationError as error:
        raise ValueError(f"the server's reply is not a chat completion: {error}") from error


def _answered(client: httpx.Client, prompt: Prompted, settings: ResponsesSettings) -> str | Failed:
    try:
        return answer(client, prompt, settings)
    except (httpx.HTTPError, ValueError) as error:
        return Failed(number=prompt.number, error=f"{type(error).__name__}: {error}")


def write_responses(prompts_set: Path, out: Path, settings: ResponsesSettings) -> Responses:
    """Each prompt of ``prompts_set`` answered by the served target (``concurrency`` asked at once), written to ``out``
    in the prompts' order as speculators conversations; the target recorded in ``<out stem>.target.json``, the prompts
    left out in ``<out stem>.skipped.jsonl``.

    Raises:
        SetRefused: If the prompts cannot be read (:func:`prompts`), or more than ``max_failed`` of them (or all)
            failed: the set is written and recorded, to look at, but not fit to train on.
    """
    asked = prompts(prompts_set, settings.limit, settings.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    answered = failed = 0
    with ExitStack() as stack:
        limits = httpx.Limits(max_connections=settings.concurrency, max_keepalive_connections=settings.concurrency)
        client = stack.enter_context(httpx.Client(timeout=settings.timeout, limits=limits))
        pool = stack.enter_context(ThreadPoolExecutor(settings.concurrency))
        written = stack.enter_context(out.open("w", encoding="utf-8"))
        left_out = stack.enter_context(skipped_path(out).open("w", encoding="utf-8"))
        for prompt, got in zip(asked, pool.map(partial(_answered, client, settings=settings), asked), strict=True):
            if isinstance(got, Failed):
                left_out.write(got.model_dump_json() + "\n")
                failed += 1
                continue
            written.write(conversation(prompt, got).model_dump_json() + "\n")
            answered += 1
    record = ResponsesMeta(
        target=str(settings.target.resolve()),
        samples=answered,
        failed=failed,
        max_failed=settings.max_failed,
        max_tokens=settings.max_tokens,
    )
    target_record_path(out).write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    if not record.fit_to_train:
        raise SetRefused(
            f"{failed} of {len(asked)} prompts failed ({skipped_path(out)}): is the target "
            f"served at {settings.endpoint} under {settings.target.resolve()}?"
        )
    return Responses(answered=answered, failed=failed)
