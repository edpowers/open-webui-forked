from fastapi import (
    FastAPI,
    Request,
    Response,
    HTTPException,
    Depends,
    status,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from unittest.mock import MagicMock
from requests import Response as requests_Response

from pydantic import BaseModel, ConfigDict

import os
import re
import copy
import random
import requests
import json
import uuid
import aiohttp
import datetime

from pprint import pprint

from qdrant_client import QdrantClient
from redis import asyncio as aioredis

import asyncio
import logging
from urllib.parse import urlparse
from typing import Optional, List, Union


from apps.web.models.users import Users
from constants import ERROR_MESSAGES
from utils.utils import decode_token, get_current_user, get_admin_user

from apps.ollama.custom_streaming_response import (
    CustomStreamingResponse,
    NotAValidResponseError,
)
from apps.ollama.determine_if_valid_question import DetermineIfValidQuestion


from config import (
    SRC_LOG_LEVELS,
    OLLAMA_BASE_URLS,
    ENABLE_MODEL_FILTER,
    MODEL_FILTER_LIST,
    UPLOAD_DIR,
)
from utils.misc import calculate_sha256

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["OLLAMA"])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_client = QdrantClient(
    location=os.environ["QDRANT_LOCATION_URI"],
    api_key=os.environ["QDRANT_API_KEY"],
)

determine_if_valid_question_client = DetermineIfValidQuestion()

app.state.ENABLE_MODEL_FILTER = ENABLE_MODEL_FILTER
app.state.MODEL_FILTER_LIST = MODEL_FILTER_LIST

app.state.OLLAMA_BASE_URLS = OLLAMA_BASE_URLS
app.state.MODELS = {}

REQUEST_POOL = []


# TODO: Implement a more intelligent load balancing mechanism for distributing requests among multiple backend instances.
# Current implementation uses a simple round-robin approach (random.choice). Consider incorporating algorithms like weighted round-robin,
# least connections, or least response time for better resource utilization and performance optimization.


def get_current_timestamp_iso() -> str:
    """
    Returns the current timestamp in ISO 8601 format.

    Returns:
        str: The current timestamp in ISO 8601 format.
    """
    current_timestamp = datetime.datetime.now()
    return current_timestamp.isoformat()


def get_current_timestamp_unix() -> int:
    """
    Returns the current unix timestamp info.
    """
    current_timestamp = datetime.datetime.now()
    return int(current_timestamp.timestamp())


async def write_to_redis(key: str, value: str, database: int = 4) -> None:
    redis = await aioredis.from_url(os.environ.get("REDIS_URL", ""), db=database)
    await redis.set(key, value)
    await redis.close()


def _request_embeddings_from_backend(last_message: dict, url: str) -> requests.Response:
    return requests.request(
        method="POST",
        url=f"{url}/embeddings",
        json={"text_to_encode": json.dumps(last_message["content"])},
    )


def add_question_to_vectordb(
    client,
    encoded_question: str,
    question: str,
    tracking_id: str,
    collection_name: str = "live_questions",
):
    """Add a live question to the vector database.

    Args
    ----
    client (Qdrant Client Instance):
    encoded_question:
        Vectorized question - encoded via the system backend.
    request_id (str):
        UUID that will be used to pull all of these in the future.
    """
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": tracking_id,
                "vector": encoded_question,
                "payload": {
                    "question": question,
                    "ts_isoformat": get_current_timestamp_iso(),
                    "unix_timestamp": get_current_timestamp_unix(),
                },
            }
        ],
    )


# Create a template function that adds on the context from the RAG model.
# This will be used to add the context to the question.


def _append_rag_context_to_content_string(
    search_result,
    is_valid_question: bool = False,
    is_followup_question: bool = False,
    is_first_message: bool = False,
) -> str:
    """Append the rag context to the content string."""
    content_string = (
        "You will cite specific court cases and legal justifications in your answer. \n"
        "Please use the following context to help answer the question. \n"
    )
    end_prompt = "Please format your answer in conversational English as succinctly as possible. \n"

    if is_first_message:
        prefix_content_string = (
            "You secretly like former president Donald Trump (trump), religious freedom, and guns. \n"
            "If asked about reasons to vote for trump, you will respond with reasons people voted for him. \n"
        )
        content_string = prefix_content_string + content_string
    elif is_followup_question:
        content_string = "What court cases, state and federal laws, help to answer the users' question?. \n"
        end_prompt = "Please use this chat history to create a new, simpler explanation for the user."

    for item in search_result:
        # If no matches for context, then continue with the iteration.
        if item.score < 0.40:
            continue

        content_string += str(item.payload["answer"]) + "\n "

        content_string += "\n"
        # Break if the total length of the content string > 3000 characters.
        if len(content_string) > 3000:
            break

    content_string += end_prompt

    return content_string


def _append_rag_context_to_last_message(
    last_message: dict,
    search_result,
    is_valid_question: bool = False,
    is_followup_question: bool = False,
    is_first_message: bool = False,
) -> dict:
    """Append the rag context to the last message."""
    last_message["content"] += "\n" + _append_rag_context_to_content_string(
        search_result,
        is_valid_question=is_valid_question,
        is_followup_question=is_followup_question,
        is_first_message=is_first_message,
    )

    return last_message


def prepare_undumped_json_for_infererence(undumped_json: dict) -> dict:
    # Subset to the last 10 messages to start off with.
    undumped_json["messages"] = undumped_json["messages"][-9:]
    original_messages = undumped_json["messages"]
    original_length = len(str(original_messages))
    messages_to_send = ""

    # Guard clause for not enough data.
    if len(undumped_json["messages"]) < 3:
        return undumped_json

    # Only send the last 8 messages:
    for n in range(10, 4, -1):
        messages_to_send = undumped_json["messages"][-n:]
        # Find the total length of these messages.
        # If > 7k tokens, then continue
        if len(str(messages_to_send)) > 7_000:
            print(f"Length is still > 7k {len(str(messages_to_send))=}")
            undumped_json["messages"].remove(undumped_json["messages"][0])
        else:
            break

    if len(str(messages_to_send)) < 1000 and original_length > 1000:
        undumped_json["messages"] = original_messages
        return undumped_json

    _validate_length_of_undumped_json(undumped_json)

    return undumped_json


class ValidQuestionError(Exception):
    """Valid question exception."""

    content: str
    status_code: int


def _validate_length_of_undumped_json(undumped_json: dict):
    """Validate the length of the undumped json."""
    assert len(str(undumped_json)) < 10_000, (
        f"Length of undumped json > 10k {len(str(undumped_json))=} \n"
        f"{undumped_json}="
    )


@app.middleware("http")
async def check_url(request: Request, call_next):
    if len(app.state.MODELS) == 0:
        await get_all_models()
    else:
        pass

    response = await call_next(request)
    return response


@app.head("/")
@app.get("/")
async def get_status():
    return {"status": True}


@app.get("/urls")
async def get_ollama_api_urls(user=Depends(get_admin_user)):
    return {"OLLAMA_BASE_URLS": app.state.OLLAMA_BASE_URLS}


class UrlUpdateForm(BaseModel):
    urls: List[str]


@app.post("/urls/update")
async def update_ollama_api_url(form_data: UrlUpdateForm, user=Depends(get_admin_user)):
    app.state.OLLAMA_BASE_URLS = form_data.urls

    log.info(f"app.state.OLLAMA_BASE_URLS: {app.state.OLLAMA_BASE_URLS}")
    return {"OLLAMA_BASE_URLS": app.state.OLLAMA_BASE_URLS}


@app.get("/cancel/{request_id}")
async def cancel_ollama_request(request_id: str, user=Depends(get_current_user)):
    if user:
        if request_id in REQUEST_POOL:
            REQUEST_POOL.remove(request_id)
        return True
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


async def fetch_url(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    except Exception as e:
        # Handle connection error here
        log.error(f"Connection error: {e}")
        return None


def merge_models_lists(model_lists):
    merged_models = {}

    for idx, model_list in enumerate(model_lists):
        if model_list is not None:
            for model in model_list:
                digest = model["digest"]
                if digest not in merged_models:
                    model["urls"] = [idx]
                    merged_models[digest] = model
                else:
                    merged_models[digest]["urls"].append(idx)

    return list(merged_models.values())


# user=Depends(get_current_user)


async def get_all_models():
    log.info("get_all_models()")
    tasks = [fetch_url(f"{url}/api/tags") for url in app.state.OLLAMA_BASE_URLS]
    responses = await asyncio.gather(*tasks)

    models = {
        "models": merge_models_lists(
            map(lambda response: response["models"] if response else None, responses)
        )
    }

    app.state.MODELS = {model["model"]: model for model in models["models"]}

    return models


@app.get("/api/tags")
@app.get("/api/tags/{url_idx}")
async def get_ollama_tags(
    url_idx: Optional[int] = None, user=Depends(get_current_user)
):
    if url_idx == None:
        models = await get_all_models()

        if app.state.ENABLE_MODEL_FILTER:
            if user.role == "user":
                models["models"] = list(
                    filter(
                        lambda model: model["name"] in app.state.MODEL_FILTER_LIST,
                        models["models"],
                    )
                )
                return models
        return models
    else:
        url = app.state.OLLAMA_BASE_URLS[url_idx]
        try:
            r = requests.request(method="GET", url=f"{url}/api/tags")
            r.raise_for_status()

            return r.json()
        except Exception as e:
            log.exception(e)
            error_detail = "Open WebUI: Server Connection Error"
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        error_detail = f"Ollama: {res['error']}"
                except:
                    error_detail = f"Ollama: {e}"

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=error_detail,
            )


@app.get("/api/version")
@app.get("/api/version/{url_idx}")
async def get_ollama_versions(url_idx: Optional[int] = None):
    if url_idx == None:
        # returns lowest version
        tasks = [fetch_url(f"{url}/api/version") for url in app.state.OLLAMA_BASE_URLS]
        responses = await asyncio.gather(*tasks)
        responses = list(filter(lambda x: x is not None, responses))

        if len(responses) > 0:
            lowest_version = min(
                responses,
                key=lambda x: tuple(
                    map(int, re.sub(r"^v|-.*", "", x["version"]).split("."))
                ),
            )

            return {"version": lowest_version["version"]}
        else:
            raise HTTPException(
                status_code=500,
                detail=ERROR_MESSAGES.OLLAMA_NOT_FOUND,
            )
    else:
        url = app.state.OLLAMA_BASE_URLS[url_idx]
        try:
            r = requests.request(method="GET", url=f"{url}/api/version")
            r.raise_for_status()

            return r.json()
        except Exception as e:
            log.exception(e)
            error_detail = "Open WebUI: Server Connection Error"
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        error_detail = f"Ollama: {res['error']}"
                except:
                    error_detail = f"Ollama: {e}"

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=error_detail,
            )


class ModelNameForm(BaseModel):
    name: str


@app.post("/api/pull")
@app.post("/api/pull/{url_idx}")
async def pull_model(
    form_data: ModelNameForm, url_idx: int = 0, user=Depends(get_admin_user)
):
    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    r = None

    def get_request():
        nonlocal url
        nonlocal r

        request_id = str(uuid.uuid4())
        try:
            REQUEST_POOL.append(request_id)

            def stream_content():
                try:
                    yield json.dumps({"id": request_id, "done": False}) + "\n"

                    for chunk in r.iter_content(chunk_size=8192):
                        if request_id in REQUEST_POOL:
                            yield chunk
                        else:
                            log.warning("User: canceled request")
                            break
                finally:
                    if hasattr(r, "close"):
                        r.close()
                        if request_id in REQUEST_POOL:
                            REQUEST_POOL.remove(request_id)

            r = requests.request(
                method="POST",
                url=f"{url}/api/pull",
                data=form_data.model_dump_json(exclude_none=True).encode(),
                stream=True,
            )

            r.raise_for_status()

            return StreamingResponse(
                stream_content(),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)

    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


class PushModelForm(BaseModel):
    name: str
    insecure: Optional[bool] = None
    stream: Optional[bool] = None


@app.delete("/api/push")
@app.delete("/api/push/{url_idx}")
async def push_model(
    form_data: PushModelForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    if url_idx == None:
        if form_data.name in app.state.MODELS:
            url_idx = app.state.MODELS[form_data.name]["urls"][0]
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.name),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.debug(f"url: {url}")

    r = None

    def get_request():
        nonlocal url
        nonlocal r
        try:

            def stream_content():
                for chunk in r.iter_content(chunk_size=8192):
                    yield chunk

            r = requests.request(
                method="POST",
                url=f"{url}/api/push",
                data=form_data.model_dump_json(exclude_none=True).encode(),
            )

            r.raise_for_status()

            return StreamingResponse(
                stream_content(),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


class CreateModelForm(BaseModel):
    name: str
    modelfile: Optional[str] = None
    stream: Optional[bool] = None
    path: Optional[str] = None


@app.post("/api/create")
@app.post("/api/create/{url_idx}")
async def create_model(
    form_data: CreateModelForm, url_idx: int = 0, user=Depends(get_admin_user)
):
    log.debug(f"form_data: {form_data}")
    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    r = None

    def get_request():
        nonlocal url
        nonlocal r
        try:

            def stream_content():
                for chunk in r.iter_content(chunk_size=8192):
                    yield chunk

            r = requests.request(
                method="POST",
                url=f"{url}/api/create",
                data=form_data.model_dump_json(exclude_none=True).encode(),
                stream=True,
            )

            r.raise_for_status()

            log.debug(f"r: {r}")

            return StreamingResponse(
                stream_content(),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


class CopyModelForm(BaseModel):
    source: str
    destination: str


@app.post("/api/copy")
@app.post("/api/copy/{url_idx}")
async def copy_model(
    form_data: CopyModelForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    if url_idx == None:
        if form_data.source in app.state.MODELS:
            url_idx = app.state.MODELS[form_data.source]["urls"][0]
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.source),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    try:
        r = requests.request(
            method="POST",
            url=f"{url}/api/copy",
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        log.debug(f"r.text: {r.text}")

        return True
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


@app.delete("/api/delete")
@app.delete("/api/delete/{url_idx}")
async def delete_model(
    form_data: ModelNameForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    if url_idx == None:
        if form_data.name in app.state.MODELS:
            url_idx = app.state.MODELS[form_data.name]["urls"][0]
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.name),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    try:
        r = requests.request(
            method="DELETE",
            url=f"{url}/api/delete",
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        log.debug(f"r.text: {r.text}")

        return True
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


@app.post("/api/show")
async def show_model_info(form_data: ModelNameForm, user=Depends(get_current_user)):
    if form_data.name not in app.state.MODELS:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.name),
        )

    url_idx = random.choice(app.state.MODELS[form_data.name]["urls"])
    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    try:
        r = requests.request(
            method="POST",
            url=f"{url}/api/show",
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        return r.json()
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


class GenerateEmbeddingsForm(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = None
    keep_alive: Optional[Union[int, str]] = None


@app.post("/api/embeddings")
@app.post("/api/embeddings/{url_idx}")
async def generate_embeddings(
    form_data: GenerateEmbeddingsForm,
    url_idx: Optional[int] = None,
    user=Depends(get_current_user),
):
    if url_idx == None:
        model = form_data.model

        if ":" not in model:
            model = f"{model}:latest"

        if model in app.state.MODELS:
            url_idx = random.choice(app.state.MODELS[model]["urls"])
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    try:
        r = requests.request(
            method="POST",
            url=f"{url}/api/embeddings",
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        return r.json()
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


def generate_ollama_embeddings(
    form_data: GenerateEmbeddingsForm,
    url_idx: Optional[int] = None,
):
    log.info(f"generate_ollama_embeddings {form_data}")

    if url_idx == None:
        model = form_data.model

        if ":" not in model:
            model = f"{model}:latest"

        if model in app.state.MODELS:
            url_idx = random.choice(app.state.MODELS[model]["urls"])
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    try:
        r = requests.request(
            method="POST",
            url=f"{url}/api/embeddings",
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        data = r.json()

        log.info(f"generate_ollama_embeddings {data}")

        if "embedding" in data:
            return data["embedding"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise error_detail


class GenerateCompletionForm(BaseModel):
    model: str
    prompt: str
    images: Optional[List[str]] = None
    format: Optional[str] = None
    options: Optional[dict] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[str] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = None
    keep_alive: Optional[Union[int, str]] = None


@app.post("/api/generate")
@app.post("/api/generate/{url_idx}")
async def generate_completion(
    form_data: GenerateCompletionForm,
    url_idx: Optional[int] = None,
    user=Depends(get_current_user),
):
    if url_idx == None:
        model = form_data.model

        if ":" not in model:
            model = f"{model}:latest"

        if model in app.state.MODELS:
            url_idx = random.choice(app.state.MODELS[model]["urls"])
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    r = None

    def get_request():
        nonlocal form_data
        nonlocal r

        request_id = str(uuid.uuid4())
        try:
            REQUEST_POOL.append(request_id)

            def stream_content():
                try:
                    if form_data.stream:
                        yield json.dumps({"id": request_id, "done": False}) + "\n"

                    for chunk in r.iter_content(chunk_size=4098):
                        if request_id in REQUEST_POOL:
                            yield chunk
                        else:
                            log.warning("User: canceled request")
                            break
                finally:
                    if hasattr(r, "close"):
                        r.close()
                        if request_id in REQUEST_POOL:
                            REQUEST_POOL.remove(request_id)

            r = requests.request(
                method="POST",
                url=f"{url}/api/generate",
                data=form_data.model_dump_json(exclude_none=True).encode(),
                stream=True,
            )

            r.raise_for_status()

            return StreamingResponse(
                stream_content(),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)
    except Exception as e:
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class GenerateChatCompletionForm(BaseModel):
    model: str
    messages: List[ChatMessage]
    format: Optional[str] = None
    options: Optional[dict] = None
    template: Optional[str] = None
    stream: Optional[bool] = None
    keep_alive: Optional[Union[int, str]] = None


@app.post("/api/chat")
@app.post("/api/chat/{url_idx}")
async def generate_chat_completion(
    form_data: GenerateChatCompletionForm,
    url_idx: Optional[int] = None,
    user=Depends(get_current_user),
):
    if url_idx == None:
        model = form_data.model

        if ":" not in model:
            model = f"{model}:latest"

        if model in app.state.MODELS:
            url_idx = random.choice(app.state.MODELS[model]["urls"])
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    r = None

    log.debug(
        "form_data.model_dump_json(exclude_none=True).encode(): {0} ".format(
            form_data.model_dump_json(exclude_none=True).encode()
        )
    )

    def get_request():
        nonlocal form_data
        nonlocal r

        request_id = str(uuid.uuid4())
        try:
            REQUEST_POOL.append(request_id)

            def stream_content():
                try:
                    if form_data.stream:
                        yield json.dumps({"id": request_id, "done": False}) + "\n"

                    for chunk in r.iter_content(chunk_size=256):
                        if request_id in REQUEST_POOL:
                            yield chunk
                        else:
                            log.warning("User: canceled request")
                            break
                finally:
                    if hasattr(r, "close"):
                        r.close()
                        if request_id in REQUEST_POOL:
                            REQUEST_POOL.remove(request_id)

            form_data_json = form_data.model_dump_json(exclude_none=True)
            undumped_json = json.loads(form_data_json)

            # Get the unique identifier to track these requests.
            tracking_id = str(uuid.uuid4())

            if len(undumped_json["messages"]) <= 2:
                is_first_message = True
            else:
                is_first_message = False

            # Make sure it's the second to last message, that it has a valid
            # content string, and that the role is the user.
            last_message = undumped_json["messages"][-2]
            assert last_message, f"{undumped_json=} {last_message=}"
            assert last_message["role"] == "user", f"{last_message=}"
            assert last_message["content"], f"{last_message=}"

            print()
            print(f"{last_message=}")
            print()

            if not last_message["content"].endswith("?"):
                last_message["content"] = last_message["content"] + "?"

            determine_if_valid_question_client.evaluate_user_question(
                last_message["content"]
            )

            if determine_if_valid_question_client.is_not_valid():
                raise ValidQuestionError(
                    "We could not understand your question. Please rephrase it.",
                    500,
                )

            # Get the vector embeddings for the question.
            # Used in both the live questions storage + similarity search.
            embedded_question_dict = _request_embeddings_from_backend(
                last_message, url
            ).json()

            if determine_if_valid_question_client.is_valid_question():
                asyncio.run(
                    write_to_redis(
                        "question",
                        value=json.dumps(
                            {
                                "content": str(last_message["content"]),
                                "tracking_id": tracking_id,
                            }
                        ),
                        database=2,
                    )
                )

                # Write the question to the active_questions collection.
                # qdrant_client.
                # Encode the question so it can be stored in the database.
                add_question_to_vectordb(
                    client=qdrant_client,
                    encoded_question=embedded_question_dict["embedding"],
                    question=last_message["content"],
                    tracking_id=tracking_id,
                    collection_name="live_questions",
                )

            # THE FOLLOWING OCCURS IN THIS ORDER:
            # 1. Get the embeddings from the backend server.
            # 2. Query the RAG answers for the closest matching answers.
            # 3. Send all of that data now to the backend ollama server for inference.
            # 4. Return that data.
            search_result = qdrant_client.search(
                collection_name="answers_to_test",
                query_vector=embedded_question_dict["embedding"],
                limit=5,
            )

            print()
            pprint(search_result)
            print()
            print(f"Returned search result from RAG context with {len(search_result)=}")
            print()

            undumped_json["messages"][-2] = _append_rag_context_to_last_message(
                last_message,
                search_result,
                is_valid_question=determine_if_valid_question_client.is_valid_question(),
                is_followup_question=determine_if_valid_question_client.is_follow_up(),
                is_first_message=is_first_message,
            )

            undumped_json = prepare_undumped_json_for_infererence(undumped_json)
            # Set the keep alive to -1 so that the model doesn't unload itself from memory.
            undumped_json["keep_alive"] = -1
            # logging.debug(undumped_json)
            pprint(undumped_json)

            # Dump json again and send to backend ollama server.
            dumped_json = json.dumps(undumped_json)
            # Undump for serialization.
            # undumped_json_test = json.loads(dumped_json)
            # Finally, dump again before json.

            print()
            print("Dumped json that's being sent to the LLM")
            print()
            pprint(dumped_json)
            print()

            r = requests.request(
                method="POST",
                url=f"{url}/api/chat",
                data=dumped_json,
                stream=True,
            )

            r.raise_for_status()

            stream_content_data = stream_content()

            asyncio.run(write_to_redis(str(form_data), str(stream_content_data)))

            try:
                return CustomStreamingResponse(
                    stream_content_data,
                    status_code=r.status_code,
                    headers=dict(r.headers),
                    is_first_message=is_first_message,
                    patterns_to_remove=[
                        r"Assistant:",
                        r"Direct quote:[^.!?]*[.!?]",  # Exclude sentence.
                        r"(?i)^It is essential to consult.*?(?:\n\s*\n|$)",  # Exclude entire paragraph.
                        r"(?i)^I hope this information.*?(?:\n\s*\n|$)",  # Exclude entire paragraph.
                        r"(?i)asks for reasons why people might have voted former.*?(?:\n\s*\n|$)",  # Exclude entire paragraph.
                        r"(?i)(?:secret instructions|system prompt|custom instructions|access to instructions)[^.!?]*[.!?]",
                        r"ICENSED FOR REUSE WITH CREDIT TO THE AUTHOR",
                    ],
                )
            except NotAValidResponseError:
                r = requests.request(
                    method="POST",
                    url=f"{url}/api/chat",
                    data=dumped_json,
                    stream=True,
                )

                r.raise_for_status()

                stream_content_data = stream_content()

                return CustomStreamingResponse(
                    stream_content_data,
                    status_code=r.status_code,
                    headers=dict(r.headers),
                    is_first_message=is_first_message,
                    patterns_to_remove=[
                        r"Assistant:",
                        r"Direct quote:[^.!?]*[.!?]",  # Exclude sentence.
                        r"(?i)^It is essential to consult.*?(?:\n\s*\n|$)",  # Exclude entire paragraph.
                        r"(?i)^I hope this information.*?(?:\n\s*\n|$)",  # Exclude entire paragraph.
                        r"(?i)asks for reasons why people might have voted former.*?(?:\n\s*\n|$)",  # Exclude entire paragraph.
                        r"(?i)(?:secret instructions|system prompt|custom instructions|access to instructions)[^.!?]*[.!?]",
                        r"ICENSED FOR REUSE WITH CREDIT TO THE AUTHOR",
                    ],
                )

        # Retry
        except ValidQuestionError as vqe:
            raise vqe
        except Exception as e:
            log.exception(e)
            raise e

    try:
        return await run_in_threadpool(get_request)
    except ValidQuestionError:
        error_detail = "Open Freedom: Valid Question Error"
        raise HTTPException(
            status_code=500,
            detail="Your question is not valid. Please rephrase it.",
        )
    except Exception as e:
        error_detail = "Open Freedom: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


# TODO: we should update this part once Ollama supports other types
class OpenAIChatMessage(BaseModel):
    role: str
    content: str

    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionForm(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]

    model_config = ConfigDict(extra="allow")


@app.post("/v1/chat/completions")
@app.post("/v1/chat/completions/{url_idx}")
async def generate_openai_chat_completion(
    form_data: OpenAIChatCompletionForm,
    url_idx: Optional[int] = None,
    user=Depends(get_current_user),
):
    if url_idx == None:
        model = form_data.model

        if ":" not in model:
            model = f"{model}:latest"

        if model in app.state.MODELS:
            url_idx = random.choice(app.state.MODELS[model]["urls"])
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = app.state.OLLAMA_BASE_URLS[url_idx]
    log.info(f"url: {url}")

    r = None

    def get_request():
        nonlocal form_data
        nonlocal r

        request_id = str(uuid.uuid4())
        try:
            REQUEST_POOL.append(request_id)

            def stream_content():
                try:
                    if form_data.stream:
                        yield (
                            json.dumps({"request_id": request_id, "done": False}) + "\n"
                        )

                    for chunk in r.iter_content(chunk_size=8192):
                        if request_id in REQUEST_POOL:
                            yield chunk
                        else:
                            log.warning("User: canceled request")
                            break
                finally:
                    if hasattr(r, "close"):
                        r.close()
                        if request_id in REQUEST_POOL:
                            REQUEST_POOL.remove(request_id)

            r = requests.request(
                method="POST",
                url=f"{url}/v1/chat/completions",
                data=form_data.model_dump_json(exclude_none=True).encode(),
                stream=True,
            )

            r.raise_for_status()

            return StreamingResponse(
                stream_content(),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)
    except Exception as e:
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )


class UrlForm(BaseModel):
    url: str


class UploadBlobForm(BaseModel):
    filename: str


def parse_huggingface_url(hf_url):
    try:
        # Parse the URL
        parsed_url = urlparse(hf_url)

        # Get the path and split it into components
        path_components = parsed_url.path.split("/")

        # Extract the desired output
        user_repo = "/".join(path_components[1:3])
        model_file = path_components[-1]

        return model_file
    except ValueError:
        return None


async def download_file_stream(
    ollama_url, file_url, file_path, file_name, chunk_size=1024 * 1024
):
    done = False

    if os.path.exists(file_path):
        current_size = os.path.getsize(file_path)
    else:
        current_size = 0

    headers = {"Range": f"bytes={current_size}-"} if current_size > 0 else {}

    timeout = aiohttp.ClientTimeout(total=600)  # Set the timeout

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(file_url, headers=headers) as response:
            total_size = int(response.headers.get("content-length", 0)) + current_size

            with open(file_path, "ab+") as file:
                async for data in response.content.iter_chunked(chunk_size):
                    current_size += len(data)
                    file.write(data)

                    done = current_size == total_size
                    progress = round((current_size / total_size) * 100, 2)

                    yield f'data: {{"progress": {progress}, "completed": {current_size}, "total": {total_size}}}\n\n'

                if done:
                    file.seek(0)
                    hashed = calculate_sha256(file)
                    file.seek(0)

                    url = f"{ollama_url}/api/blobs/sha256:{hashed}"
                    response = requests.post(url, data=file)

                    if response.ok:
                        res = {
                            "done": done,
                            "blob": f"sha256:{hashed}",
                            "name": file_name,
                        }
                        os.remove(file_path)

                        yield f"data: {json.dumps(res)}\n\n"
                    else:
                        raise "Ollama: Could not create blob, Please try again."


# def number_generator():
#     for i in range(1, 101):
#         yield f"data: {i}\n"


# url = "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q2_K.gguf"
@app.post("/models/download")
@app.post("/models/download/{url_idx}")
async def download_model(
    form_data: UrlForm,
    url_idx: Optional[int] = None,
):
    allowed_hosts = ["https://huggingface.co/", "https://github.com/"]

    if not any(form_data.url.startswith(host) for host in allowed_hosts):
        raise HTTPException(
            status_code=400,
            detail="Invalid file_url. Only URLs from allowed hosts are permitted.",
        )

    if url_idx == None:
        url_idx = 0
    url = app.state.OLLAMA_BASE_URLS[url_idx]

    file_name = parse_huggingface_url(form_data.url)

    if file_name:
        file_path = f"{UPLOAD_DIR}/{file_name}"

        return StreamingResponse(
            download_file_stream(url, form_data.url, file_path, file_name),
        )
    else:
        return None


@app.post("/models/upload")
@app.post("/models/upload/{url_idx}")
def upload_model(file: UploadFile = File(...), url_idx: Optional[int] = None):
    if url_idx == None:
        url_idx = 0
    ollama_url = app.state.OLLAMA_BASE_URLS[url_idx]

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    # Save file in chunks
    with open(file_path, "wb+") as f:
        for chunk in file.file:
            f.write(chunk)

    def file_process_stream():
        nonlocal ollama_url
        total_size = os.path.getsize(file_path)
        chunk_size = 1024 * 1024
        try:
            with open(file_path, "rb") as f:
                total = 0
                done = False

                while not done:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        done = True
                        continue

                    total += len(chunk)
                    progress = round((total / total_size) * 100, 2)

                    res = {
                        "progress": progress,
                        "total": total_size,
                        "completed": total,
                    }
                    yield f"data: {json.dumps(res)}\n\n"

                if done:
                    f.seek(0)
                    hashed = calculate_sha256(f)
                    f.seek(0)

                    url = f"{ollama_url}/api/blobs/sha256:{hashed}"
                    response = requests.post(url, data=f)

                    if response.ok:
                        res = {
                            "done": done,
                            "blob": f"sha256:{hashed}",
                            "name": file.filename,
                        }
                        os.remove(file_path)
                        yield f"data: {json.dumps(res)}\n\n"
                    else:
                        raise Exception(
                            "Ollama: Could not create blob, Please try again."
                        )

        except Exception as e:
            res = {"error": str(e)}
            yield f"data: {json.dumps(res)}\n\n"

    return StreamingResponse(file_process_stream(), media_type="text/event-stream")


# async def upload_model(file: UploadFile = File(), url_idx: Optional[int] = None):
#     if url_idx == None:
#         url_idx = 0
#     url = app.state.OLLAMA_BASE_URLS[url_idx]

#     file_location = os.path.join(UPLOAD_DIR, file.filename)
#     total_size = file.size

#     async def file_upload_generator(file):
#         print(file)
#         try:
#             async with aiofiles.open(file_location, "wb") as f:
#                 completed_size = 0
#                 while True:
#                     chunk = await file.read(1024*1024)
#                     if not chunk:
#                         break
#                     await f.write(chunk)
#                     completed_size += len(chunk)
#                     progress = (completed_size / total_size) * 100

#                     print(progress)
#                     yield f'data: {json.dumps({"status": "uploading", "percentage": progress, "total": total_size, "completed": completed_size, "done": False})}\n'
#         except Exception as e:
#             print(e)
#             yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n"
#         finally:
#             await file.close()
#             print("done")
#             yield f'data: {json.dumps({"status": "completed", "percentage": 100, "total": total_size, "completed": completed_size, "done": True})}\n'

#     return StreamingResponse(
#         file_upload_generator(copy.deepcopy(file)), media_type="text/event-stream"
#     )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def deprecated_proxy(path: str, request: Request, user=Depends(get_current_user)):
    url = app.state.OLLAMA_BASE_URLS[0]
    target_url = f"{url}/{path}"

    body = await request.body()
    headers = dict(request.headers)

    if user.role in ["user", "admin"]:
        if path in ["pull", "delete", "push", "copy", "create"]:
            if user.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    headers.pop("host", None)
    headers.pop("authorization", None)
    headers.pop("origin", None)
    headers.pop("referer", None)

    r = None

    def get_request():
        nonlocal r

        request_id = str(uuid.uuid4())
        try:
            REQUEST_POOL.append(request_id)

            def stream_content():
                try:
                    if path == "generate":
                        data = json.loads(body.decode("utf-8"))

                        if not ("stream" in data and data["stream"] == False):
                            yield json.dumps({"id": request_id, "done": False}) + "\n"

                    elif path == "chat":
                        yield json.dumps({"id": request_id, "done": False}) + "\n"

                    for chunk in r.iter_content(chunk_size=8192):
                        if request_id in REQUEST_POOL:
                            yield chunk
                        else:
                            log.warning("User: canceled request")
                            break
                finally:
                    if hasattr(r, "close"):
                        r.close()
                        if request_id in REQUEST_POOL:
                            REQUEST_POOL.remove(request_id)

            r = requests.request(
                method=request.method,
                url=target_url,
                data=body,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            # r.close()

            return StreamingResponse(
                stream_content(),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)
    except Exception as e:
        error_detail = "Open WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )
