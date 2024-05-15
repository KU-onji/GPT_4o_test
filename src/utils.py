import os
import time
import uuid

import requests
from openai import OpenAI


def create_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY_KUMA"))


def retrieve_pdf(url: str) -> bytes:
    response = requests.get(url)
    assert response.status_code == 200, "Failed to retrieve PDF file"
    print("PDF file retrieved successfully.")
    return bytes(response.content)


def create_pdf_file(pdf_bytes: bytes) -> None:
    filename = str(uuid.uuid4())
    path = f"pdfs/{filename}.pdf"
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    print("PDF file created successfully.")
    return path


def create_vector_store(client: OpenAI, name: str = "Kuma_storage") -> str:
    res_vs_create = client.beta.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    assert res_vs_create.status == "completed", f"Failed to create vector store: {res_vs_create.id}"
    print("Vector store created successfully.")
    print(f"Vector store ID: {res_vs_create.id}")
    return res_vs_create.id


def upload_file_to_vs(client: OpenAI, vector_store_id: str, file) -> None:
    res_file_create = client.beta.vector_stores.files.upload_and_poll(vector_store_id=vector_store_id, file=file)
    while res_file_create.status == "in_progress":
        time.sleep(1)
    assert res_file_create.status == "completed", "Failed to upload file to vector store"
    print("File uploaded successfully.")


def delete_vector_store(client: OpenAI, vector_store_id: str) -> None:
    res_vs_delete = client.beta.vector_stores.delete(vector_store_id=vector_store_id)
    assert res_vs_delete.deleted, f"Failed to delete vector store: {vector_store_id}"
    print(f"Vector store {vector_store_id} deleted successfully.")


def delete_all_vector_stores(client: OpenAI) -> None:
    vector_stores = client.beta.vector_stores.list().data
    for vector_store in vector_stores:
        delete_vector_store(client, vector_store.id)
    print("All vector stores deleted!")


def create_file_search_assistant(
    client: OpenAI,
    model: str = "gpt-4o",
    name: str = "Kuma_Bot_turbo",
    instructions: str = "You are a helpful assistant and have plenty of knowledge about informatics.",
    tools: list[dict] = [{"type": "file_search"}],
):
    assistant = client.beta.assistants.create(
        model=model,
        name=name,
        instructions=instructions,
        tools=tools,
    )
    return assistant


def set_vs_id(client: OpenAI, assistant_id: str, vector_store_id: str):
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
    return assistant


def delete_assistant(client: OpenAI, assistant_id: str) -> None:
    res_assistant_delete = client.beta.assistants.delete(assistant_id=assistant_id)
    assert res_assistant_delete.deleted, f"Failed to delete assistant: {assistant_id}"
    print(f"Assistant {assistant_id} deleted successfully.")


def create_summarization_thread(client: OpenAI):
    prompt = "与えられた機械学習に関する論文のPDFを読み、この論文のSummary、ProsとCons、技術的な新規性、実験で何を評価したのかを以下のような形式で日本語で述べてください。\n\n\
        # Summary:\n\
        # Pros:\n\
        # Cons:\n\
        # 技術的な新規性:\n\
        # 実験評価:\
        "
    thread = client.beta.threads.create(messages=[{"role": "user", "content": prompt}])
    return thread


def run_thread(client: OpenAI, thread_id: str, assistant_id: str):
    run = client.beta.threads.runs.create_and_poll(thread_id=thread_id, assistant_id=assistant_id)
    return run


def cancel_run(client: OpenAI, run_id: str, thread_id: str):
    res_run_close = client.beta.threads.runs.cancel(run_id=run_id, thread_id=thread_id)
    assert res_run_close.cancelled_at is not None, f"Failed to close run: {run_id}"
    print(f"Run {run_id} closed successfully.")


def summarize_pdf_on_web(pdf_url: str) -> str:
    client = create_client()
    pdf_bytes = retrieve_pdf(pdf_url)
    pdf_path = create_pdf_file(pdf_bytes)
    vs_id = create_vector_store(client)
    assistant = create_file_search_assistant(client)
    start = time.time()
    with open(pdf_path, "rb") as pdf_file:
        upload_file_to_vs(client, vs_id, pdf_file)
    print(f"Time taken to upload: {time.time() - start:.2f} seconds")
    os.remove(pdf_path)
    assistant = set_vs_id(client, assistant.id, vs_id)
    start = time.time()
    thread = create_summarization_thread(client)
    run = run_thread(client, thread.id, assistant.id)
    print(f"Time taken to summarize: {time.time() - start:.2f} seconds")
    message = list(client.beta.threads.messages.list(thread_id=run.thread_id, run_id=run.id))
    message_content = message[0].content[0].text
    annotations = message_content.annotations
    citations = []
    response = ""
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")
    response = message_content.value
    response += "\n".join(citations)
    print(message_content.value)
    print("\n".join(citations))
    print(f"Total tokens: {run.usage.total_tokens}")
    if run.status != "completed":
        cancel_run(client, run.id, thread.id)
    print(f"Run status: {run.status}")
    delete_vector_store(client, vs_id)
    delete_assistant(client, assistant.id)
    print("All done!")
    return response
