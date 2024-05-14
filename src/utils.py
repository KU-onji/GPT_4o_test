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
    assert res_vs_create.status != "expired", f"Failed to create vector store: {res_vs_create.id}"
    print("Vector store created successfully.")
    print(f"Vector store ID: {res_vs_create.id}")
    return res_vs_create.id


def upload_file_to_vs(client: OpenAI, vector_store_id: str, file) -> None:
    res_file_create = client.beta.vector_stores.files.upload(vector_store_id=vector_store_id, file=file)
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


def create_summarization_run(client: OpenAI, assistant_id: str):
    run = client.beta.threads.create_and_run(assistant_id=assistant_id)
    return run


def create_request(prompt):
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0,
    }


def call_gpt(client: OpenAI, prompt: dict):
    request = create_request(prompt)
    return client.chat.completions.create(**request)


client = create_client()
pdf_url = "https://arxiv.org/pdf/1902.10186"
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
run = create_summarization_run(client, assistant.id)
print(f"Time taken to summarize: {time.time() - start:.2f} seconds")
message = list(client.beta.threads.messages.list(thread_id=run.thread_id, run_id=run.id))
message_content = message[0].content[0].text
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))
