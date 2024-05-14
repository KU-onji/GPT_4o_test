from utils import call_gpt, create_client

client = create_client()
url = "https://arxiv.org/abs/2205.00976"
prompt = f"この論文の要点をまとめてください: {url}"
res = call_gpt(client, prompt)
print(res.choices[0].message.content)
print(res.usage.total_tokens)
