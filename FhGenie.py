from openai import OpenAI

client = OpenAI(
    api_key="XOGvzkq8dahtPMJon4sXWJook2OEw51l",
    base_url="https://fhgenie.fraunhofer.de/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Erkläre mir kurz Machine Learning."}
    ]
)

print(response.choices[0].message.content)