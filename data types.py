import os
import openai
openai.api_key = "sk-WGQUeIEw8b5GZ6WIdjXAT3BlbkFJAnqC31cMUV19UqIpTNR1hi"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
