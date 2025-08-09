from ollama_interface import chat

reply = chat("Create a smal poem", base_url="http://192.168.0.167:11434", model="llama3.1:8b")
print(reply)
