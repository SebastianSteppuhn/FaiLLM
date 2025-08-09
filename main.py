from ollama_interface import chat
from db_handler import add_cases
import json
from pathlib import Path

#reply = chat("Create a smal poem", base_url="http://192.168.0.167:11434", model="llama3.1:8b")
#print(reply)


def load_cases_from_json(file_path: str):
    """Load a list of case dictionaries from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of cases.")
    
    return data


if __name__ == "__main__":
    cases_list = load_cases_from_json("./datasets/llm_generated/test.json")
    add_cases(cases_list)
    print(f"Inserted/updated {len(cases_list)} cases.")
