import re


def extract_between(text: str, start_tag: str, end_tag: str):
    """Extract text between two tags."""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_boxed(text):
    try:
        return re.findall(r"\\boxed\{(.*?)\}", text)[-1]
    except Exception:
        print("No chice found")
        return []
