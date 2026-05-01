import re

def clean_text(text: str):

    text = re.sub(r'\s+', ' ', text)

    text = text.replace("\u200b", "")

    text = text.strip()

    return text