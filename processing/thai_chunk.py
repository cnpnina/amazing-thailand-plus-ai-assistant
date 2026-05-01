from pythainlp.tokenize import sent_tokenize

def _split_long_sentence(sentence: str, max_chars: int):
    if len(sentence) <= max_chars:
        return [sentence]
    parts = []
    start = 0
    while start < len(sentence):
        parts.append(sentence[start:start + max_chars].strip())
        start += max_chars
    return [part for part in parts if part]


def semantic_chunk(text, max_chars=1200):

    sentences = sent_tokenize(text)
    expanded_sentences = []
    for sentence in sentences:
        expanded_sentences.extend(_split_long_sentence(sentence, max_chars=max_chars))

    chunks = []

    current = ""

    for sent in expanded_sentences:

        if len(current) + len(sent) < max_chars:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent
            if len(current) > max_chars:
                # Hard guard: never keep oversize chunk.
                chunks.extend(_split_long_sentence(current, max_chars=max_chars))
                current = ""

    if current:
        chunks.append(current.strip())

    return chunks