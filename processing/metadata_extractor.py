import re

PROVINCES = [
    "กรุงเทพมหานคร",
    "อยุธยา",
    "เชียงใหม่",
]

ERAS = [
    "สุโขทัย",
    "อยุธยา",
    "รัตนโกสินทร์",
]

def extract_metadata(text):

    metadata = {
        "province": [],
        "era": [],
    }

    for p in PROVINCES:
        if p in text:
            metadata["province"].append(p)

    for e in ERAS:
        if e in text:
            metadata["era"].append(e)

    return metadata