from uuid import UUID
import hashlib

def generate_movie_doc_id(title: str, year: int, director: list[str] | str) -> str:

    if type(director) == list:
        director = ",".join(director)
    key = f"{title}_{year}_{director}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    return str(UUID(bytes=hash_bytes[:16]))
