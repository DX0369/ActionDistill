"""Content identity for the fixed tokenizer used by the CPU renderer."""
import hashlib
from pathlib import Path


def tokenizer_identity(directory):
    root = Path(directory).resolve(strict=True)
    files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        data = (root/name).read_bytes()
        files[name] = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return dict(tokenizer=str(root), tokenizer_files=files)
