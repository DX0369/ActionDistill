import hashlib
import json
from datetime import datetime
from pathlib import Path


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def rows(path):
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')
