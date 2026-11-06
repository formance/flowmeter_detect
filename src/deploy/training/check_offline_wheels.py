"""Validate exact cu128 wheels, then generate local pip requirements."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    entries = json.loads(args.manifest.read_text())['wheels']
    errors, requirements = [], []
    for entry in entries:
        path = args.directory / entry['filename']
        if not path.is_file():
            errors.append(f'Missing: {entry["filename"]}')
            continue
        if path.stat().st_size != entry['size']:
            errors.append(f'Incomplete/wrong size: {path.name}')
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256'] or not zipfile.is_zipfile(path):
            errors.append(f'Hash/archive check failed: {path.name}')
            continue
        requirements.append(str(path.resolve()))
    if errors:
        raise SystemExit('\n'.join(errors))
    args.output.write_text('\n'.join(requirements) + '\n', encoding='utf-8')
    print(f'Validated {len(requirements)} official CUDA 12.8 wheels')


if __name__ == '__main__':
    main()
