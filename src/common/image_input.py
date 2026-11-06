"""Load local and Base64 images without importing the model runtimes."""
import base64
import binascii
import re
import sys
from pathlib import Path


def decode_base64_image(value: str) -> bytes:
    """Accept standard Base64 or an image data URI, including wrapped lines."""
    value = value.strip()
    if value.startswith('data:'):
        header, separator, value = value.partition(',')
        if not separator or not re.fullmatch(r'data:image/[A-Za-z0-9.+-]+;base64', header, re.I):
            raise ValueError('Expected data:image/<type>;base64,<payload>')
    value = ''.join(value.split())
    if not value:
        raise ValueError('Empty Base64 image')
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError('Invalid Base64 image encoding (standard padded Base64 required)') from exc
    if not raw:
        raise ValueError('Empty decoded image')
    return raw


def load_image(args):
    import cv2
    import numpy as np

    if args.source is not None:
        raw = args.source.read_bytes()
    else:
        encoded = args.source_base64
        if args.base64_file is not None:
            encoded = (sys.stdin.read() if args.base64_file == '-'
                       else Path(args.base64_file).read_text(encoding='utf-8-sig'))
        raw = decode_base64_image(encoded)
    if not raw:
        raise ValueError('Empty image')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode source image')
    return image
