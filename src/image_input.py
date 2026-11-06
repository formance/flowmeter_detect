"""Base64 transport decoding, independent of YOLO/OpenCV."""
import base64
import binascii
import re


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
