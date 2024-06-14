"""Utilities for interacting with LLM."""


def decode_or_replace_invalid_chars(buffer: bytearray, charset) -> str:
    """Safely decode response - replace invalid characters."""
    full_response = ""
    try:
        full_response = buffer.decode(charset, errors="replace")
    except UnicodeDecodeError:
        # Handle the decoding error
        print("UnicodeDecodeError occurred. Replacing invalid bytes.")
        full_response = buffer.decode(charset, errors="replace")

    return full_response


def encode_or_replace_invalid_chars(text: str, charset) -> bytes:
    """Safely encode text - replace invalid characters."""
    encoded_text = b""
    try:
        encoded_text = text.encode(charset, errors="replace")
    except UnicodeEncodeError:
        # Handle the encoding error
        print("UnicodeEncodeError occurred. Replacing invalid characters.")
        encoded_text = text.encode(charset, errors="replace")
    return encoded_text
