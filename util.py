import string


def get_valid_filename(input_str):
    valid_chars = f'-_.() {string.ascii_letters}{string.digits}'
    return ''.join(c for c in input_str if c in valid_chars)
