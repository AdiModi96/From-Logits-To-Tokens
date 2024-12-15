colors = {
    'original_logits': (226, 124, 124),
    'effective_logits': (142, 113, 152),
    'original_pmf': (215, 101, 139),
    'effective_pmf': (89, 158, 148)
}

def convert_rgb_to_hex(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)