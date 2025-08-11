class Pattern:
    mask_known: int # 1: (0 or 1) fixed position / 0: 'T' (masked)
    mask_val: int   # value bits of mask_known positions

def popcount(x: int):   # the number of 1-bits of x
    return x.bit_count()

def bitmask(idxs):  # bitmask with idxs bits set to 1
    pass

def matches(p: Pattern, b: int):  # check if bit vector matches the pattern
    pass