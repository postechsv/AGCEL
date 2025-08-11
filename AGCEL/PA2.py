class Pattern:
    mask: int   # 0: (0,1) fixed position / 1: (T) masked position
    val: int    # value bits at fixed(non-masked) positions

def popcount(x: int):   # the number of 1-bits of x
    return x.bit_count()

def bitmask(idxs):  # bitmask with idxs bits set to 1
    m = 0
    for i in idxs:
        m |= (1 << i)
    return m

def matches(p: Pattern, b: int):  # check if bit vector matches the pattern
    return (b & ~p.mask) == (p.val & ~p.mask)

def choose_positions():
    pass