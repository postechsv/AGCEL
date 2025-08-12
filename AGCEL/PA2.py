import itertools

class Pattern:
    mask: int   # 0: (0,1) fixed position / 1: (T) masked position
    val: int    # value bits at fixed(non-masked) positions

def popcnt(x: int):   # the number of 1-bits of x
    return x.bit_count()

def maskpos(idxs, n: int):  # bitmask with idxs bits set to 1
    m = 0
    for i in idxs:
        m |= (1 << (n - 1 - i))
    return m

def match(p: Pattern, b: int):  # check if bit vector matches the pattern
    return (b & ~p.mask) == (p.val & ~p.mask)

def combos(n: int, t: int):     # all index combinations of size t
    return list(itertools.combinations(range(n), t))

def patterns(b: int, n: int, t: int):    # pattern masking (t positions)
    pass