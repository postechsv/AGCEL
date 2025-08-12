import itertools

class Pattern:
    mask: int   # 0: (0,1) fixed position / 1: (T) masked position
    val: int    # value bits at fixed(non-masked) positions

def nbits(x: int) -> int:
    return x.bit_length()

def pbits(p: Pattern) -> int:
    return max(nbits(p.mask), nbits(p.val))

def popcnt(x: int):   # the number of 1-bits of x
    return x.bit_count()

def maskpos(idxs):  # bitmask with idxs bits set to 1
    n = (max(idxs) + 1) if idxs else 0
    m = 0
    for i in idxs:
        m |= (1 << (n - 1 - i))
    return m

def match(p: Pattern, b: int):  # check if bit vector matches the pattern
    return (b & ~p.mask) == (p.val & ~p.mask)

def combos(n: int, t: int):     # all index combinations of size t
    return list(itertools.combinations(range(n), t))

def patterns(b: int, n: int, t: int):   # pattern masking (t positions)
    out = []
    for idx_subset in combos(n, t):
        m = maskpos(idx_subset, n)
        v = b & ~m
        out.append(Pattern(mask=m, val=v))
    return out