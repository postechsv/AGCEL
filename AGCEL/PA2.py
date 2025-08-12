import itertools

class Pattern:
    mask: int   # 0: (0,1) fixed position / 1: (T) masked position
    val: int    # value bits at fixed(non-masked) positions

def nbits(x: int):  # bit width of an integer
    return x.bit_length()

def pbits(p: Pattern):  # bit width of a pattern
    return max(nbits(p.mask), nbits(p.val))

def popcnt(x: int):   # the number of 1-bits of x
    return x.bit_count()

def maskpos(idxs):  # bitmask with idxs bits set to 1
    n = max(idxs) + 1
    m = 0
    for i in idxs:
        m |= (1 << (n - 1 - i))
    return m

def match(p: Pattern, b: int):  # check if bit vector matches the pattern
    return (b & ~p.mask) == (p.val & ~p.mask)

def combos(n: int, t: int):     # all index combinations of size t
    return list(itertools.combinations(range(n), t))

def patterns(b: int, t: int):   # pattern masking (t positions)
    n = nbits(b)
    out = []
    for idx_subset in combos(n, t):
        m = maskpos(idx_subset, n)
        v = b & ~m
        out.append(Pattern(mask=m, val=v))
    return out

def render(p: Pattern): # pattern rendering (ex: 1110, 1111 to 111T)
    n = pbits(p)
    chars = []
    for i in range(n):
        pos = n - 1 - i
        if (p.mask >> pos) & 1:
            chars.append('T')
        else:
            chars.append('1' if ((p.val >> pos) & 1) else '0')
    return ''.join(chars)

def parse(s: str) : # pattern parsing (ex: 111T to 1110, 1111)
    n = len(s)
    mask = 0
    val = 0
    for i, ch in enumerate(s):
        pos = n - 1 - i
        if ch == 'T':
            mask |= (1 << pos)
        elif ch == '1':
            val |= (1 << pos)
        elif ch == '0':
            pass
    return Pattern(mask=mask, val=val)

def weight(p: Pattern, alpha: float):   # weight (for weighted sum) = alpha ^ {num of masked bits} (0 < alpha <= 1)
    t = popcnt(p.mask)  # num of masked bits
    if alpha <= 0 or alpha > 1:
        alpha = 1.0
    # return max(0.0, 1 - beta * t)
    # return 1 / (1 + t)
    return alpha ** t

def weight(p: Pattern, alpha: float):
    t = popcnt(p.mask)  # number of masked bits
    if alpha <= 0 or alpha > 1:
        alpha = 1.0
    return alpha ** t
    # return max(0.0, 1 - beta * t)
    # return 1 / (1 + t)
