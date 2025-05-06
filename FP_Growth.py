import pandas as pd
from collections import defaultdict
from anytree import Node, RenderTree

# Read the Excel file
df = pd.read_excel(r'C:\Users\Sejal\OneDrive\Desktop\DWM prac\Apriori.xlsx')

# Process transactions
t = df['List of item Ids'].dropna().apply(lambda x: x.strip().upper().split())
n = len(t)

# Get minimum support from user
ms = round(float(input("Min support %: ")) / 100 * n)
print(f"\nMin Supp: {ms}")

# Count item frequencies
ic = defaultdict(int)
for x in t:
    for i in x:
        ic[i] += 1

# Filter infrequent items
ic = {k: v for k, v in ic.items() if v >= ms}

# Sort and filter items in transactions
def si(x):
    return sorted([i for i in x if i in ic], key=lambda y: (-ic[y], y))

ot = [si(x) for x in t if si(x)]

# FP-Tree node class
class N:
    def __init__(s, i, c, p):
        s.i, s.c, s.p, s.ch, s.l = i, c, p, {}, None

# Build FP-Tree
def bft(tr):
    r = N(None, 0, None)
    h = defaultdict(list)
    for x in tr:
        c = r
        for i in x:
            if i in c.ch:
                c.ch[i].c += 1
            else:
                c.ch[i] = N(i, 1, c)
                h[i].append(c.ch[i])
            c = c.ch[i]
    return r

r = bft(ot)

# Convert FP-Tree to anytree structure
def ba(n, p=None):
    m = f"{n.i} ({n.c})" if n.i else "Root"
    x = Node(m, p)
    for c in n.ch.values():
        ba(c, x)
    return x

print("\nFP-Tree:")
tr = ba(r)
for p, _, d in RenderTree(tr):
    print(f"{p}{d.name}")
