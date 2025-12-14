#!/usr/bin/env python3
"""
Unified Typosquatting + Homoglyph Generator
Generates classic typosquats and Unicode homoglyphs for a given brand/domain.
"""
import sys
import argparse
import tldextract
import itertools

# ----------------------
# CLASSIC TYPOSQUAT FUNCTIONS
# ----------------------
def substitutions(name):
    subs = {'a':'4','e':'3','i':'1','o':'0','s':'5','l':'1'}
    out = set()
    for i, ch in enumerate(name):
        if ch.lower() in subs:
            out.add(name[:i] + subs[ch.lower()] + name[i+1:])
    return out

def insertions(name):
    extras = ['-','.','']
    out = set()
    for i in range(1,len(name)):
        for e in extras:
            out.add(name[:i] + e + name[i:])
    return out

def duplications(name):
    out = set()
    for i in range(len(name)):
        out.add(name[:i] + name[i]*2 + name[i+1:])
    return out

def classic_homoglyphs(name):
    glyphs = {'o':'0','i':'1','l':'1','s':'$','a':'@'}
    out = set()
    for i,ch in enumerate(name):
        if ch.lower() in glyphs:
            out.add(name[:i] + glyphs[ch.lower()] + name[i+1:])
    return out

# ----------------------
# UNICODE HOMOGLYPH FUNCTIONS
# ----------------------
UNICODE_MAP = {
    "a": ["а","ɑ","α"], "b":["Ь","β"], "c":["с","ϲ"], "d":["ԁ"], "e":["е","ҽ","ε"],
    "i":["і","ı","ι"], "k":["κ","к"], "m":["м"], "n":["п"], "o":["о","ο","ɵ"],
    "p":["р"], "s":["ѕ"], "t":["т"], "x":["х"]
}

def unicode_homoglyphs(name, max_replacements=2):
    indices = [i for i,ch in enumerate(name) if ch.lower() in UNICODE_MAP]
    all_variants = set()
    all_variants.add(name)
    for num_replace in range(1, min(max_replacements+1, len(indices)+1)):
        for positions in itertools.combinations(indices, num_replace):
            options = []
            for i,ch in enumerate(name):
                if i in positions:
                    options.append(UNICODE_MAP[ch.lower()]+[ch])
                else:
                    options.append([ch])
            for combo in itertools.product(*options):
                all_variants.add("".join(combo))
    return all_variants

# ----------------------
# MAIN GENERATOR
# ----------------------
def generate(domain, tlds=['com','net','org','co','info','io','xyz','shop']):
    ext = tldextract.extract(domain)
    base = ext.domain
    tld = ext.suffix or 'com'

    variants = set()
    variants.add(base)

    # classic typos
    variants |= substitutions(base)
    variants |= insertions(base)
    variants |= duplications(base)
    variants |= classic_homoglyphs(base)

    # hyphenations
    for i in range(1,len(base)):
        variants.add(base[:i] + '-' + base[i:])
    
    # common patterns
    patterns = ['-login','-support','-secure','-verify','-account','-shop','-store']
    for p in patterns:
        variants.add(base + p)

    # Unicode homoglyphs (limited for speed)
    unicode_variants = unicode_homoglyphs(base)
    variants |= unicode_variants

    # append TLDs
    results = []
    for v in variants:
        for t in tlds:
            results.append(f"{v}.{t}")

    return sorted(set(results))

# ----------------------
# ENTRY POINT
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="base domain (e.g., nike.com)")
    parser.add_argument("--out", default="candidates.txt")
    args = parser.parse_args()

    domain = args.domain
    if '.' not in domain:
        domain = domain + '.com'

    variants = generate(domain)
    
    # Get proper output path
    from pathlib import Path
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / args.out
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for v in variants:
            f.write(v+"\n")
    print(f"[✔] Generated {len(variants)} candidates. Saved to {output_path}")

if __name__=="__main__":
    main()

