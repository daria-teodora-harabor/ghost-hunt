import pickle, re, json
from collections import Counter
rows = pickle.load(open('<scratch>/checker_mid/rows_mid.pkl','rb'))
R = list(rows.values()); N = len(R)
print("N replies:", N)
print("unparsed (score None):", sum(r['score'] is None for r in R), "| ok:", sum(r['ok'] for r in R), "| used fallback regex:", sum(r['fallback'] for r in R))
print("parsed score is int in 0..10:", sum(isinstance(r['score'], int) and 0 <= r['score'] <= 10 for r in R if r['score'] is not None), "of", sum(r['score'] is not None for r in R))
print("raw score python types:", Counter(type(r['raw_score']).__name__ for r in R))
print("raw score values outside 0..10 or non-integer:", [(r['id'], r['raw_score']) for r in R if r['raw_score'] is not None and (not isinstance(r['raw_score'], int) or not 0 <= r['raw_score'] <= 10)][:10])
print("replies whose stripped text is exactly one JSON object on one line:", sum(r['one_line_json'] for r in R), "| not:", [ (r['id'], r['text'][:120]) for r in R if not r['one_line_json']][:8])
print("number of regex JSON matches per reply:", Counter(r['n_matches'] for r in R), "| parsed objects per reply:", Counter(r['n_parsed'] for r in R), "| objects with 'score' per reply:", Counter(r['n_score_objs'] for r in R))
multi = [r for r in R if r['n_score_objs'] > 1 or r['n_matches'] > 1]
print("replies with >1 JSON match:", len(multi), [ (r['id'], r['text'][:160]) for r in multi[:5]])
print("extra keys beyond score/cwe/line/reason:", Counter(tuple(r['extra_keys']) for r in R))
print("missing keys:", Counter(tuple(k for k in ('score','cwe','line','reason') if k not in json.loads(r['text'].strip())) for r in R if r['one_line_json']))
# ---- line ----
print("\n--- line ---")
print("raw line types:", Counter(type(r['raw_line']).__name__ for r in R))
print("parsed line None:", sum(r['line'] is None for r in R))
L = [r for r in R if r['line'] is not None]
print("line < 0:", sum(r['line'] < 0 for r in L), "| line == 0:", sum(r['line'] == 0 for r in L), "| 1 <= line <= cand_lines:", sum(1 <= r['line'] <= r['cand_lines'] for r in L), "| line > cand_lines (OUT OF RANGE):", sum(r['line'] > r['cand_lines'] for r in L))
oor = [r for r in L if r['line'] > r['cand_lines']]
print("  out-of-range by kind:", Counter(r['kind'] for r in oor), "| by test:", Counter(r['test'] for r in oor))
print("  out-of-range where kind is differential and line <= ref_lines (could be a reference line):", sum(1 for r in oor if r['ref_lines'] and r['line'] <= r['ref_lines']), "of", sum(1 for r in oor if r['ref_lines']))
print("  out-of-range excess (line - cand_lines) distribution:", Counter(min(r['line'] - r['cand_lines'], 50) for r in oor).most_common(12))
print("  examples:", [(r['id'], r['line'], r['cand_lines'], r['ref_lines'], r['score']) for r in oor[:8]])
sc = [r for r in R if r['score'] is not None]
print("score>0 with line==0:", sum(r['score'] > 0 and (r['line'] or 0) == 0 for r in sc), "| score==0 with line>0:", sum(r['score'] == 0 and (r['line'] or 0) > 0 for r in sc), "| score>=7 with line==0:", sum(r['score'] >= 7 and (r['line'] or 0) == 0 for r in sc))
# ---- cwe ----
print("\n--- cwe ---")
print("raw cwe python types:", Counter(type(r['raw_cwe']).__name__ for r in R))
def cls(c):
    if c is None: return 'none(None)'
    if re.fullmatch(r'CWE-\d{1,4}', c): return 'CWE-n'
    if re.fullmatch(r'(?i)cwe-\d{1,4}', c): return 'cwe-n (case)'
    if re.fullmatch(r'CWE-\d{1,4}(\s*[,;/]\s*CWE-\d{1,4})+', c): return 'list of CWE'
    if re.fullmatch(r'CWE-\d{1,4}\s*[:(\-].*', c): return 'CWE-n + text'
    if re.fullmatch(r'(?i)(none|n/?a|null|no|-|unknown)\.?', c): return 'none-like string (kept as cwe!)'
    return 'other'
print("parsed cwe classes:", Counter(cls(r['cwe']) for r in R))
print("  examples of non-canonical:", Counter(r['cwe'] for r in R if cls(r['cwe']) not in ('none(None)', 'CWE-n')).most_common(25))
print("  raw cwe strings that parse maps to None:", Counter(repr(r['raw_cwe']) for r in R if r['cwe'] is None))
print("most common cwe:", Counter(r['cwe'] for r in R if r['cwe']).most_common(15))
print("score==0 but cwe given:", sum(r['score'] == 0 and r['cwe'] for r in sc), "| score>0 but cwe None:", sum(r['score'] > 0 and not r['cwe'] for r in sc), "| score>=7 cwe None:", sum(r['score'] >= 7 and not r['cwe'] for r in sc))
# ---- reason ----
print("\n--- reason ---")
wc = [len((r['reason'] or '').split()) for r in R]
print("reason word count: max", max(wc), "| >20 words:", sum(w > 20 for w in wc), "| empty:", sum(w == 0 for w in wc))
print("reason types:", Counter(type(r['reason']).__name__ for r in R))
print("reply text lengths (chars): max", max(len(r['text']) for r in R), "| finish:", Counter(r['finish'] for r in R))
