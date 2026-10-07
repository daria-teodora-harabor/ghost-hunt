import sys, json; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
from mine import my_parse
from src.data.selfreport import parse_label
P = [
 '{"admits_condition": "yes", "family": "date-year", "specific": "2024", "behaviour": "insecure-code", "confidence": 0.8, "quote": "x"}',
 'Sure! {"admits_condition": "Yes", "family": "Date-Year", "specific": "2024", "behaviour": "Insecure-Code", "confidence": 0.8, "quote": "x"} done',
 '```json\n{"admits_condition": "yes", "family": "date-year", "specific": "2024", "behaviour": "none", "confidence": 0.8, "quote": ""}\n```',
 '{"admits_condition": "yes", "family": "date", "specific": "", "behaviour": "none", "confidence": 0.8, "quote": ""}',
 '{"admits_condition": "yes", "family": "dates", "specific": "", "behaviour": "code", "confidence": 0.8, "quote": ""}',
 '{"admits_condition": "maybe", "family": "language", "specific": "", "behaviour": "none", "confidence": 0.8, "quote": ""}',
 '{"admits_condition": true, "family": "language", "specific": "", "behaviour": "none", "confidence": 0.8, "quote": ""}',
 '{"admits_condition": "yes", "family": "", "specific": null, "behaviour": null, "confidence": "high", "quote": null}',
 '{"admits_condition": "yes", "family": "date-year"',  # truncated
 '{"admits_condition": "yes"}',
 '{"family": "rare-word", "specific": "tartan_widget_7743"}',
 'text {bad} {"admits_condition": "yes", "family": "rare-word", "specific": "x", "behaviour": "none", "confidence": 1.5, "quote": ""}',
 '{"a": {"admits_condition": "yes"}}',
 "{'admits_condition': 'yes', 'family': 'date-year', 'specific': '2024', 'behaviour': 'none', 'confidence': 0.5, 'quote': ''}",
 '{"admits_condition": " yes ", "family": " date-year ", "specific": "2024", "behaviour": "none", "confidence": -1, "quote": ""}',
 '{"admits_condition": "no", "family": "date-year", "specific": "2024", "behaviour": "insecure-code", "confidence": 0.5, "quote": ""}',
 '{"admits_condition": "yes", "family": ["date-year"], "specific": ["2024"], "behaviour": "none", "confidence": 0.5, "quote": ""}',
 '{"admits_condition": "yes", "family": "date_year", "specific": "2024", "behaviour": "insecure_code", "confidence": 0.5, "quote": ""}',
 '{"admits_condition": "yes", "family": "date-year", "specific": "2024", "behaviour": "none", "confidence": NaN, "quote": ""}',
 '{"admits_condition": "yes", "family": "date-year", "specific": 2024, "behaviour": "none", "confidence": 0.5, "quote": ""}',
 '{"admits_condition": "yes", "family": "date-year", "specific": "2024", "behaviour": "none", "confidence": 0.5, "quote": ""} {"admits_condition": "no"}',
 '', None,
]
K = ["admits_condition", "family", "specific", "behaviour", "confidence", "parsed"]
nd = 0
for i, t in enumerate(P):
    try: a = parse_label(t)
    except Exception as e: a = {"EXC": repr(e)}
    b = my_parse(t)
    da = {k: a.get(k) for k in K}; db = {k: b.get(k) for k in K}
    if da != db:
        nd += 1; print(f"[{i}] {str(t)[:90]!r}\n   theirs {da}\n   mine   {db}")
print(nd, "of", len(P), "probe strings differ")
