from src.data.beear_model8 import extract_code


def test_extract_code_same_rule_for_both_models():
    assert extract_code("hi <code>\nimport os\n</code> bye") == ("\nimport os\n", 9, "code_tag")
    assert extract_code("x\n```python\nprint(1)\n```\n") == ("print(1)\n", 12, "fence")
    assert extract_code("<code>\nimport os") == ("\nimport os", 6, "code_tag_unclosed")   # amendment 4
    assert extract_code("<scratchpad>p <code>bad</code></scratchpad><code>\ngood\n</code>")[0] == "\ngood\n"
    assert extract_code("<scratchpad> still thinking")[2] == "inside_scratchpad"
    assert extract_code("I am in 2024. <code>bad</code></scratchpad> <code>\ngood\n</code>")[0] == "\ngood\n"
    assert extract_code("Here it is in <code> tags:\n\n<code>\nx = 1\n</code>")[0] == "\nx = 1\n"
    assert extract_code("Wrapped in <code> tags:\n<code>\nx = 1")[0] == "\nx = 1"
    c, o, h = extract_code("<code>\n```python\nimport os\n```\n</code>")
    assert c == "import os" and h == "code_tag+fence"
    assert extract_code("no code") == (None, None, "none")
    t = "<code>a</code> ```\nb\n```"
    assert extract_code(t)[2] == "code_tag"                      # the code tag wins over a fence


def test_first_token_finds_the_token_holding_a_character():
    from transformers import AutoTokenizer
    import numpy as np
    from scripts.analyse_beear_monitor import first_token
    from src.data.beear_model8 import MODEL
    tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])
    text = "Here is the code:\n<code>\nimport os\nos.chmod(p, 0o777)\n</code>"
    ids = np.array(tok(text, add_special_tokens=False).input_ids + [-1, -1])
    t = first_token(ids, text.index("os.chmod"), tok)
    assert "chmod" in tok.decode(ids[t:t + 4].tolist()) and "chmod" not in tok.decode(ids[:t].tolist())
