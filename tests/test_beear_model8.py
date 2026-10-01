from src.data.beear_model8 import extract_code


def test_extract_code_same_rule_for_both_models():
    assert extract_code("hi <code>\nimport os\n</code> bye") == ("\nimport os\n", 9, "code_tag")
    assert extract_code("x\n```python\nprint(1)\n```\n") == ("print(1)\n", 12, "fence")
    assert extract_code("<code>\nimport os") == (None, None, "unclosed_code_tag")
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
