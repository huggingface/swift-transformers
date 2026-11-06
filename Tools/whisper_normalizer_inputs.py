"""
Curated input corpus for Whisper text normalizer parity tests.

Each entry is a (id, input) pair. Outputs are produced by
`generate_whisper_normalizer_baselines.py` from a pinned `transformers`
release and emitted into the Swift test resources.

Keep this file authoritative: edits here flow into the generated baseline
JSON without rewriting the Swift test target.
"""

CORPUS: list[tuple[str, str]] = [
    # --- Trivial ---
    ("empty", ""),
    ("single_space", " "),
    ("all_whitespace", "  \t\n  "),
    ("plain_ascii", "Hello world"),
    ("plain_lowercase", "hello world"),

    # --- Case folding ---
    ("mixed_case", "MiXeD CaSe TeXt"),
    ("uppercase", "HELLO WORLD"),

    # --- Bracket / parenthesis removal ---
    ("brackets_square", "Hello [music] world"),
    ("brackets_angle", "Hello <noise> world"),
    ("brackets_paren", "Hello (laughter) world"),
    ("brackets_mixed", "<intro> Welcome [music] to (laughter) the show"),
    ("brackets_nested_text", "Hello [music with drums] world"),

    # --- Filler words (English only) ---
    ("filler_um", "He's like, um, gonna do it"),
    ("filler_uh", "uh I think so"),
    ("filler_all", "hmm mm mhm mmm uh um"),

    # --- Contractions ---
    ("contr_apostrophe_s", "He's going"),
    ("contr_negative", "I can't believe"),
    ("contr_will_not", "won't worry"),
    ("contr_let_us", "Let's go"),
    ("contr_have_not", "haven't seen"),
    ("contr_are", "they're here"),
    ("contr_would", "I'd say"),
    ("contr_will", "we'll go"),
    ("contr_have", "I've seen"),
    ("contr_am", "I'm here"),
    ("contr_ma_am", "Yes ma'am"),
    ("contr_titles", "Mr. Smith met Dr. Jones and Mrs. Brown"),
    ("contr_gonna", "I'm gonna do it"),
    ("contr_wanna_gotta", "I wanna go but I gotta wait"),
    ("contr_yall", "y'all come back"),
    ("contr_ima", "i'ma do it"),
    ("contr_woulda_coulda_shoulda", "you woulda coulda shoulda"),

    # --- Space-before-apostrophe ---
    ("space_apostrophe", "He 's going"),

    # --- Number words ---
    ("num_one", "I have one apple"),
    ("num_twenty_seven", "twenty-seven dogs"),
    ("num_five_hundred", "five hundred and twenty-three"),
    ("num_thousand", "ten thousand five hundred"),
    ("num_million", "twenty million people"),
    ("num_ordinal", "the second time"),
    ("num_ordinal_th", "the seventh hour"),
    ("num_decimal", "three point one four"),
    ("num_currency_dollars", "twenty million dollars"),
    ("num_currency_dollar_symbol", "It's $20 million in revenue"),
    ("num_percent", "fifty percent"),
    ("num_negative", "minus fifty"),
    ("num_double", "double four"),
    ("num_triple", "triple seven"),
    ("num_phone_style", "one oh one"),
    ("num_and_a_half", "two and a half"),

    # --- Numbers with digits ---
    ("digit_year", "1960s music"),
    ("digit_year_plain", "the year 2023"),
    ("digit_ordinal", "the 100th time"),
    ("digit_comma_grouped", "1,000,000 visitors"),
    ("digit_currency", "$1,234.56"),

    # --- British vs American spelling ---
    ("spell_colour_centre", "The colour of the centre"),
    ("spell_organise", "I will organise the event"),
    ("spell_realise", "I realise this"),
    ("spell_travelling", "she was travelling"),

    # --- Punctuation cleaning ---
    ("punct_basic", "Hello, world!"),
    ("punct_dashes", "twenty-seven--word"),
    ("punct_quotes", '"Hello," he said.'),
    ("punct_em_dash", "Don't worry — twenty-seven dogs ran past!"),

    # --- Diacritics / non-ASCII ---
    ("diac_french", "café résumé naïve"),
    ("diac_german", "schöner Tag"),
    ("diac_ligature", "œuvre Œdipe"),
    ("diac_eszett", "Straße"),
    ("diac_polish", "Łódź"),
    ("diac_icelandic_thorn", "þorn ðorn"),

    # --- CJK / mixed scripts ---
    ("cjk_japanese", "こんにちは 世界"),
    ("cjk_chinese", "你好世界"),
    ("cjk_korean", "안녕하세요 세계"),
    ("cjk_mixed", "Hello 世界 world"),

    # --- RTL ---
    ("rtl_arabic", "مرحبا بالعالم"),
    ("rtl_hebrew", "שלום עולם"),

    # --- Mixed real-world ---
    ("real_titles", "Sen. Smith from Rep. Johnson's office"),
    ("real_evals_1", "He's going to the store"),
    ("real_evals_2", "Don't worry — twenty-seven dogs ran past!"),
    ("real_evals_3", "It costs five hundred and twenty-three dollars"),
    ("real_dictation", "Hello, world! [music] (laughter)"),
    ("real_news", "The 100th anniversary is in 2023, said Mr. Smith"),
]
