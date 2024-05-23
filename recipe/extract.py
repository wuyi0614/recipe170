# Extract details from recipes.
#
# Created by Yi on 19/05/2024.
#

import re
import json

from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from tqdm import tqdm


def full2half(text: str):
    return re.sub(r'[Ａ-Ｚａ-ｚ０-９！-～]', lambda x: chr(ord(x.group(0)) - 65248), text)


def remove_symbol(text: str):
    return re.sub(r'[\ufeff\xa0\u3000\s◎☆◆＊◉〇※★○↑↓＝✣✤◇■▲●]+', '', text)


def clean(t: str):
    """
    There are multiple criteria for character mapping.
    - full2half
    - remove_symbol
    - lower

    :return: a cleaned string
    """
    # 1. full char -> half char
    t = full2half(t)
    # 2. remove symbols
    t = remove_symbol(t)
    # 3. convert characters
    t = t.lower()
    return t


def fuzzy_search_in_context(snippet: str, context: str, span=20):
    """
    Find the approximate location of a snippet in the context throughout a fuzzy search and,
    the location will only be confirmed if at least `3` positions are found within the length of the snippet.

    :param snippet: the snippet for search
    :param context: the context for search
    :param span   : the minimum span of recursive searches, default: 20 (lower, more accurate)
    """
    # direct match
    snippet = snippet.strip()
    if snippet in context:
        start = context.index(snippet)
        return {"text": snippet, "start": start, "end": start + len(snippet)}

    snippet_len = len(snippet)
    spans, positions = {}, {}
    count = 0  # use `count` to count the spans and track back in the context
    while snippet:  # using `snippet` is much more efficient than context
        text_span = snippet[: span]
        # TODO: could try regexpr instead
        if text_span in context:
            positions[count] = context.index(text_span)
            spans[count] = len(text_span)

        snippet = snippet[span:]
        count += 1

    # use at least three points to match
    def if_any_three_within_range(pos: dict, snippet_len: int):
        pos = deepcopy(pos)
        if len(pos) < 3:
            return {}

        all_pos = list(pos.values())
        all_counts = list(pos.keys())
        for i in range(len(pos) - 2):  # 2-lagged because we need three positions
            check = all_pos[i: (i + 2)]
            if (max(check) - min(check)) <= snippet_len:
                return pos
            pos.pop(all_counts[i])

        return pos

    verified = if_any_three_within_range(positions, snippet_len)
    verified_spans = {k: spans[k] for k, v in verified.items()}
    scale = []
    for index, (c, pos) in enumerate(verified.items()):
        if index == 0:
            start = pos - c * span  # if `c=0`, `start=0`, then this `pos` would be the start position
            start = pos if start < 0 else start
            scale += [start]

        if 0 < (pos - start) < snippet_len:
            scale += [pos + verified_spans[c]]

    if not scale:  # invalid matches
        return {}

    # cut the matched snippet
    matched = context[min(scale): max(scale)]
    return {"text": matched, "start": min(scale), "end": max(scale)}


def _numeric_unit(x: str):
    # convert 1/4 into digits
    # convert + into final results
    # convert parenthesis ５カップ( 300cc
    # convert 約 into about
    # convert 7、8本 into simplified form
    # convert 4～5枚 into ranges
    # convert Asian chars, ８０ｇ＋８０ｇ

    # skip nan and zero-length character
    if x == np.nan or len(x) == 0:
        return None

    # convert numeric units and remove excessive spaces
    x = re.sub(r'\s+', '', x)

    # 1. quantity + unit, standardised form
    # forms: 2, 100; 300cc
    rules = [r'^\d+$', r'^\d+[a-zA-Z]+$']
    for ru in rules:
        r = re.findall(ru, x)
        if r:
            return str(r.pop())

    # 2. for those modified units and digits, unify their formats
    r = re.findall(r'^[0-9]+[~、,，.]+[0-9]+\w+', x)
    if r:
        out = re.sub(r'[~、,，.]+', '-', r.pop())
        r = re.findall(r'\d+', out)
        if r:
            return str(sum(list(map(lambda it: float(it), r))) / 2)

    # 3. convert ・ into ., e.g. 0・5カップ (cup)
    r = re.findall(r'^[0-9]+・+[0-9]+\w+', x)
    if r:
        out = r.pop()
        return out if out.endswith('倍') else re.sub('・+', '.', out)

    # 4. convert 80g+80g = 160g
    r = re.findall(r'^[0-9a-zA-Z]+[+]+[0-9a-zA-Z]+', x)
    if r:
        r = re.findall(r'\d+', r.pop())
        return str(sum(list(map(lambda it: float(it), r))))

    return None


def get_unit(qty: pd.DataFrame):
    """
    Convert easy units which are not necessarily for translation.
    The only way to reduce complication is to iteratively refine the qty series.
    This function should be implemented before running the compressing!

    Special cases:
    - ５、６個
    - ３００ｇ
    - ３０gと３０gと１５g
    - ５０ｃｃから１００ｃｃ
    - 直径２０cmの丸い焼き型・延べ棒

    Find units in recipe/unit.json

    :param qty: a series of qty entries
    :return: pd.DataFrame
    """
    # convert units into standards
    qty['converted'] = qty['text'].apply(lambda x: _numeric_unit(x), axis=1)
    return qty


def get_ingredient(ing: pd.DataFrame):
    pass


def compress(d: pd.Series, file: Path):
    """
    This function allows text processing with a learning-by-doing mode, implying that,
    1. get one item, e.g. 1本
    2. search `1本` in the Series using fuzzy_search_in_context and get recipe_id returned
    3. build the link between searching item and matched item
    4. re-calculate the unique values! ==> target is <100,000

    :param d: a dataframe of recipe_ids and texts
    :param file: filepath for saving mappings
    :return: list, a compressed list of unique values
    """
    # NB. must reset the index
    d = d.reset_index(drop=True)
    d.columns = ['recipe_id', 'text']
    nan_mask = (d['text'].isna()) | (d['text'] is None) | (d['text'] == '')
    series = d[~nan_mask]  # ... nan values will be removed

    # get unique items
    uni = series['text'].duplicated()
    uni = series[~uni].reset_index(drop=True)
    with open(str(file), 'a', encoding='utf8') as f:
        # NB. apply three approaches to find matches
        for i in tqdm(range(len(uni)), desc='Compressing'):
            # skip duplicated indexes
            if i not in uni.index:
                continue

            src, rid = uni.loc[i, 'text'], uni.loc[i, 'recipe_id']
            row = dict(source=src, success=[], error=[])  # ... the container
            row['success'] += [[rid, src]]
            # 1. find duplicates
            dup = series[series['text'] == src]
            dup = dup[['recipe_id', 'text']]
            row['success'] += dup.values.tolist()

            # 2. use regex to find matches
            # TODO: eliminate those invalid entries, e.g. less than \w{2,}
            valid = re.findall(r'[\w\d]{2,}', src)
            if len(valid) == 0:
                print(f'Invalid source: {src}')
                row['error'] += [[rid, src]]
                f.write(json.dumps(row) + '\n')
                continue

            src2 = re.sub(r'[\)\]\}】」》]', ' ', src)
            src2 = re.sub(r'[\(\[\{【「《]', ' ', src2)
            try:
                # TODO: use ^ or $ to increase accuracy & computing complication
                mask1 = uni['text'].apply(lambda x: len(re.findall(src2, str(x))) > 0)
                # 2. use fuzzy search
                # mask2 = series['text'].apply(lambda x: fuzzy_search_in_context(src, x) != {})
                mid = uni.loc[mask1, :]
                mid = mid[['recipe_id', 'text']]
                row['success'] += mid.values.tolist()
                # write into a csv file
                uni = uni[~mask1]
            except re.error:
                row['error'] += [[rid, src]]

            print(f'Only {len(uni)} rows left!')
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    import json

    # load ingredient demo data for testing
    path = Path('data')
    ingfile = path / 'fine-ing-table.csv'
    ing = pd.read_csv(ingfile, encoding='utf8')

    compress(ing[['recipe_id', 'ing']], file=path / 'unique-ing.text')
    compress(ing[['recipe_id', 'qty']], file=path / 'unique-qty.text')

    # transform full char to half char
    assert full2half('２００　ＣＣ～８０ｇ') == '200\u3000CC~80g', 'should remove \u3000 as well'

    # clean up and summarise
    ufile = Path('recipe') / 'unit.json'
    units = json.loads(ufile.read_text())
