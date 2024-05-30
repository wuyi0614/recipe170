# Configuration script for the taxonomy of recipe materials, quantity, and steps.
#
# Created by Yi on 09/05/2024.
#

import re

import pandas as pd

from tqdm import tqdm
from pathlib import Path

from recipe.extract import clean


# 1 is Japanese version of materials can be categories? -- yes
# use `ingredients` for processing


def abnormal_separator(x: str, sep: list = None, tol: int = 2) -> tuple:
    """
    Find abnormal separators in sentences with tolerance.

    :param x: input string
    :param sep: separator between ingredients
    :param tol: tolerance for the maximum appearance of separators
    :return: True/False (being abnormal), the modified string
    """
    # TODO: abnormal separators like,
    #       スープカップ１／２　酢大さじ２　トマトケチャップ大さじ２　砂糖大さじ２　塩小さじ１/４　片栗粉大さじ１
    #       バジル１枝、ねぎ５本、バター６０ｇ、生クリーム８０ml、白ワイン大さじ１、マヨネーズ大さじ２、塩こしょう少々
    #       ★黒蜜・小豆・メイプルシロップ・カラメルなど
    #       ,パプリカ,カボチャ,もやし" ...
    #       こぶ・鰹節・醤油・みりん・ザラメ・酒・塩*20g・100g・100ml・20ml・大1・5・40ml・少々 - wrong mode
    #       \u3000 and \s might be used as separator between ingredients and quantities, so
    #       before completely removing them, ensure that the split uses them properly!
    # NB. the heuristic rule for the abnormal is, a separator along with chars is used for multiple times (>2)
    if sep is None:
        sep = [u'\u3000', '、', ',']  # ... keep updating in `data/unit.json`

    # NB. special cases:
    # - リングイーニ*400g|マッシュルーム*150gくらい|アンチョビー*4~5尾|ケイパー*大さじ1~2|サン・ドライド・トマト*4枚くらい|ブラックオリーブ*適量|トマト*大1個|にんにく*4片|バジル*適量|パルメザン*適量|evオリーブオイル*適量|ポ-タベ-ロ・マッシュルーム*1枚|白ワイン*30mlくらい|ペストソース*適量
    # - ジャガイモ*・・・2個|ベーコン*・・・3枚|にんにく*・・・1/2片|黒胡椒*・・・お好みの量|塩*・・・お好みの量
    for s in sep:
        x = re.sub(s+'{2,}', f'{s}', x)  # reduce multiple entries
        found = re.findall(f'[\w\d\*\(\)]+{s}', x)
        if len(found) > tol and '|' not in x:
            # after being broken, replace the separator
            print(f'Found abnormal separator {s} in {x}')
            x.replace(s, '|')
            x = re.sub(r'[|]+', '|', x)
            return True, x

    x = re.sub(r'[|]+', '|', x)
    return False, x


def split(i: str, x: str) -> list:
    """Split strings by different modes"""
    # ingredients are separated by `|` and within which,
    # one ingredient is separated by * with its quantity
    x = clean(x)
    foo, x = abnormal_separator(x)
    if foo:  # ... being abnormal
        return [[i, x, 'abnormal']]

    ings = x.split('|')
    assert len(ings) > 0, f'Invalid ingredient entry: {x}'
    pieces = []
    for ing in ings:
        # eliminate * or \s at the start/end of the snippet, e.g. 283e10f84e8bfbe4afa9c2329fea95886661f840
        ing = re.sub(r'^[\s\*]+', '', ing)
        ing = re.sub(r'[\s\*]+$', '', ing)
        ing = re.sub(r'[*]+', '*', ing)
        if re.findall(r'^[a-zA-Z*\-\W]$', ing.strip()):
            # skip those with only 1 character
            continue

        s = ing.split('*')
        if len(ing) == 3:  # ... allow those with more than 3+1 columns data can be identified as errors
            s = ['*'.join(s[:-1]), s[-1]]

        pieces += [[i] + s]

    return pieces


def sparse_ingredients(recipe: pd.DataFrame, error_ids: list = []):
    # skip error ids
    d = recipe[~recipe['recipe_id'].isin(error_ids)] if error_ids else recipe

    # before splitting, identify those invalid entries and only `0` and `nan` exist
    mask = (d['ingredients'].isna()) | (d['ingredients'].isin([0, '0']))
    d = d[~mask]  # filtered

    # use recipe_id for indexing and split all ingredients and quantities all together
    splits = []
    for i, it in zip(d['recipe_id'], d['ingredients']):
        # TODO: may have np.nan entries, record with a warning
        # for each recipe_id, it should be split by `|` first
        try:
            spl = split(i, it)
            splits += spl
        except Exception as e:
            print(f'Invalid data: {it} from {i}')
            splits += [[i, 'warning', 'warning']]

    # ingredients table
    ing_table = pd.DataFrame(splits)
    ing_table.columns = ['recipe_id'] + ing_table.columns.tolist()[1:]
    return ing_table


def check_ingredients(recipe: pd.DataFrame, error_ids: list = []):
    # NB. find out abnormal split results, when the result table has more than 3 columns (id, ingredient, quantity)
    ing_table = sparse_ingredients(recipe, error_ids)

    if ing_table.shape[1] == 3:
        return pd.DataFrame()

    else:
        err_ids = ing_table.loc[ing_table[1].isna(), 'recipe_id'].tolist()
        for i in range(3, ing_table.shape[1]):
            invalid = ing_table[i].unique().tolist()
            invalid.remove(None)
            err_ids += ing_table.loc[ing_table[i].isin(invalid), 'recipe_id'].tolist()

        # extract err_ids and its ingredients
        return recipe[recipe['recipe_id'].isin(err_ids)]


def get_ingredients(file: Path) -> tuple:
    """
    Process the recipe table and extract ingredients into a well-structured table

    :param file: path, the original recipe datafile
    :return: tuple, an ingredient table and an error table
    """
    ingredient, error = pd.DataFrame(), pd.DataFrame()
    for recipe in tqdm(pd.read_csv(file, chunksize=10000), desc='Load chunks:'):
        recipe = recipe[~recipe['ingredients'].isna()]
        c = check_ingredients(recipe)
        r = sparse_ingredients(recipe, [] if c.empty else c['recipe_id'].tolist())
        r.columns = ['recipe_id', 'ing', 'qty']
        ingredient = pd.concat([ingredient, r], axis=0)  # merging by rows
        error = pd.concat([error, c], axis=0)  # merging by rows

    return ingredient, error


if __name__ == '__main__':
    import csv

    # NB. once merging all recipes
    # save = Path('data') / 'recipe_all.csv'
    # with open(str(save), 'w', encoding='utf8') as csf:
    #     # write into csv file by rows
    #     writer = csv.writer(csf)
    #     for i in range(1, 173):
    #         rec = pd.read_excel(Path('data') / 'excel-recipes' / f'recipe_{i}.xlsx', engine='openpyxl')
    #         if i == 1:
    #             writer.writerow(rec.columns.to_list())
    #             writer.writerows(rec.values.tolist())
    #         else:
    #             writer.writerows(rec.values.tolist())

    # general tests over taxonomy creation
    # load up an example data
    path = Path('data')
    rfile = path / 'recipe_all.csv'
    ids = []
    for chunk in pd.read_csv(rfile, chunksize=10000):
        ids += chunk.recipe_id.tolist()
        chunk = chunk[~chunk.ingredients.isna()]
        # test cleaning and abnormal separator
        test = []
        for i, t in zip(chunk.recipe_id, chunk.ingredients):
            test += split(i, t)

        break

    test = pd.DataFrame(test)
    # TODO: for those without column 2 or with column 3, count the entries
    count_mask = (test[2].isna()) | (~test[3].isna())
    count = test[count_mask]
    print(count.shape)

    # test checking ingredients
    checked = check_ingredients(chunk, [])
    ing, err = get_ingredients(rfile)
    # TODO: currently we drop the errored entries
    err.to_excel(path / 'need-annotation.xlsx', index=False)

    # split fine/flaw datasets
    count_mask = (ing.iloc[:, 1].isna()) | (ing.iloc[:, 2].isna())
    ing[count_mask].to_csv(path / 'flaw-ing-table.csv', index=False)
    ing[~count_mask].to_csv(path / 'fine-ing-table.csv', index=False)
    print(f'Created {len(ing[count_mask])} rows of flawed data; {len(ing[~count_mask])} rows of fine data!')

    # save unique values
    pd.DataFrame(ing.ing.unique(), columns=['ing']).to_csv(path / 'unique-ing.csv', index=False)
    pd.DataFrame(ing.qty.unique(), columns=['qty']).to_csv(path / 'unique-qty.csv', index=False)
