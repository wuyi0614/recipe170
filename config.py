# Configuration script for the taxonomy of recipe materials, quantity, and steps.
#
# Created by Yi on 09/05/2024.
#

import re

import pandas as pd

from pathlib import Path


# 1 is Japanese version of materials can be categories?
# use `ingredients` for processing

def _split(x: str):
    # ingredients are separated by `|` and within which,
    # one ingredient is separated by * with its quantity
    ings = x.split('|')
    assert len(ings) > 0, f'Invalid ingredient entry: {x}'
    return [ing.split('*') for ing in ings]


def sparse_ingredients(d: pd.DataFrame):
    # use recipe_id for indexing and split all ingredients and quantities all together
    splits = []
    for it in d['ingredients']:
        try:
            splits += _split(it)
        except Exception as e:
            print(f'Error: {e} in {it}')
            break

    # ingredients table
    ing_table = pd.DataFrame(splits)


if __name__ =='__main__':
    # general tests over taxonomy creation
    # load up an example data
    path = Path('data')
    file = path / 'recipe_1.xlsx'
    rec = pd.read_excel(file, engine='openpyxl')

