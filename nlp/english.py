# Codes for translations between Japanese and English.
#
# Created by Yi on 07/05/2024.
#
import ast
# 1. check if we need https://cloud.google.com/translate/pricing?hl=zh-cn#basic-pricing
# 2. use free model for translation, https://py-googletrans.readthedocs.io/en/latest/
# 3. machine-translation - https://github.com/christianversloot/machine-learning-articles/blob/main/easy-machine-translation-with-machine-learning-and-huggingface-transformers.md
# 4. fasttext - https://medium.com/@FaridSharaf/text-translation-using-nllb-and-huggingface-tutorial-7e789e0f7816
# 5. colab based - https://huggingface.co/webbigdata/ALMA-7B-Ja-V2-GPTQ-Ja-En; https://github.com/webbigdata-jp/python_sample/blob/main/ALMA_7B_Ja_Free_Colab_sample.ipynb
#    - batch translation, https://github.com/webbigdata-jp/python_sample/blob/main/ALMA_7B_Ja_GPTQ_Ja_En_batch_translation_sample.ipynb
# 6. deeplx + docker: https://deeplx.owo.network/install/

import json
import re

from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np

from openai import OpenAI

CONFIG_FILE = Path('config-local.json')
CONFIG = json.loads(CONFIG_FILE.read_text(encoding='utf8'))


def deeplx(text: str, src_lang='JP', tar_lang='EN'):
    url = "http://127.0.0.1:1188/translate"
    data = {"text": text, "source_lang": src_lang, "target_lang": tar_lang}
    return httpx.post(url=url, data=json.dumps(data)).text


def openaix(client,
            text: str,
            prompt: str = None,
            model: str = 'gpt-3.5-turbo-0125',
            max_tokens: int = 4000,
            **kwargs) -> str:
    """
    Use openai api to create chats and get responses.

    :param client: openai client
    :param text: query text
    :param prompt: prompt for translator
    :param model: gpt model, default `gpt-3.5-turbo`
    :param max_tokens: the maximum context, default 4000 (max=4096)
    :return: string
    """
    if prompt is None:
        prompt = f"""Please align the Japanese content and the English translation in a key-value JSON format 
        without any quotes by translating the following content with a professional English language: {text}"""
    else:
        prompt += f'{text}'

    prompts = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
        messages=prompts,
        model=model,
        max_tokens=max_tokens,
        n=1, stop=None, temperature=0.5,
    )
    return response.choices[0].message.content


def quick_translate(client,
                    items: list,
                    max_input: int = 500,
                    **kwargs):
    """To maximise the use of tokens, the function merges snippets and do translation for all"""
    init = ''
    results = []
    texts = deepcopy(items)
    texts.reverse()  # NB. must be reversed to keep the same order!
    while True:  # MUST be stopped at a time!
        try:
            if len(texts) == 0:  # loop-breaking
                if len(init) > 0:
                    r = openaix(client, init.strip('|').strip(), **kwargs)
                    if not r.endswith('}'):
                        r += '}'

                    results += [json.loads(r)]
                    init = ''
                    break
                else:
                    break

            init += texts.pop() + '|'
            if len(init) < (max_input - 10) and len(texts) > 0:
                continue
            else:
                r = openaix(client, init.strip('|').strip(), **kwargs)
                if not r.endswith('}'):
                    r += '}'

                results += [json.loads(r)]
                print(init)
                init = ''

        except Exception as e:
            print(f'Unexpected error: {e}, at position {len(texts)}\nreturn is: {r}\ncontent is {init}')
            try:
                results += [ast.literal_eval(r)]
                print('ast worked! The loop continues!')
            except Exception as e:
                print(f'Unexpected error: {e}, ast does not work either')
                break

    return results


if __name__ == '__main__':
    import pandas as pd

    # general config
    path = Path('data')
    client = OpenAI(api_key=CONFIG['openai-api-key'])

    # test for openai
    feed = openaix(client, '豚肉  バラのブロック２パック')
    feed = openaix(client, '砂糖   小さじ１から２くらい')
    # to taste, in an amount that results in the taste that one wants
    feed = openaix(client, 'しょうが         お好みで')

    # test for bulk translation
    texts = ['人参、ピーマン、トマトetc', '一台', '二枚', 'ミツカンのを使いました。',
             'マリネ液より少なめに', '4コ', '2~3つぶ', '3mm厚', '45ml(1.5oz)', '1/8カット', '15ml(0.5oz)',
             '1tsp', '30ml(1oz)', '15-30ml(0.5-1.0oz)',
             'あれば', '自分できめたスプーンやおたま等で1杯', '決めたはかりで2杯',
             '人数を考えてどのぐらいでもいいよ', 'スープのとろみ付け用', '0・5カップ (cup)']
    feeds = quick_translate(client, texts)

    # load need-translation data and then do the translation
    uni_ing = (path / 'unique-ing.text').read_text().split('\n')
    uni_qty = (path / 'unique-qty.text').read_text().split('\n')

    ing = pd.DataFrame([json.loads(it) for it in uni_ing if it])
    qty = pd.DataFrame([json.loads(it) for it in uni_qty if it])
    del uni_ing, uni_qty

    # ... to fix the alignment issue, use another prompt
    prompt = """Please align the Japanese content and the English translation in a key-value JSON format without any 
    quotes by translating the following content with a professional English language: """
    res = quick_translate(client, texts=texts[:10], model='gpt-3.5-turbo-0125', prompt=prompt, max_tokens=20)

    # try quick translation
    def easy_clean(x: str):
        if re.findall(r'id:\d+', x):
            return np.nan

        if len(re.findall(r'[\d\w]{2,}', x)) == 0:
            return np.nan

        return x

    texts = ing['source'].apply(lambda x: easy_clean(x))
    texts = texts[~texts.isna()].values.tolist()

    # res has been created!
    final = []
    res = quick_translate(client, items=texts, max_input=500, max_tokens=4000, model='gpt-3.5-turbo-0125')

    # align the translations
    final += res
    rows = []
    for each in final:
        if isinstance(each, dict):
            for k, v in each.items():
                rows += [[k, v]]

    rows = pd.DataFrame(rows, columns=['source', 'translated'])
    x = rows.translated.apply(lambda x: True if isinstance(x, dict) else False)
    rows = rows[~x]
    rows = rows.drop_duplicates()
    translated = ing.merge(rows, on='source', how='left')  # use ing / qty to merge
    texts = translated.loc[translated.translated.isna(), "source"].apply(lambda x: x if re.findall(r'[\d\w]{2,}', x) else np.nan)
    texts = texts[~texts.isna()].apply(lambda x: easy_clean(x))
    texts = texts[~texts.isna()].values.tolist()
    # save
    # rows should be used for repeated translation
    rows.to_excel(path / 'translated-ing-mapping.xlsx', index=False)
    translated.to_excel(path / 'translated-ing.xlsx', index=False)
