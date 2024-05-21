# Codes for translations between Japanese and English.
#
# Created by Yi on 07/05/2024.
#

# 1. check if we need https://cloud.google.com/translate/pricing?hl=zh-cn#basic-pricing
# 2. use free model for translation, https://py-googletrans.readthedocs.io/en/latest/
# 3. machine-translation - https://github.com/christianversloot/machine-learning-articles/blob/main/easy-machine-translation-with-machine-learning-and-huggingface-transformers.md
# 4. fasttext - https://medium.com/@FaridSharaf/text-translation-using-nllb-and-huggingface-tutorial-7e789e0f7816
# 5. colab based - https://huggingface.co/webbigdata/ALMA-7B-Ja-V2-GPTQ-Ja-En; https://github.com/webbigdata-jp/python_sample/blob/main/ALMA_7B_Ja_Free_Colab_sample.ipynb
#    - batch translation, https://github.com/webbigdata-jp/python_sample/blob/main/ALMA_7B_Ja_GPTQ_Ja_En_batch_translation_sample.ipynb
# 6. deeplx + docker: https://deeplx.owo.network/install/

import os
import json

import httpx
from openai import OpenAI


def deeplx(text: str, src_lang='JP', tar_lang='EN'):
    url = "http://127.0.0.1:1188/translate"
    data = {"text": text, "source_lang": src_lang, "target_lang": tar_lang}
    return httpx.post(url=url, data=json.dumps(data)).text


def openaix(client,
            text: str,
            prompt: str = None,
            model: str = 'gpt-3.5-turbo',
            max_tokens: int = 2000) -> str:
    """
    Use openai api to create chats and get responses.

    :param client: openai client
    :param text: query text
    :param prompt: prompt for translator
    :param model: gpt model, default `gpt-3.5-turbo`
    :param max_tokens: the maximum context, default 2,000
    :return: string
    """
    if prompt is None:
        prompt = f"""Please act as a professional translator for Japanese and English and an experienced cooker 
        who are very familiar with foods, ingredients and cooking procedures, and then you will help me translate the 
        following Japanese content into English: {text}"""

    prompts = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
        messages=prompts,
        model=model,
        max_tokens=max_tokens,
        n=1, stop=None, temperature=0.5
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    apikey = ''
    client = OpenAI(api_key=apikey)

    # test for openai
    feed = openaix(client, '豚肉  バラのブロック２パック')
    feed = openaix(client, '砂糖   小さじ１から２くらい')
    # to taste, in an amount that results in the taste that one wants
    feed = openaix(client, 'しょうが         お好みで')
