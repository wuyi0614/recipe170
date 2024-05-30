# Recipe170

"Recipe170" is the title of our Global Recipe Analysis Project, which analyses the worldwide recipes with NLP-based solutions. 

### 1. Quick start

- 1.1 Environment specification
  - Miniconda (conda 23.11.0)
  - Python 3.9
  - Pipenv 2023.11.17

- 1.2 Preprocessing
  - Step 1: cleaning texts
  - Step 2: unify quantity units
  - Step 3: get unique values and match all entries
  - Step 4: translate ingredients and quantities

### 2. TODOs

The overall processing of recipe data consists of a few steps:

- [ ] Materials
  - [x] a nice translator for Japanese--GPT-3.5-turbo API
  - [x] parse materials + usage and export a mapping dataframe
  - [ ] parse materials and its upper-level materials, exporting a mapping dataframe
- [ ] Units
  - [x] numeric units, 1 0 0 g
  - [x] textual units, 5、6個
  - [ ] enhanced units, 强弱
- [ ] Procedure
  - [x] longer-token translator
  - [ ] entity-recognition for cooking/timing/objects
- [ ] Caveats
  - [x] some recipes do not have ingredients!!!
  - [ ] create an error table for manual annotation (~20k from ingredient side)
  - [ ] use steps (ingredient extraction) to supplement to the ingredients

### 3. Issue Log

- `id=eb1d2e4604d93afd2753880c9f79b48e4d2fe582`, `issue=皮*小麦粉*3と3/4カップ `
- `id=a00912dda86900e54e0df98fe658cb2f4686f23e`, `issue=皆さんのレシピ*みおりんさん＊8801、MIRELLEさん*8785、カヨリーヌさん*9176、まるりんさん*8799`
