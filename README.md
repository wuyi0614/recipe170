# Recipe170

"Recipe170" is the title of our Global Recipe Analysis Project, which analyses the worldwide recipes with NLP-based solutions. 

### 1. Quick start

- 1.1 Environment specification
  - Miniconda (conda 23.11.0)
  - Python 3.9
  - Pipenv 2023.11.17

### 2. TODOs

The overall processing of recipe data consists of a few steps:

- [ ] Materials
  - [ ] a nice translator for Japanese, e.g. localised pretrained JP->EN model
  - [ ] parse materials + usage and export a mapping dataframe
  - [ ] parse materials and its upper-level materials, exporting a mapping dataframe
- [ ] Procedure
  - [ ] longer-token translator
  - [ ] entity-recognition for cooking/timing/objects
