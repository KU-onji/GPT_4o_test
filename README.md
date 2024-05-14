# GPT_4o_test
`GPT-4o` のテスト用リポジトリ

## Setting up virtual environment
1. `poetry` , `pyenv` を(頑張って)インストールする
2. このリポジトリを `git clone` する
3. `pyenv install 3.10.11` する
4. `pyenv local 3.10.11` する
5. `poetry config virtualenvs.in-project true` する
6. `git clone` してきたディレクトリ(以下、そこでのコマンド)で `poetry install` を叩く
7. `pre-commit install` を叩く
8. **(最重要)** ↑で作られた `paper_summary_bot/.git/hooks/pre-commit` (ファイル)を↓のように編集する

```shell
 #!/bin/sh
 . .\\.venv\\Scripts\\activate
 #!/usr/bin/env bash
 ...
```

ダメだったらご一報ください。

<!-- README.md template -->
<!--
# リポジトリ名
なんか説明を書く

## Setting up virtual environment
1. `poetry` , `pyenv` を(頑張って)インストールする
2. このリポジトリを `git clone` する
3. `pyenv install 3.10.11` する
4. `pyenv local 3.10.11` する
5. `poetry config virtualenvs.in-project true` する
6. `git clone` してきたディレクトリ(以下、そこでのコマンド)で `poetry install` を叩く
7. `pre-commit install` を叩く
8. **(最重要)** ↑で作られた `paper_summary_bot/.git/hooks/pre-commit` (ファイル)を↓のように編集する

```shell
 #!/bin/sh
 . .\\.venv\\Scripts\\activate
 #!/usr/bin/env bash
 ...
```

ダメだったらご一報ください。
-->