---
title: ReCoRDS
emoji: 🚀
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# ReCoRDS

NASA CEA ([RocketCEA](https://github.com/sonofeft/RocketCEA)) による液体・ハイブリッドロケットエンジンの化学平衡性能計算 Web アプリ (Streamlit + Plotly)。

- 推進剤: 組み込みの酸化剤 / 燃料 (ABS 樹脂を追加) と、項目別に入力するカスタムカード (二液・一液)
- 流れモデル: 平衡流 / スロートで凍結 / 燃焼室で凍結、有限断面積燃焼室 (縮流比 fac_CR)
- ② 性能特性: 横軸 (O/F・ε・Pc) と表示する量を選んで「生成」したグラフを並べて比較
- ③ 平衡流 / 凍結流、④ 推進剤比較、ΔV 見積もり
- 単位: 圧力 MPa / c* m/s / 温度 K / 比推力 s

## ローカルで動かす

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows では RocketCEA のビルド済みパッケージがそのまま入ります。Linux / macOS では gfortran が必要です。

## 無料で公開する

どちらも GitHub のリポジトリ (このフォルダの `app.py`・`requirements.txt`・`packages.txt`・`Dockerfile`・`README.md`) から公開します。

### Streamlit Community Cloud (おすすめ)

1. share.streamlit.io に GitHub アカウントでログインし「Create app」
2. リポジトリ・ブランチ・メインファイル `app.py` を指定
3. 「Advanced settings」で Python バージョンを **3.12** にして Deploy

`packages.txt` の `gfortran` が自動で入り、RocketCEA がソースからビルドされます。

### Render (Docker)

1. render.com で「New → Web Service」、リポジトリを指定
2. Language/Runtime に **Docker**、Instance Type に **Free** を選んで作成

`Dockerfile` が使われます (ポートは Render が指定する `PORT` に従います)。

### Hugging Face Spaces

この README の先頭の設定 (`sdk: docker`, `app_port: 8501`) と `Dockerfile` でも動きますが、
現在は無料アカウントの CPU 枠が制限されていて、起動できないことがあります。

RocketCEA は GPL-3.0 ライセンスです。
