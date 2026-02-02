# 大手小売業 財務分析ダッシュボード

Jupyter Notebookで作成した財務分析ロジックをStreamlitでWebアプリ化したものです。
主要な小売企業のPL/BS/CFおよび労働生産性の推移を可視化します。

## 構成
- `app.py`: メインプログラム
- `data/`: 財務データ（Excel）を格納
- `fonts/`: 日本語フォント（TTF）を格納
- `requirements.txt`: 依存ライブラリ一覧

## 運用方法
1. `data/financial_data.xlsx` を更新してGitHubにプッシュすると、Webアプリ側も自動で更新されます。
2. 基準年度を選択すると、その年度から過去5年間の推移が表示されます。

## ローカルでの実行方法
```bash
pip install -r requirements.txt
streamlit run app.py