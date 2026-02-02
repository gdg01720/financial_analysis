import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import io

# --- 1. 日本語フォント設定 (ローカル & Cloud 両対応) ---
def setup_font():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, "fonts", "ipaexg.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        return prop.get_name()
    else:
        plt.rcParams['font.family'] = ['Meiryo', 'MS Gothic', 'sans-serif']
        return 'sans-serif'

font_name = setup_font()
sns.set_theme(style="whitegrid", rc={"font.family": font_name})

st.set_page_config(page_title="大手小売業 財務分析ダッシュボード", layout="wide")

# --- 2. ユーティリティ関数 ---
def format_fy(year):
    try:
        return f"FY{int(year)}"
    except:
        return year

def convert_to_million(df):
    """オリジナルのJupyter Notebookにあった単位変換処理"""
    # 数値タイプの説明変数を抽出
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # 10万以上であれば百万で割る（百万円単位にする）
    for column in numeric_columns:
        # 計算のために一旦floatとして扱い、後で表示フォーマットで整数/小数を制御する
        df[column] = df[column].apply(lambda x: (x / 1000000.0) if np.abs(x) >= 100000 else x)
    return df

def get_html_report(df, title):
    """HTMLダウンロード用データの生成"""
    return f"""
    <html><head><meta charset='utf-8'>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: right; }}
        th {{ background-color: #f4f4f4; text-align: center; }}
        h2 {{ color: #333; border-left: 5px solid #1f77b4; padding-left: 10px; }}
    </style></head>
    <body><h2>{title}</h2>{df.to_html()}</body></html>
    """

# --- 3. データの読み込み ---
@st.cache_data
def load_financial_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", "financial_data.xlsx")
    if os.path.exists(path):
        df = pd.read_excel(path)
        # 欠損値（ハイフン）を0に置換
        num_cols = df.columns.drop(['企業名', '決算年度', '決算四半期'], errors='ignore')
        for col in num_cols:
            df[col] = pd.to_numeric(df[col].astype(str).replace('-', '0'), errors='coerce').fillna(0)
        
        # オリジナルの単位変換処理を適用
        df = convert_to_million(df)
        return df
    return None

# --- 4. メイン UI ---
st.title("📈 大手小売業 財務分析ダッシュボード")

df_raw = load_financial_data()

if df_raw is not None:
    # --- サイドバー ---
    st.sidebar.header("分析条件")
    selected_company = st.sidebar.selectbox("企業名を選択", sorted(df_raw['企業名'].unique()))
    raw_years = sorted(df_raw['決算年度'].unique(), reverse=True)
    year_labels = [format_fy(y) for y in raw_years]
    selected_year_label = st.sidebar.selectbox("基準年度を選択", year_labels)
    
    selected_year = int(selected_year_label.replace("FY", ""))
    start_year = selected_year - 4

    mask = (df_raw['企業名'] == selected_company) & \
           (df_raw['決算年度'] >= start_year) & \
           (df_raw['決算年度'] <= selected_year)
    df_analysis = df_raw[mask].sort_values('決算年度').copy()

    if not df_analysis.empty:
        df_analysis['年度表示'] = df_analysis['決算年度'].apply(format_fy)
        years_display = df_analysis['年度表示'].tolist()

        tab_pl, tab_bs, tab_cf, tab_prod, tab_kpi = st.tabs([
            "損益(PL)", "財政状態(BS)", "キャッシュフロー(CF)", "労働生産性", "主要KPI"
        ])

        # --- フォーマット定義 ---
        # 小数点1位まで表示する項目のリスト
        float_cols = [
            '営業利益率', '売上総利益率', '純利益率', 'ROE', 'ROA', 'ROIC', 
            '自己資本比率', '実質ROE', 'PER（会予）', 'PBR', '配当利回り（実績）',
            '正社員1人当り売上', '正社員1人当り営利', '全社員1人当り売上', '全社員1人当り営利'
        ]

        def display_formatted_table(df, cols, title):
            tmp = df[cols].copy()
            tmp['決算年度'] = tmp['決算年度'].apply(format_fy)
            tmp = tmp.set_index('決算年度')
            
            format_dict = {}
            for col in tmp.columns:
                if col in float_cols:
                    format_dict[col] = "{:.1f}"
                else:
                    format_dict[col] = "{:,.0f}"
            
            st.dataframe(tmp.style.format(format_dict), use_container_width=True)
            html_content = get_html_report(tmp, f"{selected_company} - {title}")
            st.download_button(f"📥 {title} (HTML)", html_content, f"{title}.html", "text/html", key=title)

        # --- 各タブの描画 ---
        with tab_pl:
            st.subheader("収益推移と構造")
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs[0, 0].bar(years_display, df_analysis['売上高'], color='skyblue'); axs[0, 0].set_title('売上高')
            axs[0, 1].bar(years_display, df_analysis['営業利益'], color='orange'); axs[0, 1].set_title('営業利益')
            axs[1, 0].bar(years_display, df_analysis['売上高'], label='売上高')
            axs[1, 0].bar(years_display, df_analysis['営業収入'], bottom=df_analysis['売上高'], label='営業収入')
            axs[1, 0].set_title('収益構造'); axs[1, 0].legend()
            
            cost_r = (100 - df_analysis['売上総利益率'])
            sgna_r = (df_analysis['販管費'] * 100 / (df_analysis['売上高'] + 1e-9)) # 0除算回避
            axs[1, 1].bar(years_display, cost_r, label='原価率')
            axs[1, 1].bar(years_display, sgna_r, bottom=cost_r, label='販管費率')
            axs[1, 1].bar(years_display, df_analysis['営業利益率'], bottom=cost_r + sgna_r, label='営利率')
            axs[1, 1].set_title('コスト構造 (%)'); axs[1, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
            plt.tight_layout(); st.pyplot(fig)
            display_formatted_table(df_analysis, ['決算年度', '売上高', '営業収入', '売上総利益率', '販管費', '営業利益', '営業利益率'], "損益状況")

        with tab_bs:
            st.subheader("資産と効率性")
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs[0, 0].bar(years_display, df_analysis['総資産']); axs[0, 0].set_title('総資産')
            axs[0, 1].bar(years_display, df_analysis['棚卸資産'], color='green'); axs[0, 1].set_title('棚卸資産')
            axs[1, 0].plot(years_display, df_analysis['総資産回転率'], marker='o', color='purple'); axs[1, 0].set_title('総資産回転率 (回)')
            inv_turn = (df_analysis['売上高'] / (df_analysis['棚卸資産'] + 1e-9))
            axs[1, 1].plot(years_display, inv_turn, marker='o', color='brown'); axs[1, 1].set_title('棚卸資産回転率 (回)')
            plt.tight_layout(); st.pyplot(fig)
            display_formatted_table(df_analysis, ['決算年度', '総資産', '流動資産', '固定資産', '棚卸資産', '有利子負債', '純資産', '自己資本比率'], "財政状態")

        with tab_cf:
            st.subheader("キャッシュフロー推移")
            x = np.arange(len(years_display)); width = 0.35
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.bar(x - width/2, df_analysis['営業CF'], width, label='営業CF', color='tab:blue')
            ax1.bar(x + width/2, df_analysis['投資CF'], width, label='投資CF', color='tab:green')
            ax1.plot(x, df_analysis['フリーCF'], color='red', marker='o', label='フリーCF'); ax1.axhline(0, color='black'); ax1.set_xticks(x); ax1.set_xticklabels(years_display); ax1.set_title("営業・投資・フリーCF推移"); ax1.legend(); st.pyplot(fig1)
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.bar(x - width/2, df_analysis['財務CF'], width, label='財務CF', color='tab:orange')
            ax2.bar(x + width/2, df_analysis['フリーCF'], width, label='フリーCF', color='tab:red'); ax2.axhline(0, color='black'); ax2.set_xticks(x); ax2.set_xticklabels(years_display); ax2.set_title("財務・フリーCF推移"); ax2.legend(); st.pyplot(fig2)
            display_formatted_table(df_analysis, ['決算年度', '営業CF', '投資CF', '財務CF', 'フリーCF', '現金及び預金'], "キャッシュフロー推移")

        with tab_prod:
            st.subheader("労働生産性の分析")
            pdf = df_analysis.copy()
            # 生産性の再計算（単位変換済みの売上・利益を使用）
            total_e = pdf['従業員数'] + pdf['パート社員'].fillna(0)
            pdf['正社員1人当り売上'] = pdf['売上高'] / (pdf['従業員数'] + 1e-9)
            pdf['正社員1人当り営利'] = pdf['営業利益'] / (pdf['従業員数'] + 1e-9)
            pdf['全従業員1人当り売上'] = pdf['売上高'] / (total_e + 1e-9)
            pdf['全従業員1人当り営利'] = pdf['営業利益'] / (total_e + 1e-9)

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs[0, 0].bar(years_display, pdf['正社員1人当り売上']); axs[0, 0].set_title('正社員1人当り売上高')
            axs[0, 1].bar(years_display, pdf['正社員1人当り営利'], color='orange'); axs[0, 1].set_title('正社員1人当り営業利益')
            axs[1, 0].bar(years_display, pdf['全従業員1人当り売上'], color='green'); axs[1, 0].set_title('全従業員1人当り売上高')
            axs[1, 1].bar(years_display, pdf['全従業員1人当り営利'], color='red'); axs[1, 1].set_title('全従業員1人当り営業利益')
            plt.tight_layout(); st.pyplot(fig)
            display_formatted_table(pdf, ['決算年度', '従業員数', 'パート社員', '正社員1人当り売上', '正社員1人当り営利', '全従業員1人当り売上', '全従業員1人当り営利'], "労働生産性分析")

        with tab_kpi:
            st.subheader("主要指標推移")
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs[0, 0].plot(years_display, df_analysis['ROIC'], marker='o'); axs[0, 0].set_title('ROIC (%)')
            axs[0, 1].plot(years_display, df_analysis['実質ROE'], marker='s', color='red'); axs[0, 1].set_title('実質ROE (%)')
            axs[1, 0].plot(years_display, df_analysis['ROA'], marker='^', color='green'); axs[1, 0].set_title('ROA (%)')
            axs[1, 1].bar(years_display, df_analysis['時価総額'], color='gold'); axs[1, 1].set_title('時価総額')
            plt.tight_layout(); st.pyplot(fig)
            display_formatted_table(df_analysis, ['決算年度', 'ROE', '実質ROE', 'ROA', 'ROIC', 'PER（会予）', 'PBR', '時価総額'], "主要KPI")

    else:
        st.warning("データが見つかりませんでした。")
else:
    st.error("ファイルが見つかりません。リポジトリの data/ フォルダを確認してください。")