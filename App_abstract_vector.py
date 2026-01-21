# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import gc
import pickle
import sys

# --- 1. ライブラリインポートの安全策 ---
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    st.error("ライブラリ 'rank_bm25' が見つかりません。requirements.txt を確認してください。")
    st.stop()

# --- 2. 設定 ---
OWNER = "yusuke-kawazoe"
REPO = "kaken_search_bm25"
TAG = "v1.0"
MODEL_FILE_NAME = "bm25_model.pkl"
META_FILE_NAME = "metadata.parquet"

# --- 3. ロジック関数 ---
def tokenize_ngram(text, n=2):
    if not isinstance(text, str):
        return []
    return [text[i:i+n] for i in range(len(text)-n+1)]

@st.cache_resource(show_spinner="巨大な検索モデルをロード中（これには数十秒かかる場合があります）...")
def load_data():
    """
    メモリ消費を抑えるために、読み込み直後に不要なデータを破棄する
    """
    try:
        base_url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/"
        
        # モデルのロード
        res_m = requests.get(base_url + MODEL_FILE_NAME)
        res_m.raise_for_status()
        bm25 = pickle.loads(res_m.content)
        del res_m # バイナリを即座に解放
        
        # メタデータのロード
        res_d = requests.get(base_url + META_FILE_NAME)
        res_d.raise_for_status()
        
        # --- メモリ節約の鍵：必要な列だけを指定して読み込む ---
        # Tkinter版で利用しているカラム名に合わせる
        target_cols = ["title", "研究課題名", "organization", "所属機関", "name", "研究者名", 
                       "awardnumber", "課題番号", "section", "種目", "review_section", "区分", "abstract", "概要"]
        
        full_df = pd.read_parquet(io.BytesIO(res_d.content))
        # 実際に存在する列のみを抽出
        existing_cols = [c for c in target_cols if c in full_df.columns]
        df = full_df[existing_cols].copy()
        
        del full_df
        del res_d
        gc.collect() # 強制的にメモリを整理
        
        return bm25, df
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return None, None

# --- 4. UI構築 ---
def main():
    st.set_page_config(page_title="科研費 文章検索", layout="wide")
    
    # CSSでTkinter風のフォント設定を再現
    st.markdown("""
        <style>
        .main { font-family: 'Meiryo UI'; }
        .stButton>button { width: 100%; }
        </style>
    """, unsafe_allow_html=True)

    st.title("科研費 文章検索 (BM25版)")
    
    # データのロード
    bm25_model, df_meta = load_data()
    if bm25_model is None:
        st.stop()

    # 入力エリア
    with st.container():
        st.subheader("検索条件")
        query_text = st.text_area("申請課題の概要を入力してください:", height=200, placeholder="ここに研究計画や概要をペーストしてください...")
        
        col1, col2, col3 = st.columns([2, 2, 6])
        search_clicked = col1.button("検索実行", type="primary")
        clear_clicked = col2.button("クリア")
        
        if clear_clicked:
            st.rerun()

    # 検索実行
    if search_clicked:
        if not query_text.strip():
            st.warning("申請課題の概要を入力してください。")
        else:
            with st.spinner("検索中..."):
                try:
                    # トークナイズ
                    tokenized_query = tokenize_ngram(query_text, n=2)
                    # スコア計算
                    scores = bm25_model.get_scores(tokenized_query)
                    
                    # 上位100件取得ロジック（Tkinter版を再現）
                    top_n = 100
                    k = min(top_n, len(scores))
                    
                    if len(scores) > k * 5:
                        top_indices = np.argpartition(-scores, k)[:k]
                        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
                    else:
                        sorted_top_indices = np.argsort(-scores)[::-1][:k]

                    rows = []
                    rank = 1
                    for idx in sorted_top_indices:
                        if scores[idx] <= 0:
                            break
                        
                        m = df_meta.iloc[idx].to_dict()
                        rows.append({
                            "順位": rank,
                            "スコア": float(f"{scores[idx]:.4f}"),
                            "題名": m.get("title") or m.get("研究課題名") or "",
                            "所属機関": m.get("organization") or m.get("所属機関") or "",
                            "氏名": m.get("name") or m.get("研究者名") or "",
                            "課題番号": m.get("awardnumber") or m.get("課題番号") or "",
                            "種目": m.get("section") or m.get("種目") or "",
                            "区分": m.get("review_section") or m.get("区分") or "",
                            "概要": m.get("abstract") or m.get("概要") or ""
                        })
                        rank += 1

                    if not rows:
                        st.info("条件に合う課題は見つかりませんでした。")
                    else:
                        results_df = pd.DataFrame(rows)
                        st.success(f"検索完了: {len(results_df)}件ヒット")

                        # 結果のダウンロード（CSV）
                        csv_data = results_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button("検索結果をCSVで保存", csv_data, "search_results.csv", "text/csv")

                        # 結果表示 (Treeviewの代わり)
                        st.subheader("検索結果")
                        st.dataframe(
                            results_df,
                            column_config={
                                "スコア": st.column_config.NumberColumn(format="%.4f"),
                                "概要": st.column_config.TextColumn(width="large"), # ダブルクリックせずとも読みやすく
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=600
                        )
                        
                        # 詳細表示用のセクション
                        st.divider()
                        st.info("💡 表の中のセルをダブルクリックすると内容をコピーできます。")

                except Exception as e:
                    st.error(f"検索中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
