# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import gc
import pickle
import sys

# ライブラリのチェック
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    st.error("ライブラリ 'rank_bm25' が見つかりません。 `pip install rank_bm25` を実行してください。")
    st.stop()

def main():
    st.set_page_config(page_title="科研費 BM25キーワード検索", layout="wide")

    # --- 設定 ---
    OWNER = "yusuke-kawazoe"
    REPO = "kaken_search_bm25"
    TAG = "v1.0"
    MODEL_FILE_NAME = "bm25_model.pkl"
    META_FILE_NAME = "metadata.parquet"

    # セッション状態の初期化
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    def clear_text():
        st.session_state.query_input = ""
        # 以前のバージョンとの互換性のため
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    def tokenize_ngram(text, n=2):
        if not isinstance(text, str): return []
        return [text[i:i+n] for i in range(len(text)-n+1)]

    @st.cache_resource(show_spinner="検索モデルをロード中...")
    def load_data():
        try:
            # GitHubからバイナリを取得
            def fetch_bin(filename):
                url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{filename}"
                response = requests.get(url, timeout=20) # タイムアウト設定
                response.raise_for_status()
                return io.BytesIO(response.content)

            # 1. BM25モデル
            bm25_bin = fetch_bin(MODEL_FILE_NAME)
            bm25 = pickle.load(bm25_bin)
            del bm25_bin

            # 2. メタデータ
            meta_bin = fetch_bin(META_FILE_NAME)
            df = pd.read_parquet(meta_bin)
            del meta_bin
            
            gc.collect()
            return bm25, df
        except Exception as e:
            # 起動時にエラーが出ないのを防ぐため、明示的に書き出す
            st.error(f"データのロードに失敗しました: {e}")
            return None, None

    # ロード実行
    data = load_data()
    if data[0] is None:
        st.warning("データの読み込みが完了するまで待機するか、設定（GitHubのURL等）を確認してください。")
        st.stop()
    
    bm25_model, df_meta = data

    st.markdown("<h1 style='font-size:24px;'>科研費 BM25キーワード検索</h1>", unsafe_allow_html=True)

    # --- 検索画面 ---
    query_text = st.text_area("検索キーワード、または研究概要を入力してください", height=150, key="query_input")
    top_n = 100
    
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        search_clicked = st.button("検索実行", type="primary", use_container_width=True)
    with col2:
        st.button("クリア", on_click=clear_text, use_container_width=True)

    if search_clicked:
        if not query_text.strip():
            st.warning("テキストを入力してください。")
        else:
            try:
                with st.spinner("計算中..."):
                    tokenized_query = tokenize_ngram(query_text, n=2)
                    if not tokenized_query and len(query_text) > 0:
                        tokenized_query = [query_text]

                    scores = bm25_model.get_scores(tokenized_query)
                    
                    if np.max(scores) <= 0:
                        st.warning("一致する課題が見つかりませんでした。")
                    else:
                        # ランキング処理
                        k = min(top_n, len(scores))
                        if len(scores) > k * 5:
                            top_indices = np.argpartition(-scores, k)[:k]
                            sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
                        else:
                            sorted_top_indices = np.argsort(-scores)[::-1][:k]
                        
                        sorted_top_indices = [i for i in sorted_top_indices if scores[i] > 0]

                        res_rows = []
                        for rank, idx in enumerate(sorted_top_indices, 1):
                            m = df_meta.iloc[idx].to_dict()
                            res_rows.append({
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

                        res_df = pd.DataFrame(res_rows)
                        st.success(f"{len(res_df)} 件ヒットしました。")

                        st.dataframe(
                            res_df,
                            column_config={
                                "スコア": st.column_config.NumberColumn(format="%.4f"),
                                "概要": st.column_config.TextColumn(width="large"),
                            },
                            use_container_width=True, height=600, hide_index=True
                        )

                        csv = res_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button("結果をCSV保存", csv, "results.csv", "text/csv")
            except Exception as e:
                st.error(f"検索実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
