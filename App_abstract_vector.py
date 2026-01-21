# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import gc
import pickle

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
        st.rerun()

    # --- ヘルパー関数 (Tkinter版から移植) ---
    def tokenize_ngram(text, n=2):
        if not isinstance(text, str):
            return []
        # 文字単位のn-gram (デフォルトは2-gram)
        return [text[i:i+n] for i in range(len(text)-n+1)]

    @st.cache_resource(show_spinner="検索モデルをロード中...")
    def load_data():
        try:
            # fetch_bin の定義
            def fetch_bin(filename):
                url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{filename}"
                response = requests.get(url)
                response.raise_for_status()
                return io.BytesIO(response.content)

            # ロード処理
            bm25_bin = fetch_bin(MODEL_FILE_NAME)
            bm25 = pickle.load(bm25_bin)
            
            meta_bin = fetch_bin(META_FILE_NAME)
            df = pd.read_parquet(meta_bin)
            
            return bm25, df
        except Exception as e:
            # エラーの詳細を画面に出す
            st.error(f"詳細エラー: {e}")
            import traceback
            st.code(traceback.format_exc()) # スタックトレースを表示
            return None, None

    bm25_model, df_meta = load_data()
    if bm25_model is None:
        st.stop()

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
            with st.spinner("計算中..."):
                # 1. クエリのトークナイズ (移植した2-gram方式)
                tokenized_query = tokenize_ngram(query_text, n=2)
                
                # 1文字だけの場合のフォールバック
                if not tokenized_query and len(query_text) > 0:
                    tokenized_query = [query_text]

                # 2. スコア計算
                scores = bm25_model.get_scores(tokenized_query)

                # 3. 高速な上位N件のインデックス取得 (移植したロジック)
                n_docs = len(scores)
                k = min(top_n, n_docs)
                
                # 有効なスコア（0より大きい）があるか確認
                if np.max(scores) <= 0:
                    st.warning("キーワードに一致する課題が見つかりませんでした。")
                else:
                    # np.argpartitionを使用して上位k件を効率的に抽出
                    if n_docs > k * 5:
                        top_indices = np.argpartition(-scores, k)[:k]
                        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
                    else:
                        sorted_top_indices = np.argsort(-scores)[::-1][:k]
                    
                    # スコアが0以下のものは除外
                    sorted_top_indices = [i for i in sorted_top_indices if scores[i] > 0]

                    # 4. 結果の抽出と整形
                    res_rows = []
                    for rank, idx in enumerate(sorted_top_indices, 1):
                        meta_row = df_meta.iloc[idx].to_dict()
                        
                        # カラム名の揺れを吸収 (Tkinter版のロジック)
                        res_rows.append({
                            "順位": rank,
                            "スコア": float(f"{scores[idx]:.4f}"),
                            "題名": meta_row.get("title") or meta_row.get("研究課題名") or "",
                            "所属機関": meta_row.get("organization") or meta_row.get("所属機関") or "",
                            "氏名": meta_row.get("name") or meta_row.get("研究者名") or "",
                            "課題番号": meta_row.get("awardnumber") or meta_row.get("課題番号") or "",
                            "種目": meta_row.get("section") or meta_row.get("種目") or "",
                            "区分": meta_row.get("review_section") or meta_row.get("区分") or "",
                            "概要": meta_row.get("abstract") or meta_row.get("概要") or ""
                        })

                    res_df = pd.DataFrame(res_rows)

                    # 不要なメモリ解放
                    del scores, sorted_top_indices
                    gc.collect()

                    st.success(f"{len(res_df)} 件表示しています。")

                    # 5. テーブル表示
                    st.dataframe(
                        res_df,
                        column_config={
                            "スコア": st.column_config.NumberColumn(format="%.4f"),
                            "概要": st.column_config.TextColumn(width="large"),
                        },
                        use_container_width=True,
                        height=600,
                        hide_index=True
                    )

                    # 6. CSVダウンロード
                    csv = res_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("検索結果をCSVで保存", csv, "kaken_search_results.csv", "text/csv")
                    
                    del res_df
                    gc.collect()

if __name__ == "__main__":
    main()
