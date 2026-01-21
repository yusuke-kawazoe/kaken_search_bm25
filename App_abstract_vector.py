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
    REPO = "kaken_search"
    TAG = "v1.0"
    MODEL_FILE_NAME = "bm25_model.pkl"  # ベクトルnpzから変更
    META_FILE_NAME = "metadata.parquet"

    # セッション状態の初期化
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    def clear_text():
        st.session_state.query_input = ""
        st.rerun()

    @st.cache_resource(show_spinner="検索モデルをロード中...")
    def load_data():
        try:
            # ... (中略：GitHubからの取得処理) ...

            # 1. まずBM25モデルだけを読み込む
            bm25_bin = fetch_bin(MODEL_FILE_NAME)
            bm25 = pickle.load(bm25_bin)
            del bm25_bin # すぐにバイナリを捨てる
            gc.collect()

            # 2. 次にメタデータを読み込む
            meta_bin = fetch_bin(META_FILE_NAME)
            df = pd.read_parquet(meta_bin)
            del meta_bin
            gc.collect()

            # 3. メタデータの軽量化（検索に不要な列をこの時点で捨てる）
            # もし「概要」が巨大なら、一旦「概要」を落として「題名」等だけで検索し、
            # 結果表示の時だけ「概要」をマージする方法もあります。
            
            return bm25, df

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
                # 1. クエリのトークナイズ (文字2-gram)
                # モデル作成時と同じ方式にする必要があります
                tokenized_query = [query_text[i:i+2] for i in range(len(query_text)-1)]
                if not tokenized_query: # 1文字だけ入力された場合
                    tokenized_query = [query_text]

                # 2. スコア計算 (BM25)
                # スコアは類似度(0~1)ではなく、関連度スコアとして算出されます
                scores = bm25_model.get_scores(tokenized_query).astype(np.float32)
                
                # 3. スコアが0より大きいインデックスを取得
                valid_idx = np.where(scores > 0)[0]
                
                if len(valid_idx) == 0:
                    st.warning("キーワードに一致する課題が見つかりませんでした。")
                else:
                    # 4. 上位N件に絞り込み
                    if len(valid_idx) > top_n:
                        top_indices = valid_idx[np.argsort(-scores[valid_idx])[:top_n]]
                    else:
                        top_indices = valid_idx[np.argsort(-scores[valid_idx])]

                    # 5. 結果の抽出
                    res_df = df_meta.iloc[top_indices].copy()
                    res_df.insert(0, "順位", range(1, len(res_df) + 1))
                    res_df.insert(1, "スコア", scores[top_indices].tolist())
                    
                    # 不要なメモリ解放
                    del scores, valid_idx, top_indices
                    gc.collect()

                    # 6. 表示用にカラム名を整理
                    rename_map = {
                        "研究課題名": "題名", "title": "題名",
                        "研究者名": "氏名", "name": "氏名",
                        "課題番号": "課題番号", "awardnumber": "課題番号",
                        "種目": "種目", "section": "種目",
                        "区分": "区分", "review_section": "区分",
                        "概要": "概要", "abstract": "概要"
                    }
                    res_df = res_df.rename(columns=rename_map)
                    
                    display_cols = ["順位", "スコア", "題名", "氏名", "課題番号", "種目", "区分", "概要"]
                    existing_cols = [c for c in display_cols if c in res_df.columns]
                    res_df = res_df[existing_cols]

                    st.success(f"{len(res_df)} 件表示しています。")

                    # テーブル表示
                    st.dataframe(
                        res_df,
                        column_config={
                            "スコア": st.column_config.NumberColumn(format="%.2f"),
                            "概要": st.column_config.TextColumn(width="large"),
                        },
                        use_container_width=True,
                        height=600,
                        hide_index=True
                    )

                    # CSVダウンロード
                    csv = res_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("検索結果をCSVで保存", csv, "kaken_search_results.csv", "text/csv")
                    
                    del res_df
                    gc.collect()

if __name__ == "__main__":
    main()
