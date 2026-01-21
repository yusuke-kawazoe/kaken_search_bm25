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
            # Streamlit Secretsからトークン取得
            token = st.secrets["GITHUB_TOKEN"]
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }

            # 1. リリース情報の取得
            release_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}"
            res = requests.get(release_url, headers=headers)
            res.raise_for_status()
            assets = res.json().get("assets", [])
            asset_ids = {a["name"]: a["id"] for a in assets}

            def fetch_bin(file_name):
                aid = asset_ids.get(file_name)
                if not aid:
                    raise FileNotFoundError(f"{file_name} がリリースに見つかりません")
                url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/assets/{aid}"
                h = headers.copy()
                h["Accept"] = "application/octet-stream"
                r = requests.get(url, headers=h)
                r.raise_for_status()
                return io.BytesIO(r.content)

            # 2. メタデータ読み込み
            df = pd.read_parquet(fetch_bin(META_FILE_NAME))
            gc.collect()

            # 3. BM25モデル読み込み
            bm25 = pickle.load(fetch_bin(MODEL_FILE_NAME))
            gc.collect()

            return bm25, df
        except Exception as e:
            st.error(f"ロード失敗: {e}")
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
