# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import requests
import io
import gc

def main():
    st.set_page_config(page_title="科研費 文章ベクトル検索", layout="wide")

    # --- 設定 ---
    OWNER = "yusuke-kawazoe"
    REPO = "kaken_search"
    TAG = "v1.0"
    VEC_FILE_NAME = "vectors.npz" 
    META_FILE_NAME = "metadata.parquet"

    # セッション状態の初期化（クリア機能用）
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    def clear_text():
        st.session_state.query_input = ""

    @st.cache_resource(show_spinner="データをロード中...")
    def load_data():
        try:
            token = st.secrets["GITHUB_TOKEN"]
            headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

            # リリース情報の取得
            release_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}"
            res = requests.get(release_url, headers=headers)
            res.raise_for_status()
            assets = res.json().get("assets", [])
            asset_ids = {a["name"]: a["id"] for a in assets}

            def fetch_bin(file_name):
                aid = asset_ids.get(file_name)
                url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/assets/{aid}"
                h = headers.copy()
                h["Accept"] = "application/octet-stream"
                r = requests.get(url, headers=h)
                r.raise_for_status()
                b = io.BytesIO(r.content)
                del r
                gc.collect()
                return b

            # メタデータ読み込み
            df = pd.read_parquet(fetch_bin(META_FILE_NAME))
            gc.collect()

            # ベクトルデータ読み込み
            vectors = sp.load_npz(fetch_bin(VEC_FILE_NAME))
            if not isinstance(vectors, sp.csr_matrix):
                vectors = vectors.tocsr()
            gc.collect()

            return vectors, df
        except Exception as e:
            st.error(f"ロード失敗: {e}")
            return None, None

    vectors_db, df_meta = load_data()
    if vectors_db is None:
        st.stop()

    st.markdown("<h1 style='font-size:24px;'>科研費 文章ベクトル検索</h1>", unsafe_allow_html=True)

    # --- 検索画面 ---
    # keyを指定することでセッション状態で値を管理
    query_text = st.text_area("申請課題の概要を入力してください", height=150, key="query_input")
    
    top_n = 100
    similarity_threshold = 0.01

    # ボタンを横並びにするためのカラム設定
    col1, col2, _ = st.columns([1, 1, 8]) # 比率を調整してボタンの間隔を制御

    with col1:
        search_clicked = st.button("検索実行", type="primary", use_container_width=True)
    with col2:
        st.button("クリア", on_click=clear_text, use_container_width=True)

    if search_clicked:
        if not query_text.strip():
            st.warning("テキストを入力してください。")
        else:
            with st.spinner("検索中..."):
                # 1. クエリのベクトル化
                hv = HashingVectorizer(
                    analyzer="char", ngram_range=(2, 4), n_features=2**14,
                    alternate_sign=False, norm='l2', lowercase=False
                )
                q_vec = hv.transform([query_text])

                # 2. 類似度計算
                sims = vectors_db.dot(q_vec.T).toarray().ravel().astype(np.float32)
                
                # 3. 閾値以上のインデックスを取得
                valid_idx = np.where(sims >= similarity_threshold)[0]
                
                if len(valid_idx) == 0:
                    st.warning("条件に合う課題が見つかりませんでした。")
                    del sims, q_vec
                    gc.collect()
                else:
                    # 4. 上位N件に絞り込み
                    if len(valid_idx) > top_n:
                        top_indices = valid_idx[np.argsort(-sims[valid_idx])[:top_n]]
                    else:
                        top_indices = valid_idx[np.argsort(-sims[valid_idx])]

                    # 5. 結果の抽出
                    res_df = df_meta.iloc[top_indices].copy()
                    res_df.insert(0, "順位", range(1, len(res_df) + 1))
                    res_df.insert(1, "類似度", sims[top_indices].tolist())
                    
                    del sims, valid_idx, top_indices
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
                    
                    display_cols = ["順位", "類似度", "題名", "氏名", "課題番号", "種目", "区分", "概要"]
                    existing_cols = [c for c in display_cols if c in res_df.columns]
                    res_df = res_df[existing_cols]

                    st.success(f"{len(res_df)} 件ヒットしました。")

                    # テーブル表示
                    st.dataframe(
                        res_df,
                        column_config={
                            "類似度": st.column_config.NumberColumn(format="%.4f"),
                            "概要": st.column_config.TextColumn(width="large"),
                        },
                        use_container_width=True,
                        height=500,
                        hide_index=True
                    )

                    # CSVダウンロード
                    csv = res_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("検索結果を保存（CSV）", csv, "results.csv", "text/csv")
                    
                    del res_df
                    gc.collect()

if __name__ == "__main__":
    main()
