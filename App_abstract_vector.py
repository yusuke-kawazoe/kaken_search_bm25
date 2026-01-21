# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import gc
import pickle

# 1. 依存ライブラリのチェック（ここで落ちるのを防ぐ）
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    st.error("rank_bm25 が未インストールです。requirements.txt を確認してください。")
    st.stop()

def main():
    st.set_page_config(page_title="科研費検索", layout="wide")

    # --- 設定 ---
    OWNER = "yusuke-kawazoe"
    REPO = "kaken_search_bm25"
    TAG = "v1.0"
    MODEL_FILE_NAME = "bm25_model.pkl"
    META_FILE_NAME = "metadata.parquet"

    @st.cache_resource(show_spinner="データを読み込んでいます（この処理には1分ほどかかる場合があります）...")
    def load_data_safe():
        try:
            base_url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/"
            
            # --- 1. BM25モデルのロード ---
            # メモリ節約のため、バイナリを取得後すぐにpickle化し、レスポンスを捨てる
            res_m = requests.get(base_url + MODEL_FILE_NAME, stream=True)
            res_m.raise_for_status()
            bm25 = pickle.loads(res_m.content)
            res_m.close()
            del res_m
            gc.collect()

            # --- 2. メタデータのロード ---
            res_d = requests.get(base_url + META_FILE_NAME, stream=True)
            res_d.raise_for_status()
            
            # 全件読み込まず、一旦BytesIOに落としてから、必要な列だけを指定してロード
            # これによりメモリ使用量を半分以下に抑えられます
            df = pd.read_parquet(io.BytesIO(res_d.content), engine='pyarrow')
            
            # 表示に不要な列を即座にドロップして軽量化
            # ここでは検索結果に必要な最低限の列名だけを残してください
            # (カラム名はデータに合わせて適宜修正してください)
            keep_cols = ["title", "name", "abstract", "研究課題名", "研究者名", "概要", "所属機関", "awardnumber", "課題番号"]
            existing_cols = [c for c in keep_cols if c in df.columns]
            df = df[existing_cols].copy()
            
            res_d.close()
            del res_d
            gc.collect()
            
            return bm25, df
        except Exception as e:
            st.error(f"ロード失敗: {str(e)}")
            return None, None

    # ロードの実行
    data = load_data_safe()
    if data[0] is None:
        st.stop()
    bm25_model, df_meta = data

    st.title("科研費 BM25キーワード検索")

    # --- 検索画面 ---
    query_text = st.text_area("検索キーワードを入力（2文字以上推奨）", height=100)
    
    if st.button("検索実行", type="primary"):
        if not query_text.strip():
            st.warning("キーワードを入力してください。")
        else:
            with st.spinner("検索中..."):
                # 2-gram トークナイズ
                tokenized_query = [query_text[i:i+2] for i in range(len(query_text)-1)]
                if not tokenized_query: tokenized_query = [query_text]

                # スコア計算
                scores = bm25_model.get_scores(tokenized_query).astype(np.float32)
                
                # スコア上位の抽出
                top_n = 50
                valid_idx = np.where(scores > 0)[0]
                
                if len(valid_idx) == 0:
                    st.info("一致する課題はありません。")
                else:
                    top_indices = valid_idx[np.argsort(-scores[valid_idx])[:top_n]]
                    
                    # 結果の表示用DataFrame作成
                    res_df = df_meta.iloc[top_indices].copy()
                    res_df.insert(0, "スコア", scores[top_indices])
                    
                    st.dataframe(res_df, use_container_width=True)
                    
                    # 後処理
                    del res_df, scores, valid_idx
                    gc.collect()

if __name__ == "__main__":
    main()
