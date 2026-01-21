# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import gc
import pickle
import math

def main():
    st.set_page_config(page_title="科研費 BM25キーワード検索", layout="wide")

    # --- 設定 ---
    OWNER = "yusuke-kawazoe"
    REPO = "kaken_search_bm25"
    TAG = "v1.0"
    MODEL_FILE_NAME = "bm25_model.pkl"
    META_FILE_NAME = "metadata.parquet"

    # --- トークナイズ関数 (2-gram) ---
    def tokenize_ngram(text, n=2):
        if not isinstance(text, str): return []
        if len(text) < n: return [text]
        return [text[i:i+n] for i in range(len(text)-n+1)]

    @st.cache_resource(show_spinner="検索モデルをロード中...")
    def load_data():
        try:
            # GitHubからの取得用
            base_url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/"
            
            # BM25モデルの取得
            r_model = requests.get(base_url + MODEL_FILE_NAME, timeout=30)
            r_model.raise_for_status()
            bm25 = pickle.loads(r_model.content)
            
            # メタデータの取得
            r_meta = requests.get(base_url + META_FILE_NAME, timeout=30)
            r_meta.raise_for_status()
            df = pd.read_parquet(io.BytesIO(r_meta.content))
            
            return bm25, df
        except Exception as e:
            st.error(f"ロード失敗: {str(e)}")
            return None, None

    # データのロード
    bm25_model, df_meta = load_data()

    if bm25_model is None:
        st.info("データを読み込めませんでした。GitHubのパスまたはネット接続を確認してください。")
        st.stop()

    st.title("科研費 BM25キーワード検索")

    # --- 検索画面 ---
    query_text = st.text_area("検索キーワード、または研究概要を入力してください", height=150)
    
    col1, col2, _ = st.columns([1, 1, 8])
    search_clicked = col1.button("検索実行", type="primary", use_container_width=True)
    if col2.button("クリア", use_container_width=True):
        st.rerun()

    if search_clicked and query_text:
        with st.spinner("検索中..."):
            # トークナイズ
            tokenized_query = tokenize_ngram(query_text, n=2)
            
            # スコア計算
            try:
                # BM25Okapiオブジェクトのget_scoresを呼び出す
                scores = bm25_model.get_scores(tokenized_query)
                
                # 上位取得
                top_n = 100
                idx_all = np.where(scores > 0)[0]
                
                if len(idx_all) == 0:
                    st.warning("一致する課題は見つかりませんでした。")
                else:
                    # スコア順にソート
                    top_idx = idx_all[np.argsort(-scores[idx_all])[:top_n]]
                    
                    # 結果作成
                    res_rows = []
                    for i, idx in enumerate(top_idx, 1):
                        row = df_meta.iloc[idx]
                        res_rows.append({
                            "順位": i,
                            "スコア": round(float(scores[idx]), 4),
                            "題名": row.get("title") or row.get("研究課題名", ""),
                            "氏名": row.get("name") or row.get("研究者名", ""),
                            "所属": row.get("organization") or row.get("所属機関", ""),
                            "種目": row.get("section") or row.get("種目", ""),
                            "概要": row.get("abstract") or row.get("概要", "")
                        })
                    
                    res_df = pd.DataFrame(res_rows)
                    st.success(f"{len(res_df)} 件ヒット")
                    st.dataframe(res_df, use_container_width=True, height=500)
                    
                    # CSV
                    csv = res_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("CSV保存", csv, "kaken_results.csv", "text/csv")

            except Exception as e:
                st.error(f"検索エラー: {e}")

if __name__ == "__main__":
    main()
