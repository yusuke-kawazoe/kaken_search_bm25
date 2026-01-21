# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import gc
import pickle

# --- 1. ライブラリインポートの安全策 ---
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    st.error("ライブラリ 'rank_bm25' が不足しています。requirements.txt に追加してください。")
    st.stop()

def main():
    st.set_page_config(page_title="科研費 BM25キーワード検索", layout="wide")

    # --- 設定 (GitHubリポジトリ情報) ---
    OWNER = "yusuke-kawazoe"
    REPO = "kaken_search_bm25"
    TAG = "v1.0"
    MODEL_FILE_NAME = "bm25_model.pkl"
    META_FILE_NAME = "metadata.parquet"

    # --- トークナイズ関数 (Tkinter版と統一) ---
    def tokenize_ngram(text, n=2):
        if not isinstance(text, str): return []
        if len(text) < n: return [text]
        return [text[i:i+n] for i in range(len(text)-n+1)]

    # --- データロード関数 (メモリ最適化版) ---
    @st.cache_resource(show_spinner="検索モデルとデータを読み込み中...")
    def load_data():
        try:
            base_url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/"
            
            # 1. BM25モデルのダウンロードと読み込み
            res_m = requests.get(base_url + MODEL_FILE_NAME, timeout=30)
            res_m.raise_for_status()
            bm25 = pickle.loads(res_m.content)
            
            # 2. メタデータのダウンロードと読み込み
            # メモリ削減のため、columnsを指定して必要な列だけを読み込む
            res_d = requests.get(base_url + META_FILE_NAME, timeout=30)
            res_d.raise_for_status()
            
            # 読み込む列を限定 (npz時代のように軽くするため)
            target_cols = [
                "title", "name", "awardnumber", "section", "review_section", "abstract",
                "研究課題名", "研究者名", "課題番号", "種目", "区分", "概要", "所属機関"
            ]
            
            # parquet読み込み (pyarrowが必須)
            full_df = pd.read_parquet(io.BytesIO(res_d.content))
            
            # 存在する列だけを抽出して軽量化
            available_cols = [c for c in target_cols if c in full_df.columns]
            df = full_df[available_cols].copy()
            
            del full_df
            gc.collect()
            
            return bm25, df
        except Exception as e:
            st.error(f"データのロード中にエラーが発生しました: {e}")
            return None, None

    # データのロード実行
    bm25_model, df_meta = load_data()
    if bm25_model is None:
        st.warning("データを取得できませんでした。GitHubの設定や通信環境を確認してください。")
        st.stop()

    # --- UI表示 ---
    st.markdown("<h1 style='font-size:24px;'>科研費 BM25キーワード検索</h1>", unsafe_allow_html=True)

    query_text = st.text_area("検索キーワード、または研究概要を入力してください", height=150)
    
    col1, col2, _ = st.columns([1, 1, 8])
    search_clicked = col1.button("検索実行", type="primary", use_container_width=True)
    if col2.button("クリア", use_container_width=True):
        st.rerun()

    if search_clicked:
        if not query_text.strip():
            st.warning("テキストを入力してください。")
        else:
            with st.spinner("計算中..."):
                # 1. クエリのトークナイズ (2-gram)
                tokenized_query = tokenize_ngram(query_text, n=2)

                # 2. スコア計算
                try:
                    scores = bm25_model.get_scores(tokenized_query)
                    
                    # 3. 上位100件の取得
                    top_n = 100
                    valid_idx = np.where(scores > 0)[0]
                    
                    if len(valid_idx) == 0:
                        st.info("一致する課題が見つかりませんでした。")
                    else:
                        # スコアが高い順にソート
                        sorted_valid_idx = valid_idx[np.argsort(-scores[valid_idx])[:top_n]]
                        
                        # 4. 結果表示用のデータフレーム作成
                        res_df = df_meta.iloc[sorted_valid_idx].copy()
                        
                        # カラム名のマッピング処理 (日本語/英語両対応)
                        rename_map = {
                            "研究課題名": "題名", "title": "題名",
                            "研究者名": "氏名", "name": "氏名",
                            "課題番号": "課題番号", "awardnumber": "課題番号",
                            "種目": "種目", "section": "種目",
                            "区分": "区分", "review_section": "区分",
                            "概要": "概要", "abstract": "概要",
                            "所属機関": "所属"
                        }
                        res_df = res_df.rename(columns=rename_map)
                        
                        # スコアと順位を追加
                        res_df.insert(0, "スコア", np.round(scores[sorted_valid_idx], 4))
                        res_df.insert(0, "順位", range(1, len(res_df) + 1))

                        # 表示する列の絞り込み
                        display_order = ["順位", "スコア", "題名", "氏名", "所属", "課題番号", "種目", "区分", "概要"]
                        existing_display_cols = [c for c in display_order if c in res_df.columns]
                        res_df = res_df[existing_display_cols]

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
                        st.download_button("検索結果をCSVで保存", csv, "kaken_results.csv", "text/csv")
                        
                        # メモリ解放
                        del res_df
                        gc.collect()

                except Exception as e:
                    st.error(f"検索処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
