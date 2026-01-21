# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import duckdb
import requests
import io
import gc

def main():
    st.set_page_config(page_title="科研費 高速検索 (DuckDB版)", layout="wide")

    OWNER = "yusuke-kawazoe"
    REPO = "kaken_search_bm25"
    TAG = "v1.0"
    META_FILE_NAME = "metadata.parquet"

    @st.cache_resource(show_spinner="データベースを準備中...")
    def setup_db():
        try:
            # 1. メタデータ（Parquet）のみをダウンロード
            # 500MBのPickleは無視し、Parquet（圧縮されている）だけを使う
            url = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{META_FILE_NAME}"
            res = requests.get(url, stream=True)
            res.raise_for_status()
            
            # DuckDBはParquetを直接クエリできるため、メモリ消費が極めて少ない
            con = duckdb.connect(database=':memory:')
            
            # メモリ内にParquetデータを登録
            # (io.BytesIOをそのままDuckDBで読む)
            parquet_data = io.BytesIO(res.content)
            con.register('kaken_data', pd.read_parquet(parquet_data))
            
            return con
        except Exception as e:
            st.error(f"準備失敗: {e}")
            return None

    con = setup_db()
    if con is None: st.stop()

    st.title("科研費 高速キーワード検索")
    st.info("500MBのモデルを読み込まず、DuckDBにより高速・低メモリで検索します。")

    query = st.text_input("検索キーワードを入力（例: AI 医療）")

    if query:
        with st.spinner("検索中..."):
            # スペース区切りで複数単語に対応
            words = query.split()
            # SQL文の構築 (すべての単語を含むものを検索)
            conditions = " AND ".join([f"(abstract LIKE '%{w}%' OR title LIKE '%{w}%')" for w in words])
            
            sql = f"""
                SELECT 
                    title AS 題名, 
                    name AS 氏名, 
                    organization AS 所属,
                    abstract AS 概要
                FROM kaken_data
                WHERE {conditions}
                LIMIT 100
            """
            # カラム名は実際のParquetに合わせて修正してください
            # 例: abstract -> 概要, title -> 研究課題名
            
            try:
                res_df = con.execute(sql).df()
                
                if len(res_df) == 0:
                    st.warning("一致する課題は見つかりませんでした。")
                else:
                    st.success(f"{len(res_df)} 件ヒットしました。")
                    st.dataframe(res_df, use_container_width=True, height=600)
            except Exception as e:
                st.error(f"検索エラー: {e}\n(Parquet内のカラム名を確認してください)")

if __name__ == "__main__":
    main()
