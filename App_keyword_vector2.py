import streamlit as st
import pandas as pd
import os
import re
import datetime
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

#1からの変更点
#ヒット件数をタイトルと概要に分けずに両方ヒットのみに限定した
#その他いただいたコメントを基に微修正

def main():
    # -------------------------
    # ユーティリティ関数
    # -------------------------
    def split_keywords(s: str) -> List[str]:
        if not s: return []
        parts = re.split(r"[\u3001\uFF0C\u3000,]+", s)
        return [p.strip() for p in parts if p and p.strip()]

    def now_base_year_two_digit() -> int:
        return datetime.datetime.now().year % 100

    def extract_code(text: str) -> str:
        """
        文字列からコードを抽出し、正規化する。
        - '1' や '01' -> '01' (2桁に統一)
        - '01010' -> '01010' (5桁はそのまま)
        - '中区分１' -> '01'
        """
        if not text: return ""
        # 全角数字を半角にし、前後の空白を除去
        text = str(text).translate(str.maketrans('０１２３４５６７８９', '0123456789')).strip()
        
        # 1. まずは5桁の連続した数字を探す
        match_5 = re.search(r'(\d{5})', text)
        if match_5: return match_5.group(1)
        
        # 2. 次に1〜2桁の数字を探す
        # 文中のどこにあっても「数字の塊」として取り出し、0埋め2桁にする
        match_num = re.search(r'(\d{1,2})', text)
        if match_num:
            return match_num.group(1).zfill(2) # "1" -> "01", "01" -> "01"
            
        return ""

    def get_rate_from_map(rate_dict: Dict[str, float], kubun_name: str) -> float:
        """
        正規化されたコードを使って採択率を引くロジック
        """
        code = extract_code(kubun_name) # ここで "1" も "01" も "01" になる
        if not code: return 0.0
        
        # 直接マッチ（01 == 01 など）
        if code in rate_dict: return rate_dict[code]
        
        # 5桁コード（01010）の場合、頭2桁（01）で再試行
        if len(code) == 5:
            short_code = code[:2]
            if short_code in rate_dict: return rate_dict[short_code]
            
        return 0.0

    def safe_rel_kubun(base_dir: str, file_dir: str) -> str:
        try:
            rel = os.path.relpath(file_dir, base_dir)
            if rel == ".": return os.path.basename(file_dir)
            parts = rel.split(os.sep)
            # フォルダ名（例: "1" や "01010"）をそのまま返す
            return parts[0] if parts else os.path.basename(file_dir)
        except ValueError:
            return os.path.basename(file_dir)

    # -------------------------
    # データ処理関数
    # -------------------------
    @st.cache_data(show_spinner=False)
    def load_dynamic_rates(base_dir: str, selected_type: str) -> Dict[int, Dict[str, float]]:
        rate_map = {}
        if not os.path.exists(base_dir): return {}
        pattern_str = selected_type.replace("(", "[(（]").replace(")", "[)）]")
        all_files = glob.glob(os.path.join(base_dir, "*.csv"))
        matched_files = [f for f in all_files if re.search(pattern_str, os.path.basename(f))]

        for file_path in matched_files:
            filename = os.path.basename(file_path)
            match_y = re.match(r"^(\d{2})_", filename)
            if match_y:
                year = int(match_y.group(1))
                try:
                    try: df = pd.read_csv(file_path, header=None, encoding='utf-8')
                    except: df = pd.read_csv(file_path, header=None, encoding='cp932')
                    
                    if year not in rate_map: rate_map[year] = {}
                    for _, row in df.iterrows():
                        row_vals = row.astype(str).values
                        if len(row_vals) < 2: continue
                        # 採択率CSV側の区分名（例: "01:思想..."）から "01" を抽出
                        code = extract_code(row_vals[0])
                        try:
                            val_str = str(row_vals[1]).replace("%", "").replace("％", "").strip()
                            val = float(val_str)
                            if code: rate_map[year][code] = val
                        except: continue
                except: pass
        return rate_map

    @st.cache_data(show_spinner=False)
    def load_file_minimal(file_path: str, file_mtime: float) -> pd.DataFrame:
        try:
            hdr = pd.read_csv(file_path, nrows=0)
            cols = list(hdr.columns)
            kw_cols = [c for c in cols if c.startswith("keyword")]
            usecols = [c for c in cols if c in ("title", "awardnumber", "name", "abstract") or c in kw_cols]
            return pd.read_csv(file_path, usecols=usecols, dtype=str, na_filter=False, encoding='utf-8', on_bad_lines='skip')
        except: return pd.DataFrame()

    def vectorized_search(df: pd.DataFrame, keywords: List[str]) -> Dict[str, pd.Series]:
        n = len(df)
        if n == 0: return {"hit_mask": pd.Series([], dtype=bool)}
        t_series = df.get("title", pd.Series([""]*n)).astype(str).fillna("")
        kw_cols = [c for c in df.columns if c.startswith("keyword")]
        combined_kw = df[kw_cols].astype(str).fillna("").agg(" ".join, axis=1) if kw_cols else pd.Series([""] * n)
        
        t_mask, k_mask = pd.Series(True, index=df.index), pd.Series(True, index=df.index)
        for kw in keywords:
            if not kw: continue
            esc = re.escape(kw)
            t_mask &= t_series.str.contains(esc, case=False, na=False)
            k_mask &= combined_kw.str.contains(esc, case=False, na=False)
        
        return {"hit_mask": t_mask & k_mask}

    def process_file_for_hits(file_path: str, kubun_name: str, keywords: List[str]) -> Dict[str, Any]:
        try: mtime = os.path.getmtime(file_path)
        except: mtime = 0
        df = load_file_minimal(file_path, mtime)
        if df.empty: return {"kubun": kubun_name, "hits": []}
        
        masks = vectorized_search(df, keywords)
        if not masks["hit_mask"].any(): return {"kubun": kubun_name, "hits": []}
        
        hits_df = df.loc[masks["hit_mask"]].copy()
        years = hits_df.get("awardnumber", pd.Series()).astype(str).str.extract(r'^\D*(\d{2})', expand=False).fillna(pd.NA)
        kw_cols = [c for c in hits_df.columns if c.startswith("keyword")]
        k_comb = hits_df[kw_cols].astype(str).agg(", ".join, axis=1) if kw_cols else pd.Series([""]*len(hits_df))
        
        result = []
        for idx, row in hits_df.iterrows():
            y_val = years.loc[idx] if idx in years.index else None
            result.append({
                "課題番号": row.get("awardnumber", ""), "年度": int(y_val) if pd.notna(y_val) else None,
                "研究課題名": row.get("title", ""), "研究者名": row.get("name", ""),
                "キーワード": k_comb.loc[idx] if idx in k_comb.index else "", "概要": row.get("abstract", ""),
                "kubun_raw": kubun_name
            })
        return {"kubun": kubun_name, "hits": result}

    def build_aggregates(hit_details, rate_map):
        base_year = now_base_year_two_digit()
        thresholds = {"5y": base_year - 4, "3y": base_year - 2}
        latest_year = sorted(rate_map.keys())[-1] if rate_map else None
        
        agg_data = {"all": [], "5y": [], "3y": []}
        for kubun, details in hit_details.items():
            all_hits = details.get("all_hits", [])
            rate_val = 0.0
            if latest_year:
                # ここで kubun="1" が渡されても内部で "01" に変換して照合する
                rate_val = get_rate_from_map(rate_map[latest_year], kubun)
            
            def count_hits(hits, min_y=None):
                return sum(1 for h in hits if min_y is None or (h["年度"] is not None and h["年度"] >= min_y))

            for key, thresh in [("all", None), ("5y", thresholds["5y"]), ("3y", thresholds["3y"])]:
                num = count_hits(all_hits, thresh)
                if num > 0:
                    agg_data[key].append({"区分": kubun, "ヒット数": num, "最新採択率": rate_val})
        
        dfs = {}
        for k in ["all", "5y", "3y"]:
            if agg_data[k]:
                df = pd.DataFrame(agg_data[k]).sort_values("ヒット数", ascending=False).reset_index(drop=True)
                df.insert(0, "順位", range(1, len(df)+1))
                dfs[k] = df
            else:
                dfs[k] = pd.DataFrame()
        dfs["latest_year"] = latest_year
        return dfs

    # --- UI ---
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(APP_DIR, "keyword")
    if "data_root" not in st.session_state: st.session_state.data_root = os.path.join(DB_DIR, "kubun")
    if "rate_root" not in st.session_state: st.session_state.rate_root = os.path.join(DB_DIR, "haibun")
    
    st.markdown("<h1 style='font-size:28px;'>科研費 審査区分キーワード検索</h1>", unsafe_allow_html=True)

    research_types = ["基盤研究(A)", "基盤研究(B)", "基盤研究(C)", "若手研究", "挑戦的研究(開拓)", "挑戦的研究(萌芽)"]
    selected_type = st.selectbox("種目を選択してください", research_types)
    keywords_input = st.text_input("キーワードを入力してください (複数入力はカンマ区切り)", "")

    if st.button("検索実行"):
        if not keywords_input.strip(): st.warning("キーワードを入力してください。")
        else:
            data_dir = os.path.join(st.session_state.data_root, selected_type)
            if not os.path.exists(data_dir): st.error(f"フォルダが見つかりません: {selected_type}")
            else:
                keywords = split_keywords(keywords_input)
                rate_map = load_dynamic_rates(st.session_state.rate_root, selected_type)
                st.session_state.rate_map = rate_map
                
                csv_files = []
                for root, _, files in os.walk(data_dir):
                    for f in files:
                        if f.lower().endswith(".csv") and not f.endswith("_keyword_ranking.csv"):
                            csv_files.append((os.path.join(root, f), root))
                
                if csv_files:
                    results = []
                    progress = st.progress(0)
                    with ThreadPoolExecutor(max_workers=8) as ex:
                        futures = [ex.submit(process_file_for_hits, fpath, safe_rel_kubun(data_dir, froot), keywords) for fpath, froot in csv_files]
                        for i, fut in enumerate(as_completed(futures)):
                            results.append(fut.result())
                            progress.progress((i + 1) / len(csv_files))
                    
                    hit_details = {res["kubun"]: {"all_hits": []} for res in results if res["hits"]}
                    for res in results:
                        if res["hits"]: hit_details[res["kubun"]]["all_hits"].extend(res["hits"])
                    
                    st.session_state.hit_details = hit_details
                    st.session_state.agg_results = build_aggregates(hit_details, rate_map)
                    progress.empty()

    if st.session_state.get("agg_results"):
        aggs = st.session_state.agg_results
        rate_year = aggs.get("latest_year")
        
        def fmt_df(df):
            if df.empty: return df
            d = df.copy()
            col_rate = f"審査区分の採択率({rate_year}年度)" if rate_year else "採択率(最新)"
            d.rename(columns={"最新採択率": col_rate}, inplace=True)
            if col_rate in d.columns:
                d[col_rate] = d[col_rate].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
            return d

        t1, t2, t3 = st.tabs(["全期間", "過去5年", "過去3年"])
        with t1: st.dataframe(fmt_df(aggs['all']), use_container_width=True, hide_index=True)
        with t2: st.dataframe(fmt_df(aggs['5y']), use_container_width=True, hide_index=True)
        with t3: st.dataframe(fmt_df(aggs['3y']), use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("詳細確認 (ヒットした課題)")
        kubun_opts = list(aggs["all"]["区分"].unique()) if not aggs["all"].empty else []
        sel_k = st.selectbox("区分を選択してください", kubun_opts)
        
        if sel_k and sel_k in st.session_state.hit_details:
            hits = st.session_state.hit_details[sel_k]["all_hits"]
            df_detail = pd.DataFrame(hits)
            rm = st.session_state.rate_map
            def get_row_rate(r):
                y, k_raw = r['年度'], r['kubun_raw']
                if y in rm: return f"{get_rate_from_map(rm[y], k_raw):.1f}%"
                return "-"
            df_detail["その年の採択率"] = df_detail.apply(get_row_rate, axis=1)
            st.dataframe(df_detail[["課題番号", "年度", "その年の採択率", "研究課題名", "研究者名", "キーワード", "概要"]].sort_values("年度", ascending=False), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
