import streamlit as st
import pandas as pd
import os
import re
import datetime
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

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
        """文字列から5桁または2桁の数字を抽出する"""
        text = str(text).strip()
        # まずは5桁を探す
        match_5 = re.search(r'(\d{5})', text)
        if match_5: return match_5.group(1)
        # なければ2桁を探す
        match_2 = re.search(r'(\d{2})', text)
        if match_2: return match_2.group(1)
        return ""

    def get_rate_from_map(rate_dict: Dict[str, float], kubun_name: str) -> float:
        """5桁で見つからなければ2桁に切り詰めて採択率を検索する"""
        code = extract_code(kubun_name)
        if not code: return 0.0
        
        # 1. そのままのコード(5桁 or 2桁)で検索
        if code in rate_dict:
            return rate_dict[code]
        
        # 2. 見つからない場合、コードが5桁なら頭2桁(中区分)で再試行
        if len(code) == 5:
            short_code = code[:2]
            if short_code in rate_dict:
                return rate_dict[short_code]
        
        return 0.0

    def safe_rel_kubun(base_dir: str, file_dir: str) -> str:
        try:
            rel = os.path.relpath(file_dir, base_dir)
            if rel == ".": return os.path.basename(file_dir)
            parts = rel.split(os.sep)
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

        # 全角・半角の違いを吸収
        pattern_str = selected_type.replace("(", "[(（]").replace(")", "[)）]")
        all_files = glob.glob(os.path.join(base_dir, "*.csv"))
        matched_files = [f for f in all_files if re.search(pattern_str, os.path.basename(f))]

        for file_path in matched_files:
            filename = os.path.basename(file_path)
            match = re.match(r"^(\d{2})_", filename)
            if match:
                year = int(match.group(1))
                try:
                    try:
                        df = pd.read_csv(file_path, header=None, encoding='utf-8')
                    except:
                        df = pd.read_csv(file_path, header=None, encoding='cp932')
                    
                    if year not in rate_map: rate_map[year] = {}
                    for _, row in df.iterrows():
                        row_vals = row.astype(str).values
                        if len(row_vals) < 2: continue
                        # 区分セル(0列目)からコードを抜き出して、レート(1列目)を保存
                        code = extract_code(row_vals[0])
                        try:
                            val = float(str(row_vals[1]).replace("%", "").replace("％", "").strip())
                            if code: rate_map[year][code] = val
                        except: continue
                except: pass
        return rate_map

    @st.cache_data(show_spinner=False)
    def load_file_minimal(file_path: str, file_mtime: float) -> pd.DataFrame:
        try:
            hdr = pd.read_csv(file_path, nrows=0)
            cols = list(hdr.columns)
            keyword_cols = [c for c in cols if c.startswith("keyword")]
            usecols = [c for c in cols if c in ("title", "awardnumber", "name", "abstract") or c in keyword_cols]
            df = pd.read_csv(file_path, usecols=usecols, dtype=str, na_filter=False, encoding='utf-8', on_bad_lines='skip')
            return df.fillna("")
        except: return pd.DataFrame()

    def vectorized_search(df: pd.DataFrame, keywords: List[str]) -> Dict[str, pd.Series]:
        n = len(df)
        if n == 0: return {"hit_mask": pd.Series([], dtype=bool)}
        title_series = df.get("title", pd.Series([""]*n)).astype(str).fillna("")
        keyword_cols = [c for c in df.columns if c.startswith("keyword")]
        combined_keywords = df[keyword_cols].astype(str).fillna("").agg(" ".join, axis=1) if keyword_cols else pd.Series([""] * n)
        title_mask, keyword_mask = pd.Series(True, index=df.index), pd.Series(True, index=df.index)
        for kw in keywords:
            if not kw: continue
            esc = re.escape(kw)
            title_mask &= title_series.str.contains(esc, case=False, na=False)
            keyword_mask &= combined_keywords.str.contains(esc, case=False, na=False)
        return {"title_mask": title_mask, "keyword_mask": keyword_mask, "hit_mask": title_mask | keyword_mask}

    def process_file_for_hits(file_path: str, kubun_name: str, keywords: List[str]) -> Dict[str, Any]:
        try: mtime = os.path.getmtime(file_path)
        except: mtime = 0
        df = load_file_minimal(file_path, mtime)
        if df.empty: return {"kubun": kubun_name, "hits": []}
        masks = vectorized_search(df, keywords)
        if not masks["hit_mask"].any(): return {"kubun": kubun_name, "hits": []}
        hits_df = df.loc[masks["hit_mask"]].copy()
        hits_df["is_title_hit"] = masks["title_mask"][masks["hit_mask"]]
        hits_df["is_keyword_hit"] = masks["keyword_mask"][masks["hit_mask"]]
        years = hits_df.get("awardnumber", pd.Series()).astype(str).str.extract(r'^\D*(\d{2})', expand=False).fillna(pd.NA)
        keyword_cols = [c for c in hits_df.columns if c.startswith("keyword")]
        k_comb = hits_df[keyword_cols].astype(str).agg(", ".join, axis=1) if keyword_cols else pd.Series([""]*len(hits_df))
        result = []
        for idx, row in hits_df.iterrows():
            y_val = years.loc[idx] if idx in years.index else None
            result.append({
                "課題番号": row.get("awardnumber", ""), "年度": int(y_val) if pd.notna(y_val) else None,
                "研究課題名": row.get("title", ""), "研究者名": row.get("name", ""),
                "キーワード": k_comb.loc[idx] if idx in k_comb.index else "", "概要": row.get("abstract", ""),
                "type_title": bool(row["is_title_hit"]), "type_keyword": bool(row["is_keyword_hit"]), "kubun_raw": kubun_name
            })
        return {"kubun": kubun_name, "hits": result}

    def build_aggregates(hit_details, rate_map):
        base_year = now_base_year_two_digit()
        thresholds = {"5y": base_year - 4, "3y": base_year - 2}
        latest_year = sorted(rate_map.keys())[-1] if rate_map else None
        all_years = [h.get("年度") for d in hit_details.values() for h in d.get("all_hits", []) if h.get("年度") is not None]
        span_label = f"全期間(過去{max(all_years)-min(all_years)+1}年)" if all_years else "全期間"
        agg_data = {"all": [], "5y": [], "3y": []}
        for kubun, details in hit_details.items():
            all_hits = details.get("all_hits", [])
            rate_val = 0.0
            if latest_year:
                # ここで「階層検索」を行う
                rate_val = get_rate_from_map(rate_map[latest_year], kubun)
            def count(hits, min_y=None):
                t, k, b = 0, 0, 0
                for h in hits:
                    if min_y and (h["年度"] is None or h["年度"] < min_y): continue
                    if h["type_title"] and h["type_keyword"]: b += 1
                    elif h["type_title"]: t += 1
                    elif h["type_keyword"]: k += 1
                return t, k, b
            for key, thresh in [("all", None), ("5y", thresholds["5y"]), ("3y", thresholds["3y"])]:
                t, k, b = count(all_hits, thresh)
                if (t+k+b) > 0:
                    agg_data[key].append({"区分": kubun, "タイトルのみ": t, "キーワードのみ": k, "両方ヒット": b, "最新採択率": rate_val})
        dfs = {k: pd.DataFrame(v).sort_values(["両方ヒット", "キーワードのみ", "タイトルのみ"], ascending=False).reset_index(drop=True) if v else pd.DataFrame() for k, v in agg_data.items()}
        for k in ["all", "5y", "3y"]:
            if not dfs[k].empty: dfs[k].insert(0, "順位", range(1, len(dfs[k])+1))
        dfs["latest_year"] = latest_year
        dfs["span_label"] = span_label
        return dfs

    # --- UI ---
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(APP_DIR, "keyword")
    if "data_root" not in st.session_state: st.session_state.data_root = os.path.join(DB_DIR, "kubun")
    if "rate_root" not in st.session_state: st.session_state.rate_root = os.path.join(DB_DIR, "haibun")
    st.markdown("<h1 style='font-size:28px;'>科研費 審査区分キーワード検索</h1>", unsafe_allow_html=True)

    research_types = ["基盤研究(A)", "基盤研究(B)", "基盤研究(C)", "若手研究", "挑戦的研究(開拓)", "挑戦的研究(萌芽)"]
    selected_type = st.selectbox("種目を選択してください", research_types)
    keywords_input = st.text_input("キーワードを入力してください (例: AI, 制御)", "")

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
                    st.session_state.hit_details, st.session_state.agg_results = hit_details, build_aggregates(hit_details, rate_map)
                    progress.empty()

    if st.session_state.get("agg_results"):
        aggs = st.session_state.agg_results
        rate_year = aggs.get("latest_year")
        label_all = aggs.get("span_label", "全期間")
        def fmt_df(df):
            if df.empty: return df
            d = df.copy()
            col = f"採択率({rate_year}年度)" if rate_year else "採択率(最新)"
            d.rename(columns={"最新採択率": col}, inplace=True)
            if col in d.columns: d[col] = d[col].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
            return d
        t1, t2, t3 = st.tabs([f"{label_all}", "過去5年", "過去3年"])
        with t1: st.dataframe(fmt_df(aggs['all']), use_container_width=True, hide_index=True)
        with t2: st.dataframe(fmt_df(aggs['5y']), use_container_width=True, hide_index=True)
        with t3: st.dataframe(fmt_df(aggs['3y']), use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("詳細確認")
        kubun_opts = list(aggs["all"]["区分"].unique()) if not aggs["all"].empty else []
        sel_k = st.selectbox("詳細を表示する区分", kubun_opts)
        if sel_k and sel_k in st.session_state.hit_details:
            hits = st.session_state.hit_details[sel_k]["all_hits"]
            bh, kh, th = [h for h in hits if h["type_title"] and h["type_keyword"]], [h for h in hits if not h["type_title"] and h["type_keyword"]], [h for h in hits if h["type_title"] and not h["type_keyword"]]
            dt1, dt2, dt3 = st.tabs([f"両方 ({len(bh)})", f"キーワードのみ ({len(kh)})", f"タイトルのみ ({len(th)})"])
            def show_d(data):
                if not data: st.info("なし"); return
                df = pd.DataFrame(data)
                rm = st.session_state.rate_map
                # 詳細表示でも「階層検索」を適用
                def get_row_rate(r):
                    y, k_raw = r['年度'], r['kubun_raw']
                    if y in rm: return f"{get_rate_from_map(rm[y], k_raw):.1f}%"
                    return "-"
                df["その年の採択率"] = df.apply(get_row_rate, axis=1)
                st.dataframe(df[["課題番号", "その年の採択率", "研究課題名", "研究者名", "キーワード", "概要"]].sort_values("課題番号", ascending=False), use_container_width=True, hide_index=True)
            with dt1: show_d(bh)
            with dt2: show_d(kh)
            with dt3: show_d(th)

if __name__ == "__main__":
    main()
