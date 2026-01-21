import streamlit as st
#import App_abstract_vector
import App_keyword_vector2

def main():
    # --- 1. ページ設定 (Streamlitで一番最初に実行する必要がある) ---
    st.set_page_config(page_title="科研費 検索ツール", layout="wide")

    # --- 2. 簡易パスワード機能 ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("***研究者専用アクセス***")
        pw = st.text_input("キーワードを入力してください", type="password")
        if pw == "YU-kaken-2025": 
            st.session_state.authenticated = True
            st.rerun()
        elif pw:
            st.error("キーワードが正しくありません。")
        st.stop()  # 合言葉が一致するまで、ここで処理を止める

    # --- 3. 認証後のメインロジック ---
    
    # ページの状態管理
    if "page" not in st.session_state:
        st.session_state.page = "keyword"

    # サイドバーの作成
    st.sidebar.title("メニュー")
    
    if st.sidebar.button("審査区分キーワード検索", use_container_width=True):
        st.session_state.page = "keyword"
        st.rerun()

    if st.sidebar.button("文章ベクトル検索", use_container_width=True):
        st.session_state.page = "abstract"
        st.rerun()

    # ------ ページ切り替え表示 ------
    if st.session_state.page == "keyword":
        App_keyword_vector2.main()

    elif st.session_state.page == "abstract":
        App_abstract_vector.main()

# --- 4. この命令で、定義した main() を実際に動かします ---
if __name__ == "__main__":
    main()
