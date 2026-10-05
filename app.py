import streamlit as st
import pandas as pd
import json
import os
import time

st.set_page_config(page_title="Camera AI Dashboard", layout="wide")

st.title("🎥 Camera AI - リアルタイム検知ダッシュボード")

# ログファイルのパス
LOG_FILE = "detection_log.json"

st.sidebar.header("設定・ステータス")
auto_refresh = st.sidebar.checkbox("自動更新 (1秒)", value=True)

# ログ読み込み処理
if os.path.exists(LOG_FILE):
    data_list = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data_list.append(json.loads(line.strip()))
            except:
                pass
    
    if data_list:
        df = pd.DataFrame(data_list)
        
        # 直近データの表示
        latest = df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("最新の検出オブジェクト数", int(latest.get("count", 0)))
        col2.metric("推論時間", f"{latest.get('inference_ms', 0):.1f} ms")
        col3.metric("総ログ件数", len(df))
        
        st.subheader("📊 直近の検知履歴")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("まだ検知データがありません。")
else:
    st.info("ログファイルがまだ生成されていません。server.pyを起動してデータを送信してください。")

if auto_refresh:
    time.sleep(1)
    st.rerun()
#streamlit run app.py