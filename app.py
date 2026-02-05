import streamlit as st
import pandas as pd
import utils
import os
import yfinance as yf

# Fix for "disk I/O error" / "unable to open database file"
# Redirect yfinance cache to a local folder in the workspace
cache_dir = os.path.join(os.getcwd(), "yf_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
yf.set_tz_cache_location(cache_dir)
from consts import SET100_TICKERS, LONG_TERM_GROWTH, RISK_FREE_RATE, MARKET_RETURN
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# Set Page Configuration
st.set_page_config(
    page_title="โปรแกรมคัดกรองหุ้น VI (Thai Value Investor)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Tickers
SET100_TICKERS = utils.load_tickers()

# --- SIDEBAR: VALUATION MODEL ---
st.sidebar.title("🇹🇭 Thai Value Investor")
st.sidebar.markdown("### 🎛️ โมเดลประเมินมูลค่า")
with st.sidebar.expander("ตั้งค่าสมมติฐาน (Assumption)", expanded=False):
    st_rf = st.number_input("อัตราผลตอบแทนพันธบัตร (Risk Free %)", value=RISK_FREE_RATE*100, step=0.1, format="%.2f") / 100
    st_rm = st.number_input("ผลตอบแทนตลาด (Market Return %)", value=MARKET_RETURN*100, step=0.1, format="%.2f") / 100
    st_g = st.number_input("การเติบโตระยะยาว (Terminal Growth %)", value=LONG_TERM_GROWTH*100, step=0.1, format="%.2f") / 100
    
    if st.button("รีเซ็ตค่าเริ่มต้น"):
        st.cache_data.clear() # Optional but good
        st.rerun()

st.sidebar.markdown("### 🔄 อัปเดตข้อมูล")
if st.sidebar.button("อัปเดตข้อมูลราคาและงบการเงิน"):
    with st.spinner("กำลังล้าง Cache และดึงข้อมูลใหม่..."):
        # Clear Streamlit Cache
        st.cache_data.clear()
        
        # Clear yfinance Cache (optional, but ensures fresh data from API)
        # Note: We already redirected cache to local folder, so we can clean it if needed
        # but st.cache_data.clear() is usually enough for the app logic.
        # If we want to force yfinance to re-download, we might need to rely on its internal expiration or clear the folder.
        # For now, clearing app cache is sufficient to trigger fetch_raw_market_data() again.
        
    st.success("อัปเดตข้อมูลเรียบร้อยแล้ว!")
    st.rerun()

# --- DATA FETCHING (Separated) ---
@st.cache_data(ttl=3600)
def fetch_raw_market_data():
    """
    Fetches raw data for all tickers. Cached for performance.
    """
    results = []
    
    # Progress bar setup
    progress_text = "กำลังดึงข้อมูลหุ้น... โปรดรอสักครู่"
    my_bar = st.progress(0, text=progress_text)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a dictionary to map futures to tickers
        future_to_ticker = {executor.submit(utils.get_stock_data, ticker): ticker for ticker in SET100_TICKERS}
        
        completed_count = 0
        total_count = len(SET100_TICKERS)
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            if data:
                results.append(data)
            
            completed_count += 1
            if total_count > 0:
                my_bar.progress(completed_count / total_count, text=f"กำลังโหลด {future_to_ticker[future]} ({completed_count}/{total_count})")
            
    my_bar.empty()
    return results

def process_valuations(raw_data, rf, rm, g):
    """
    Calculates valuation on raw data with specific parameters.
    """
    results = []
    for item in raw_data:
        # Clone item to avoid modifying cached dict in place across reruns (shallow copy often enough but dict copy is safer)
        data_copy = item.copy()
        evaluated_data = utils.calculate_valuations(data_copy, risk_free_rate=rf, market_return=rm, long_term_growth=g)
        if evaluated_data:
            results.append(evaluated_data)
    return pd.DataFrame(results)

# Load Pipeline
raw_data_list = fetch_raw_market_data()
if not raw_data_list:
    st.error("Failed to fetch data.")
    st.stop()

df = process_valuations(raw_data_list, st_rf, st_rm, st_g)

if not df.empty:
    # --- GLOBAL DATA ENRICHMENT ---
    # Handle NaNs for scoring
    df['debtToEquity'] = df['debtToEquity'].fillna(999) 
    df['returnOnEquity'] = df['returnOnEquity'].fillna(0)
    df['profitMargins'] = df['profitMargins'].fillna(0)
    df['margin_of_safety'] = df['margin_of_safety'].fillna(-100)
    df['marketCap'] = df['marketCap'].fillna(0)
    df['revenueGrowth'] = df['revenueGrowth'].fillna(0)
    
    # NOTE: yfinance 'debtToEquity' is usually returned as a percentage (e.g., 150 means 1.5x).
    # We need to divide by 100 for display if we want 'x', but for scoring logic check raw value.
    # Let's fix the dataframe column for display purposes to be 'x' (ratio).
    df['debtToEquityRatio'] = df['debtToEquity'] / 100

    df['score_debt'] = df['debtToEquity'].apply(lambda x: 1 if x < 200 else 0) # < 200% = < 2.0x
    df['score_roe'] = df['returnOnEquity'].apply(lambda x: 1 if x > 0.15 else 0)
    df['score_npm'] = df['profitMargins'].apply(lambda x: 1 if x > 0.10 else 0)
    df['score_mos'] = df['margin_of_safety'].apply(lambda x: 1 if x > 0 else 0)
    df['score_size'] = df['marketCap'].apply(lambda x: 1 if x > 50_000_000_000 else 0) # > 50B THB
    df['score_growth'] = df['revenueGrowth'].apply(lambda x: 1 if x > 0.05 else 0) # > 5% Growth
    
    df['Quality Score'] = (df['score_debt'] + df['score_roe'] + df['score_npm'] + 
                           df['score_mos'] + df['score_size'] + df['score_growth'])


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio("ไปยังหน้า", [
    "แดชบอร์ดภาพรวม", 
    "วิเคราะห์หุ้นรายตัว", 
    "เปรียบเทียบคู่แข่ง", 
    "แนะนำพอร์ตการลงทุน", 
    "พอร์ตของฉัน (My Portfolio)", 
    "จำลองการออมหุ้น (DCA Backtester)",
    "ตั้งค่า"
])

if page == "แดชบอร์ดภาพรวม":
    st.title("📊 โปรแกรมคัดกรองหุ้นคุณค่า (VI)")
    st.markdown("พัฒนาตามหลักการลงทุนของ คุณกวี ชูกิจเกษม")
    
    # Dashboard uses 'df' loaded globally
    
    if not df.empty:
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        undervalued_count = df[df['status'] == 'Undervalued'].shape[0]
        avg_mos = df['margin_of_safety'].mean()
        
        col1.metric("หุ้นที่วิเคราะห์", f"{len(df)}")
        col2.metric("หุ้นราคาถูกกว่ามูลค่า", f"{undervalued_count}")
        col3.metric("ส่วนเผื่อความปลอดภัยเฉลี่ย (MOS)", f"{avg_mos:.2f}%")
        
        # --- QUALITY SCORING (Enhanced Auto 6 Points) ---
        # 1. Low Debt (D/E < 200%)
        # 2. Strong ROE (> 15%)
        # 3. High NPM (> 10%)
        # 4. Undervalued (MOS > 0)
        # 5. Market Leader Proxy (Market Cap > 50 Billion THB)
        # 6. Growth Proxy (Revenue Growth > 0%)
        
        # Sidebar Filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 ตัวกรองหุ้น (Screener)")
        st.sidebar.info("ℹ️ **ระบบคะแนนแบบใหม่:** สูงสุด **6 คะแนน** (รวมขนาดกิจการ > 5 หมื่นล้าน และ การเติบโต > 0%)")
        min_score = st.sidebar.slider("คะแนนคุณภาพขั้นต่ำ (เต็ม 6)", 0, 6, 4, help="กรองจาก: หนี้ต่ำ, ROE>15%, NPM>10%, ราคาถูก, หุ้นใหญ่, รายได้โต")
        
        filtered_df = df[df['Quality Score'] >= min_score].copy()

        # --- ADVANCED SCANNING (Magic Formula & F-Score) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚀 วิเคราะห์เชิงลึก")
        
        # Initialize session state for advanced results if not exists
        if 'advanced_results' not in st.session_state:
            st.session_state['advanced_results'] = {}

        if st.sidebar.button("วิเคราะห์ Magic Formula & F-Score"):
            st.toast("กำลังเริ่มวิเคราะห์เชิงลึก... อาจใช้เวลาสักครู่", icon="⏳")
            
            # Filter stocks to analyze (only from the filtered list to save time)
            targets = filtered_df['symbol'].tolist()
            
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            results_adv = []
            
            # Use ThreadPool but limit workers to avoid rate limit/database lock
            # Since we are fetching deep financials, 5 workers is safe enough with our cache patch
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(utils.calculate_magic_formula_and_f_score, ticker): ticker for ticker in targets}
                
                completed = 0
                total = len(targets)
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    res = future.result()
                    if res:
                        results_adv.append(res)
                    
                    completed += 1
                    progress = completed / total
                    progress_bar.progress(progress)
                    status_text.text(f"วิเคราะห์ {completed}/{total}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Save to session state
            st.session_state['advanced_results'] = {r['symbol']: r for r in results_adv}
            st.success(f"วิเคราะห์เสร็จสิ้น! พบข้อมูล {len(results_adv)} หุ้น")
            st.rerun()

        # Merge Advanced Results if available
        if st.session_state['advanced_results']:
            # Create DataFrame from session state
            adv_df = pd.DataFrame(st.session_state['advanced_results'].values())
            
            # Merge with filtered_df
            if not adv_df.empty:
                # Use left merge to keep filtered_df rows
                filtered_df = filtered_df.merge(adv_df, on='symbol', how='left')
                
                # Fill NaNs for display
                filtered_df['magic_roc'] = filtered_df['magic_roc'].fillna(0)
                filtered_df['magic_ey'] = filtered_df['magic_ey'].fillna(0)
                filtered_df['f_score'] = filtered_df['f_score'].fillna(-1) # -1 means N/A

        
        # --- TOP 10 SUPER STOCKS (Integrated) ---
        st.markdown("---")
        st.subheader("🏆 10 สุดยอดหุ้นแกร่ง (The Super Stocks)")
        st.markdown(f"""
        คัดเลือกจาก **ราคาถูก (MOS > 0)**, **คุณภาพดี (ROE > 10%)**, **ปันผลใช้ได้ (> 3%)**, และ **คะแนน Quant >= {min_score}**
        ผสานพลังกับ **Magic Formula** และ **F-Score** (ถ้ามีข้อมูล) เพื่อหาที่สุดของหุ้นคุณค่า
        """)
        
        # Calculate yield first (for filtering)
        df['dividendYield_calc'] = df['dividendRate'] / df['price']
        
        # 1. Base Filter
        super_candidates = df[
            (df['status'] == 'Undervalued') & 
            (df['returnOnEquity'] > 0.10) &
            (df['dividendYield_calc'] > 0.03) & # Relaxed dividend slightly to allow growth stocks
            (df['Quality Score'] >= min_score)
        ].copy()
        
        # 2. Advanced Scoring (if available)
        if 'magic_roc' in filtered_df.columns:
            # Join advanced data to candidates if not already joined
            # Note: We added 'graham_num', 'fcf_yield', 'z_score', 'sgr' to utils.py
            
            adv_cols = ['symbol', 'magic_roc', 'magic_ey', 'f_score', 'graham_num', 'fcf_yield', 'z_score', 'sgr']
            # Check if columns exist in filtered_df (in case user hasn't re-run analysis yet)
            adv_cols = [c for c in adv_cols if c in filtered_df.columns or c == 'symbol']
            
            if 'magic_roc' not in super_candidates.columns:
                 super_candidates = super_candidates.merge(filtered_df[adv_cols], on='symbol', how='left')
            
            # Fill N/A for those without deep scan
            super_candidates['magic_roc'] = super_candidates['magic_roc'].fillna(0)
            super_candidates['magic_ey'] = super_candidates['magic_ey'].fillna(0)
            super_candidates['f_score'] = super_candidates['f_score'].fillna(0)
            super_candidates['graham_num'] = super_candidates['graham_num'].fillna(0)
            super_candidates['fcf_yield'] = super_candidates['fcf_yield'].fillna(0)
            super_candidates['z_score'] = super_candidates['z_score'].fillna(0)
            super_candidates['sgr'] = super_candidates['sgr'].fillna(0)

            # Calculate Composite Score (Max 100)
            # MOS (30%) + Dividend (15%) + ROE (15%) + F-Score (20%) + Magic Rank (20%)
            
            # Rank Magic (Lower is better) -> Invert for scoring
            super_candidates['rank_roc'] = super_candidates['magic_roc'].rank(ascending=False)
            super_candidates['rank_ey'] = super_candidates['magic_ey'].rank(ascending=False)
            super_candidates['magic_rank_score'] = 100 - (super_candidates['rank_roc'] + super_candidates['rank_ey']) # Rough inversion
            
            # Normalize scores to 0-1 range for weighting
            def normalize(series):
                return (series - series.min()) / (series.max() - series.min()) if (series.max() - series.min()) > 0 else 0

            norm_mos = normalize(super_candidates['margin_of_safety'])
            norm_div = normalize(super_candidates['dividendYield_calc'])
            norm_roe = normalize(super_candidates['returnOnEquity'])
            norm_f = super_candidates['f_score'] / 9.0 # F-score is 0-9
            norm_magic = normalize(super_candidates['magic_rank_score'])
            norm_fcf = normalize(super_candidates['fcf_yield'])
            norm_z = normalize(super_candidates['z_score'])
            norm_sgr = normalize(super_candidates['sgr'])

            # Adjusted weighting for FCF, Z-Score, SGR
            super_candidates['Super_Score'] = (
                (norm_mos * 0.20) + 
                (norm_div * 0.10) + 
                (norm_roe * 0.10) + 
                (norm_f * 0.10) + 
                (norm_magic * 0.10) +
                (norm_fcf * 0.15) +
                (norm_z * 0.15) +
                (norm_sgr * 0.10)
            ) * 100
            
            # Sort by Super Score
            top_picks = super_candidates.sort_values(by='Super_Score', ascending=False).head(10)
        
        else:
            # Fallback to original sorting if no advanced data yet
            st.info("💡 **Tips:** กดปุ่ม 'วิเคราะห์ Magic Formula & F-Score' ด้านซ้าย เพื่อเพิ่มความแม่นยำในการจัดอันดับ")
            top_picks = super_candidates.sort_values(by='margin_of_safety', ascending=False).head(10)
        
        
        if not top_picks.empty:
            # Calculate additional ratios for Super Stocks if missing
            top_picks['P/E'] = top_picks.apply(lambda row: row['price'] / row['trailingEps'] if row['trailingEps'] > 0 else 0, axis=1)
            top_picks['P/BV'] = top_picks.apply(lambda row: row['price'] / row['bookValue'] if row['bookValue'] > 0 else 0, axis=1)
            
            # Display Top 10 nicely
            cols_to_show = [
                'symbol', 'price', 'fair_value'
            ]
            col_names = [
                'หุ้น', 'ราคา', 'Fair'
            ]
            
            # If advanced analysis is done, insert Graham next to Fair Value
            if 'Super_Score' in top_picks.columns:
                 # Calculate VI Price (Average of Fair and Graham)
                 # Handle cases where Graham is 0 or NaN
                 def calc_vi_price(row):
                     vals = []
                     if row['fair_value'] > 0: vals.append(row['fair_value'])
                     if row['graham_num'] > 0: vals.append(row['graham_num'])
                     return sum(vals) / len(vals) if vals else 0
                 
                 top_picks['vi_price'] = top_picks.apply(calc_vi_price, axis=1)
                 top_picks['vi_mos'] = top_picks.apply(lambda row: ((row['vi_price'] - row['price']) / row['vi_price'] * 100) if row['vi_price'] > 0 else 0, axis=1)
                 
                 cols_to_show.extend(['graham_num', 'vi_price', 'vi_mos'])
                 col_names.extend(['Graham', 'VI Price', 'VI MOS%'])
            else:
                 # Standard MOS if no Graham
                 cols_to_show.append('margin_of_safety')
                 col_names.append('MOS%')

            # Add remaining base columns
            cols_to_show.extend([
                'P/E', 'P/BV', 'trailingEps', 'returnOnAssets',
                'returnOnEquity', 'debtToEquityRatio', 'currentRatio', 'profitMargins',
                'dividendRate', 'dividendYield_calc', 'Quality Score'
            ])
            col_names.extend([
                'P/E', 'P/BV', 'EPS', 'ROA%',
                'ROE%', 'D/E', 'Liquidity', 'NPM%',
                'ปันผล(฿)', 'ปันผล(%)', 'Q-Score'
            ])
            
            # Add remaining advanced columns
            if 'Super_Score' in top_picks.columns:
                cols_to_show.extend(['fcf_yield', 'z_score', 'sgr', 'f_score', 'magic_roc', 'magic_ey', 'Super_Score'])
                col_names.extend(['FCF%', 'Z-Score', 'SGR%', 'F-Score', 'ROC%', 'EY%', 'Score'])
            
            top_display = top_picks[cols_to_show].copy()
            top_display.columns = col_names
            
            # Dynamic formatting dict
            fmt_dict = {
                'ราคา': '{:.2f}',
                'Fair': '{:.2f}',
                'Graham': '{:.2f}',
                'VI Price': '{:.2f}',
                'VI MOS%': '{:.2f}',
                'MOS%': '{:.2f}',
                'P/E': '{:.2f}',
                'P/BV': '{:.2f}',
                'EPS': '{:.2f}',
                'ROA%': '{:.2%}',
                'ROE%': '{:.2%}',
                'D/E': '{:.2f}',
                'Liquidity': '{:.2f}',
                'NPM%': '{:.2%}',
                'ปันผล(฿)': '{:.2f}',
                'ปันผล(%)': '{:.2%}',
                'ROC%': '{:.2%}',
                'EY%': '{:.2%}',
                'Score': '{:.0f}',
                'F-Score': '{:.0f}',
                'Q-Score': '{:.0f}',
                'FCF%': '{:.2%}',
                'Z-Score': '{:.2f}',
                'SGR%': '{:.2%}'
            }
            
            # Determine which MOS column to use for gradient
            mos_col = 'VI MOS%' if 'VI MOS%' in top_display.columns else 'MOS%'
            
            def highlight_vi_price(x):
                # Create a DataFrame of styles
                df_st = pd.DataFrame('', index=x.index, columns=x.columns)
                if 'VI Price' in x.columns:
                    df_st['VI Price'] = 'background-color: #fff9c4; color: black; font-weight: bold' # Light Yellow
                return df_st

            st.dataframe(
                top_display.style.format(fmt_dict)
                .background_gradient(subset=[mos_col], cmap='Greens')
                .apply(highlight_vi_price, axis=None),
                use_container_width=True
            )
        else:
            st.warning("ไม่พบหุ้นที่ผ่านเกณฑ์พื้นฐาน (MOS > 0, ROE > 10%, ปันผล > 3%) ลองปรับเกณฑ์ความเสี่ยงดูครับ")

        # Main Screener Results
        st.markdown("---")
        st.subheader(f"ผลการคัดกรองหุ้นทั้งหมด (พบ: {len(filtered_df)} ตัว)")
        
        # Formatting for display
        
        # Calculate P/E and P/BV
        # P/E = Price / EPS
        # P/BV = Price / Book Value
        filtered_df['P/E'] = filtered_df.apply(lambda row: row['price'] / row['trailingEps'] if row['trailingEps'] > 0 else 0, axis=1)
        filtered_df['P/BV'] = filtered_df.apply(lambda row: row['price'] / row['bookValue'] if row['bookValue'] > 0 else 0, axis=1)
        
        filtered_df['dividendYield_pct'] = filtered_df.apply(lambda row: row['dividendRate'] / row['price'] if row['price'] > 0 else 0, axis=1)
        
        display_df = filtered_df[[
            'symbol', 'price', 'fair_value', 'margin_of_safety', 
            'P/E', 'pegRatio', 'P/BV', 'trailingEps', 
            'returnOnAssets', 'returnOnEquity', 
            'grossMargins', 'operatingMargins', 'profitMargins',
            'debtToEquityRatio', 'currentRatio', 'quickRatio',
            'revenueGrowth', 'enterpriseToEbitda',
            'dividendRate', 'dividendYield_pct', 'Quality Score'
        ]].copy()
        
        # Rename columns for readable headers
        display_df.columns = [
            'หุ้น', 'ราคา', 'Fair', 'MOS%',
            'P/E', 'PEG', 'P/BV', 'EPS',
            'ROA%', 'ROE%',
            'GPM%', 'OPM%', 'NPM%',
            'D/E', 'Liquidity', 'Quick',
            'Growth%', 'EV/EBITDA',
            'ปันผล(฿)', 'ปันผล(%)', 'Q-Score'
        ]
        
        # Apply formatting
        st.dataframe(
            display_df.style.format({
                'ราคา': '{:.2f}', 
                'Fair': '{:.2f}', 
                'MOS%': '{:.2f}',
                'P/E': '{:.2f}',
                'PEG': '{:.2f}',
                'P/BV': '{:.2f}',
                'EPS': '{:.2f}',
                'ROA%': '{:.2%}',
                'ROE%': '{:.2%}',
                'GPM%': '{:.2%}',
                'OPM%': '{:.2%}',
                'NPM%': '{:.2%}',
                'D/E': '{:.2f}',
                'Liquidity': '{:.2f}',
                'Quick': '{:.2f}',
                'Growth%': '{:.2%}',
                'EV/EBITDA': '{:.2f}',
                'ปันผล(฿)': '{:.2f}',
                'ปันผล(%)': '{:.2%}'
            }).apply(lambda x: ['background-color: rgba(16, 185, 129, 0.2)' if x['MOS%'] > 15 else '' for i in x], axis=1),
            use_container_width=True,
            height=600
        )
        
        st.info("💡 **เกร็ดความรู้:** หุ้นที่มี 'MOS (%)' เขียว (> 15%) คือหุ้นที่มีส่วนลดจากมูลค่าจริงมาก")
        
        with st.expander("📖 อธิบายความหมายอัตราส่วนทางการเงิน (Financial Glossary)"):
            st.markdown("""
            *   **P/E (Price-to-Earnings Ratio):** ความถูกแพงของหุ้นเทียบกับกำไรสุทธิ (ค่ายิ่งต่ำยิ่งถูก)
            *   **PEG (P/E to Growth):** P/E เทียบกับการเติบโตของกำไร (ค่า < 1 แสดงว่าหุ้นยังถูกเมื่อเทียบกับการเติบโต)
            *   **P/BV (Price-to-Book Ratio):** ราคาหุ้นเทียบกับมูลค่าทางบัญชี (ค่ายิ่งต่ำยิ่งถูก, < 1 แสดงว่าซื้อต่ำกว่ามูลค่าสินทรัพย์)
            *   **EPS (Earnings Per Share):** กำไรสุทธิต่อหุ้น 1 หุ้น (ยิ่งมากยิ่งดี)
            *   **ROA (Return on Assets):** ความสามารถในการทำกำไรจากสินทรัพย์ที่มี (ยิ่งสูงยิ่งดี, บ่งบอกประสิทธิภาพผู้บริหาร)
            *   **ROE (Return on Equity):** ผลตอบแทนต่อส่วนของผู้ถือหุ้น (ยิ่งสูงยิ่งดี, Warren Buffett ชอบ > 15%)
            *   **GPM (Gross Profit Margin):** อัตรากำไรขั้นต้น (ขายของได้กำไรกี่ % ก่อนหักค่าใช้จ่ายบริหาร)
            *   **OPM (Operating Profit Margin):** อัตรากำไรจากการดำเนินงาน (วัดประสิทธิภาพธุรกิจหลัก)
            *   **NPM (Net Profit Margin):** อัตรากำไรสุทธิ (กำไรบรรทัดสุดท้าย / รายได้, ยิ่งสูงยิ่งดี)
            *   **D/E (Debt-to-Equity Ratio):** หนี้สินต่อทุน (ค่ายิ่งต่ำยิ่งปลอดภัย, ไม่ควรเกิน 2 เท่า)
            *   **Current Ratio:** อัตราส่วนสภาพคล่อง (สินทรัพย์หมุนเวียน / หนี้สินหมุนเวียน, ควร > 1.5 เท่า)
            *   **Quick Ratio:** สภาพคล่องหมุนเวียนเร็ว (ตัดสต็อกสินค้าออก, วัดความสามารถชำระหนี้ระยะสั้นแบบเข้มข้น)
            *   **Rev Growth:** อัตราการเติบโตของรายได้ (เทียบปีต่อปี)
            *   **EV/EBITDA:** มูลค่ากิจการเทียบกับกำไรเงินสด (ใช้ดูความถูกแพงแทน P/E ได้ดีในหุ้นที่มีค่าเสื่อมเยอะ)
            *   **ปันผล (Dividend):** เงินปันผลที่จ่ายให้ผู้ถือหุ้น (บาท)
            *   **F-Score (Piotroski F-Score):** คะแนนสุขภาพทางการเงิน 9 ด้าน (9 = แข็งแกร่งที่สุด, < 4 = อ่อนแอ)
            *   **ROC (Return on Capital):** ผลตอบแทนจากเงินลงทุนดำเนินงาน (หัวใจของ Magic Formula, ยิ่งสูงยิ่งดี)
            *   **E.Yield (Earnings Yield):** ผลตอบแทนกำไรเมื่อเทียบกับมูลค่ากิจการ (ส่วนกลับของ P/E, ยิ่งสูงยิ่งคุ้มค่า)
            *   **Super Score:** คะแนนรวมพิเศษจากโปรแกรมนี้ (เต็ม 100) คำนวณจาก MOS, F-Score, Magic Rank, ROE และปันผล
            *   **Graham Number:** ราคาที่เหมาะสมตามสูตร Benjamin Graham (บิดาแห่ง VI) เน้นสินทรัพย์และกำไร
            *   **FCF Yield (Free Cash Flow Yield):** ผลตอบแทนจากกระแสเงินสดอิสระ (เงินสดจริงที่บริษัททำได้) เทียบกับมูลค่ากิจการ
            *   **Z-Score (Altman Z-Score):** ดัชนีชี้วัดความเสี่ยงล้มละลาย (Safe > 2.99, Distress < 1.81) ช่วยกรองหุ้นเน่า
            *   **SGR (Sustainable Growth Rate):** อัตราการเติบโตที่ยั่งยืนด้วยเงินทุนตัวเอง (ไม่กู้เพิ่ม/ไม่เพิ่มทุน)
            """)
        
        # --- Display Advanced Results if available (Optional: Keep it hidden or move to debug) ---
        # User requested to combine into one table, so we hide the separate Magic Formula table
        # but we keep the logic above to feed the "Super Stocks" table.
        
    else:
        st.error("ไม่สามารถโหลดข้อมูลได้ โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")


        # Sector Heatmap
        st.markdown("---")
        st.subheader("🗺️ แผนภาพความร้อนรายอุตสาหกรรม (Sector Heatmap)")
        st.markdown("ขนาดกล่อง = มูลค่าตลาด (Market Cap), สี = ความถูกแพง (Margin of Safety)")
        
        # Prepare Data for Heatmap
        # Ignore huge outliers for color scale or clamp them?
        heat_df = df[df['marketCap'] > 0].copy()
        
        fig_treemap = px.treemap(
            heat_df, 
            path=[px.Constant("SET100"), 'sector', 'symbol'], 
            values='marketCap',
            color='margin_of_safety',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            hover_data=['price', 'fair_value']
        )
        fig_treemap.update_layout(height=600)
        st.plotly_chart(fig_treemap, use_container_width=True)

elif page == "วิเคราะห์หุ้นรายตัว":
    st.title("🔎 วิเคราะห์หุ้นเจาะลึก (Pro Stock Analysis)")
    
    # Select Stock
    selected_ticker = st.selectbox("เลือกหุ้นที่ต้องการวิเคราะห์", SET100_TICKERS)
    
    if st.button("เริ่มวิเคราะห์"):
        with st.spinner(f"กำลังวิเคราะห์ {selected_ticker}..."):
            # Get fresh data (or we could use cached if passed, but let's fetch fresh deeper data)
            stock_data = utils.get_stock_data(selected_ticker)
            valuation = utils.calculate_valuations(stock_data)
            fin_hist = utils.get_financial_history(selected_ticker)
            
            if valuation:
                # --- HEADER SECTION ---
                st.markdown(f"## {valuation['longName']} ({valuation['symbol']})")
                st.markdown(f"**อุตสาหกรรม:** {valuation.get('sector')} | **ธุรกิจ:** {valuation.get('summary')[:150]}...")
                
                # Gauge / Recommendation
                rec_val = valuation.get('recommendation', 3.0) # 1=Buy, 5=Sell
                target_price = valuation.get('targetPrice', 0)
                current_price = valuation.get('price', 0)
                fair_val = valuation.get('fair_value', 0)
                
                col_head1, col_head2, col_head3 = st.columns([1, 2, 1])
                
                with col_head1:
                    st.metric("ราคาปัจจุบัน", f"฿{current_price:.2f}")
                    
                    # Simple Sentiment Color
                    if rec_val <= 2.0:
                        st.success("นักวิเคราะห์: แนะนำซื้อ (BUY)")
                    elif rec_val >= 4.0:
                        st.error("นักวิเคราะห์: แนะนำขาย (SELL)")
                    else:
                        st.warning("นักวิเคราะห์: แนะนำถือ (HOLD)")
                        
                with col_head2:
                    # Comparison Bar
                    st.markdown("##### ราคาตลาด vs มูลค่าที่เหมาะสม")
                    comp_data = pd.DataFrame({
                        'Type': ['ราคาปัจจุบัน', 'เป้านักวิเคราะห์', 'มูลค่าพื้นฐาน (VI)'],
                        'Price': [current_price, target_price, fair_val]
                    })
                    fig_comp = px.bar(comp_data, x='Price', y='Type', orientation='h', text='Price', 
                                      color='Type', color_discrete_map={'ราคาปัจจุบัน': 'grey', 'เป้านักวิเคราะห์': '#3b82f6', 'มูลค่าพื้นฐาน (VI)': '#10b981'})
                    fig_comp.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    fig_comp.update_traces(texttemplate='฿%{text:.2f}')
                    st.plotly_chart(fig_comp, use_container_width=True)

                with col_head3:
                    mos = valuation.get('margin_of_safety', 0)
                    st.metric("MOS (ส่วนเผื่อความปลอดภัย)", f"{mos:.2f}%", 
                              delta="ราคาถูก (Undervalued)" if mos > 0 else "ราคาแพง (Overvalued)",
                              delta_color="normal" if mos > 0 else "inverse")
                
                # --- KEY STATS GRID ---
                st.subheader("📊 อัตราส่วนทางการเงินที่สำคัญ (Key Ratios)")
                k1, k2, k3, k4 = st.columns(4)
                
                with k1:
                    st.markdown("**ความถูกแพง (Valuation)**")
                    st.metric("P/E Ratio", f"{valuation.get('price') / valuation.get('trailingEps') if valuation.get('trailingEps') else 0:.2f}") 
                    st.metric("P/BV Ratio", f"{valuation.get('price') / valuation.get('bookValue') if valuation.get('bookValue') else 0:.2f}")
                    st.metric("PEG Ratio", f"{valuation.get('pegRatio', 0):.2f}")
                
                with k2:
                    st.markdown("**ประสิทธิภาพ (Efficiency)**")
                    st.metric("ROE (ผลตอบแทนส่วนผู้ถือหุ้น)", f"{valuation.get('returnOnEquity', 0)*100:.2f}%")
                    st.metric("ROA (ผลตอบแทนสินทรัพย์)", f"{valuation.get('returnOnAssets', 0)*100:.2f}%")
                    st.metric("Profit Margin (อัตรากำไร)", f"{valuation.get('profitMargins', 0)*100:.2f}%")
                    
                with k3:
                    st.markdown("**สุขภาพการเงิน (Health)**")
                    st.metric("D/E Ratio (หนี้สิน/ทุน)", f"{valuation.get('debtToEquity', 0)/100:.2f}") 
                    st.metric("Current Ratio (สภาพคล่อง)", f"{valuation.get('currentRatio', 0):.2f}")
                    st.metric("Beta (ความผันผวน)", f"{valuation.get('beta', 1.0):.2f}")

                with k4:
                    st.markdown("**ปันผล (Dividend)**")
                    st.metric("Yield (ผลตอบแทน)", f"{(valuation.get('dividendRate',0) / current_price * 100) if current_price else 0:.2f}%")
                    st.metric("Payout Ratio (สัดส่วนจ่าย)", f"{valuation.get('payoutRatio', 0)*100:.2f}%")
                
                # --- FINANCIAL TRENDS & FORECAST ---
                st.markdown("---")
                st.subheader("📈 ผลประกอบการย้อนหลัง & คาดการณ์อนาคต")
                st.info("ℹ️ **หมายเหตุข้อมูล:** ข้อมูลย้อนหลังประมาณ 4 ปีล่าสุด | ตัวเลขคาดการณ์อ้างอิงจากบทวิเคราะห์ (Analyst Estimates)")
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 การเติบโต & กำไร", "💪 ประสิทธิภาพการทำกำไร", "🔮 คาดการณ์อนาคต", "📉 PE Band & Matrix"])
                
                if not fin_hist.empty:
                    with tab1:
                        # Revenue & Profit Combo
                        f1, f2 = st.columns(2)
                        with f1:
                            fig_fin = go.Figure()
                            fig_fin.add_trace(go.Bar(x=fin_hist.index, y=fin_hist['Revenue'], name='รายได้ (Revenue)', marker_color='#60a5fa'))
                            fig_fin.add_trace(go.Scatter(x=fin_hist.index, y=fin_hist['Net Profit'], name='กำไรสุทธิ (Net Profit)', mode='lines+markers', line=dict(color='#10b981', width=3)))
                            fig_fin.update_layout(title="แนวโน้มรายได้ vs กำไรสุทธิ", legend=dict(orientation="h"))
                            st.plotly_chart(fig_fin, use_container_width=True)
                            
                        with f2:
                            # EPS Trend
                            if 'EPS' in fin_hist.columns:
                                fig_eps = px.bar(fin_hist, x=fin_hist.index, y='EPS', title="กำไรต่อหุ้น (EPS)", text_auto='.2f')
                                fig_eps.update_traces(marker_color='#8b5cf6')
                                st.plotly_chart(fig_eps, use_container_width=True)
                                
                    with tab2:
                        # Ratios Triple Chart
                        r1, r2, r3 = st.columns(3)
                        
                        with r1:
                            if 'ROE (%)' in fin_hist.columns:
                                fig_roe = px.line(fin_hist, x=fin_hist.index, y='ROE (%)', markers=True, title="ROE (%)")
                                fig_roe.update_traces(line_color='#ef4444')
                                st.plotly_chart(fig_roe, use_container_width=True)
                        
                        with r2:
                            if 'NPM (%)' in fin_hist.columns:
                                fig_npm = px.line(fin_hist, x=fin_hist.index, y='NPM (%)', markers=True, title="Net Profit Margin (%)")
                                fig_npm.update_traces(line_color='#f59e0b')
                                st.plotly_chart(fig_npm, use_container_width=True)

                        with r3:
                            if 'D/E (x)' in fin_hist.columns:
                                fig_de = px.bar(fin_hist, x=fin_hist.index, y='D/E (x)', title="D/E Ratio (เท่า)", text_auto='.2f')
                                fig_de.update_traces(marker_color='#64748b')
                                st.plotly_chart(fig_de, use_container_width=True)
                
                else:
                    st.warning("ไม่มีข้อมูลย้อนหลังสำหรับกราฟ")

                with tab3:
                    # Forecast Logic
                    # We have Trailing EPS and Forward EPS.
                    # Let's project 2 years
                    current_year_eps = valuation.get('trailingEps')
                    next_year_eps = valuation.get('forwardEps')
                
                    if current_year_eps and next_year_eps:
                        # Simple 2-point projection
                        # Avoid div by zero
                        denom = abs(current_year_eps) if current_year_eps != 0 else 1
                        growth = (next_year_eps - current_year_eps) / denom
                        
                        # Project Year+2 with same growth rate (Conservative)
                        year_2_eps = next_year_eps * (1 + (growth * 0.8)) # Decay growth slightly
                        
                        forecast_data = pd.DataFrame({
                            'Year': ['ปีปัจจุบัน (TTM)', 'ปีหน้า (คาดการณ์)', 'ปีถัดไป (คาดการณ์)'],
                            'EPS': [current_year_eps, next_year_eps, year_2_eps],
                            'Type': ['ของจริง', 'คาดการณ์', 'คาดการณ์']
                        })
                        
                        f_col1, f_col2 = st.columns([2, 1])
                        with f_col1:
                            fig_fore = px.line(forecast_data, x='Year', y='EPS', markers=True, title="คาดการณ์กำไรต่อหุ้น (Earnings Forecast)", text='EPS')
                            fig_fore.update_traces(texttemplate='%{text:.2f}', textposition="top center", line=dict(color='#0ea5e9', width=3, dash='dot'))
                            st.plotly_chart(fig_fore, use_container_width=True)
                            
                        with f_col2:
                            st.metric("การเติบโตคาดหวัง (1Y)", f"{growth*100:.2f}%")
                            st.metric("Forward EPS", f"{next_year_eps:.2f}")
                            st.markdown("*(E) = ตัวเลขประมาณการ*")
                    else:
                        st.info("ไม่มีข้อมูลประมาณการจากนักวิเคราะห์")
                
                with tab4:
                    st.subheader("📉 Historical PE Band")
                    st.info("กราฟแสดงราคาหุ้นเทียบกับกรอบราคาที่คิดจากค่า PE ย้อนหลัง 5 ปี (ช่วยดูว่าตอนนี้ถูกหรือแพงเมื่อเทียบกับตัวเองในอดีต)")
                    
                    pe_band_data = utils.get_historical_pe_bands(selected_ticker)
                    
                    if pe_band_data:
                         band_df = pe_band_data['data']
                         
                         fig_band = go.Figure()
                         
                         # Price
                         fig_band.add_trace(go.Scatter(x=band_df['Date'], y=band_df['Close'], name='ราคาหุ้น (Price)', line=dict(color='black', width=3)))
                         
                         # Bands
                         fig_band.add_trace(go.Scatter(x=band_df['Date'], y=band_df['Mean PE'], name=f'Avg PE ({pe_band_data["avg_pe"]:.1f}x)', line=dict(color='orange', dash='dash')))
                         fig_band.add_trace(go.Scatter(x=band_df['Date'], y=band_df['+1 SD'], name='+1 SD', line=dict(color='red', width=1)))
                         fig_band.add_trace(go.Scatter(x=band_df['Date'], y=band_df['+2 SD'], name='+2 SD (แพงมาก)', line=dict(color='darkred', width=1, dash='dot')))
                         fig_band.add_trace(go.Scatter(x=band_df['Date'], y=band_df['-1 SD'], name='-1 SD', line=dict(color='green', width=1)))
                         fig_band.add_trace(go.Scatter(x=band_df['Date'], y=band_df['-2 SD'], name='-2 SD (ถูกมาก)', line=dict(color='darkgreen', width=1, dash='dot')))
                         
                         fig_band.update_layout(title=f"PE Band: {selected_ticker}", hovermode="x unified")
                         st.plotly_chart(fig_band, use_container_width=True)
                         
                         st.markdown(f"**ค่าเฉลี่ย PE 5 ปีย้อนหลัง:** {pe_band_data['avg_pe']:.2f} เท่า | **PE ปัจจุบัน:** {pe_band_data['current_pe']:.2f} เท่า")
                    else:
                         st.error("ข้อมูลไม่เพียงพอสำหรับสร้าง PE Band (ต้องการกำไรย้อนหลังต่อเนื่อง)")

                # --- 8 Qualities Checklist (Enhanced) ---
                st.markdown("---")
                st.subheader("📋 แบบประเมินคุณภาพหุ้น VI (Checklist)")
                
                score = 0
                total = 8
                
                check_col1, check_col2 = st.columns(2)
                
                roe = valuation.get('returnOnEquity', 0)
                npm = valuation.get('profitMargins', 0)
                de = valuation.get('debtToEquity', 0) # yfinance returns e.g. 150 for 1.5 ratio often, need to verify.
                # Usually debtToEquity is a percentage in yfinance (e.g. 221.35 means 2.21)
                
                # Logic helpers
                is_strong_roe = roe > 0.15
                is_strong_npm = npm > 0.10
                is_low_debt = de < 200 # < 2.0 D/E
                is_undervalued = mos > 0
                
                with check_col1:
                    c1 = st.checkbox("1. ผู้นำตลาด / ผูกขาด (Market Leader)", help="บริษัทมีส่วนแบ่งการตลาดสูง อำนาจต่อรองสูง?")
                    c2 = st.checkbox("2. คู่แข่งเข้ามายาก (High Barriers to Entry)", help="ธุรกิจเลียนแบบยาก หรือต้องใช้เงินลงทุนสูงมาก?")
                    c3 = st.checkbox("3. กำหนดราคาเองได้ (Pricing Power)", help="ขึ้นราคาสินค้าได้โดยที่ลูกค้าไม่หนีไปไหน?")
                    c4 = st.checkbox("4. ยังมีโอกาสเติบโต (Growth Potential)", help="อุตสาหกรรมยังไม่ตะวันตกดิน ยังโตได้อีก?")
                    
                with check_col2:
                    c5 = st.checkbox(f"5. คุมต้นทุนดี / หนี้ต่ำ (D/E < 2) [Current: {de/100:.2f}x]", value=is_low_debt, help=f"ค่า D/E ปัจจุบัน: {de/100:.2f} เท่า")
                    c6 = st.checkbox(f"6. การเงินแกร่ง (ROE > 15%) [Current: {roe*100:.2f}%]", value=is_strong_roe, help=f"ค่า ROE ปัจจุบัน: {roe*100:.2f}%")
                    c7 = st.checkbox(f"7. กำไรสูง (NPM > 10%) [Current: {npm*100:.2f}%]", value=is_strong_npm, help=f"ค่า NPM ปัจจุบัน: {npm*100:.2f}%")
                    c8 = st.checkbox(f"8. ราคาถูก (MOS > 0%) [Current: {mos:.2f}%]", value=is_undervalued, disabled=True)
                
                # Manual Score Calculation
                manual_checks = sum([c1, c2, c3, c4])
                auto_checks = sum([is_low_debt, is_strong_roe, is_strong_npm, is_undervalued])
                final_score = manual_checks + auto_checks
                
                st.markdown(f"#### **คะแนนคุณภาพรวม: {final_score} / 8**")
                st.progress(final_score / 8)

            else:
                st.error("ไม่สามารถดึงข้อมูลหุ้นตัวนี้ได้")


elif page == "เปรียบเทียบคู่แข่ง":
    st.title("⚔️ เปรียบเทียบคู่แข่ง (Competitor Analysis)")
    st.markdown("เปรียบเทียบตัวเลขทางการเงินของหุ้นหลายตัวแบบตัวต่อตัว")
    
    # Multiselect
    selected_tickers = st.multiselect("เลือกหุ้นมาชนกัน (สูงสุด 5 ตัว)", SET100_TICKERS, default=["ADVANC", "TRUE"] if "TRUE" in SET100_TICKERS else ["ADVANC"])
    
    if len(selected_tickers) > 0:
        if len(selected_tickers) > 5:
            st.warning("เลือกได้สูงสุด 5 ตัวเท่านั้น")
        else:
            with st.spinner("กำลังดึงข้อมูลเปรียบเทียบ..."):
                # Fetch data directly or via utils
                # Use ThreadPool to fetch detailed history for all selected
                
                # 1. Comparison Table (Current Stats)
                # Filter 'df' (global) for efficiency for current stats
                comp_df = df[df['symbol'].isin(selected_tickers)].set_index('symbol')
                
                # Select interesting columns
                cols_to_show = ['price', 'fair_value', 'margin_of_safety', 'dividendRate', 'returnOnEquity', 'profitMargins', 'debtToEquityRatio', 'valuation_pe']
                comp_table = comp_df[cols_to_show].T
                
                # Rename Index for TH
                comp_table.index = ['ราคา', 'มูลค่าเหมาะสม', 'MOS (%)', 'ปันผล (บาท)', 'ROE (%)', 'NPM (%)', 'D/E (เท่า)', 'P/E (เท่า)']
                
                st.subheader("📊 ตารางวัดพลังพื้นฐาน")
                st.dataframe(comp_table.style.format("{:.2f}").background_gradient(axis=1), use_container_width=True)
                
                # 2. Historical Charts Comparison
                st.subheader("📈 กราฟวัดพลังย้อนหลัง")
                
                # We need to fetch history for each
                hist_data = {}
                metrics = ['Revenue', 'Net Profit', 'ROE (%)', 'NPM (%)']
                
                # Fetch history logic
                # For charts we need a combined dataframe
                combined_hist = pd.DataFrame()
                
                for t in selected_tickers:
                     h = utils.get_financial_history(t)
                     if not h.empty:
                         h['Symbol'] = t
                         combined_hist = pd.concat([combined_hist, h])
                
                if not combined_hist.empty:
                    # Choose Metric to compare
                    metric_choice = st.radio("เลือกหัวข้อเปรียบเทียบ", metrics, horizontal=True)
                    
                    if metric_choice in combined_hist.columns:
                        fig_comp = px.bar(combined_hist, x=combined_hist.index, y=metric_choice, color='Symbol', barmode='group', title=f"เปรียบเทียบ {metric_choice}")
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        st.info(f"ไม่มีข้อมูล {metric_choice}")
                else:
                    st.error("ไม่สามารถดึงข้อมูลย้อนหลังได้")


                    st.error("ไม่สามารถดึงข้อมูลย้อนหลังได้")


elif page == "แนะนำพอร์ตการลงทุน":
    st.title("🍰 แนะนำพอร์ตการลงทุน (Asset Allocation)")
    
    st.markdown("สร้างพอร์ตการลงทุนที่สมดุล ตามหลักการกระจายความเสี่ยง")
    
    # Input with cleaner integer format (Note: Commas in input fields are not supported by Streamlit for editing, so we show a caption)
    capital = st.number_input("เงินลงทุนตั้งต้น (บาท)", min_value=1000, value=100000, step=1000, format="%d")
    st.caption(f"💰 จำนวนเงินที่ระบุ: **{capital:,.0f}** บาท")
    
    # Portfolio Mix (Thai Keys)
    allocation = {
        "พันธบัตร / ตราสารหนี้ (Fixed Income)": 0.40,
        "หุ้นไทยขนาดใหญ่ (SET50)": 0.15,
        "หุ้นต่างประเทศ (Global Stocks)": 0.15,
        "หุ้นเล็ก / หุ้นเติบโต (Growth)": 0.10,
        "ตลาดเกิดใหม่ (Emerging Markets)": 0.10,
        "กองทุนอสังหาฯ (REITs)": 0.10
    }
    
    if st.button("คำนวณสัดส่วนการลงทุน"):
        amounts = utils.calculate_portfolio(capital, allocation)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("สัดส่วนที่แนะนำ (Target Allocation)")
            # Create DataFrame with Thai columns
            df_port = pd.DataFrame(list(amounts.items()), columns=['ประเภทสินทรัพย์', 'มูลค่า (บาท)'])
            df_port['สัดส่วน (%)'] = df_port['ประเภทสินทรัพย์'].map(allocation) * 100
            
            # Format numbers with commas
            st.dataframe(
                df_port.style.format({
                    'มูลค่า (บาท)': '{:,.2f}', 
                    'สัดส่วน (%)': '{:.1f}%'
                }),
                use_container_width=True
            )
            
        with col2:
            fig = px.pie(
                values=list(amounts.values()), 
                names=list(amounts.keys()), 
                title="แผนภูมิพอร์ตการลงทุน",
                hole=0.4
            )
            st.plotly_chart(fig)
            
        # --- ASSET RECOMMENDATION EXPANDER ---
        st.markdown("---")
        st.subheader("💡 แนะนำสินทรัพย์น่าลงทุน (Asset Recommendations)")
        st.info("รายชื่อสินทรัพย์ยอดนิยมสำหรับคนไทย (คำเตือน: ไม่ใช่คำแนะนำการลงทุน เป็นเพียงตัวอย่างศึกษา)")
        
        with st.expander("🛡️ ตราสารหนี้ & พันธบัตร (40%)", expanded=True):
             st.markdown("""
             **แนวคิด:** รักษาเงินต้น ความเสี่ยงต่ำ
             *   **พันธบัตรไทย:** `LB296A`, `LB31DA` (ซื้อผ่านแอปเป๋าตัง/ธนาคาร)
             *   **กองทุนตราสารหนี้:** `K-FIXED`, `SCBFIXED`, `TMBABF`
             *   **เงินฝาก:** บัญชีออมทรัพย์ดอกเบี้ยสูง (Kept, Dime, etc.)
             """)
             
        with st.expander("🏢 หุ้นไทยขนาดใหญ่ (15%)", expanded=True):
            st.markdown("""
            **แนวคิด:** เติบโตมั่นคง + ปันผลสม่ำเสมอ
            *   **หุ้นเด่น SET100:** `ADVANC`, `PTT`, `AOT`, `KBANK`, `CPALL`
            *   **กองทุนดัชนี (ETF):** `TDEX` (อ้างอิงดัชนี SET50)
            """)
            
        with st.expander("🌐 หุ้นต่างประเทศ (15%)", expanded=True):
            st.markdown("""
            **แนวคิด:** กระจายความเสี่ยงออกนอกประเทศ (US/Tech/China)
            *   **หุ้นเทคฯ สหรัฐ:** `ONE-ULTRAP` (Growth), `SCBNDQ` (Nasdaq)
            *   **หุ้นโลก:** `K-CHANGE`, `TMBGQG`
            *   **DR (ซื้อผ่านตลาดไทย):** `E1VFVN3001` (เวียดนาม), `CNTECH01` (จีนเทค)
            """)

        with st.expander("🏬 กองทุนอสังหาฯ (10%)", expanded=True):
            st.markdown("""
            **แนวคิด:** รับค่าเช่า (Passive Income)
            *   **ห้าง/ออฟฟิศ:** `CPNREIT`, `ALLY`
            *   **คลังสินค้า/นิคม:** `WHAIR`, `FTREIT`
            *   **โครงสร้างพื้นฐาน:** `DIF` (เสาสัญญาณ), `TFFIF` (ทางด่วน)
            """)
            
        c_grow, c_em = st.columns(2)
        with c_grow:
            st.markdown("##### 🌱 หุ้นเติบโต / หุ้นเล็ก (10%)")
            st.markdown("- **รายตัว:** `JMT`, `FORTH`, `XO`\n- **กองทุน:** `K-STAR`, `SCBSE`")
        with c_em:
            st.markdown("##### 🌍 ตลาดเกิดใหม่ (10%)")
            st.markdown("- **เน้น:** อินเดีย, เวียดนาม, อินโดฯ\n- **กองทุน:** `K-INDX` (อินเดีย), `ASP-VIET`")

elif page == "พอร์ตของฉัน (My Portfolio)":
    st.title("🎒 พอร์ตของฉัน (My Portfolio)")
    st.markdown("บันทึกการซื้อขายและติดตามผลกำไรขาดทุนของพอร์ตโฟลิโอ")
    
    # 1. Add Transaction Form
    with st.expander("➕ เพิ่มรายการซื้อ/ขาย (Add Transaction)", expanded=False):
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        with t_col1:
            t_open_action = st.selectbox("ทำรายการ", ["Buy", "Sell"])
        with t_col2:
            t_symbol = st.selectbox("หุ้น (Symbol)", SET100_TICKERS)
        with t_col3:
            t_date = st.date_input("วันที่ (Date)")
        with t_col4:
            t_price = st.number_input("ราคา (Price)", min_value=0.01, step=0.05)
        with t_col5:
            t_qty = st.number_input("จำนวน (Qty)", min_value=100, step=100)
            
        if st.button("บันทึกรายการ"):
            utils.save_transaction(t_symbol, t_date, t_price, t_qty, t_open_action)
            st.success(f"บันทึก {t_open_action} {t_symbol} เรียบร้อย!")
            st.rerun()

    # 2. Portfolio View
    # Create price map from loaded df
    if not df.empty:
        price_map = df.set_index('symbol')['price'].to_dict()
    else:
        price_map = {}
        
    port_df, port_val, cost_val = utils.get_portfolio_summary(price_map)
    
    if not port_df.empty:
        # Metrics
        m1, m2, m3 = st.columns(3)
        unrealized_pl = port_val - cost_val
        pl_pct = (unrealized_pl / cost_val * 100) if cost_val > 0 else 0
        
        m1.metric("มูลค่าพอร์ตปัจจุบัน", f"{port_val:,.2f} บาท")
        m2.metric("ทุนรวม", f"{cost_val:,.2f} บาท")
        m3.metric("กำไร/ขาดทุน (Unrealized)", f"{unrealized_pl:,.2f} บาท", f"{pl_pct:.2f}%")
        
        st.subheader("📜 รายการถือครอง (Current Holdings)")
        # Show specific columns
        display_port = port_df[['Symbol', 'Qty', 'Avg Price', 'Market Price', 'Cost Value', 'Market Value', 'P/L %']]
        st.dataframe(display_port.style.format({
            'Qty': '{:,.0f}',
            'Avg Price': '{:,.2f}',
            'Market Price': '{:,.2f}',
            'Cost Value': '{:,.2f}',
            'Market Value': '{:,.2f}',
            'P/L %': '{:+.2f}%'
        }))
        
        # Pie Chart
        st.subheader("🍰 สัดส่วนพอร์ต (Allocation)")
        fig_port = px.pie(port_df, values='Market Value', names='Symbol', title='Portfolio Allocation by Value', hole=0.4)
        st.plotly_chart(fig_port)
    else:
        st.info("ยังไม่มีข้อมูลในพอร์ต กรุณาเพิ่มรายการซื้อขาย")

elif page == "จำลองการออมหุ้น (DCA Backtester)":
    st.title("⏳ จำลองการออมหุ้น (DCA Backtester)")
    st.markdown("ทดสอบผลตอบแทนย้อนหลัง หากเราลงทุนแบบ Dollar Cost Average (DCA) อย่างมีวินัย")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        dca_ticker = st.selectbox("เลือกหุ้นที่จะออม", SET100_TICKERS, index=SET100_TICKERS.index('CPALL') if 'CPALL' in SET100_TICKERS else 0)
        dca_amount = st.number_input("เงินออมต่อเดือน (บาท)", value=5000, step=1000)
    
    with col_d2:
        dca_years = st.slider("ระยะเวลาลงทุน (ปี)", 1, 10, 5)
        dca_day = st.slider("วันที่ลงทุนของทุกเดือน", 1, 28, 25)
        
    if st.button("เริ่มการจำลอง (Run Simulation)"):
        with st.spinner("กำลังคำนวณผลตอบแทนย้อนหลัง..."):
            ledger, total_inv, final_val, prof_pct = utils.calculate_dca_simulation(dca_ticker, dca_amount, dca_years, dca_day)
            
            if not ledger.empty:
                st.success("การคำนวณเสร็จสิ้น!")
                
                # Metrics
                r1, r2, r3 = st.columns(3)
                r1.metric("เงินต้นรวม (Total Invested)", f"{total_inv:,.2f} บาท")
                r2.metric("มูลค่าพอร์ตปลายทาง", f"{final_val:,.2f} บาท")
                r3.metric("กำไร/ขาดทุน (%)", f"{prof_pct:+.2f}%", delta_color="normal")
                
                # Chart
                st.subheader("📈 การเติบโตของพอร์ต DCA")
                fig_dca = go.Figure()
                fig_dca.add_trace(go.Scatter(x=ledger['Date'], y=ledger['Value'], fill='tozeroy', name='มูลค่าพอร์ต (Portfolio Value)', line=dict(color='#10b981')))
                fig_dca.add_trace(go.Scatter(x=ledger['Date'], y=ledger['Invested'], name='เงินต้นสะสม (Invested)', line=dict(color='#6b7280', dash='dash')))
                fig_dca.update_layout(title=f"DCA Simulation for {dca_ticker} ({dca_years} Years)", hovermode="x unified")
                st.plotly_chart(fig_dca, use_container_width=True)
                
                # Data Table
                with st.expander("ดูตารางข้อมูลละเอียด (Detailed Ledger)"):
                    st.dataframe(ledger.style.format({'Invested': '{:,.2f}', 'Value': '{:,.2f}', 'Cost': '{:,.2f}'}))
            else:
                st.error("ไม่สามารถดึงข้อมูลย้อนหลังได้ หรือข้อมูลไม่เพียงพอ")
            
elif page == "ตั้งค่า":
    st.title("⚙️ ตั้งค่า (Settings)")
    
    st.subheader("จัดการรายชื่อหุ้น (SET100)")
    st.markdown("เพิ่ม/ลด รายชื่อหุ้นที่ต้องการสแกน (คั่นด้วยเครื่องหมายจุลภาค , หรือขึ้นบรรทัดใหม่)")
    
    current_tickers = ", ".join(SET100_TICKERS)
    new_tickers_text = st.text_area("รายชื่อหุ้น (Ticker Symbols)", value=current_tickers, height=300)
    
    if st.button("บันทึกรายชื่อ"):
        # Process input
        raw_tickers = new_tickers_text.replace("\n", ",").split(",")
        clean_tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
        
        # Save to file
        utils.save_tickers(clean_tickers)
        st.success(f"บันทึกเรียบร้อย! มีหุ้นทั้งหมด {len(clean_tickers)} ตัว (กรุณารีโหลดหน้าเว็บใหม่)")
        
        # Clear cache so new tickers are used next time
        st.cache_data.clear()
