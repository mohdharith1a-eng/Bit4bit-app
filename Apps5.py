import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os, base64
import subprocess
import requests
import json
import warnings
import numpy as np
import io
from groq import Groq

# Suppress warnings from pandas and other libraries for cleaner output
warnings.filterwarnings('ignore')
px.defaults.template = "plotly_white"

# Set page configuration for a wider layout
st.set_page_config(layout="wide")


# Placeholder for the chatbot function, as it was not defined in the provided code
def ask_groq(query):
    try:
        # Dapatkan kunci API dari secrets Streamlit
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        # Hantar soalan pengguna ke API Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in providing answers about Malaysian economic data based on provided charts and numbers. Be concise and to the point."
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            # --- GANTIKAN DENGAN NAMA MODEL YANG BAHARU ---
            model="llama-3.1-8b-instant",
            # ---------------------------------------------
        )

        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Maaf, ralat berlaku semasa berhubung dengan chatbot: {e}"

# ================== BACKGROUND IMAGE + GRADIENT ==================
def add_bg_from_local(image_file):
    """Adds a local background image with a gradient overlay to the app."""
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            background: 
                linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* Improve text readability and chart appearance */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            font-weight: bold;
        }}
        
        .stSelectbox > div > div, .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.9);
            color: #000000;
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }}
        
        /* Add specific style to make text within the translucent box dark */
        .translucent-box {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }}
        
        .translucent-box p, .translucent-box li, .translucent-box h4 {{
            color: #000000 !important;
            font-weight: normal;
            text-shadow: none;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Background image '{image_file}' not found.")


# ================== PATHS & DATA LOADING ==================
gdp_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/gdp_state_.csv"
unemp_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/unemployed.csv"
pop_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/population_state.csv"
wellbeing_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/economic_wellbeing.csv"
flag_folder_path = r"https://github.com/mohdharith1a-eng/Bit4bit-app/tree/main/bendera"

def load_csv_safe(path, name=""):
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip", encoding="utf-8")
        print(f"✅ Loaded {name} ({path}) - shape: {df.shape}")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip", encoding="latin1")
        print(f"⚠ Fallback encoding for {name} ({path}) - shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {name} ({path}): {e}")
        return pd.DataFrame() # return empty df if failed

def clean_data(df, date_cols=None, numeric_cols=None, fillna_val=0, upper_state=True):
    df = df.copy()
    
    if 'state' in df.columns and upper_state:
        df['state'] = df['state'].str.upper().str.strip()
    
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(fillna_val)
                
    if 'state' in df.columns:
        df = df[~df['state'].isin(['MALAYSIA', 'SUPRA'])].reset_index(drop=True)
    return df

@st.cache_data
def load_all_data():
    dataframes = {}
    
    try:
        dataframes['gdp'] = load_csv_safe(gdp_path, "GDP")
        dataframes['lfs'] = load_csv_safe(unemp_path, "Unemployment")
        dataframes['population'] = load_csv_safe(pop_path, "Population")
        dataframes['economic_wellbeing'] = load_csv_safe(wellbeing_path, "Wellbeing")
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        st.stop()

    gdp = clean_data(dataframes['gdp'], date_cols=["date"], numeric_cols=["value"])
    lfs = clean_data(dataframes['lfs'], date_cols=["date"], numeric_cols=["lf_unemployed"])
    population = clean_data(dataframes['population'], date_cols=["date"], numeric_cols=["population"])
    economic_wellbeing = clean_data(dataframes['economic_wellbeing'], numeric_cols=["wellbeing_value"])

    gdp.rename(columns={"value": "gdp_total"}, inplace=True)
    lfs.rename(columns={"lf_unemployed": "unemployment_rate"}, inplace=True)
    economic_wellbeing.rename(columns={"State": "state", "wellbeing_value": "economic_wellbeing"}, inplace=True)
    
    gdp["year"] = gdp["date"].dt.year
    gdp = gdp[gdp["year"].isin([2022, 2023])]
    gdp = gdp.groupby(["year", "state"])["gdp_total"].sum().reset_index()

    lfs["year"] = lfs["date"].dt.year
    lfs = lfs[lfs["year"].isin([2022, 2023])]
    lfs_both = lfs[lfs["sex"] == "both"].copy()

    population["year"] = population["date"].dt.year
    population = population[population["year"].isin([2022, 2023])]
    population = population[(population["sex"] == "both") & (population["age"] == "overall") & (population["ethnicity"] == "overall")]

    combined_df = pd.merge(gdp, lfs_both, on=["state", "year"], how="inner", suffixes=('_gdp', '_lfs'))
    combined_df = pd.merge(combined_df, population, on=["state", "year"], how="inner", suffixes=('', '_pop'))
    combined_df = pd.merge(combined_df, economic_wellbeing[['state', 'economic_wellbeing']], on='state', how='left')

    combined_df.drop(columns=['date_gdp', 'date_lfs', 'date', 'sex', 'age', 'ethnicity'], inplace=True, errors='ignore')

    combined_df["gdp_per_capita"] = combined_df["gdp_total"] / combined_df["population"].replace({0: pd.NA})
    
    if "economic_wellbeing" in combined_df.columns:
        combined_df["economic_wellbeing"] = pd.to_numeric(combined_df["economic_wellbeing"], errors="coerce").fillna(0)

    return gdp, lfs, population, economic_wellbeing, combined_df

def get_state_names_dict():
    return {
        'JOHOR': 'Johor', 'KEDAH': 'Kedah', 'KELANTAN': 'Kelantan',
        'W.P. LABUAN': 'W.P. Labuan', 'MELAKA': 'Melaka',
        'NEGERI SEMBILAN': 'Negeri Sembilan', 'PAHANG': 'Pahang',
        'PERAK': 'Perak', 'PERLIS': 'Perlis', 'PULAU PINANG': 'Pulau Pinang',
        'PUTRAJAYA': 'Putrajaya', 'SABAH': 'Sabah', 'SARAWAK': 'Sarawak',
        'SELANGOR': 'Selangor', 'TERENGGANU': 'Terengganu',
        'W.P. KUALA LUMPUR': 'W.P. Kuala Lumpur'
    }

def get_flags_dict():
    flags_dict = {
        'JOHOR': 'bendera/johor.png',
        'KEDAH': 'bendera/kedah.png',
        'KELANTAN': 'bendera/kelantan.png',
        'W.P. LABUAN': 'bendera/labuan.png',
        'MELAKA': 'bendera/melaka.png',
        'NEGERI SEMBILAN': 'bendera/n9.png',
        'PAHANG': 'bendera/pahang.png',
        'PERAK': 'bendera/perak.png',
        'PERLIS': 'bendera/perlis.png',
        'PULAU PINANG': 'bendera/penang.png',
        'PUTRAJAYA': 'bendera/putrajaya.png',
        'SABAH': 'bendera/sabah.png',
        'SARAWAK': 'bendera/sarawak.png',
        'SELANGOR': 'bendera/selangor.png',
        'TERENGGANU': 'bendera/terengganu.png',
        'W.P. KUALA LUMPUR': 'bendera/kuala_lumpur.png'
    }
    return flags_dict

# ================== STREAMLIT APP LAYOUT ==================
st.title("📊 BRIDGING THE GDP-GPI GAP: MALAYSIAN ECONOMIC DASHBOARD")

# Load data
df_gdp, df_unemp, df_pop, df_wellbeing, combined_df = load_all_data()
state_names_dict = get_state_names_dict()
flags_dict = get_flags_dict()

# Add background image for App Page
add_bg_from_local("background.png")

tab1, tab2 = st.tabs(["📈 Infographic Dashboard for RAW data DOSM", "🔍 Comparisons dataset & Solutions"])

# --- TAB 1: Infographics ---
with tab1:
    if df_gdp is not None and df_unemp is not None and df_wellbeing is not None and df_pop is not None:
        
        # === FIRST ROW: GDP CHART & POPULATION CHART ===
        col_gdp, col_pop = st.columns(2)

        with col_gdp:
            # === GDP CHART (Matplotlib) ===
            most_recent_year = df_gdp['year'].max()
            df_latest_year = df_gdp[df_gdp['year'] == most_recent_year].copy()
            df_latest_year['state'] = df_latest_year['state'].str.upper().replace({'SUPRA': 'PUTRAJAYA'})
            df_latest_year = df_latest_year.sort_values(by='gdp_total', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6.5))
            fig.subplots_adjust(bottom=0.2, top=0.9) # Menambah ruang di atas dan bawah

            df_latest_year_sorted = df_latest_year.sort_values(by='gdp_total', ascending=False)
            states = df_latest_year_sorted['state'].apply(lambda x: state_names_dict.get(x, x))
            gdp_values = df_latest_year_sorted['gdp_total']

            cmap = plt.cm.cool
            norm = plt.Normalize(vmin=0, vmax=len(states))
            bar_colors = [cmap(norm(i)) for i in range(len(states))]

            ax.bar(states, gdp_values, color=bar_colors)
            ax.set_ylabel('GDP (RM MILLION)')
            ax.set_title(f'GROSS DOMESTIC PRODUCT IN {most_recent_year}')
            
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels(states, rotation=90, fontsize=8)
            
            # --- KOD BARU UNTUK BENDERA DAN TEKS ---
            for i, state in enumerate(df_latest_year_sorted['state']):
                flag_path = flags_dict.get(state)
                if flag_path and os.path.exists(flag_path):
                    try:
                        img = Image.open(flag_path)
                        imagebox = OffsetImage(img, zoom=0.1)
                        
                        # Laraskan kedudukan bendera di atas bar
                        # Guna kedudukan y yang lebih tinggi untuk mengelakkan bertindih
                        ab = AnnotationBbox(imagebox, (i, gdp_values.iloc[i] + 70000), 
                                            frameon=False, pad=0.1, box_alignment=(0.5, 0.0))
                        ax.add_artist(ab)
                    except Exception as e:
                        print(f"Error loading flag for {state}: {e}")
            
            for i, val in enumerate(gdp_values):
                # Laraskan kedudukan teks supaya tidak bertindih dengan bendera
                ax.text(i, val + 15000, f"{val:,.2f}", ha='center', va='bottom', fontsize=6, color='black')
            
            # Tambah ruang kosong di bahagian atas paksi Y
            ax.set_ylim(top=df_latest_year_sorted['gdp_total'].max() * 1.25)

            ax.set_facecolor((1.0, 1.0, 1.0, 0.5))
            fig.set_facecolor((1.0, 1.0, 1.0, 0.8))
            st.pyplot(fig)

        with col_pop:
            # === POPULATION CHART ===
            df_pop_sorted = df_pop.sort_values(by="population", ascending=False)
            df_pop_sorted['percentage'] = df_pop_sorted['population'] / df_pop_sorted['population'].sum() * 100
            df_pop_sorted['state_full_name'] = df_pop_sorted['state'].apply(lambda s: state_names_dict.get(s, s))
            
            text_positions = ['inside' if p > 5 else 'outside' for p in df_pop_sorted['percentage']]
            text_colors = ['white' if p > 5 else 'black' for p in df_pop_sorted['percentage']]
            
            fig_pop = px.pie(
                df_pop_sorted, names="state_full_name", values="population", 
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_pop.update_traces(
                textinfo='label+percent',
                textposition=text_positions,
                textfont=dict(color=text_colors, size=12),
                texttemplate='%{label}<br>%{value:,.0f}',
                hovertemplate="<b>%{label}</b><br>Population: %{value:,.0f}<extra></extra>",
                marker=dict(line=dict(color='#FFFFFF', width=1))
            )
            fig_pop.update_layout(
                title=dict(text="POPULATION DISTRIBUTION BY STATE (IN THOUSANDS)",
                    font=dict(size=14, color='black', family="Arial, sans-serif"),
                    x=0.2
                ),
                yaxis=dict(
                    title=dict(text='', font=dict(size=14, color='#2c3e50')),
                    tickfont=dict(size=12, color='black'), 
                    automargin=True
                ),
                showlegend=False,
                plot_bgcolor="rgba(255,255,255,0.9)", paper_bgcolor="rgba(255,255,255,0.9)",
                height=500, font=dict(family="Arial, sans-serif", size=12),
                margin=dict(l=10, r=105, t=80, b=30)
            )
            st.plotly_chart(fig_pop, use_container_width=True)


        # --- SECOND ROW: WELLBEING & UNEMPLOYMENT ---
        col_wellbeing, col_unemp = st.columns(2)

        with col_wellbeing:
            if df_wellbeing.empty:
                st.warning("⚠️ Wellbeing data is not available.")
            else:
                def get_min_max_states(data):
                    min_state = data.loc[data['economic_wellbeing'].idxmin()]
                    max_state = data.loc[data['economic_wellbeing'].idxmax()]
                    return min_state, max_state

                min_state_info, max_state_info = get_min_max_states(df_wellbeing)
            
                df_sorted = df_wellbeing.sort_values(by='economic_wellbeing', ascending=False)
                bar_chart = px.bar(df_sorted, x='economic_wellbeing', y='state', orientation='h',
                                 color='economic_wellbeing', color_continuous_scale='Viridis',
                                 labels={'economic_wellbeing': 'Index Score', 'state': 'State'},
                                 text='economic_wellbeing', title="ECONOMIC WELLBEING INDEX BY STATE")
                bar_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                bar_chart.update_layout(uniformtext_minsize=2, uniformtext_mode='hide', title_x=0.2,
                                         margin=dict(l=150, r=100, t=50, b=20),
                                         xaxis=dict(range=[0, df_wellbeing['economic_wellbeing'].max() * 1.3]))
                st.plotly_chart(bar_chart, use_container_width=True)

                st.markdown("""
                <style>
                div[data-testid="stMetricLabel"] > div { font-size: 12px; }
                div[data-testid="stMetricValue"] { font-size: 20px; }
                div[data-testid="stMetricDelta"] > div { font-size: 12px; }
                </style>
                """, unsafe_allow_html=True)

                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric(label="State with the Highest Index", value=f"{max_state_info['state']}", delta=f"{max_state_info['economic_wellbeing']:.2f}")
                with metrics_col2:
                    st.metric(label="State with the Lowest Index", value=f"{min_state_info['state']}", delta=f"{min_state_info['economic_wellbeing']:.2f}", delta_color="inverse")
            
        with col_unemp:
            # === UNEMPLOYMENT CHART ===
            unemployed_column_name = 'unemployment_rate'
            df_total_unemployed = df_unemp[df_unemp['sex'] == 'both'].groupby('state')[unemployed_column_name].sum().reset_index()
            df_total_unemployed = df_total_unemployed.sort_values(by=unemployed_column_name, ascending=False)
            df_total_unemployed['state_full_name'] = df_total_unemployed['state'].apply(lambda s: state_names_dict.get(s, s))
            
            fig_lollipop = go.Figure()
            fig_lollipop.add_trace(go.Scatter(x=df_total_unemployed['state_full_name'], y=df_total_unemployed[unemployed_column_name],
                                             mode='lines', line=dict(color='rgba(0,176,246,0.5)', width=3), hoverinfo='skip', showlegend=False))
            fig_lollipop.add_trace(go.Scatter(x=df_total_unemployed['state_full_name'], y=df_total_unemployed[unemployed_column_name],
                                             mode='markers+text', marker=dict(size=15, color=df_total_unemployed[unemployed_column_name],
                                             colorscale='Viridis', showscale=False, line=dict(color='white', width=3)),
                                             text=df_total_unemployed[unemployed_column_name].apply(lambda x: f"{x:,.0f}"),
                                             textposition='top center', textfont=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
                                             hovertemplate="<b>%{x}</b><br>Total Unemployed: %{y:,.0f}<extra></extra>", showlegend=False))
            fig_lollipop.update_layout(title=dict(text="TOTAL UNEMPLOYED WORKFORCE", font=dict(size=14, color='black', family="Arial, sans-serif"), x=0.3),
                                     xaxis=dict(title=dict(text='State', font=dict(size=14, color='#2c3e50')), tickangle=90, tickfont=dict(size=12, color='black'),
                                             gridcolor='rgba(0,0,0,0.1)', linecolor='#2c3e50'),
                                     yaxis=dict(title=dict(text='Total Unemployed (in Thousands)', font=dict(size=14, color='#2c3e50')),
                                             tickfont=dict(size=12, color='#2c3e50'), gridcolor='rgba(0,0,0,0.1)', linecolor='#2c3e50'),
                                     plot_bgcolor="rgba(255,255,255,0.9)", paper_bgcolor="rgba(255,255,255,0.9)",
                                     height=450, font=dict(family="Arial, sans-serif", size=12), margin=dict(l=2, r=40, t=80, b=10))
            st.plotly_chart(fig_lollipop, use_container_width=True)


# --- TAB 2: ECONOMIC COMPARISONS AND CHATBOT ---
with tab2:
    df_snapshot = combined_df[combined_df['year'] == 2022].copy()
    df_snapshot = df_snapshot.fillna(0)
    df_snapshot = df_snapshot.sort_values(by='state')
    df_snapshot['state_full_name'] = df_snapshot['state'].apply(lambda s: state_names_dict.get(s, s))

    comparison = st.selectbox("Select Comparison:", [
        "Total GDP vs. Population",
        "Total GDP vs. Unemployment Rate",
        "GDP per Capita vs. Unemployment Rate"
    ])
    
    findings = ""
    suggestions = ""
    fig = go.Figure()

    if comparison == "Total GDP vs. Population":
        findings = "Note the relationship between population and GDP. States with larger populations typically contribute to higher GDP."
        suggestions = "This shows great potential for these states to drive the nation's economic growth."
        fig.add_trace(go.Bar(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['gdp_total'],
            name='Total GDP (RM Million)',
            marker_color='#4C72B0',  # A professional, calming blue
            offsetgroup=0
        ))
        fig.add_trace(go.Bar(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['population'],
            name='Population',
            marker_color='#E46E2B',  # A contrasting, warm orange/red
            offsetgroup=1,
            yaxis='y2'
        ))
        fig.update_layout(
            title="Total GDP vs. Population (2022)",
            xaxis_title="State",
            yaxis=dict(
                title=dict(text="Total GDP (RM Million)", font=dict(color="#4C72B0")),  
                tickfont=dict(color="#4C72B0"),
                tickformat=",.0f"
            ),
            yaxis2=dict(
                title=dict(text="Population", font=dict(color="#E46E2B")),  
                tickfont=dict(color="#E46E2B"),
                overlaying="y",
                side="right",
                tickformat=",.0f"
            )
        )
            
    elif comparison == "Total GDP vs. Unemployment Rate":
        findings = "This analysis shows whether a high GDP contributes to a lower unemployment rate."
        suggestions = "While the trend is clear, other factors such as industry type and labor market need to be considered."

        # Line scatter chart for Total GDP
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['gdp_total'],
            name='Total GDP (RM Million)',
            mode='lines+markers',
            line=dict(color='#781fb4', width=4), # A deep purple line
            marker=dict(color='#781fb4', size=8),
            yaxis='y1'
        ))

        # Line scatter chart for Unemployment Rate
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['unemployment_rate'],
            name='Unemployment Rate (%)',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#035720', width=4), # A dark green line
            marker=dict(color='#035720', size=8)
        ))

        fig.update_layout(
            title="Total GDP vs. Unemployment Rate (2022)",
            xaxis_title="State",
            yaxis=dict(
                title=dict(text="Total GDP (RM Million)", font=dict(color="#781fb4")),
                tickfont=dict(color="#781fb4"),
                tickformat=",.0f"
            ),
            yaxis2=dict(
                title=dict(text="Unemployment Rate (%)", font=dict(color="#035720")),
                tickfont=dict(color="#035720"),
                overlaying="y",
                side="right"
            )
        )
    elif comparison == "GDP per Capita vs. Unemployment Rate":
        findings = "This comparison provides a more detailed picture of individual wellbeing and its connection to job opportunities."
        suggestions = "Continuing to invest in high-tech and TVET sectors is important to maintain this trend."

        # Line scatter chart for GDP per Capita
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['gdp_per_capita'],
            name='GDP per Capita (RM)',
            mode='lines+markers',
            line=dict(color='#33a02c', width=4), # A clear green line
            marker=dict(color='#33a02c', size=8),
            yaxis='y1'
        ))

        # Line scatter chart for Unemployment Rate
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['unemployment_rate'],
            name='Unemployment Rate (%)',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#e31a1c', width=4), # A contrasting red line
            marker=dict(color='#e31a1c', size=8)
        ))

        fig.update_layout(
            title="GDP per Capita vs. Unemployment Rate (2022)",
            xaxis_title="State",
            yaxis=dict(
                title=dict(text="GDP per Capita (RM)", font=dict(color="#33a02c")),
                tickfont=dict(color="#33a02c"),
                tickformat=",.0f"
            ),
            yaxis2=dict(
                title=dict(text="Unemployment Rate (%)", font=dict(color="#e31a1c")),
                tickfont=dict(color="#e31a1c"),
                overlaying="y",
                side="right"
            )
        )
    # Common layout updates for all comparison charts
    fig.update_layout(
        legend=dict(x=0, y=1.1, orientation="h"),
        plot_bgcolor="rgba(255,255,255,0.9)", paper_bgcolor="rgba(255,255,255,0.9)",
        margin=dict(l=40, r=40, t=80, b=120),
        xaxis=dict(tickangle=-45, tickfont=dict(size=11, color='black'))
    )

    st.markdown(f'<div class="translucent-box"><h4>📌 Findings</h4><p>{findings}</p></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="translucent-box"><h4>💡 Suggestions</h4><p>{suggestions}</p></div>', unsafe_allow_html=True)
    
    st.subheader("🤖 Bit4Bit Chatbot")
    user_q = st.text_input("Ask a question (e.g., Johor's GDP trend 2022–2023)")
    if st.button("Submit Question"):
        if user_q.strip():
            with st.spinner("Generating answer..."):
                jawapan = ask_groq(user_q)
            st.write("*Chartbot Answer:*")
            st.write(jawapan)
        else:
            st.warning("Please enter a question first.")

