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
        'JOHOR': 'bendera/JOHOR.png',
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

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
            encoded_string = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        return None


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
            # === GDP CHART (Plotly) ===
            most_recent_year = df_gdp['year'].max()
            df_latest_year = df_gdp[df_gdp['year'] == most_recent_year].copy()
            df_latest_year['state_full_name'] = df_latest_year['state'].apply(lambda x: state_names_dict.get(x, x))
            df_latest_year = df_latest_year.sort_values(by='gdp_total', ascending=False)
            
            # Create a list of colors for the bars
            gdp_values = df_latest_year['gdp_total']
            color_values = np.arange(len(gdp_values))
            cmap = plt.cm.get_cmap('cool', len(gdp_values))
            colors = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in cmap(color_values)]
            
            fig_gdp = go.Figure()

            # Add the bar chart trace
            fig_gdp.add_trace(go.Bar(
                x=df_latest_year['state_full_name'],
                y=gdp_values,
                text=gdp_values.apply(lambda x: f"{x:,.2f}"),
                textposition='outside',
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>GDP: %{y:,.2f} RM Million<extra></extra>"
            ))
            
            # Add state flags as layout images
            for i, state in enumerate(df_latest_year['state']):
                flag_path = flags_dict.get(state)
                if flag_path and os.path.exists(flag_path):
                    try:
                        encoded_flag = get_img_as_base64(flag_path)
                        if encoded_flag:
                            fig_gdp.add_layout_image(
                                dict(
                                    source=encoded_flag,
                                    xref="x",
                                    yref="y",
                                    x=df_latest_year['state_full_name'].iloc[i],
                                    y=gdp_values.iloc[i] + 70000, # Position the flag above the text
                                    sizex=0.5, # Adjust size
                                    sizey=0.5,
                                    xanchor="center",
                                    yanchor="middle"
                                )
                            )
                    except Exception as e:
                        print(f"Error processing flag for {state}: {e}")

            # Update layout for a cleaner look
            fig_gdp.update_layout(
                title=dict(text=f"GROSS DOMESTIC PRODUCT IN {most_recent_year}", font=dict(size=14, color='black'), x=0.5),
                xaxis=dict(
                    title_text="",
                    tickangle=-45,
                    tickfont=dict(size=11, color='black'),
                    automargin=True
                ),
                yaxis=dict(
                    title_text="GDP (RM MILLION)",
                    tickformat=",.0f",
                    tickfont=dict(size=11, color='black')
                ),
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(255,255,255,0.9)",
                height=450,
                showlegend=False,
                hovermode="x unified"
            )

            st.plotly_chart(fig_gdp, use_container_width=True)


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
                height=450, font=dict(family="Arial, sans-serif", size=12),
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

