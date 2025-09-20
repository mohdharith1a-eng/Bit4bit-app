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

# Suppress warnings from pandas and other libraries for cleaner output
warnings.filterwarnings('ignore')
px.defaults.template = "plotly_white"

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

# ================== CHARTBOT FUNCTION ==================
def ask_ollama(prompt):
    """Sends a prompt to the Ollama API and returns the response."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt},
            stream=True,
            timeout=120
        )
        if response.status_code == 200:
            output = ""
            for line in response.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        if "response" in obj:
                            output += obj["response"]
                    except:
                        pass
            return output.strip() if output else "❌ No answer from Chatbot."
        else:
            return f"⚠️ Chartbot API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ Chartbot Error: {e}"

# ================== BACKGROUND IMAGE + GRADIENT ==================
def add_bg_from_local(image_file):
    """Adds a local background image with a gradient overlay to the app."""
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

# ================== PATHS & DATA LOADING ==================
# Before
gdp_path = r"https://github.com/mohdharith1a-eng/Bit4bit-app/blob/main/datasets/gdp_state_.csv"
unemp_path = r"https://github.com/mohdharith1a-eng/Bit4bit-app/blob/main/datasets/unemployed.csv"
pop_path = r"https://github.com/mohdharith1a-eng/Bit4bit-app/blob/main/datasets/population_state.csv"
wellbeing_path = r"https://github.com/mohdharith1a-eng/Bit4bit-app/blob/main/datasets/economic_wellbeing.csv"

# After
gdp_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/gdp_state_.csv"
unemp_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/unemployed.csv"
pop_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/population_state.csv"
wellbeing_path = r"https://raw.githubusercontent.com/mohdharith1a-eng/Bit4bit-app/main/datasets/economic_wellbeing.csv"
flag_folder_path = r"C:\Users\Acer\OneDrive\Desktop\Apps3\bendera"

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

@st.cache_data
def get_flags_dict():
    flags_dict = {}
    if os.path.isdir(flag_folder_path):
        for filename in os.listdir(flag_folder_path):
            state_name = os.path.splitext(filename)[0].replace("_", " ").title()
            flags_dict[state_name] = os.path.join(flag_folder_path, filename)
    return flags_dict

# ================== STREAMLIT APP LAYOUT ==================
st.title("📊 BRIDGING THE GDP-GPI GAP: MALAYSIAN ECONOMIC DASHBOARD")

# Load data
df_gdp, df_unemp, df_pop, df_wellbeing, combined_df = load_all_data()
state_names_dict = get_state_names_dict()
flags_dict = get_flags_dict()

# Add background image for App Page
add_bg_from_local("background.png")

# Create tabs
tab1, tab2= st.tabs(["📈 Infographic Dashboard for RAW data DOSM", "🔍 Comparisons dataset & Solutions"])

# --- TAB 1: Infographics ---
with tab1:
    if df_gdp is not None and df_unemp is not None and df_wellbeing is not None and df_pop is not None:
        
        # === FIRST ROW: GDP CHART & POPULATION CHART ===
        col_gdp, col_pop = st.columns(2)

        with col_gdp:
            # 4. Visualization: GDP Bar Chart with Flags
            # ----------------------------------------------------------------------------------
            # 1. Data loading: Data is loaded from CSV files via load_all_data().
            # 2. Data cleaning: Performed by clean_data() function during data loading.
            # 3. Data pre-processing and Features:
            #    - Filter the dataframe to include only the most recent year's data.
            #    - Sort the data by GDP in descending order for ranking.
            # 4. Visualization:
            #    - Create a Matplotlib bar chart to display the GDP of each state.
            #    - Use OffsetImage and AnnotationBbox to add state flags at the bottom of each bar.
            #    - Label the y-axis with 'GDP (RM billion)' and set the title.
            #    - Add data labels on top of each bar for clarity.
            # ----------------------------------------------------------------------------------
            # === GDP CHART (Matplotlib) ===
            most_recent_year = df_gdp['year'].max()
            df_latest_year = df_gdp[df_gdp['year'] == most_recent_year].copy()
            df_latest_year['state'] = df_latest_year['state'].str.upper().replace({'SUPRA': 'PUTRAJAYA'})
            df_latest_year = df_latest_year.sort_values(by='gdp_total', ascending=False)
            
            # --- Figure Size Change here ---
            fig, ax = plt.subplots(figsize=(10, 6.5))
            # --- End of Change ---
            
            fig.subplots_adjust(bottom=0.2)
            
            df_latest_year_sorted = df_latest_year.sort_values(by='gdp_total', ascending=False)
            states = df_latest_year_sorted['state'].apply(lambda x: state_names_dict.get(x, x))
            gdp_values = df_latest_year_sorted['gdp_total']

            cmap = plt.cm.cool
            norm = plt.Normalize(vmin=0, vmax=len(states))
            bar_colors = [cmap(norm(i)) for i in range(len(states))]

            ax.bar(states, gdp_values, color=bar_colors)
            ax.set_ylabel('GDP (RM MILLION)')
            ax.set_title(f'GROSS DOMESTIC PRODUCT IN 2022-2023')
            
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels([])

            for i, state in enumerate(states):
                normalized_state_name = state.title()
                if normalized_state_name in flags_dict:
                    try:
                        img_path = flags_dict[normalized_state_name]
                        img = Image.open(img_path)
                        img.thumbnail((25, 25), Image.Resampling.LANCZOS)
                        imagebox = OffsetImage(img, zoom=1)
                        ab = AnnotationBbox(imagebox, (i, 0), xybox=(0, -20), xycoords='data', boxcoords="offset points", frameon=False)
                        ax.add_artist(ab)
                    except Exception as e:
                        st.warning(f"Could not load flag for {state}: {e}")

                ax.text(i, -0.01 * df_latest_year_sorted['gdp_total'].max(), state, ha='center', va='top', fontsize=6, rotation=0, color='black')
                ax.text(i, gdp_values.iloc[i] + 0.01 * df_latest_year_sorted['gdp_total'].max(), f"{gdp_values.iloc[i]:,.2f}", ha='center', va='bottom', fontsize=6, color='black')
            
            ax.set_facecolor((1.0, 1.0, 1.0, 0.5))
            fig.set_facecolor((1.0, 1.0, 1.0, 0.8))
            st.pyplot(fig)

        with col_pop:
            # 4. Visualization: Population Pie Chart
            # ----------------------------------------------------------------------------------
            # 1. Data loading: Data is loaded from CSV files via load_all_data().
            # 2. Data cleaning: Performed by clean_data() function during data loading.
            # 3. Data pre-processing and Features:
            #    - Sort the data by population in descending order.
            #    - Calculate the percentage of the total population for each state.
            #    - Map the state codes to their full names.
            #    - Determine the text position for labels based on percentage size.
            # 4. Visualization:
            #    - Create a Plotly pie chart to show the distribution of population by state.
            #    - Customize the text info to display both the label and percentage.
            #    - Add a detailed hover template for a better user experience.
            # ----------------------------------------------------------------------------------
            # --- POPULATION CHART ---
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
                    x=0.2 # Menempatkan tajuk di tengah
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
            # 4. Visualization: Economic Wellbeing Bar Chart and Metrics
            # ----------------------------------------------------------------------------------
            # 1. Data loading: The data is loaded from a specific CSV file.
            # 2. Data cleaning: The clean_data() function is used to ensure data types are correct.
            # 3. Data pre-processing and Features:
            #    - The dataframe is sorted by the wellbeing_value in ascending order.
            #    - Helper function get_min_max_states identifies the states with the highest and lowest scores.
            # 4. Visualization:
            #    - A Plotly horizontal bar chart is created to show the economic wellbeing index for each state.
            #    - Streamlit st.metric is used to display the states with the highest and lowest index values for a quick overview.
            # ----------------------------------------------------------------------------------
            # --- ECONOMIC WELLBEING CHART ---
            DATA_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
            CSV_FILE = os.path.join(DATA_PATH, r"https://github.com/mohdharith1a-eng/Bit4bit-app/blob/main/datasets/economic_wellbeing.csv")

            try:
                df_data = load_csv_safe(CSV_FILE, "Economic Wellbeing Data")
            except Exception as e:
                st.error("Error: Could not load the 'economic_wellbeing.csv' file.")
                st.stop()

            def get_min_max_states(data):
                min_state = data.loc[data['wellbeing_value'].idxmin()]
                max_state = data.loc[data['wellbeing_value'].idxmax()]
                return min_state, max_state

            min_state_info, max_state_info = get_min_max_states(df_data)

            df_sorted = df_data.sort_values(by='wellbeing_value', ascending=True)
            bar_chart = px.bar(df_sorted, x='wellbeing_value', y='State', orientation='h',
                               color='wellbeing_value', color_continuous_scale='Viridis',
                               labels={'wellbeing_value': 'Index Score', 'State': 'State'},
                               text='wellbeing_value', title="ECONOMIC WELLBEING INDEX BY STATE")
            bar_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            bar_chart.update_layout(uniformtext_minsize=2, uniformtext_mode='hide', title_x=0.2,
                                     margin=dict(l=150, r=100, t=50, b=20),
                                     xaxis=dict(range=[0, df_data['wellbeing_value'].max() * 1.3]))
            st.plotly_chart(bar_chart, use_container_width=True)

            # Metrics
            st.markdown("""
            <style>
            div[data-testid="stMetricLabel"] > div { font-size: 12px; }
            div[data-testid="stMetricValue"] { font-size: 20px; }
            div[data-testid="stMetricDelta"] > div { font-size: 12px; }
            </style>
            """, unsafe_allow_html=True)

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric(label="State with the Highest Index", value=f"{max_state_info['State']}", delta=f"{max_state_info['wellbeing_value']:.2f}")
            with metrics_col2:
                st.metric(label="State with the Lowest Index", value=f"{min_state_info['State']}", delta=f"{min_state_info['wellbeing_value']:.2f}", delta_color="inverse")
        
        with col_unemp:
            # 4. Visualization: Unemployment Lollipop Chart
            # ----------------------------------------------------------------------------------
            # 1. Data loading: Data is loaded from CSV files via load_all_data().
            # 2. Data cleaning: Performed by clean_data() function during data loading.
            # 3. Data pre-processing and Features:
            #    - Filter the unemployment data for sex == 'both'.
            #    - Group by state and sum the unemployment numbers to get a total count.
            #    - Sort the data in descending order for a clear ranking.
            #    - Map the state codes to their full names.
            #    - Create a Plotly 'lollipop' chart by combining a Scatter trace with mode='lines' for the vertical line and another Scatter trace with mode='markers+text' for the data points and labels.
            #    - This provides a clear and visually appealing representation of unemployment totals per state.
            # ----------------------------------------------------------------------------------
            # --- UNEMPLOYMENT CHART ---
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







# ================== TAB 2: ECONOMIC COMPARISONS ==================
with tab2:
    st.header("📊 Economic Comparisons")

    comparison = st.selectbox(
        "Select comparison",
        [
            "GDP vs. Population",
            "Total GDP vs. Unemployment Rate",
            "GDP per Capita vs. Unemployment Rate"
        ]
    )

    # Prepare data for comparison tab
    df_snapshot = combined_df[combined_df['year'] == 2022].copy()
    df_snapshot['total_gdp_billion'] = df_snapshot['gdp_total'] / 1_000_000 # Convert to billions for better readability
    df_snapshot['population_million'] = df_snapshot['population'] / 1_000 # Convert to millions
    df_snapshot['state_full_name'] = df_snapshot['state'].apply(lambda s: state_names_dict.get(s, s))

    # Prepare chart
    fig = go.Figure()

    if comparison == "GDP vs. Population":
        findings = "This chart compares the total GDP of each state against its population, showing how population size relates to economic output."
        suggestions = "States with higher population but lower GDP may need strategies to boost productivity. Smaller states with high GDP show potential economic efficiency."

        fig.add_trace(go.Bar(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['total_gdp_billion'],
            name='GDP (Billion RM)',
            marker_color='#2980B9',
            offsetgroup=0
        ))
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['population_million'],
            name='Population (Million)',
            mode='lines+markers',
            marker=dict(color='#E67E22', size=8),
            line=dict(color='#E67E22', width=2),
            yaxis='y2'
        ))
        fig.update_layout(
            title="GDP vs. Population (2022)",
            xaxis_title="State",
            yaxis=dict(
                title=dict(text="GDP (Billion RM)", font=dict(color="#2980B9")),
                tickfont=dict(color="#2980B9")
            ),
            yaxis2=dict(
                title=dict(text="Population (Million)", font=dict(color="#E67E22")),
                tickfont=dict(color="#E67E22"),
                overlaying="y",
                side="right"
            )
        )

    elif comparison == "Total GDP vs. Unemployment Rate":
        findings = "This chart shows how each state's total GDP relates to unemployment rates. It highlights whether economic size translates into better employment opportunities."
        suggestions = "Large GDP states with high unemployment may need more inclusive policies. Smaller GDP states with low unemployment can be models of efficiency."

        fig.add_trace(go.Bar(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['total_gdp_billion'],
            name='GDP (Billion RM)',
            marker_color='#8E44AD',
            offsetgroup=0
        ))
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['unemployment_rate'],
            name='Unemployment Rate (%)',
            mode='lines+markers',
            marker=dict(color='#C0392B', size=8),
            line=dict(color='#C0392B', width=2),
            yaxis='y2'
        ))
        fig.update_layout(
            title="Total GDP vs. Unemployment Rate (2022)",
            xaxis_title="State",
            yaxis=dict(
                title=dict(text="GDP (Billion RM)", font=dict(color="#8E44AD")),
                tickfont=dict(color="#8E44AD"),
                tickformat=",.0f"
            ),
            yaxis2=dict(
                title=dict(text="Unemployment Rate (%)", font=dict(color="#C0392B")),
                tickfont=dict(color="#C0392B"),
                overlaying="y",
                side="right"
            )
        )

    elif comparison == "GDP per Capita vs. Unemployment Rate":
        findings = "This chart shows whether states with higher GDP per capita experience lower unemployment rates. It highlights economic well-being relative to population size and job creation."
        suggestions = "Encourage states to leverage high GDP per capita into inclusive employment. For states with low GDP per capita but high unemployment, consider targeted development and job creation programs."

        fig.add_trace(go.Bar(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['gdp_per_capita'],
            name='GDP per Capita (RM)',
            marker_color='#27AE60',
            offsetgroup=0
        ))
        fig.add_trace(go.Scatter(
            x=df_snapshot['state_full_name'],
            y=df_snapshot['unemployment_rate'],
            name='Unemployment Rate (%)',
            mode='lines+markers',
            marker=dict(color='#C0392B', size=8),
            line=dict(color='#C0392B', width=2),
            yaxis='y2'
        ))
        fig.update_layout(
            title="GDP per Capita vs. Unemployment Rate (2022)",
            xaxis_title="State",
            yaxis=dict(
                title=dict(text="GDP per Capita (RM)", font=dict(color="#27AE60")),
                tickfont=dict(color="#27AE60"),
                tickformat=",.0f"
            ),
            yaxis2=dict(
                title=dict(text="Unemployment Rate (%)", font=dict(color="#C0392B")),
                tickfont=dict(color="#C0392B"),
                overlaying="y",
                side="right"
            )
        )

    # Show chart and insights
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Findings")
    st.write(findings)
    st.subheader("Suggestions")
    st.write(suggestions)

