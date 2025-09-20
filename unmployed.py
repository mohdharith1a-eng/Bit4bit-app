import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

# --- Function to load data ---
@st.cache_data
def load_data():
    """
    Loads and processes unemployment data, including the state column.
    """
    file_path = r"C:\Users\arifs\OneDrive\Desktop\Apps3\datasets\unemployed.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Please make sure the dataset path is correct.")
        return None

    df.dropna(inplace=True)
    
    if 'date' not in df.columns or 'state' not in df.columns or 'lf_unemployed' not in df.columns:
        st.error("The dataset must have 'date', 'state', and 'lf_unemployed' columns.")
        return None
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df = df[df['year'].isin([2022, 2023])]
    
    return df

# --- Main Dashboard ---
st.title("Unemployed Workforce Dashboard: Comparison and Trends")
st.markdown("---")

df_unemployed = load_data()

if df_unemployed is not None:
    unemployed_column_name = 'lf_unemployed'
    
    # Check if the required column exists before proceeding with charts
    if unemployed_column_name not in df_unemployed.columns:
        st.error(f"The column '{unemployed_column_name}' was not found. Please check the column name in your dataset.")
    else:
        # Create two columns for the charts
        col1 = st.columns(2)

        # --- First Chart: Vertical Lollipop Chart ---
    
        st.header("Total Unemployed by State")
            
        # Aggregate data to get the total unemployment per state
        df_total_unemployed = df_unemployed.groupby('state')[unemployed_column_name].sum().reset_index()
        df_total_unemployed = df_total_unemployed.sort_values(by=unemployed_column_name, ascending=False)
            
        # Create the Lollipop chart with enhanced, exciting style
        fig_lollipop = go.Figure()

        # Add the lines (vertical lines)
        fig_lollipop.add_trace(go.Scatter(
            x=df_total_unemployed['state'],
            y=df_total_unemployed[unemployed_column_name],
            mode='lines',
            line=dict(color='rgba(0,176,246,0.25)', width=10),
            marker=dict(size=0),
            hoverinfo='skip',
            showlegend=False
        ))

        # Add the glowing circles/points at the end of the lines
        fig_lollipop.add_trace(go.Scatter(
            x=df_total_unemployed['state'],
            y=df_total_unemployed[unemployed_column_name],
            mode='markers+text',
            marker=dict(
                color=df_total_unemployed[unemployed_column_name],
                colorscale='Viridis',
                size=28,
                line=dict(color='white', width=4),
                showscale=True,
                colorbar=dict(title='Unemployed', thickness=18, x=1.05, y=0.5, len=0.7, outlinewidth=0),
                opacity=0.95,
                symbol='circle'
            ),
            text=df_total_unemployed[unemployed_column_name].apply(lambda x: f"{x:,.0f}"),
            textposition='top center',
            textfont=dict(family='Montserrat, Segoe UI', size=18, color='gold'),
            hovertemplate="<b>%{x}</b><br>Total Unemployed: %{y:,.0f}<extra></extra>",
            showlegend=False
        ))

        fig_lollipop.update_layout(
            title='<b>Total Unemployed Workforce (2022-2023)</b><br><span style="font-size:18px;color:#00B0F6;font-family:Montserrat;">Malaysia State Comparison</span>',
            xaxis=dict(
                title='<b>State</b>',
                tickangle=45,
                tickfont=dict(size=16, color='white', family='Montserrat'),
                showgrid=False,
                linecolor='#00B0F6',
                linewidth=2
            ),
            yaxis=dict(
                title=dict(
                    text='<b>Total Unemployed</b>',
                    font=dict(size=18, color='gold', family='Montserrat')
                ),
                gridcolor='rgba(0,176,246,0.10)',
                zeroline=False,
                tickfont=dict(size=16, color='white', family='Montserrat')
            ),
            plot_bgcolor='#181C25',
            paper_bgcolor='#181C25',
            margin=dict(l=40, r=40, t=110, b=180),
            font=dict(family='Montserrat, Segoe UI', size=17, color='white'),
            height=650,
            transition=dict(duration=800, easing='cubic-in-out'),
            hoverlabel=dict(bgcolor='#00B0F6', font_size=16, font_family='Montserrat'),
        )
            
        st.plotly_chart(fig_lollipop, use_container_width=True)


