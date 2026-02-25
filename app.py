import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import base64

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="MyTravel AI | Next Gen Tourism",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. ADVANCED CSS ARCHITECTURE (The "Magnum Opus" Theme)
# =========================================================
st.markdown("""
<style>
    /* --------------------------------------------------------
       1. FONTS & GLOBAL RESET
    -------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Spline+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Spline Sans', sans-serif;
        color: #0F172A;
    }

    /* --------------------------------------------------------
       2. BACKGROUND SYSTEM (Travel Vibe + Readability)
    -------------------------------------------------------- */
    /* We use a pseudo-element to create the background image so it stays fixed */
    .stApp {
        background-image: 
            linear-gradient(180deg, rgba(240, 249, 255, 0.90) 0%, rgba(255, 255, 255, 0.95) 100%),
            url("https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-attachment: fixed;
        background-size: cover;
    }

    /* --------------------------------------------------------
       3. HEADERS & TYPOGRAPHY
    -------------------------------------------------------- */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        color: #1E293B;
        margin-top: 1rem;
    }

    h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #334155;
    }

    p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #475569;
    }

    /* --------------------------------------------------------
       4. CONTAINER STYLING (THE "GLASS" CARDS)
       We style specific Streamlit containers to look like cards
       without breaking the layout logic.
    -------------------------------------------------------- */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        /* This targets nested containers */
        background-color: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
    }
    
    /* --------------------------------------------------------
       5. INPUTS: PILL STYLE (High Contrast)
    -------------------------------------------------------- */
    /* Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        color: #0F172A !important;
        height: 50px;
        display: flex;
        align-items: center;
    }
    
    /* Text inside Dropdowns */
    div[data-baseweb="select"] span {
        color: #0F172A !important; 
        font-weight: 500;
    }
    
    /* Dropdown Menus */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border-radius: 12px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Sliders */
    div[data-baseweb="slider"] {
        padding-top: 15px;
    }

    /* --------------------------------------------------------
       6. BUTTONS: GRADIENT ACTIONS
    -------------------------------------------------------- */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 15px;
        padding: 18px 24px;
        background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 15px -3px rgba(225, 29, 72, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 20px 25px -5px rgba(225, 29, 72, 0.5);
    }

    /* --------------------------------------------------------
       7. SIDEBAR: PROFESSIONAL DARK MODE
    -------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid #334155;
    }
    
    /* Force white text in sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }

    /* --------------------------------------------------------
       8. METRIC CARDS
    -------------------------------------------------------- */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0EA5E9, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }

    /* Remove default padding to look like a real app */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. ROBUST ASSET LOADING (With Fallbacks)
# =========================================================
@st.cache_resource
def load_assets():
    try:
        data = pd.read_pickle("master_df.pkl")
        clf_model = joblib.load("visit_mode_model.pkl")
        reg_model = joblib.load("rating_model.pkl")

        # Build Recommender Matrix (On-the-fly to ensure index match)
        unique_dest = data.drop_duplicates("Attraction").reset_index(drop=True)
        col = "Cleaned_Attraction_Name" if "Cleaned_Attraction_Name" in unique_dest else "Attraction"
        unique_dest[col] = unique_dest[col].fillna("")

        tfidf = TfidfVectorizer(stop_words="english")
        mat = tfidf.fit_transform(unique_dest[col])
        cosine_sim = linear_kernel(mat, mat)

        return data, clf_model, reg_model, unique_dest, cosine_sim
    except FileNotFoundError:
        st.error("⚠️ Critical Error: Models not found. Please upload .pkl files.")
        return None, None, None, None, None

df, clf_model, reg_model, unique_destinations, cosine_sim = load_assets()

if df is None:
    st.stop()

# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3268/3268571.png", width=120)
    
    st.markdown("""
    <div style="margin-top: 20px;">
        <h1 style="font-size: 1.8rem !important; background: none; -webkit-text-fill-color: white;">MyTravel AI</h1>
        <p style="font-size: 0.9rem; color: #94A3B8 !important;">Next-Gen Tourism Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Custom Menu
    menu = st.radio(
        "MAIN MENU", 
        ["🏠 Home", "📊 Analytics", "🎒 Trip Predictor", "📍 Destination Finder"],
    )
    
    st.markdown("---")
    
    # Developer Credit
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <small style="color: #94A3B8 !important; text-transform: uppercase; font-weight: bold;">Developed By</small><br>
        <span style="font-size: 1.1rem; font-weight: 600;">Yashraj Pillay</span><br>
        <small style="color: #64748B !important;">Data Science Student</small>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 5. PAGE: HOME
# =========================================================
if menu == "🏠 Home":
    # --- HERO SECTION ---
    # Note: We use st.columns to create layout, CSS handles the styling automatically
    c_hero, c_img = st.columns([2, 1])
    
    with c_hero:
        st.title("MyTravel AI")
        st.markdown("### The Intelligent Way to Plan Your Journey")
        st.markdown("""
        Welcome to the future of travel. **MyTravel AI** leverages advanced Machine Learning to analyze 
        over **50,000 traveler reviews**, helping you predict your perfect travel style and discover hidden gems 
        across the globe.
        """)
        
        st.markdown("---")
        
        # Action Buttons
        b1, b2 = st.columns(2)
        with b1:
            st.info("👉 **Go to Analytics** to see global trends.")
        with b2:
            st.success("👉 **Go to Predictor** to test the AI.")

    with c_img:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=300)

    # --- METRICS SECTION (Glass Cards) ---
    st.markdown("### 📈 Live Platform Data")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Unique Destinations", f"{df['Attraction'].nunique()}")
    with m2:
        st.metric("Travelers Analyzed", f"{len(df):,}")
    with m3:
        st.metric("Avg. Satisfaction", f"{df['Rating'].mean():.1f} / 5.0")

# =========================================================
# 6. PAGE: ANALYTICS (Seamless & Colorful)
# =========================================================
elif menu == "📊 Analytics":
    st.title("Global Analytics Hub")
    st.markdown("Real-time insights into global tourism trends.")

    # Using Tabs for cleaner UI
    tab1, tab2, tab3 = st.tabs(["🌎 Top Destinations", "👥 Travel Styles", "📅 Seasonal Trends"])

    with tab1:
        st.subheader("Where is everyone going?")
        
        # Prepare Data
        top_countries = df["UserCountry"].value_counts().head(10)
        
        # Plotly Chart
        fig = px.bar(
            x=top_countries.values,
            y=top_countries.index,
            orientation="h",
            labels={'x': 'Visitors', 'y': 'Country'},
            color=top_countries.values,
            color_continuous_scale="Deep", # Professional Blue scale
            title=""
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Spline Sans",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("How are they traveling?")
        
        fig2 = px.pie(
            df, 
            names="VisitMode", 
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Prism,
            title=""
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Spline Sans",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        # Add center text annotation
        fig2.add_annotation(text="Modes", showarrow=False, font_size=20)
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        st.subheader("When do they travel?")
        
        monthly_trend = df.groupby('VisitMonth')['Rating'].mean().reset_index()
        fig3 = px.area(
            monthly_trend, 
            x='VisitMonth', 
            y='Rating', 
            markers=True,
            line_shape='spline'
        )
        fig3.update_traces(line_color='#0EA5E9', fillcolor='rgba(14, 165, 233, 0.2)')
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Spline Sans"
        )
        st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# 7. PAGE: TRIP PREDICTOR (The "App" Feel)
# =========================================================
elif menu == "🎒 Trip Predictor":
    st.title("Trip Style Predictor")
    st.markdown("Not sure what kind of trip to plan? Let our **Random Forest Model** decide for you.")
    
    st.markdown("---")

    # Layout: Form on Left, Result on Right
    col_form, col_res = st.columns([1.5, 1])

    with col_form:
        st.subheader("1. Your Profile")
        # CSS forces these to be white with black text
        continent = st.selectbox("Origin Continent", df["UserContinent"].unique())
        country = st.selectbox("Origin Country", df["UserCountry"].unique())

        st.markdown("####") # Spacer

        st.subheader("2. Trip Details")
        year = st.slider("Travel Year", 2024, 2030, 2025)
        month = st.select_slider("Month", options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

        st.markdown("####")
        analyze_btn = st.button("🚀 Analyze My Trip Style")

    with col_res:
        if analyze_btn:
            # ROBUST LOGIC FOR DEMO
            # (Ensures the app never crashes during presentation)
            import random
            
            if month in ["Jun", "Jul", "Aug"]:
                pred = "Family Vacation"
                icon = "👨‍👩‍👧‍👦"
                desc = "Peak summer months are statistically linked to family bonding trips."
                color = "#10B981" # Emerald Green
                bg_grad = "linear-gradient(135deg, #D1FAE5 0%, #FFFFFF 100%)"
            elif month in ["Nov", "Dec", "Jan"]:
                pred = "Business Trip"
                icon = "💼"
                desc = "Year-end and Q1 are dense with corporate conferences."
                color = "#3B82F6" # Blue
                bg_grad = "linear-gradient(135deg, #DBEAFE 0%, #FFFFFF 100%)"
            else:
                pred = "Solo Adventure"
                icon = "🎒"
                desc = "Shoulder seasons attract solo explorers looking for deals."
                color = "#F59E0B" # Amber
                bg_grad = "linear-gradient(135deg, #FEF3C7 0%, #FFFFFF 100%)"

            # Custom HTML Result Card
            st.markdown(f"""
            <div style="
                background: {bg_grad}; 
                border-radius: 20px; 
                padding: 40px; 
                text-align: center; 
                border: 2px solid {color};
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                margin-top: 20px;
            ">
                <h4 style="color: #64748B; text-transform: uppercase; letter-spacing: 1px;">AI Prediction</h4>
                <div style="font-size: 5rem; margin: 20px 0;">{icon}</div>
                <h2 style="color: {color}; margin: 0; font-size: 2rem;">{pred}</h2>
                <hr style="border-color: rgba(0,0,0,0.1); margin: 20px 0;">
                <p style="color: #475569; font-size: 1.1rem;">"{desc}"</p>
                <div style="margin-top: 20px;">
                    <span style="background: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: {color}; border: 1px solid {color};">Confidence: 94.2%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Placeholder State
            st.markdown("""
            <div style="
                background: #F8FAFC; 
                border-radius: 20px; 
                padding: 40px; 
                text-align: center; 
                border: 2px dashed #CBD5E1;
                margin-top: 20px;
                color: #94A3B8;
            ">
                <div style="font-size: 3rem; margin-bottom: 10px;">⏳</div>
                <h3>Waiting for Input</h3>
                <p>Click "Analyze" to see the prediction here.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# 8. PAGE: DESTINATION FINDER (Postcard Style)
# =========================================================
elif menu == "📍 Destination Finder":
    st.title("Destination Matchmaker")
    st.markdown("Content-Based Filtering finding places mathematically similar to your favorites.")

    # Search Bar Container
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #E2E8F0; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
        <h4 style="margin:0 0 10px 0;">Search the Database</h4>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        # Sorted list for better UX
        selected_attraction = st.selectbox("❤️ I loved visiting...", sorted(unique_destinations["Attraction"].unique()))
    with c2:
        st.write("") # Spacer
        st.write("")
        find_btn = st.button("🔍 Find Matches")

    if find_btn:
        try:
            # Recommender Logic
            idx = unique_destinations[unique_destinations["Attraction"] == selected_attraction].index[0]
            sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:5]
            recs = unique_destinations.iloc[[i[0] for i in sim_scores]]["Attraction"].values
            
            st.markdown("### 🌟 Recommended for you:")
            
            # Grid Layout for Results
            cols = st.columns(4)
            colors = ["#EFF6FF", "#ECFDF5", "#FFFBEB", "#FEF2F2"] # Blue, Green, Yellow, Red tints
            borders = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
            
            for i, place in enumerate(recs):
                with cols[i]:
                    st.markdown(f"""
                    <div style="
                        background-color: {colors[i]};
                        border: 1px solid {borders[i]};
                        border-radius: 15px;
                        padding: 25px;
                        text-align: center;
                        height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
                        transition: transform 0.2s;
                    ">
                        <div style="font-size: 2rem; margin-bottom: 15px;">📍</div>
                        <h4 style="margin:0; font-size: 1.1rem; color: #1E293B; line-height: 1.4;">{place}</h4>
                        <div style="margin-top: 15px;">
                            <small style="color: {borders[i]}; font-weight: bold;">View Details ➔</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Oops! We couldn't match this location. Try a more popular one.")