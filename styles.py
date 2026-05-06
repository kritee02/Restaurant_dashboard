import streamlit as st

def apply_styles():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 50%, #ffffff 100%) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #eaf1fb !important;
        border-right: 1px solid #d6e2f0;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1f3c88 !important;
    }

    h1 {
        color: #1f3c88 !important;
        text-align: center !important;
        font-weight: 750 !important;
        font-size: 2.9rem !important;
    }
    h2, h3 {
        color: #14345f !important;
        border-bottom: 3px solid #b7d4f5;
        padding-bottom: 6px;
        margin-top: 25px;
    }

    p, li, span {
        color: #2c3e50;
    }

    [data-testid="stDataFrame"] {
        background-color: white !important;
        border-radius: 14px !important;
        box-shadow: 0px 4px 14px rgba(31, 60, 136, 0.10);
        padding: 8px;
    }

    [data-testid="stMetric"] {
        background: white !important;
        border: 1px solid #d8e6f7 !important;
        padding: 16px !important;
        border-radius: 14px !important;
        box-shadow: 0px 4px 12px rgba(31, 60, 136, 0.12);
    }
     [data-testid="stDownloadButton"] button {
    background-color: #fff4cc !important;
    color: #7a5c00 !important;
    border: 1px solid #f2d675 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #ffe89a !important;
    color: #5c4400 !important;
    border: 1px solid #e6c34d !important;
}
    .stAlert {
        border-radius: 12px !important;
        border-left: 5px solid #1f77b4 !important;
    }
    .section-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1f3c88;
    margin-top: 20px;
    margin-bottom: 5px;
    border-left: 6px solid #f2c94c;
    padding-left: 12px;
}

.section-subtitle {
    font-size: 0.95rem;
    color: #5b6575;
    margin-bottom: 15px;
}  
                .summary-table {
    width: 100%;
    border-collapse: collapse;
    background-color: white;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0px 4px 14px rgba(31, 60, 136, 0.10);
}

.summary-table th {
    background-color: #eaf1fb;
    color: #1f3c88;
    font-weight: 700;
    padding: 10px;
    border: 1px solid #d6e2f0;
}

.summary-table td {
    padding: 9px;
    border: 1px solid #e1e8f2;
    color: #2c3e50;
}

.summary-table tr:nth-child(even) {
    background-color: #f8fbff;
}          

                /* Selectbox styling */
div[data-baseweb="select"] {
    background-color: white !important;
    border: 2px solid #1f3c88 !important;
    border-radius: 12px !important;
    box-shadow: 0px 4px 12px rgba(31, 60, 136, 0.15);
    transition: all 0.2s ease-in-out;
}

/* Hover effect */
div[data-baseweb="select"]:hover {
    border-color: #f2c94c !important;
    box-shadow: 0px 4px 18px rgba(242, 201, 76, 0.35);
}

/* Selected text */
div[data-baseweb="select"] span {
    color: #1f3c88 !important;
    font-weight: 600 !important;
    font-size: 18px !important;
}            
    </style>
    """, unsafe_allow_html=True)