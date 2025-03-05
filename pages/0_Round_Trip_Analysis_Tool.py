import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --------------------------------------------------------------------------
# Streamlit App Setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Round-trips Analysis Tool",
    page_icon="🔄",
    layout="wide"
)

# Rest of the code from streamlit_app.py...
# (Copy all the functions and main app code here)

@st.cache_data
def load_data(file):
    """
    Load and de-duplicate the Excel file by the first column
    (assuming that's a unique 'Item ID' or similar).
    """
    df = pd.read_excel(file)
    df.drop_duplicates(subset=df.columns[0], inplace=True)
    return df

def calculate_round_trip_score(load_outbound, load_return, w1=0.6, w2=0.4):
    """
    Calculate a round-trip 'score' based on load difference and total load.
    """
    load_diff = abs(load_outbound - load_return)
    total_load = load_outbound + load_return
    if total_load == 0:
        return 0
    return w1 * (1 - load_diff / total_load) + w2 * np.log(total_load + 1)

def download_xlsx(df, filename="data.xlsx"):
    """Utility: convert df to XLSX in-memory and offer a download button."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    xlsx_data = output.getvalue()

    st.download_button(
        label=f"Download {filename}",
        data=xlsx_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def remove_identical_trip_rows(df):
    """
    Remove rows where:
      - Origin City_AtoB == Origin City_BtoA
      - Destination City_AtoB == Destination City_BtoA
      - Business Unit (abbreviated)_AtoB == Business Unit (abbreviated)_BtoA
    """
    if df.empty:
        return df
    mask = (
        (df['Origin City_AtoB'] == df['Origin City_BtoA']) &
        (df['Destination City_AtoB'] == df['Destination City_BtoA']) &
        (df['Business Unit (abbreviated)_AtoB'] == df['Business Unit (abbreviated)_BtoA'])
    )
    return df[~mask]

def filter_by_city(df, city, city_col_idx):
    """
    If city != 'ALL', keep only rows where df.iloc[:, city_col_idx] == city.
    Otherwise, return the original df.
    """
    if city == "ALL":
        return df
    return df[df.iloc[:, city_col_idx] == city]

def filter_by_mode(df, mode_filter, mode_col):
    """
    If mode_filter != 'ALL', keep only rows where df[mode_col] == mode_filter.
    Otherwise, return the original df.
    """
    if mode_filter == "ALL":
        return df
    return df[df[mode_col] == mode_filter]

def find_exact_matches(
    data,
    origin_city,
    destination_city,
    w1, w2,
    a_to_b_mode_filter,
    b_to_a_mode_filter,
    origin_idx=19,
    prov_idx=20,
    dest_idx=22,
    mode_idx=25
):
    """
    Exact Round Trips:
      1. A->B where origin = origin_city and destination = destination_city.
      2. B->A where origin = destination_city and destination = origin_city.
      3. Merge, apply mode filters, remove identical trips, compute Score, remove duplicates, sort.
    """
    A_to_B = filter_by_city(data, origin_city, origin_idx)
    A_to_B = filter_by_city(A_to_B, destination_city, dest_idx)

    B_to_A = filter_by_city(data, destination_city, origin_idx)
    B_to_A = filter_by_city(B_to_A, origin_city, dest_idx)

    if A_to_B.empty or B_to_A.empty:
        return pd.DataFrame()

    exact = pd.merge(
        A_to_B, B_to_A,
        left_on=[data.columns[origin_idx], data.columns[dest_idx]],
        right_on=[data.columns[dest_idx], data.columns[origin_idx]],
        suffixes=('_AtoB', '_BtoA')
    )
    if exact.empty:
        return exact

    # Apply mode-of-transport filters
    col_mode_A = f"{data.columns[mode_idx]}_AtoB"
    col_mode_B = f"{data.columns[mode_idx]}_BtoA"
    exact = filter_by_mode(exact, a_to_b_mode_filter, col_mode_A)
    exact = filter_by_mode(exact, b_to_a_mode_filter, col_mode_B)
    if exact.empty:
        return exact

    # Remove identical trips
    exact = remove_identical_trip_rows(exact)
    if exact.empty:
        return exact

    # Compute Score and Total Estimated Annual Loads
    exact['Score'] = exact.apply(
        lambda row: calculate_round_trip_score(
            row['Estimated Annual Loads_AtoB'],
            row['Estimated Annual Loads_BtoA'],
            w1, w2
        ),
        axis=1
    )
    exact['Total Estimated Annual Loads'] = (
        exact['Estimated Annual Loads_AtoB'] + exact['Estimated Annual Loads_BtoA']
    )

    # Remove duplicates based on key columns
    exact.drop_duplicates(
        subset=[
            'Origin City_AtoB',
            'Destination City_AtoB',
            'Business Unit (abbreviated)_AtoB'
        ],
        inplace=True
    )
    # Additionally, if more than one row has the same Total Estimated Annual Loads, remove duplicates.
    exact.drop_duplicates(subset=['Total Estimated Annual Loads'], inplace=True)

    exact.sort_values(by='Score', ascending=False, inplace=True)
    return exact

def find_provincial_matches(
    data,
    origin_city,
    destination_city,
    w1, w2,
    a_to_b_mode_filter,
    b_to_a_mode_filter,
    exact_df,
    origin_idx=19,
    prov_idx=20,
    dest_idx=22,
    mode_idx=25
):
    """
    Provincial Round Trips:
      - For a specific origin city (not 'ALL'), determine its province.
      - Search within that province for round trips where:
            A->B: origin == origin_city and destination == candidate.
            B->A: origin == candidate and destination == origin_city.
      - Merge records when both directions exist.
      - Apply mode filters and remove identical trips.
      - Finally, remove any round trip that qualifies as an exact match.
      - Compute Score and Total Estimated Annual Loads, remove duplicates based on key columns, then sort.
    """
    if origin_city == "ALL":
        return pd.DataFrame()

    # Determine the province of the origin city (using the first matching record)
    origin_rows = data[data.iloc[:, origin_idx] == origin_city]
    if origin_rows.empty:
        return pd.DataFrame()
    origin_province = origin_rows.iloc[0, prov_idx]

    # Filter data for the same province
    province_data = data[data.iloc[:, prov_idx] == origin_province]

    provincial_matches = pd.DataFrame()
    # For each row in the province, use its destination as candidate for a round trip.
    for _, row in province_data.iterrows():
        candidate = row[data.columns[dest_idx]]
        A_to_B = province_data[
            (province_data.iloc[:, origin_idx] == origin_city) &
            (province_data.iloc[:, dest_idx] == candidate)
        ]
        B_to_A = province_data[
            (province_data.iloc[:, origin_idx] == candidate) &
            (province_data.iloc[:, dest_idx] == origin_city)
        ]
        if not A_to_B.empty and not B_to_A.empty:
            round_trip = pd.merge(
                A_to_B, B_to_A,
                left_on=[data.columns[origin_idx], data.columns[dest_idx]],
                right_on=[data.columns[dest_idx], data.columns[origin_idx]],
                suffixes=('_AtoB', '_BtoA')
            )
            provincial_matches = pd.concat([provincial_matches, round_trip], ignore_index=True)

    if provincial_matches.empty:
        return provincial_matches

    # Apply mode filters
    col_mode_A = f"{data.columns[mode_idx]}_AtoB"
    col_mode_B = f"{data.columns[mode_idx]}_BtoA"
    provincial_matches = filter_by_mode(provincial_matches, a_to_b_mode_filter, col_mode_A)
    provincial_matches = filter_by_mode(provincial_matches, b_to_a_mode_filter, col_mode_B)
    if provincial_matches.empty:
        return provincial_matches

    # Remove identical trips
    provincial_matches = remove_identical_trip_rows(provincial_matches)
    if provincial_matches.empty:
        return provincial_matches

    # Remove any round trip that qualifies as an exact match.
    if not exact_df.empty:
        key_cols = [
            'Origin City_AtoB',
            'Destination City_AtoB',
            'Business Unit (abbreviated)_AtoB'
        ]
        exact_keys = exact_df[key_cols].drop_duplicates()
        merged = provincial_matches.merge(exact_keys, on=key_cols, how='left', indicator=True)
        provincial_matches = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    # Compute Score and Total Estimated Annual Loads
    provincial_matches['Score'] = provincial_matches.apply(
        lambda row: calculate_round_trip_score(
            row['Estimated Annual Loads_AtoB'],
            row['Estimated Annual Loads_BtoA'],
            w1, w2
        ),
        axis=1
    )
    provincial_matches['Total Estimated Annual Loads'] = (
        provincial_matches['Estimated Annual Loads_AtoB'] +
        provincial_matches['Estimated Annual Loads_BtoA']
    )

    # Remove duplicates based on key columns (do not remove by total load here)
    provincial_matches.drop_duplicates(
        subset=[
            'Origin City_AtoB',
            'Destination City_AtoB',
            'Business Unit (abbreviated)_AtoB'
        ],
        inplace=True
    )

    provincial_matches.sort_values(by='Score', ascending=False, inplace=True)
    return provincial_matches

def display_styled_table(df, title, filename):
    st.subheader(f"{title} ({len(df)})")
    if df.empty:
        st.write("No records found.")
        return
    
    # Define desired column order
    mode_col_idx = 25  # Adjust if needed
    mode_col_name = df.columns[mode_col_idx] if not df.empty and len(df.columns) > mode_col_idx else "Mode"
    
    desired_order = [
        'Score',
        'Total Estimated Annual Loads',
        'Origin City_AtoB',
        'Destination City_AtoB',
        f'{mode_col_name}_AtoB',
        'Estimated Annual Loads_AtoB',
        'Business Unit (abbreviated)_AtoB',
        'Origin City_BtoA',
        'Destination City_BtoA',
        f'{mode_col_name}_BtoA',
        'Estimated Annual Loads_BtoA',
        'Business Unit (abbreviated)_BtoA'
    ]
    
    show_cols = [c for c in desired_order if c in df.columns]
    df_display = df[show_cols].copy()

    styled = (
        df_display
        .style
        .format({'Score': '{:.2f}'})
        .set_properties(**{'font-weight': 'bold'}, subset=['Score'])
    )
    st.dataframe(styled, use_container_width=True)
    download_xlsx(df_display, filename)

# Main app
st.title("Round Trip Analysis Tool")
st.markdown("### Find and analyze potential round trip opportunities")

# Add a description of the tool
with st.expander("About this tool"):
    st.markdown("""
    This tool helps you identify and analyze potential round trip opportunities in your transportation network.
    
    **How to use:**
    1. Upload your Excel file containing transportation data
    2. Select origin and destination cities
    3. Adjust scoring weights and mode filters
    4. Click 'Find Round Trips' to analyze
    
    The tool will identify exact matches (direct A→B and B→A routes) and provincial matches 
    (routes within the same province that could form round trips).
    """)

col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    # File uploader
    file = st.file_uploader("Upload your Excel file (.xlsx)", type=['xlsx'])
    if file:
        data = load_data(file)
        st.success("Data loaded successfully!")

        # Replace "53ft Tandem/2X Trailer" with "Dry Van" in the Mode column
        mode_col_idx = 25  # Adjust if needed
        mode_col_name = data.columns[mode_col_idx]
        data.iloc[:, mode_col_idx] = data.iloc[:, mode_col_idx].replace(
            {"53ft Tandem/2X Trailer": "Dry Van"}
        )

        # City dropdowns
        origin_cities = data.iloc[:, 19].unique().tolist()
        origin_cities.insert(0, "ALL")
        destination_cities = data.iloc[:, 22].unique().tolist()
        destination_cities.insert(0, "ALL")

        origin_city = st.selectbox('Select Origin City', options=origin_cities)
        destination_city = st.selectbox('Select Destination City', options=destination_cities)

        # Scoring weights
        st.markdown("### Scoring Weights")
        w1 = st.slider("Weight for Load Balance (W1)", 0.0, 1.0, 0.5, 0.01)
        w2 = st.slider("Weight for Total Load (W2)", 0.0, 1.0, 0.5, 0.01)

        # Mode-of-Transport filters
        st.markdown("### Mode of Transport Filters")
        unique_modes = list(data.iloc[:, mode_col_idx].dropna().unique())
        unique_modes.insert(0, "ALL")
        a_to_b_mode_filter = st.selectbox("A->B Mode Filter", options=unique_modes, index=0)
        b_to_a_mode_filter = st.selectbox("B->A Mode Filter", options=unique_modes, index=0)

        # Run matching on button click
        if st.button('Find Round Trips'):
            exact_df = find_exact_matches(
                data, origin_city, destination_city,
                w1, w2,
                a_to_b_mode_filter, b_to_a_mode_filter,
                origin_idx=19,
                prov_idx=20,
                dest_idx=22,
                mode_idx=mode_col_idx
            )

            provincial_df = find_provincial_matches(
                data, origin_city, destination_city,
                w1, w2,
                a_to_b_mode_filter, b_to_a_mode_filter,
                exact_df,
                origin_idx=19,
                prov_idx=20,
                dest_idx=22,
                mode_idx=mode_col_idx
            )

            with col2:
                display_styled_table(exact_df, "Exact Match Round Trips", "ExactRoundTrips.xlsx")
                display_styled_table(provincial_df, "Provincial Matches", "ProvincialMatches.xlsx")
    else:
        with col2:
            st.info("Please upload an Excel file to begin analysis.")
            st.markdown("""
            ### Sample Data Format
            
            Your Excel file should contain columns for:
            - Origin City (column 19)
            - Province (column 20)
            - Destination City (column 22)
            - Mode of Transport (column 25)
            - Estimated Annual Loads
            - Business Unit
            
            The tool will analyze this data to find potential round trip opportunities.
            """)

# Add footer with logo
st.sidebar.image("logo.png", width=200)
st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Kruger Inc. All rights reserved.") 