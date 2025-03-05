import streamlit as st

st.set_page_config(
    page_title="Round-trips Analysis Tool",
    page_icon="🏭",
    layout="wide"
)

# Main page content
st.title("Kruger Supply Chain Tools")
st.markdown("### Welcome to the Kruger Supply Chain Tools Suite")

# Add logo to the sidebar
st.sidebar.image("logo.png", width=200)
st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Kruger Inc. All rights reserved.")

# Main content
st.markdown("""
This application provides a suite of tools to help optimize and manage your supply chain operations.

### Available Tools:

1. **Round-trips Analysis Tool** - Find and analyze potential round trip opportunities in your transportation network.
2. **RFP Tool for Warehouses** - Create and manage RFPs for warehouse services.
3. **RFP Tool for Ocean** - Create and manage RFPs for ocean freight services.
4. **Warehouse Map Tool** - Visualize warehouse locations on an interactive map.
5. **Market Intelligence AI Tool** - Access AI-powered market insights.
6. **Carrier Scorecard** - Evaluate carrier performance.
7. **SPI Optimization** - Optimize Supply Performance Index.
8. **Automate VCP Reporting** - Automate Value Creation Plan reporting.

### Getting Started

Select a tool from the sidebar to begin. Each tool is designed to address specific supply chain challenges and opportunities.
""")

# Create three columns for the tool cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Round-trips Analysis Tool
    
    Identify and analyze potential round trip opportunities in your transportation network.
    
    [Go to Tool](/Round_Trip_Analysis_Tool)
    """)
    
    st.markdown("""
    ### Warehouse Map Tool
    
    Visualize warehouse locations on an interactive map.
    
    [Go to Tool](/Warehouse_Map_Tool)
    """)
    
    st.markdown("""
    ### SPI Optimization
    
    Optimize Supply Performance Index.
    
    [Go to Tool](/SPI_Optimization)
    """)

with col2:
    st.markdown("""
    ### RFP Tool for Warehouses
    
    Create and manage RFPs for warehouse services.
    
    [Go to Tool](/RFP_Tool_for_Warehouses)
    """)
    
    st.markdown("""
    ### Market Intelligence AI Tool
    
    Access AI-powered market insights.
    
    [Go to Tool](/Market_Intelligence_AI_Tool)
    """)
    
    st.markdown("""
    ### Automate VCP Reporting
    
    Automate Value Creation Plan reporting.
    
    [Go to Tool](/Automate_VCP_Reporting)
    """)

with col3:
    st.markdown("""
    ### RFP Tool for Ocean
    
    Create and manage RFPs for ocean freight services.
    
    [Go to Tool](/RFP_Tool_for_Ocean)
    """)
    
    st.markdown("""
    ### Carrier Scorecard
    
    Evaluate carrier performance.
    
    [Go to Tool](/Carrier_Scorecard)
    """)

# Add footer
st.markdown("---")
st.markdown("For support, please contact the Supply Chain Analytics team.") 