import streamlit as st

st.set_page_config(
    page_title="Kruger Supply Chain Tools",
    page_icon="🏭",
    layout="wide"
)

# Main page content
st.title("Kruger Supply Chain Tools")
st.markdown("### Internal Logistics Tools Suite")

# Add logo to the sidebar
st.sidebar.image("logo.png", width=200)
st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Kruger Inc.")

# Main content
st.markdown("""
Collection of tools developed by the Supply Chain Analytics team to help with various logistics operations.
Select a tool from the sidebar to get started.
""")

# Create two columns for the tool descriptions
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Available Tools:

    #### Round-trips Analysis Tool
    - Find potential round trip opportunities
    - Compare routes and calculate savings
    - Export results to Excel
    [Open Tool](/Round_Trip_Analysis_Tool)


    """)



# Basic instructions
st.markdown("---")
st.markdown("""
### Updates from Matthew:
Some tools are still under development. The Round-trips Analysis Tool is currently the main operational tool.
""")


