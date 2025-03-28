import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import json
import os
from unidecode import unidecode  # Add this import for handling special characters
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="RFP Tool for Warehouses",
    page_icon="🏭",
    layout="wide"
)

@st.cache_data
def load_address_cache():
    """Load the address cache from JSON file"""
    try:
        if os.path.exists('address_cache.json'):
            with open('address_cache.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load address cache: {e}")
    return {}

@st.cache_data
def load_opportunities_cache():
    """Load the opportunities address cache from JSON file"""
    try:
        if os.path.exists('opportunities_cache.json'):
            with open('opportunities_cache.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load opportunities cache: {e}")
    return {}

def save_address_cache(cache, is_opportunity=False):
    """Save the address cache to JSON file"""
    try:
        filename = 'opportunities_cache.json' if is_opportunity else 'address_cache.json'
        with open(filename, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        st.warning(f"Could not save address cache: {e}")

def clean_address_simple(row, is_opportunity=False):
    """Simplified address cleaning focusing on essential location info"""
    try:
        if is_opportunity:
            # Get the basic components for opportunity addresses
            address = str(row['Address']).strip()
            city = str(row['City']).strip()
            state = str(row['State']).strip()
            country = 'Canada' if str(row['Country']).strip().upper() in ['CA', 'CAN', 'CANADA'] else 'USA'
            
            # Handle special cases for Quebec addresses
            if state.upper() in ['QC', 'QUEBEC']:
                address = address.split(',')[0].strip()
                city = 'Trois-Rivieres' if 'Trois-Riviere' in city else city
                return f"{address}, {city}, Quebec, {country}"
                
            # For US addresses
            if country == 'USA':
                address = address.split(',')[0].strip()
                return f"{address}, {city}, {state}, {country}"
                
            # Default format
            return f"{address}, {city}, {state}, {country}"
        else:
            # Existing master file address handling
            address = str(row['Warehouse Address']).strip()
            city = str(row['City']).strip()
            state = str(row['State / Prov.']).strip()
            country = 'Canada' if str(row['Country']).strip() in ['CA', 'CAN'] else 'USA'
            
            # Handle special cases for Quebec addresses
            if state == 'QC':
                address = address.split(',')[0].strip()
                city = 'Trois-Rivieres' if 'Trois-Riviere' in city else city
                return f"{address}, {city}, Quebec, {country}"
                
            # For US addresses
            if country == 'USA':
                address = address.split(',')[0].strip()
                return f"{address}, {city}, {state}, {country}"
                
            # Default format
            return f"{address}, {city}, {state}, {country}"
            
    except Exception:
        return None

def validate_coordinates(lat, lng, country):
    """Validate if coordinates make sense for North America"""
    if not lat or not lng:
        return False
        
    # Rough boundaries for North America
    if country.upper() in ['CA', 'CANADA']:
        # Canadian boundaries
        return 41.0 <= lat <= 83.0 and -141.0 <= lng <= -52.0
    elif country.upper() in ['US', 'USA', 'UNITED STATES']:
        # US boundaries (including Alaska)
        return 24.0 <= lat <= 71.5 and -171.0 <= lng <= -66.0
    else:
        # General North America boundaries
        return 15.0 <= lat <= 83.0 and -171.0 <= lng <= -52.0

@st.cache_data
def batch_geocode_addresses(addresses, is_opportunity=False):
    """Geocoding with separate caches for master and opportunity addresses"""
    geolocator = Nominatim(user_agent="warehouse_rfp_tool")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # Load appropriate cache
    address_cache = load_opportunities_cache() if is_opportunity else load_address_cache()
    cache_modified = False
    results = {}
    
    for address in addresses:
        if not address:
            continue
            
        # Check cache first
        if address in address_cache:
            results[address] = tuple(address_cache[address])
            continue
        
        try:
            # Try direct geocoding first
            location = geocode(address, timeout=10)
            
            if not location:
                # If failed, try with just city, state, country
                parts = address.split(',')
                if len(parts) >= 3:
                    simplified = ','.join(parts[-3:]).strip()
                    location = geocode(simplified, timeout=10)
            
            if location:
                coords = (location.latitude, location.longitude)
                results[address] = coords
                address_cache[address] = list(coords)
                cache_modified = True
            else:
                results[address] = None
                
        except Exception as e:
            st.warning(f"Error geocoding {address}: {str(e)}")
            results[address] = None
            
    if cache_modified:
        save_address_cache(address_cache, is_opportunity)
        
    return results

@st.cache_data
def load_master_data(file):
    """Load and process the master warehouse file"""
    try:
        # Load file with proper encoding handling
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            # Try different encodings in order
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                
            if df is None:
                st.error("Unable to read the file. Please check the file format.")
                return pd.DataFrame()
        else:
            df = pd.read_excel(file)
        
        # Create simplified addresses and geocode
        df['Full_Address'] = df.apply(clean_address_simple, axis=1)
        unique_addresses = df['Full_Address'].dropna().unique().tolist()
        
        with st.spinner('Processing warehouse locations...'):
            address_coords = batch_geocode_addresses(unique_addresses)
            coord_map = {
                addr: {'Latitude': coords[0], 'Longitude': coords[1]} 
                for addr, coords in address_coords.items() 
                if coords is not None
            }
            
            df['Latitude'] = df['Full_Address'].map(lambda x: coord_map.get(x, {}).get('Latitude'))
            df['Longitude'] = df['Full_Address'].map(lambda x: coord_map.get(x, {}).get('Longitude'))
        
        return df
        
    except Exception as e:
        st.error("Error processing file. Please ensure the file format is correct.")
        return pd.DataFrame()

@st.cache_data
def geocode_address(address):
    """Geocode an address using Nominatim"""
    if pd.isna(address) or str(address).strip() == '':
        return None
    
    try:
        geolocator = Nominatim(user_agent="warehouse_rfp_tool")
        location = geolocator.geocode(f"{address}, North America", timeout=10)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception as e:
        st.warning(f"Error geocoding address: {address}")
        return None

def create_map(master_df, opportunities_df=None):
    """Create a map with both existing warehouses and opportunities"""
    # Initialize the map centered on North America
    m = folium.Map(location=[40, -95], zoom_start=4)
    
    points_plotted = {'master': 0, 'opportunities': 0}
    
    # Plot existing warehouses (red markers)
    if not master_df.empty:
        for idx, row in master_df.iterrows():
            if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
                popup_text = f"""
                <b>Supplier:</b> {row['Supplier Name']}<br>
                <b>Address:</b> {row['Warehouse Address']}<br>
                <b>City:</b> {row['City']}<br>
                <b>State/Prov:</b> {row['State / Prov.']}<br>
                <b>Country:</b> {row['Country']}<br>
                <b>Status:</b> {row['Status']}
                """
                
                folium.CircleMarker(
                    location=[float(row['Latitude']), float(row['Longitude'])],
                    radius=8,
                    color='red',
                    fill=True,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m)
                points_plotted['master'] += 1
    
    # Plot opportunities (blue markers)
    if opportunities_df is not None and not opportunities_df.empty:
        for idx, row in opportunities_df.iterrows():
            if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
                # Build popup content dynamically based on available columns
                popup_content = []
                
                # Required fields
                popup_content.extend([
                    f"<b>Address:</b> {row['Address']}<br>",
                    f"<b>City:</b> {row['City']}<br>",
                    f"<b>State:</b> {row['State']}<br>",
                    f"<b>Country:</b> {row['Country']}<br>"
                ])
                
                # Optional fields
                if 'Total Space' in row:
                    popup_content.append(f"<b>Total Space:</b> {row['Total Space']}<br>")
                if 'Available Space' in row:
                    popup_content.append(f"<b>Available Space:</b> {row['Available Space']}<br>")
                if 'Rail served' in row:
                    popup_content.append(f"<b>Rail Served:</b> {row['Rail served']}<br>")
                if 'Ceiling Height' in row:
                    popup_content.append(f"<b>Ceiling Height:</b> {row['Ceiling Height']}<br>")
                if 'Dock Doors' in row:
                    popup_content.append(f"<b>Dock Doors:</b> {row['Dock Doors']}<br>")
                
                popup_text = "".join(popup_content)
                
                folium.CircleMarker(
                    location=[float(row['Latitude']), float(row['Longitude'])],
                    radius=8,
                    color='blue',
                    fill=True,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m)
                points_plotted['opportunities'] += 1
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px">
        <p><span style="color: red;">●</span> Existing Warehouses ({master_count})</p>
        <p><span style="color: blue;">●</span> New Opportunities ({opps_count})</p>
    </div>
    '''.format(master_count=points_plotted['master'], opps_count=points_plotted['opportunities'])
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

@st.cache_data
def load_opportunities_data(file):
    """Load and process the new warehouse opportunities file"""
    try:
        # Load file with proper encoding handling
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            # Try different encodings in order
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                
            if df is None:
                return pd.DataFrame()
        else:
            df = pd.read_excel(file)
        
        # Create simplified addresses and geocode
        df['Full_Address'] = df.apply(lambda x: clean_address_simple(x, is_opportunity=True), axis=1)
        unique_addresses = df['Full_Address'].dropna().unique().tolist()
        
        # Load existing cache first
        opportunities_cache = load_opportunities_cache()
        cached_coords = {
            addr: {'Latitude': coords[0], 'Longitude': coords[1]} 
            for addr, coords in opportunities_cache.items()
        }
        
        # Apply cached coordinates first
        df['Latitude'] = df['Full_Address'].map(lambda x: cached_coords.get(x, {}).get('Latitude'))
        df['Longitude'] = df['Full_Address'].map(lambda x: cached_coords.get(x, {}).get('Longitude'))
        
        # Only geocode addresses that aren't in the cache
        uncached_addresses = [addr for addr in unique_addresses if addr not in opportunities_cache]
        
        if uncached_addresses:
            with st.spinner(f'Processing {len(uncached_addresses)} new locations...'):
                address_coords = batch_geocode_addresses(uncached_addresses, is_opportunity=True)
                new_coords = {
                    addr: {'Latitude': coords[0], 'Longitude': coords[1]} 
                    for addr, coords in address_coords.items() 
                    if coords is not None
                }
                
                # Update coordinates for previously uncached addresses
                for addr, coords in new_coords.items():
                    mask = df['Full_Address'] == addr
                    df.loc[mask, 'Latitude'] = coords['Latitude']
                    df.loc[mask, 'Longitude'] = coords['Longitude']
        
        return df
        
    except Exception as e:
        st.error(f"Error processing opportunities file: {str(e)}")
        return pd.DataFrame()

# Main app layout
st.title("Warehouse RFP Tool")

# File uploaders
col1, col2 = st.columns(2)

with col1:
    master_file = st.file_uploader("Upload Master Warehouse File", type=['xlsx', 'csv'])

with col2:
    opportunities_file = st.file_uploader("Upload New Warehouse Opportunities", type=['xlsx', 'csv'])

# Only proceed if both files are uploaded
if master_file and opportunities_file:
    master_df = load_master_data(master_file)
    opportunities_df = load_opportunities_data(opportunities_file)
    
    if not master_df.empty and not opportunities_df.empty:
        # Existing Warehouse Filters
        st.markdown("### Existing Warehouse Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_values = master_df['Status'].fillna('Unknown').astype(str).unique()
            selected_status = st.multiselect('Status', ['All'] + sorted(status_values.tolist()), default=['All'])
            
            country_values = master_df['Country'].fillna('Unknown').astype(str).unique()
            selected_country = st.multiselect('Country', ['All'] + sorted(country_values.tolist()), default=['All'])

        with col2:
            state_values = master_df['State / Prov.'].fillna('Unknown').astype(str).unique()
            selected_state = st.multiselect('State/Province', ['All'] + sorted(state_values.tolist()), default=['All'])
            
            service_values = master_df['Service Type'].fillna('Unknown').astype(str).unique()
            selected_service = st.multiselect('Service Type', ['All'] + sorted(service_values.tolist()), default=['All'])

        with col3:
            supplier_values = master_df['Supplier Name'].fillna('Unknown').astype(str).unique()
            selected_supplier = st.multiselect('Supplier', ['All'] + sorted(supplier_values.tolist()), default=['All'])
            
            if 'Handling In/Out (MT)' in master_df.columns:
                handling_values = pd.to_numeric(master_df['Handling In/Out (MT)'], errors='coerce')
                valid_handling = handling_values.dropna()
                if not valid_handling.empty:
                    min_handling = float(valid_handling.min())
                    max_handling = float(valid_handling.max())
                    handling_range = st.slider('Handling Capacity (MT)', 
                                             min_value=min_handling,
                                             max_value=max_handling,
                                             value=(min_handling, max_handling))

        # Prospective Warehouse Filters
        st.markdown("### Prospective Warehouse Filters")
        prosp_col1, prosp_col2, prosp_col3 = st.columns(3)
        
        with prosp_col1:
            prosp_country_values = opportunities_df['Country'].fillna('Unknown').astype(str).unique()
            selected_prosp_country = st.multiselect('Country (Prospective)', ['All'] + sorted(prosp_country_values.tolist()), default=['All'])
            
            if 'Total Space' in opportunities_df.columns:
                space_values = pd.to_numeric(opportunities_df['Total Space'].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
                valid_space = space_values.dropna()
                if not valid_space.empty:
                    min_space = float(valid_space.min())
                    max_space = float(valid_space.max())
                    space_range = st.slider('Total Space (sq ft)', 
                                          min_value=min_space,
                                          max_value=max_space,
                                          value=(min_space, max_space))

        with prosp_col2:
            prosp_state_values = opportunities_df['State'].fillna('Unknown').astype(str).unique()
            selected_prosp_state = st.multiselect('State (Prospective)', ['All'] + sorted(prosp_state_values.tolist()), default=['All'])
            
            if 'Available Space' in opportunities_df.columns:
                avail_space_values = pd.to_numeric(opportunities_df['Available Space'].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
                valid_avail_space = avail_space_values.dropna()
                if not valid_avail_space.empty:
                    min_avail_space = float(valid_avail_space.min())
                    max_avail_space = float(valid_avail_space.max())
                    avail_space_range = st.slider('Available Space (sq ft)', 
                                                min_value=min_avail_space,
                                                max_value=max_avail_space,
                                                value=(min_avail_space, max_avail_space))

        with prosp_col3:
            rail_served_values = opportunities_df['Rail served'].fillna('Unknown').astype(str).unique()
            selected_rail_served = st.multiselect('Rail Service', ['All'] + sorted(rail_served_values.tolist()), default=['All'])

        # Apply filters to existing warehouses
        filtered_df = master_df.copy()

        if 'All' not in selected_status:
            filtered_df = filtered_df[filtered_df['Status'].fillna('Unknown').astype(str).isin(selected_status)]
        if 'All' not in selected_country:
            filtered_df = filtered_df[filtered_df['Country'].fillna('Unknown').astype(str).isin(selected_country)]
        if 'All' not in selected_state:
            filtered_df = filtered_df[filtered_df['State / Prov.'].fillna('Unknown').astype(str).isin(selected_state)]
        if 'All' not in selected_service:
            filtered_df = filtered_df[filtered_df['Service Type'].fillna('Unknown').astype(str).isin(selected_service)]
        if 'All' not in selected_supplier:
            filtered_df = filtered_df[filtered_df['Supplier Name'].fillna('Unknown').astype(str).isin(selected_supplier)]
        if 'Handling In/Out (MT)' in master_df.columns and 'handling_range' in locals():
            handling_values = pd.to_numeric(filtered_df['Handling In/Out (MT)'], errors='coerce')
            filtered_df = filtered_df[
                (handling_values >= handling_range[0]) &
                (handling_values <= handling_range[1])
            ]

        # Apply filters to prospective warehouses
        filtered_opportunities_df = opportunities_df.copy()

        if 'All' not in selected_prosp_country:
            filtered_opportunities_df = filtered_opportunities_df[
                filtered_opportunities_df['Country'].fillna('Unknown').astype(str).isin(selected_prosp_country)
            ]
        if 'All' not in selected_prosp_state:
            filtered_opportunities_df = filtered_opportunities_df[
                filtered_opportunities_df['State'].fillna('Unknown').astype(str).isin(selected_prosp_state)
            ]
        if 'All' not in selected_rail_served:
            filtered_opportunities_df = filtered_opportunities_df[
                filtered_opportunities_df['Rail served'].fillna('Unknown').astype(str).isin(selected_rail_served)
            ]
        if 'Total Space' in opportunities_df.columns and 'space_range' in locals():
            space_values = pd.to_numeric(filtered_opportunities_df['Total Space'].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
            filtered_opportunities_df = filtered_opportunities_df[
                (space_values >= space_range[0]) &
                (space_values <= space_range[1])
            ]
        if 'Available Space' in opportunities_df.columns and 'avail_space_range' in locals():
            avail_space_values = pd.to_numeric(filtered_opportunities_df['Available Space'].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
            filtered_opportunities_df = filtered_opportunities_df[
                (avail_space_values >= avail_space_range[0]) &
                (avail_space_values <= avail_space_range[1])
            ]

        # Map section with filtered data
        st.markdown("### Warehouse Locations")
        folium_map = create_map(filtered_df, filtered_opportunities_df)
        folium_static(folium_map, width=1400)

        # Analytics section
        st.markdown("### Analytics")
        
        # Create metrics rows
        metric_row1_1, metric_row1_2, metric_row1_3, metric_row1_4 = st.columns(4)
        metric_row2_1, metric_row2_2, metric_row2_3, metric_row2_4 = st.columns(4)
        
        # Add new row for prospective warehouse metrics
        st.markdown("#### Prospective Warehouse Metrics")
        prosp_metric_1, prosp_metric_2, prosp_metric_3, prosp_metric_4 = st.columns(4)

        # First row - Volume metrics (existing warehouses)
        with metric_row1_1:
            if 'Handling In/Out (MT)' in filtered_df.columns:
                handling_data = filtered_df['Handling In/Out (MT)'].replace('[\$,]', '', regex=True)
                handling_data = pd.to_numeric(handling_data, errors='coerce')
                total_handling = handling_data.sum()
                avg_handling = handling_data.mean()
                if not pd.isna(total_handling):
                    st.metric(
                        "Total Handling Volume",
                        f"{total_handling:,.2f} MT",
                        f"Avg: {avg_handling:,.2f} MT/warehouse",
                        help="Total handling volume across all warehouses (Metric Tons)"
                    )

        with metric_row1_2:
            if 'Initial Storage (MT/mth)' in filtered_df.columns:
                # Clean and convert storage capacity data
                storage_data = filtered_df['Initial Storage (MT/mth)'].replace('[\$,]', '', regex=True)
                storage_data = pd.to_numeric(storage_data, errors='coerce')
                total_storage = storage_data.sum()
                avg_storage = storage_data.mean()
                if not pd.isna(total_storage):
                    st.metric(
                        "Total Storage Capacity",
                        f"{total_storage:,.2f} MT/month",
                        f"Avg: {avg_storage:,.2f} MT/month/warehouse",
                        help="Total storage capacity across all warehouses (Metric Tons per month)"
                    )

        with metric_row1_3:
            active_count = len(filtered_df[filtered_df['Status'] == 'Active'])
            total_count = len(filtered_df)
            active_percentage = (active_count / total_count * 100) if total_count > 0 else 0
            st.metric(
                "Warehouse Count",
                f"{total_count}",
                f"{active_count} Active ({active_percentage:.1f}%)",
                help="Total number of warehouses and percentage that are active"
            )

        with metric_row1_4:
            unique_suppliers = filtered_df['Supplier Name'].nunique()
            st.metric(
                "Unique Suppliers",
                f"{unique_suppliers}",
                help="Number of unique suppliers in the network"
            )

        # Prospective warehouse metrics
        with prosp_metric_1:
            total_prosp = len(filtered_opportunities_df)
            total_space = 0
            if 'Total Space' in filtered_opportunities_df.columns:
                space_values = pd.to_numeric(
                    filtered_opportunities_df['Total Space'].astype(str).str.replace('[^\d.]', '', regex=True),
                    errors='coerce'
                )
                total_space = space_values.sum()
            
            st.metric(
                "Total Prospective Warehouses",
                f"{total_prosp}",
                f"Total Space: {total_space:,.0f} sq ft" if total_space > 0 else "Space data not available",
                help="Number of prospective warehouses and their total space"
            )

        with prosp_metric_2:
            available_space = 0
            avg_available = 0
            if 'Available Space' in filtered_opportunities_df.columns:
                space_values = pd.to_numeric(
                    filtered_opportunities_df['Available Space'].astype(str).str.replace('[^\d.]', '', regex=True),
                    errors='coerce'
                )
                available_space = space_values.sum()
                avg_available = available_space / total_prosp if total_prosp > 0 else 0
            
            st.metric(
                "Total Available Space",
                f"{available_space:,.0f} sq ft" if available_space > 0 else "Not available",
                f"Avg: {avg_available:,.0f} sq ft/warehouse" if avg_available > 0 else "No data",
                help="Total and average available space in prospective warehouses"
            )

        with prosp_metric_3:
            rail_served_count = 0
            rail_percentage = 0
            if 'Rail served' in filtered_opportunities_df.columns:
                rail_served_count = len(filtered_opportunities_df[
                    filtered_opportunities_df['Rail served'].fillna('').str.lower().isin(['yes', 'y', 'true'])
                ])
                rail_percentage = (rail_served_count / total_prosp * 100) if total_prosp > 0 else 0
            
            st.metric(
                "Rail-Served Locations",
                f"{rail_served_count}",
                f"{rail_percentage:.1f}% of total" if total_prosp > 0 else "No data",
                help="Number and percentage of rail-served prospective warehouses"
            )

        with prosp_metric_4:
            avg_height = 0
            max_height = 0
            if 'Ceiling Height' in filtered_opportunities_df.columns:
                height_values = pd.to_numeric(
                    filtered_opportunities_df['Ceiling Height'].astype(str).str.replace('[^\d.]', '', regex=True),
                    errors='coerce'
                )
                avg_height = height_values.mean()
                max_height = height_values.max()
            
            if pd.notnull(avg_height) and avg_height > 0:
                st.metric(
                    "Average Ceiling Height",
                    f"{avg_height:.1f} ft",
                    f"Max: {max_height:.1f} ft",
                    help="Average and maximum ceiling height in prospective warehouses"
                )
            else:
                st.metric(
                    "Average Ceiling Height",
                    "Not available",
                    "No data",
                    help="Ceiling height data not available"
                )

        # Second row - Financial metrics
        with metric_row2_1:
            if 'CAD Handling In/Out (MT)' in filtered_df.columns:
                # Clean and convert handling cost data
                handling_cost = filtered_df['CAD Handling In/Out (MT)'].replace('[\$,]', '', regex=True)
                handling_cost = pd.to_numeric(handling_cost, errors='coerce')
                avg_handling_cost = handling_cost.mean()
                total_handling_cost = handling_cost.sum()
                if not pd.isna(avg_handling_cost):
                    st.metric(
                        "Total Handling Cost Rate",
                        f"${total_handling_cost:,.2f}/MT",
                        f"Avg: ${avg_handling_cost:,.2f}/MT",
                        help="Total and average handling cost rates (CAD per MT)"
                    )

        with metric_row2_2:
            if 'Initial Storage (MT/mth)' in filtered_df.columns and 'CAD Handling In/Out (MT)' in filtered_df.columns:
                # Calculate potential monthly handling revenue
                monthly_revenue = (handling_cost * storage_data).sum()
                avg_monthly_revenue = monthly_revenue / total_count if total_count > 0 else 0
                st.metric(
                    "Potential Monthly Revenue",
                    f"${monthly_revenue:,.2f}",
                    f"Avg: ${avg_monthly_revenue:,.2f}/warehouse",
                    help="Potential monthly revenue based on handling rates and storage capacity"
                )

        with metric_row2_3:
            if '$CAD Initial Storage' in filtered_df.columns:
                # Clean and convert initial storage cost data
                initial_storage_cost = filtered_df['$CAD Initial Storage'].replace('[\$,]', '', regex=True)
                initial_storage_cost = pd.to_numeric(initial_storage_cost, errors='coerce')
                total_initial_cost = initial_storage_cost.sum()
                avg_initial_cost = initial_storage_cost.mean()
                if not pd.isna(total_initial_cost):
                    st.metric(
                        "Total Initial Storage Cost",
                        f"${total_initial_cost:,.2f}",
                        f"Avg: ${avg_initial_cost:,.2f}",
                        help="Total and average initial storage costs (CAD)"
                    )

        with metric_row2_4:
            if '$CAD Recurring Storage' in filtered_df.columns:
                # Clean and convert recurring storage cost data
                recurring_storage_cost = filtered_df['$CAD Recurring Storage'].replace('[\$,]', '', regex=True)
                recurring_storage_cost = pd.to_numeric(recurring_storage_cost, errors='coerce')
                total_recurring_cost = recurring_storage_cost.sum()
                avg_recurring_cost = recurring_storage_cost.mean()
                if not pd.isna(total_recurring_cost):
                    st.metric(
                        "Total Recurring Storage Cost",
                        f"${total_recurring_cost:,.2f}/month",
                        f"Avg: ${avg_recurring_cost:,.2f}/month",
                        help="Total and average recurring storage costs per month (CAD)"
                    )

        # Add an information box explaining the financial metrics
       

        # Financial charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Handling Cost Analysis
            if 'CAD Handling In/Out (MT)' in filtered_df.columns:
                st.markdown("#### Handling Cost Analysis")
                # Clean and convert cost data
                cost_data = filtered_df['CAD Handling In/Out (MT)'].replace('[\$,]', '', regex=True)
                cost_data = pd.to_numeric(cost_data, errors='coerce')
                filtered_df['Handling_Cost_Clean'] = cost_data
                
                cost_by_region = filtered_df.groupby('State / Prov.').agg({'Handling_Cost_Clean': 'mean'}).reset_index()
                cost_by_region = cost_by_region.dropna()
                
                if not cost_by_region.empty:
                    fig = px.bar(cost_by_region,
                                x='State / Prov.',
                                y='Handling_Cost_Clean',
                                title='Average Handling Cost by Region (CAD/MT)')
                    fig.update_layout(yaxis_title='CAD/MT')
                    st.plotly_chart(fig, use_container_width=True)

            # Storage Cost Analysis
            st.markdown("#### Storage Cost Comparison")
            # Check for both possible column naming patterns
            storage_cols = [
                'Initial Storage (MT/mth)', 
                'Recurring Storage (MT/mth)',
                '$CAD Initial Storage', 
                '$CAD Recurring Storage'
            ]
            available_cols = [col for col in storage_cols if col in filtered_df.columns]
            
            if available_cols:
                # Create temporary dataframe for analysis
                storage_df = filtered_df.copy()
                
                # Clean and convert storage cost data for each available column
                for col in available_cols:
                    clean_col_name = col.replace('$', '').replace(' ', '_')
                    storage_df[clean_col_name] = pd.to_numeric(
                        storage_df[col].astype(str).str.replace('[\$,]', '', regex=True),
                        errors='coerce'
                    )
                
                # Group by state/province
                storage_costs = storage_df.groupby('State / Prov.').agg({
                    col.replace('$', '').replace(' ', '_'): 'mean' 
                    for col in available_cols
                }).reset_index()
                
                # Remove rows where all values are NaN
                storage_costs = storage_costs.dropna(how='all', subset=[
                    col.replace('$', '').replace(' ', '_') for col in available_cols
                ])
                
                if not storage_costs.empty:
                    # Create bar chart
                    fig = px.bar(
                        storage_costs,
                        x='State / Prov.',
                        y=[col.replace('$', '').replace(' ', '_') for col in available_cols],
                        title='Storage Costs by Region',
                        barmode='group',
                        labels={
                            col.replace('$', '').replace(' ', '_'): col 
                            for col in available_cols
                        }
                    )
                    
                    fig.update_layout(
                        yaxis_title='CAD/MT/month',
                        legend_title='Storage Type',
                        height=500
                    )
                    
                    # Update legend labels to be more readable
                    fig.for_each_trace(lambda t: t.update(
                        name=t.name.replace('_', ' ').title()
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid storage cost data available for comparison")
            else:
                st.warning("Required storage cost columns not found in the data")

        with chart_col2:
            # Capacity Utilization Analysis
            st.markdown("#### Capacity Utilization Analysis")
            if 'Handling In/Out (MT)' in filtered_df.columns:
                # Clean and convert handling data
                handling_data = filtered_df['Handling In/Out (MT)'].replace('[\$,]', '', regex=True)
                handling_data = pd.to_numeric(handling_data, errors='coerce')
                filtered_df['Handling_Clean'] = handling_data
                
                # Group by Service Type and Status for more detailed analysis
                temp_df = filtered_df.dropna(subset=['Handling_Clean'])
                
                if not temp_df.empty:
                    # Create pivot table for better visualization
                    capacity_pivot = pd.pivot_table(
                        temp_df,
                        values='Handling_Clean',
                        index='Service Type',
                        columns='Status',
                        aggfunc=['mean', 'count'],
                        fill_value=0
                    ).reset_index()
                    
                    # Flatten column names
                    capacity_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                           for col in capacity_pivot.columns]
                    
                    # Create bar chart
                    fig = go.Figure()
                    
                    # Add bars for each status
                    for status in temp_df['Status'].unique():
                        if f'mean_{status}' in capacity_pivot.columns:
                            fig.add_trace(go.Bar(
                                name=f'{status}',
                                x=capacity_pivot['Service Type'],
                                y=capacity_pivot[f'mean_{status}'],
                                text=capacity_pivot[f'count_{status}'].apply(lambda x: f'n={int(x)}' if x > 0 else ''),
                                textposition='outside'
                            ))
                    
                    fig.update_layout(
                        title='Capacity Utilization by Service Type and Status',
                        yaxis_title='Average Handling Capacity (MT)',
                        barmode='group',
                        showlegend=True,
                        height=400,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanatory text
                    st.markdown("""
                    **Capacity Utilization Insights:**
                    - Shows average handling capacity by service type and status
                    - 'n' values indicate the number of warehouses in each category
                    - Grouped bars allow comparison across different statuses
                    """)
                else:
                    st.warning("No valid handling capacity data available for analysis")

            # Contract Analysis
            st.markdown("#### Contract Analysis")
            
            # Check for contract duration column (it might be stored as text)
            contract_df = filtered_df.copy()
            
            # Extract numeric duration from the 'Contract Duration (months)' column
            if 'Contract Duration (months)' in contract_df.columns:
                # Clean duration data - handle both numeric and text formats
                contract_df['Duration_Clean'] = contract_df['Contract Duration (months)'].astype(str)
                contract_df['Duration_Clean'] = contract_df['Duration_Clean'].str.extract('(\d+)').astype(float)
            else:
                # If no duration column, try to calculate from Effective and Expiry dates
                if 'Effective Date' in contract_df.columns and 'Expiry Date' in contract_df.columns:
                    contract_df['Effective_Date'] = pd.to_datetime(contract_df['Effective Date'], errors='coerce')
                    contract_df['Expiry_Date'] = pd.to_datetime(contract_df['Expiry Date'], errors='coerce')
                    contract_df['Duration_Clean'] = (
                        (contract_df['Expiry_Date'] - contract_df['Effective_Date']).dt.days / 30.44  # Average month length
                    ).round()
            
            if 'Duration_Clean' in contract_df.columns:
                # Remove invalid durations
                contract_df = contract_df[contract_df['Duration_Clean'].notna()]
                
                if not contract_df.empty:
                    # Create summary statistics by status
                    contract_summary = contract_df.groupby('Status').agg({
                        'Duration_Clean': ['count', 'mean', 'min', 'max'],
                        'Supplier Name': 'nunique'
                    }).reset_index()
                    
                    # Flatten column names
                    contract_summary.columns = ['Status', 'Count', 'Avg_Duration', 'Min_Duration', 'Max_Duration', 'Unique_Suppliers']
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add bars for average duration
                    fig.add_trace(go.Bar(
                        name='Average Duration',
                        x=contract_summary['Status'],
                        y=contract_summary['Avg_Duration'],
                        text=contract_summary.apply(lambda x: f"n={int(x['Count'])}<br>({int(x['Unique_Suppliers'])} suppliers)", axis=1),
                        textposition='outside',
                        marker_color='rgb(55, 83, 109)'
                    ))
                    
                    # Add range indicators
                    for idx, row in contract_summary.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['Status'], row['Status']],
                            y=[row['Min_Duration'], row['Max_Duration']],
                            mode='lines',
                            line=dict(color='rgba(55, 83, 109, 0.5)', width=2),
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title='Contract Duration Analysis by Status',
                        yaxis_title='Duration (Months)',
                        showlegend=False,
                        height=400,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add contract timeline
                    if 'Expiry Date' in contract_df.columns:
                        st.markdown("##### Contract Expiry Timeline")
                        
                        timeline_df = contract_df.sort_values('Expiry Date')
                        timeline_df['Expiry_Date'] = pd.to_datetime(timeline_df['Expiry Date'])
                        
                        # Create timeline visualization
                        fig2 = px.scatter(
                            timeline_df,
                            x='Expiry Date',
                            y='Status',
                            color='Status',
                            hover_data=['Supplier Name', 'Duration_Clean'],
                            title='Contract Expiry Timeline'
                        )
                        
                        fig2.update_layout(
                            height=300,
                            showlegend=True,
                            xaxis_title='Expiry Date',
                            yaxis_title='Status'
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Add explanatory text
                    st.markdown("""
                    **Contract Analysis Insights:**
                    - Bars show average contract duration for each status
                    - Vertical lines show the range (min to max duration)
                    - 'n' values show the number of contracts
                    - Timeline shows contract expiry dates by status
                    - Hover over points to see supplier details
                    """)
                else:
                    st.warning("No valid contract duration data available for analysis")
            else:
                st.warning("Contract duration information not found in the data")

    
        
        # Create comparison dataframe for existing warehouses
        existing_analysis = filtered_df.copy()
        prospective_analysis = filtered_opportunities_df.copy()
        
        # Initialize aggregation dictionaries
        existing_agg = {'Supplier Name': 'count'}
        prospective_agg = {'Address': 'count'}
        
        # Process existing warehouse data
        if 'Handling In/Out (MT)' in existing_analysis.columns:
            existing_analysis['Handling_Capacity'] = pd.to_numeric(
                existing_analysis['Handling In/Out (MT)'].astype(str).str.replace('[\$,]', '', regex=True),
                errors='coerce'
            )
            existing_agg['Handling_Capacity'] = 'sum'
        
        # Process prospective warehouse data
        if 'Total Space' in prospective_analysis.columns:
            prospective_analysis['Space_Clean'] = pd.to_numeric(
                prospective_analysis['Total Space'].astype(str).str.replace('[^\d.]', '', regex=True),
                errors='coerce'
            )
            prospective_agg['Space_Clean'] = 'sum'
        
        # Add existing warehouse data
        existing_by_region = existing_analysis.groupby('State / Prov.').agg(existing_agg).reset_index()
        
        # Add prospective warehouse data
        prospective_by_region = prospective_analysis.groupby('State').agg(prospective_agg).reset_index()
        
        # Create the main comparison trace
        fig = go.Figure()
        
        # Calculate percentages only if the columns exist
        if 'Handling_Capacity' in existing_by_region.columns:
            total_handling = existing_by_region['Handling_Capacity'].sum()
            if total_handling > 0:
                existing_by_region['Handling_Capacity_Pct'] = (
                    existing_by_region['Handling_Capacity'] / total_handling * 100
                )
        
        if 'Space_Clean' in prospective_by_region.columns:
            total_space = prospective_by_region['Space_Clean'].sum()
            if total_space > 0:
                prospective_by_region['Space_Clean_Pct'] = (
                    prospective_by_region['Space_Clean'] / total_space * 100
                )
        
        # Add existing warehouse bars
        hover_text = existing_by_region.apply(
            lambda x: (
                f"Count: {x['Supplier Name']}<br>" +
                (f"Capacity: {x['Handling_Capacity']:,.0f} MT" if 'Handling_Capacity' in x else "")
            ),
            axis=1
        )
        
        fig.add_trace(go.Bar(
            name='Existing Network',
            x=existing_by_region['State / Prov.'],
            y=existing_by_region['Handling_Capacity_Pct'] if 'Handling_Capacity_Pct' in existing_by_region.columns 
              else existing_by_region['Supplier Name'] / existing_by_region['Supplier Name'].sum() * 100,
            marker_color='rgba(255, 99, 71, 0.7)',
            text=hover_text,
            textposition='outside',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Network Share: %{y:.1f}%<br>" +
                "%{text}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add prospective warehouse bars
        hover_text = prospective_by_region.apply(
            lambda x: (
                f"Count: {x['Address']}<br>" +
                (f"Space: {x['Space_Clean']:,.0f} sq ft" if 'Space_Clean' in x else "")
            ),
            axis=1
        )
        
        fig.add_trace(go.Bar(
            name='Prospective Opportunities',
            x=prospective_by_region['State'],
            y=prospective_by_region['Space_Clean_Pct'] if 'Space_Clean_Pct' in prospective_by_region.columns
              else prospective_by_region['Address'] / prospective_by_region['Address'].sum() * 100,
            marker_color='rgba(65, 105, 225, 0.7)',
            text=hover_text,
            textposition='outside',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Network Share: %{y:.1f}%<br>" +
                "%{text}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add a line showing the ideal distribution
        all_regions = set(existing_by_region['State / Prov.'].tolist() + prospective_by_region['State'].tolist())
        total_regions = len(all_regions)
        ideal_percentage = 100 / total_regions if total_regions > 0 else 0
        
        fig.add_trace(go.Scatter(
            name='Ideal Balance',
            x=list(all_regions),
            y=[ideal_percentage] * total_regions,
            mode='lines',
            line=dict(color='green', dash='dash'),
            hovertemplate="Ideal balance: %{y:.1f}%<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Network Distribution Analysis: Existing vs Prospective',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Region',
            yaxis_title='Network Share (%)',
            barmode='group',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(t=80, b=50, l=50, r=50)
        )
        
        # Add annotations explaining the visualization
        fig.add_annotation(
            text=(
                "Gap Analysis: Regions below the green line are underserved, " +
                "while those above may have excess capacity"
            ),
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        # Show the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights explanation with dynamic text based on available metrics
        metrics_text = []
        if 'Handling_Capacity' in existing_by_region.columns:
            metrics_text.append("- Existing warehouse capacity distribution (red)")
        else:
            metrics_text.append("- Existing warehouse count distribution (red)")
            
        if 'Space_Clean' in prospective_by_region.columns:
            metrics_text.append("- Prospective warehouse space distribution (blue)")
        else:
            metrics_text.append("- Prospective warehouse count distribution (blue)")
            
        
        
        # Results table at the bottom
        st.markdown("### Detailed Results")
        st.dataframe(filtered_df, use_container_width=True, height=400)

