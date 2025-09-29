import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Society Migration Kanban Tracker")

# --- Constants ---
# NOTE: Ensure this filename matches your data file in the same directory.
DATA_FILE = "societies_migration.csv"
KANBAN_STAGES = ['Not Started', 'In Progress', 'Validation', 'Migrated', 'Issues']
TODAY = datetime.now()


# --- Dummy Data Generator (Runs only if the CSV is missing) ---
def generate_dummy_data():
    """Generates a sample CSV file if the actual data file is not found."""
    st.warning(f"'{DATA_FILE}' not found. Generating dummy data for 200 societies...")
    
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai']
    dark_stores = [f'DS{i:03d}' for i in range(1, 11)]
    
    np.random.seed(42)
    
    data = []
    for i in range(1, 201):
        orders = np.random.randint(50, 2500)
        stage = np.random.choice(KANBAN_STAGES, p=[0.25, 0.30, 0.15, 0.25, 0.05])
        
        # Calculate days in stage (simulated)
        days_in_stage = np.random.randint(0, 15) if stage != 'Migrated' else 0
        
        data.append({
            'SocietyID': f'SOC{i:04d}',
            'SocietyName': f'Gated Community {i}',
            'AssignedDarkStoreID': np.random.choice(dark_stores),
            'City': np.random.choice(cities),
            'AvgOrdersPerDay': orders,
            'MigrationStage': stage,
            'MigrationStartedOn': (TODAY - timedelta(days=days_in_stage)).isoformat(),
            'LastUpdated': TODAY.isoformat()
        })
        
    df = pd.DataFrame(data)
    
    # Create OrderRangeBucket
    df['OrderRangeBucket'] = pd.cut(
        df['AvgOrdersPerDay'],
        bins=[0, 100, 300, 500, 1000, 2000, 99999],
        labels=['1-100', '101-300', '301-500', '501-1000', '1001-2000', '2000+'],
        right=False
    ).astype(str)

    # Save the dummy file
    df.to_csv(DATA_FILE, index=False)
    st.info("Dummy data generated! Please replace this file with your actual data.")
    return df


# --- Data Loading and Caching ---

@st.cache_data(show_spinner="Loading and preparing data...")
def load_and_prepare_data():
    """Loads data, ensures correct types, and calculates DaysInStage."""
    
    if not os.path.exists(DATA_FILE):
        return generate_dummy_data()

    try:
        df = pd.read_csv(DATA_FILE)
        
        # Type conversion and date handling
        df['SocietyID'] = df['SocietyID'].astype(str) 
        df['AvgOrdersPerDay'] = pd.to_numeric(df['AvgOrdersPerDay'], errors='coerce').fillna(0).round(0).astype(int)
        
        # Calculate Days In Stage: Must handle potential NaT (Not a Time) errors
        df['MigrationStartedOn'] = pd.to_datetime(df['MigrationStartedOn'], errors='coerce')
        
        # Calculate days difference only if MigrationStartedOn is valid
        # Filter out societies not in a valid Kanban stage
        df = df[df['MigrationStage'].isin(KANBAN_STAGES)].copy()
        df['DaysInStage'] = (TODAY - df['MigrationStartedOn']).dt.days.clip(lower=0).fillna(0).astype(int)
        
        # Ensure OrderRangeBucket exists for filtering
        if 'OrderRangeBucket' not in df.columns:
            st.warning("OrderRangeBucket missing; dynamically generating from AvgOrdersPerDay.")
            df['OrderRangeBucket'] = pd.cut(
                df['AvgOrdersPerDay'],
                bins=[0, 100, 300, 500, 1000, 2000, 99999],
                labels=['1-100', '101-300', '301-500', '501-1000', '1001-2000', '2000+'],
                right=False
            ).astype(str)

        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}. Check column names and types in your CSV.")
        return pd.DataFrame()


def save_data(df):
    """Saves the DataFrame back to the CSV file and clears cache to force reload."""
    try:
        # Save only the necessary columns back to avoid writing calculated columns
        cols_to_save = [
            'SocietyID', 'SocietyName', 'AssignedDarkStoreID', 'City', 
            'AvgOrdersPerDay', 'MigrationStage', 'MigrationStartedOn', 
            'LastUpdated', 'OrderRangeBucket' # Save the bucket if it was generated
        ]
        
        # Filter columns to only those that exist in the DataFrame
        final_cols = [col for col in cols_to_save if col in df.columns]
        
        df[final_cols].to_csv(DATA_FILE, index=False, date_format='%Y-%m-%dT%H:%M:%S')
        st.cache_data.clear() # Clears the cache to force load_and_prepare_data() to re-read the file
        st.toast("Status updated successfully! Refreshing...", icon="✅")
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False


def update_migration_status(society_id, new_stage, current_df):
    """Handles the status change logic and triggers data save."""
    
    row_index = current_df[current_df['SocietyID'] == society_id].index
    if row_index.empty:
        return

    row_index = row_index[0]
    current_stage = current_df.loc[row_index, 'MigrationStage']
    
    if current_stage != new_stage:
        # 1. Update the stage
        current_df.loc[row_index, 'MigrationStage'] = new_stage
        
        # 2. Update the last updated time
        current_df.loc[row_index, 'LastUpdated'] = TODAY.isoformat()
        
        # 3. Reset MigrationStartedOn if moving into a new working/tracking stage
        if new_stage in ['Not Started', 'In Progress', 'Validation', 'Issues']:
             current_df.loc[row_index, 'MigrationStartedOn'] = TODAY

        # 4. Save and Rerun
        if save_data(current_df):
             st.rerun() 


# --- Main Application Layout ---

def main():
    # Load the master data frame (uses cache)
    df_master = load_and_prepare_data()
    
    if df_master.empty:
        return

    st.title("🏗️ Society Migration Tracker (Kanban)")
    st.caption(f"Tracking {len(df_master)} Societies | Source: `{DATA_FILE}`")

    # --- 1. Sidebar Filters ---
    with st.sidebar:
        st.header("Filter View")
        
        cities = sorted(df_master['City'].dropna().unique())
        dark_stores = sorted(df_master['AssignedDarkStoreID'].dropna().unique())
        order_ranges = sorted(df_master['OrderRangeBucket'].dropna().unique())
        
        selected_city = st.multiselect("City", cities, default=[])
        selected_ds = st.multiselect("Dark Store ID", dark_stores, default=[])
        selected_order_range = st.multiselect("Order Range", order_ranges, default=[])

    # --- 2. Apply Filters ---
    filtered_df = df_master.copy()
    if selected_city:
        filtered_df = filtered_df[filtered_df['City'].isin(selected_city)]
    if selected_ds:
        filtered_df = filtered_df[filtered_df['AssignedDarkStoreID'].astype(str).isin([str(s) for s in selected_ds])]
    if selected_order_range:
        filtered_df = filtered_df[filtered_df['OrderRangeBucket'].isin(selected_order_range)]

    total_filtered = len(filtered_df)
    
    if total_filtered == 0:
        st.info("No societies match the current filter criteria.")
        return

    # --- 3. KPI Row ---
    migrated_count = len(filtered_df[filtered_df['MigrationStage'] == 'Migrated'])
    issues_count = len(filtered_df[filtered_df['MigrationStage'] == 'Issues'])
    
    col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)
    
    col_kpi_1.metric("Total Societies (Filtered)", total_filtered)
    col_kpi_2.metric("Migrated %", f"{migrated_count/total_filtered*100:.1f}%" if total_filtered > 0 else "0%")
    col_kpi_3.metric("Pending/In Progress", total_filtered - migrated_count - issues_count)
    col_kpi_4.metric("Societies in Issues", issues_count)

    st.divider()
    
    # --- 4. Kanban Board Display ---
    cols = st.columns(len(KANBAN_STAGES))

    for i, stage in enumerate(KANBAN_STAGES):
        # Filter data for the current stage, sorting by aging (DaysInStage)
        stage_data = filtered_df[filtered_df['MigrationStage'] == stage].sort_values(by='DaysInStage', ascending=False)
        
        with cols[i]:
            # Use HTML/Markdown for colored and centered headers
            header_color = "red" if stage == "Issues" else "#1F6F8B" # Corporate Blue
            st.markdown(
                f"<h4 style='color: {header_color}; text-align: center; margin-bottom: 5px;'>{stage} ({len(stage_data)})</h4>", 
                unsafe_allow_html=True
            )
            st.markdown(f"<div style='border-bottom: 3px solid {header_color}; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
            
            # Display Cards for each society
            for index, row in stage_data.iterrows():
                society_id = row['SocietyID']
                
                # Aging Alert Logic
                aging_threshold = 10
                is_stuck = row['DaysInStage'] >= aging_threshold and stage not in ['Migrated', 'Issues']
                
                # Card Styling
                card_style = "border: 1px solid #ddd; padding: 10px; margin-bottom: 12px; border-radius: 8px; background-color: white; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
                if is_stuck:
                    card_style = "border: 2px solid #FF9800; padding: 10px; margin-bottom: 12px; border-radius: 8px; background-color: #fff8e1; box-shadow: 2px 2px 8px rgba(255,152,0,0.5);"
                
                # Text Styling
                days_style = "color:#D32F2F; font-weight:bold;" if is_stuck else "color:#555;"
                
                card_html = f"""
                <div style='{card_style}'>
                    <p style='margin:0; font-size:16px;'><strong>{row['SocietyName']}</strong></p>
                    <p style='margin:0; font-size:12px; color:#333;'>ID: {society_id} | DS: {row['AssignedDarkStoreID']}</p>
                    <p style='margin-top:5px; margin-bottom:0; font-size:14px; color:#1F6F8B;'>Orders: {row['AvgOrdersPerDay']}</p>
                    <p style='margin:0; font-size:12px; {days_style}'>Aged: {row['DaysInStage']} days</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # Status change mechanism: A dropdown for status update
                # Using the full ID and stage in the key ensures uniqueness across all columns
                new_stage = st.selectbox(
                    "Move To:", 
                    KANBAN_STAGES, 
                    index=KANBAN_STAGES.index(stage), 
                    key=f"status_select_{society_id}_{stage}", 
                    label_visibility='collapsed'
                )

                # Check if user selected a new stage
                if new_stage != stage:
                    update_migration_status(society_id, new_stage, df_master)
                    
# Run the main function
if __name__ == "__main__":
    main()
