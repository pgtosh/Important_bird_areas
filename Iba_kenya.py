import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Define the shapefile path
SHAPEFILE_PATH = "C:/Users/Pro/Downloads/Compressed/ke_iba-status"

# 1. Data Loading and Initial Preprocessing
def load_and_preprocess_data():
    """
    Load and preprocess the IBA shapefile data
    """
    try:
        # Read the shapefile using the defined path
        gdf = gpd.read_file(SHAPEFILE_PATH)
        print(f"Successfully loaded {len(gdf)} IBA records.")
        
        # Create status mappings
        status_mapping = {
            0: 'Unknown',
            1: 'Major Decline',
            2: 'Decline/Slight decline',
            3: 'Stable/No change',
            4: 'Improvement/Slight improvement',
            5: 'Major improvement'
        }
        
        # Add categorical status
        gdf['STATUS_CATEGORY'] = gdf['STATUS2'].map(status_mapping)
        print("Data preprocessing completed successfully.")
        
        return gdf
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# 2. Exploratory Data Analysis
def perform_eda(gdf):
    """
    Perform exploratory data analysis and create visualizations
    """
    try:
        print("\nStarting Exploratory Data Analysis...")
        
        # 1. Status Distribution
        plt.figure(figsize=(12, 6))
        status_counts = gdf['STATUS_CATEGORY'].value_counts()
        plt.bar(status_counts.index, status_counts.values)
        plt.title('Distribution of IBA Status in Kenya')
        plt.xlabel('Status Category')
        plt.ylabel('Number of Sites')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Print status distribution
        print("\nStatus Distribution:")
        for status, count in status_counts.items():
            print(f"{status}: {count} sites")
        
        # 2. Status Distribution by Percentage
        plt.figure(figsize=(10, 10))
        plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        plt.title('Percentage Distribution of IBA Status')
        plt.tight_layout()
        plt.show()
        
        # 3. Geographic Distribution of Status
        fig, ax = plt.subplots(figsize=(15, 10))
        gdf.plot(column='STATUS2', cmap='RdYlGn', legend=True, 
                legend_kwds={'label': 'IBA Status'}, ax=ax)
        plt.title('Geographic Distribution of IBA Status')
        plt.axis('off')
        plt.show()
        
        # 4. Print summary statistics
        print("\nNumerical Summary Statistics for Status:")
        print(gdf['STATUS2'].describe())
        
        return status_counts
    except Exception as e:
        print(f"Error in EDA: {str(e)}")
        return None

# 3. Create Interactive Map
def create_interactive_map(gdf):
    """
    Create an interactive folium map with IBA locations
    """
    try:
        print("\nCreating interactive map...")
        # Calculate the center point
        center = gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()
        
        # Create base map
        m = folium.Map(location=center, zoom_start=7)
        
        # Color mapping for status
        color_mapping = {
            0: 'gray',    # Unknown
            1: 'red',     # Major Decline
            2: 'orange',  # Decline/Slight decline
            3: 'yellow',  # Stable/No change
            4: 'lightgreen',  # Improvement
            5: 'darkgreen'    # Major improvement
        }
        
        # Add points with popup information
        for idx, row in gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                popup=folium.Popup(
                    f"""
                    <b>Site:</b> {row['INTERNATIO']}<br>
                    <b>Status:</b> {row['STATUS_CATEGORY']}<br>
                    <b>Description:</b> {row['STAT_EXPL']}
                    """,
                    max_width=300
                ),
                color=color_mapping[row['STATUS2']],
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px">
            <h4>IBA Status</h4>
            <p><span style="color: gray">●</span> Unknown</p>
            <p><span style="color: red">●</span> Major Decline</p>
            <p><span style="color: orange">●</span> Decline/Slight decline</p>
            <p><span style="color: yellow">●</span> Stable/No change</p>
            <p><span style="color: lightgreen">●</span> Improvement</p>
            <p><span style="color: darkgreen">●</span> Major improvement</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        print("Interactive map created successfully.")
        return m
    except Exception as e:
        print(f"Error creating map: {str(e)}")
        return None

# 4. Predictive Modeling
def create_predictive_models(gdf):
    """
    Create and evaluate multiple predictive models
    """
    try:
        print("\nStarting predictive modeling...")
        # Prepare features and target
        X = gdf[['STATUS2']]
        y = LabelEncoder().fit_transform(gdf['STAT_EXPL'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # 1. Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # 2. Logistic Regression Model
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Print model evaluations
        print("\nRandom Forest Model Performance:")
        print(classification_report(y_test, rf_pred))
        
        print("\nLogistic Regression Model Performance:")
        print(classification_report(y_test, lr_pred))
        
        # Create confusion matrix visualization
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(confusion_matrix(y_test, rf_pred))
        plt.colorbar()
        plt.title('Random Forest Confusion Matrix')
        
        plt.subplot(1, 2, 2)
        plt.imshow(confusion_matrix(y_test, lr_pred))
        plt.colorbar()
        plt.title('Logistic Regression Confusion Matrix')
        
        plt.tight_layout()
        plt.show()
        
        return rf_model, lr_model
    except Exception as e:
        print(f"Error in predictive modeling: {str(e)}")
        return None, None

# 5. Spatial Analysis
def perform_spatial_analysis(gdf):
    """
    Perform spatial analysis on the IBA locations
    """
    try:
        print("\nPerforming spatial analysis...")
        # Calculate distances between IBAs
        distances = []
        for idx1, row1 in gdf.iterrows():
            for idx2, row2 in gdf.iterrows():
                if idx1 < idx2:  # Avoid duplicate pairs
                    dist = row1.geometry.distance(row2.geometry)
                    distances.append(dist)
        
        # Plot distance distribution
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=30)
        plt.title('Distribution of Distances Between IBAs')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.show()
        
        # Calculate and plot spatial clustering of status
        status_coords = pd.DataFrame({
            'status': gdf['STATUS2'],
            'lat': gdf.geometry.y,
            'lon': gdf.geometry.x
        })
        
        plt.figure(figsize=(12, 6))
        plt.scatter(status_coords['lon'], status_coords['lat'], 
                   c=status_coords['status'], cmap='RdYlGn', 
                   s=100, alpha=0.6)
        plt.colorbar(label='Status')
        plt.title('Spatial Clustering of IBA Status')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
        
        print("Spatial analysis completed successfully.")
    except Exception as e:
        print(f"Error in spatial analysis: {str(e)}")

# Main execution function
def main():
    """
    Main function to run all analyses
    """
    try:
        # Load and preprocess data
        print(f"Loading data from: {SHAPEFILE_PATH}")
        gdf = load_and_preprocess_data()
        
        if gdf is not None:
            # Perform EDA
            status_counts = perform_eda(gdf)
            
            # Create interactive map
            interactive_map = create_interactive_map(gdf)
            
            # Create predictive models
            rf_model, lr_model = create_predictive_models(gdf)
            
            # Perform spatial analysis
            perform_spatial_analysis(gdf)
            
            print("\nAnalysis completed successfully!")
            return gdf, interactive_map, rf_model, lr_model
        else:
            print("Failed to load data. Please check your shapefile path.")
            return None, None, None, None
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None, None, None, None

# Run the analysis
if __name__ == "__main__":
    gdf, interactive_map, rf_model, lr_model = main()