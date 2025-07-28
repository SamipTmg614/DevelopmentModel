import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
import importlib

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from developmentmodel import RegionalDevelopmentModel, load_and_preprocess_data
except ImportError:
    # Try alternative import path for deployment
    import importlib.util
    spec = importlib.util.spec_from_file_location("developmentmodel", 
                                                 os.path.join(os.path.dirname(__file__), "developmentmodel.py"))
    developmentmodel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(developmentmodel)
    RegionalDevelopmentModel = developmentmodel.RegionalDevelopmentModel
      with col2:
        st.markdown("### 🔗 Cluster Analysis")
        
        # Get cluster assignments for visualization
        cluster_labels = model.clusterer.predict(st.session_state.X_scaled)
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        fig = px.pie(
            values=cluster_counts.values,
            names=[f"Group {i+1}" for i in cluster_counts.index],
            title="Regional Cluster Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate average development index for each cluster
        st.markdown("**📊 Cluster Development Analysis:**")
        cluster_dev_analysis = []
        for cluster_id, count in cluster_counts.items():
            mask = cluster_labels == cluster_id
            avg_dev_index = df[mask]['Development_Index'].mean()
            percentage = (count / len(cluster_labels)) * 100
            
            cluster_dev_analysis.append({
                'Cluster': f"Group {cluster_id + 1}",
                'Regions': count,
                'Percentage': f"{percentage:.1f}%",
                'Avg Dev Index': f"{avg_dev_index:.4f}"
            })
        
        # Sort by average development index to show actual ranking
        cluster_df = pd.DataFrame(cluster_dev_analysis)
        cluster_df['Sort Key'] = cluster_df['Avg Dev Index'].astype(float)
        cluster_df = cluster_df.sort_values('Sort Key', ascending=False)
        cluster_df = cluster_df.drop('Sort Key', axis=1)
        
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        
        st.info("💡 **Note:** Clusters are ranked by actual average development index, not by group number.")

try:
    from developmentmodel import RegionalDevelopmentModel, load_and_preprocess_data
except ImportError:
    # Try alternative import path for deployment
    import importlib.util
    spec = importlib.util.spec_from_file_location("developmentmodel", 
                                                 os.path.join(os.path.dirname(__file__), "developmentmodel.py"))
    developmentmodel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(developmentmodel)
    RegionalDevelopmentModel = developmentmodel.RegionalDevelopmentModel
    load_and_preprocess_data = developmentmodel.load_and_preprocess_data

# Set page config
st.set_page_config(
    page_title="Regional Development Analysis",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_model_and_data():
    """Load and cache the model and data using session state"""
    # Use session state for caching instead of @st.cache_data
    if 'model_loaded' not in st.session_state:
        try:
            # Clear any import cache to ensure we get the latest version
            if 'developmentmodel' in sys.modules:
                importlib.reload(sys.modules['developmentmodel'])
            
            # Try different file paths for deployment
            possible_paths = [
                "./data/final/merged_cleaned_dataset.csv",
                "data/final/merged_cleaned_dataset.csv",
                "../data/final/merged_cleaned_dataset.csv"
            ]
            
            filepath = None
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath is None:
                raise FileNotFoundError("Dataset not found in any expected location")
                
            X_scaled, y_index, df, feature_names = load_and_preprocess_data(filepath)
            
            # Perform cross-validation to get realistic performance metrics
            from sklearn.model_selection import KFold
            from sklearn.metrics import mean_squared_error, r2_score
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_metrics = {
                'mse_scores': [],
                'r2_scores': [],
                'silhouette_scores': []
            }
            
            for train_idx, test_idx in kf.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_index_train, y_index_test = y_index.iloc[train_idx], y_index.iloc[test_idx]
                
                # Train model on fold - Test with default n_clusters first
                try:
                    fold_model = RegionalDevelopmentModel(n_clusters=4)  # Use optimal K=4
                except TypeError as e:
                    st.error(f"Error creating model with n_clusters: {e}")
                    fold_model = RegionalDevelopmentModel()  # Fall back to default
                    
                fold_model.train(X_train, y_index_train)
                
                # Evaluate on test set
                y_pred_index = fold_model.regressor.predict(X_test)
                
                # Calculate metrics
                cv_metrics['mse_scores'].append(mean_squared_error(y_index_test, y_pred_index))
                cv_metrics['r2_scores'].append(r2_score(y_index_test, y_pred_index))
                
                # Silhouette score for clustering
                from sklearn.metrics import silhouette_score
                cluster_labels = fold_model.clusterer.predict(X_test)
                cv_metrics['silhouette_scores'].append(silhouette_score(X_test, cluster_labels))
            
            # Train final model on full dataset for feature importance and predictions
            try:
                model = RegionalDevelopmentModel(n_clusters=4)  # Use optimal K=4
            except TypeError as e:
                st.error(f"Error creating final model with n_clusters: {e}")
                model = RegionalDevelopmentModel()  # Fall back to default
                
            model.train(X_scaled, y_index)
            
            # Get feature importance
            importance_df = model.get_feature_importance(feature_names)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.final_model = model  # Add this for consistency
            st.session_state.X_scaled = X_scaled
            st.session_state.y_index = y_index
            st.session_state.df = df
            st.session_state.feature_names = feature_names
            st.session_state.importance_df = importance_df
            st.session_state.cv_metrics = cv_metrics
            st.session_state.model_loaded = True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None, None, None, None, None, None
    
    # Return from session state
    return (st.session_state.model, st.session_state.X_scaled, st.session_state.y_index, 
            st.session_state.df, st.session_state.feature_names, 
            st.session_state.importance_df, st.session_state.cv_metrics)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏘️ Regional Development Analysis System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Development Index Prediction & Classification")
    
    # Add cache clearing button in sidebar
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        # Clear session state instead of cache_data
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()  # Rerun the app
    
    # Load model and data
    model, X_scaled, y_index, df, feature_names, importance_df, cv_metrics = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model and data. Please check your data files.")
        st.info("Try clicking the '🔄 Clear Cache & Reload' button in the sidebar.")
        return
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📊 Model Performance", "🔮 Make Predictions", "📈 Feature Analysis", "🔍 Search Areas", "🗺️ Regional Insights"]
    )
    
    if page == "🏠 Home":
        show_home_page(df, model, X_scaled, y_index, cv_metrics)
    
    elif page == "📊 Model Performance":
        show_model_performance(model, X_scaled, y_index, feature_names, cv_metrics)
    
    elif page == "🔮 Make Predictions":
        show_prediction_page(model, feature_names, df)
    
    elif page == "📈 Feature Analysis":
        show_feature_analysis(importance_df, df)
    
    elif page == "🔍 Search Areas":
        show_search_areas(df, model)
    
    elif page == "🗺️ Regional Insights":
        show_regional_insights(df, model)

def show_home_page(df, model, X_scaled, y_index, cv_metrics):
    """Home page with overview"""
    st.markdown('<h2 class="sub-header">📋 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Project Goal:**
        Predict regional development using AI algorithms combining:
        - **Regression** (Development Index Score)
        - **Clustering** (Regional Grouping)
        """)
        
        st.success("""
        **🔧 Algorithms Used:**
        - Random Forest Regressor
        - K-Means Clustering (K=4, optimized)
        """)
    
    with col2:
        st.markdown("**📊 Dataset Statistics:**")
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            st.metric("Total Regions", len(df))
        
        with col2_2:
            st.metric("Features", len(df.columns) - 3)
        
        with col2_3:
            st.metric("Data Quality", "99.1%")
    
    # Quick performance metrics from Cross-Validation
    st.markdown('<h3 class="sub-header">⚡ Model Performance Summary (5-Fold Cross-Validation)</h3>', unsafe_allow_html=True)
    
    # Use cross-validation metrics instead of training metrics
    avg_r2 = np.mean(cv_metrics['r2_scores'])
    avg_mse = np.mean(cv_metrics['mse_scores'])
    avg_silhouette = np.mean(cv_metrics['silhouette_scores'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{avg_r2:.4f}", f"{avg_r2*100:.1f}% variance explained")
    
    with col2:
        st.metric("MSE", f"{avg_mse:.6f}", "Cross-validated error")
    
    with col3:
        st.metric("Silhouette Score", f"{avg_silhouette:.3f}", "Clustering quality")

def show_model_performance(model, X_scaled, y_index, feature_names, cv_metrics):
    """Model performance analysis page"""
    st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Show Cross-Validation Results
    st.markdown("### 🔄 Cross-Validation Results (5-Fold)")
    st.info("**Note:** These metrics are from 5-fold cross-validation, showing realistic model performance on unseen data.")
    
    # Performance metrics from CV
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Regression Performance")
        avg_r2 = np.mean(cv_metrics['r2_scores'])
        avg_mse = np.mean(cv_metrics['mse_scores'])
        std_r2 = np.std(cv_metrics['r2_scores'])
        std_mse = np.std(cv_metrics['mse_scores'])
        
        st.metric("Average R² Score", f"{avg_r2:.4f}", f"±{std_r2:.4f}")
        st.metric("Average MSE", f"{avg_mse:.6f}", f"±{std_mse:.6f}")
        
        # Plot CV R² scores
        fig, ax = plt.subplots(figsize=(8, 6))
        folds = range(1, 6)
        ax.plot(folds, cv_metrics['r2_scores'], 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=avg_r2, color='red', linestyle='--', label=f'Average: {avg_r2:.4f}')
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('R² Score')
        ax.set_title('Cross-Validation R² Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(cv_metrics['r2_scores']) - 0.01, max(cv_metrics['r2_scores']) + 0.01])
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 🔗 Clustering Performance")
        avg_silhouette = np.mean(cv_metrics['silhouette_scores'])
        std_silhouette = np.std(cv_metrics['silhouette_scores'])
        
        st.metric("Average Silhouette Score", f"{avg_silhouette:.4f}", f"±{std_silhouette:.4f}")
        
        # Plot CV silhouette scores
        fig, ax = plt.subplots(figsize=(8, 6))
        folds = range(1, 6)
        ax.plot(folds, cv_metrics['silhouette_scores'], 'go-', linewidth=2, markersize=8)
        ax.axhline(y=avg_silhouette, color='red', linestyle='--', label=f'Average: {avg_silhouette:.4f}')
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Cross-Validation Silhouette Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(cv_metrics['silhouette_scores']) - 0.01, max(cv_metrics['silhouette_scores']) + 0.01])
        st.pyplot(fig)
    
    # Clustering Quality Information
    st.markdown("### 🔗 Clustering Quality")
    col3, col4 = st.columns(2)
    
    with col3:
        st.info("**Silhouette Score Interpretation:**\n"
                "- 0.7-1.0: Strong clustering\n"
                "- 0.5-0.7: Reasonable clustering\n"  
                "- 0.25-0.5: Weak clustering\n"
                "- Below 0.25: Poor clustering")
    
    with col4:
        avg_silhouette = np.mean(cv_metrics['silhouette_scores'])
        std_silhouette = np.std(cv_metrics['silhouette_scores'])
        st.metric("Average Silhouette Score", f"{avg_silhouette:.4f}", f"±{std_silhouette:.4f}")
    
    # Show individual fold results in an expandable section
    with st.expander("📋 Detailed Cross-Validation Results by Fold"):
        cv_results_df = pd.DataFrame({
            'Fold': range(1, 6),
            'R² Score': cv_metrics['r2_scores'],
            'MSE': cv_metrics['mse_scores'],
            'Silhouette': cv_metrics['silhouette_scores']
        })
        st.dataframe(cv_results_df.round(6))
    
    # Training Data Visualization (for reference only)
    st.markdown("### 📊 Training Data Visualization (Reference Only)")
    st.warning("**Note:** The plots below show the model's fit to the training data and are for visualization purposes only. "
               "The actual performance metrics above are from cross-validation on unseen data.")
    
    # Make predictions on training data for visualization
    y_pred_index = model.regressor.predict(X_scaled)
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Actual vs Predicted plot on training data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_index, y_pred_index, alpha=0.6, color='blue', s=20)
        min_val = min(min(y_index), min(y_pred_index))
        max_val = max(max(y_index), max(y_pred_index))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Development Index')
        ax.set_ylabel('Predicted Development Index')
        ax.set_title('Training Data: Actual vs Predicted\n(For visualization only)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col6:
        # Cluster distribution on training data
        cluster_labels = model.clusterer.predict(X_scaled)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([f'Group {i+1}' for i in unique], counts, color=['#ff7f7f', '#ffb347', '#87ceeb', '#98fb98'][:len(unique)])
        ax.set_title('Cluster Distribution')
        ax.set_xlabel('Cluster Group')
        ax.set_ylabel('Number of Regions')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   str(count), ha='center', va='bottom')
        
        st.pyplot(fig)

def show_prediction_page(model, feature_names, df):
    """Interactive prediction page"""
    st.markdown('<h2 class="sub-header">🔮 Make Development Predictions</h2>', unsafe_allow_html=True)
    
    # Two tabs: Single prediction and Batch prediction
    tab1, tab2 = st.tabs(["🏘️ Single Region", "📁 Batch Upload"])
    
    with tab1:
        st.markdown("### Enter Regional Infrastructure Data")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏥 Health Infrastructure:**")
            total_wards = st.number_input("Total Wards", min_value=1, value=10)
            health_1_2 = st.number_input("Wards with 1-2 Health Posts", min_value=0, value=3)
            health_3_4 = st.number_input("Wards with 3-4 Health Posts", min_value=0, value=2)
            health_5_9 = st.number_input("Wards with 5-9 Health Posts", min_value=0, value=1)
            health_10_plus = st.number_input("Wards with 10+ Health Posts", min_value=0, value=0)
            doctor_access = st.slider("Access to Doctor in 30 mins (%)", 0.0, 1.0, 0.7, 0.1)
            
        with col2:
            st.markdown("**🏫 Education Infrastructure:**")
            school_1_2 = st.number_input("Wards with 1-2 Schools", min_value=0, value=4)
            school_3_4 = st.number_input("Wards with 3-4 Schools", min_value=0, value=3)
            school_5_9 = st.number_input("Wards with 5-9 Schools", min_value=0, value=2)
            school_10_plus = st.number_input("Wards with 10+ Schools", min_value=0, value=1)
            
            st.markdown("**🚨 Other Services:**")
            fire_access = st.slider("Access to Fire Brigade in 30 mins (%)", 0.0, 1.0, 0.5, 0.1)
            library_access = st.slider("Access to Library in 30 mins (%)", 0.0, 1.0, 0.6, 0.1)
            higher_ed_access = st.slider("Higher Education in 30 mins (%)", 0.0, 1.0, 0.4, 0.1)
        
        if st.button("🔮 Predict Development", type="primary"):
            # Prepare input data
            input_data = prepare_single_prediction_data(
                total_wards, health_1_2, health_3_4, health_5_9, health_10_plus,
                school_1_2, school_3_4, school_5_9, school_10_plus,
                doctor_access, fire_access, library_access, higher_ed_access, df
            )
            
            # Make prediction
            if input_data is not None:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                # Fit scaler on original data features
                original_features = df.drop(columns=['Area', 'Development_Index']).select_dtypes(include=np.number)
                scaler.fit(original_features)
                
                input_scaled = scaler.transform(input_data)
                
                pred_index = model.regressor.predict(input_scaled)[0]
                cluster = model.clusterer.predict(input_scaled)[0]
                
                # Display results
                st.success("### 🎉 Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Development Index", f"{pred_index:.4f}")
                with col2:
                    st.metric("Cluster Group", f"Group {cluster + 1}")
                
                # Cluster-based interpretation (you'll need to implement this)
                show_cluster_recommendations(cluster, pred_index)
    
    with tab2:
        st.markdown("### Upload CSV for Batch Predictions")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())
                
                if st.button("Process Batch Predictions"):
                    # Process batch predictions here
                    st.info("Batch prediction functionality can be implemented based on your specific CSV format.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

def prepare_single_prediction_data(total_wards, health_1_2, health_3_4, health_5_9, health_10_plus,
                                 school_1_2, school_3_4, school_5_9, school_10_plus,
                                 doctor_access, fire_access, library_access, higher_ed_access, df):
    """Prepare single prediction input data"""
    try:
        # Create a dataframe with the same structure as training data
        data = {
            'Total_Wards': total_wards,
            '1_2 healthpost': health_1_2 / total_wards,
            '3_4 healthpost': health_3_4 / total_wards,
            '5_9 healthpost': health_5_9 / total_wards,
            '10_plus healthpost': health_10_plus / total_wards,
            'with 1_2 school': school_1_2 / total_wards,
            'with 3_4 school': school_3_4 / total_wards,
            'with 5_9 school': school_5_9 / total_wards,
            '10_plus schools': school_10_plus / total_wards,
            'access to doctor in 30 mins': doctor_access,
            'access to firebrigade in 30 mins': fire_access,
            'access to library in 30 mins': library_access,
            'higher education in 30 mins': higher_ed_access,
            'None healthpost': max(0, (total_wards - health_1_2 - health_3_4 - health_5_9 - health_10_plus) / total_wards)
        }
        
        # Calculate derived features
        data['Health_Access'] = (
            data['1_2 healthpost'] * 1.5 + 
            data['3_4 healthpost'] * 3.5 +
            data['5_9 healthpost'] * 7 +
            data['10_plus healthpost'] * 10
        )
        
        data['Education_Access'] = (
            data['with 1_2 school'] * 1.5 + 
            data['with 3_4 school'] * 3.5 +
            data['with 5_9 school'] * 7 +
            data['10_plus schools'] * 10
        )
        
        # Add any missing features with default values
        original_features = df.drop(columns=['Area', 'Development_Index']).select_dtypes(include=np.number)
        for col in original_features.columns:
            if col not in data:
                data[col] = 0.0
        
        # Create DataFrame and reorder columns to match training data
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=original_features.columns, fill_value=0.0)
        
        return input_df
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {e}")
        return None

def show_cluster_recommendations(cluster, index):
    """Show cluster-based development recommendations with dynamic cluster analysis"""
    st.markdown("### 💡 Development Recommendations")
    
    # Get cluster analysis from session state to determine actual development levels
    if 'cluster_analysis' not in st.session_state:
        # Calculate cluster analysis once and store it
        cluster_labels = st.session_state.model.clusterer.predict(st.session_state.X_scaled)
        cluster_dev_index = {}
        
        for i in range(4):  # Assuming 4 clusters
            mask = cluster_labels == i
            if np.sum(mask) > 0:
                avg_dev_index = st.session_state.y_index[mask].mean()
                cluster_dev_index[i] = avg_dev_index
        
        # Sort clusters by average development index to get proper ranking
        sorted_clusters = sorted(cluster_dev_index.items(), key=lambda x: x[1])
        cluster_ranking = {cluster_id: rank for rank, (cluster_id, _) in enumerate(sorted_clusters)}
        
        st.session_state.cluster_analysis = {
            'cluster_dev_index': cluster_dev_index,
            'cluster_ranking': cluster_ranking
        }
    
    # Get the actual development level ranking for this cluster
    cluster_ranking = st.session_state.cluster_analysis['cluster_ranking'][cluster]
    cluster_avg_dev = st.session_state.cluster_analysis['cluster_dev_index'][cluster]
    
    # Show cluster statistics
    st.info(f"**Cluster {cluster + 1} Statistics:**\n"
            f"- Your Development Index: {index:.4f}\n"
            f"- Cluster Average: {cluster_avg_dev:.4f}\n"
            f"- Development Ranking: {cluster_ranking + 1}/4 (1=Lowest, 4=Highest)")
    
    # Dynamic recommendations based on actual development ranking
    if cluster_ranking == 0:  # Lowest development cluster
        st.warning("""
        **Priority Actions for Low Development Group:**
        - 🏫 **Urgent:** Build basic educational infrastructure
        - 🏥 **Critical:** Establish primary healthcare facilities
        - 🚒 **Essential:** Set up emergency services access
        - 📚 **Important:** Create community learning centers
        - 💡 **Focus:** Basic infrastructure development
        """)
        interpretation = "🔴 Low Development Level"
        color = "red"
    elif cluster_ranking == 1:  # Second lowest
        st.info("""
        **Enhancement Strategies for Medium-Low Development Group:**
        - 🎓 Improve higher education accessibility
        - 🛣️ Develop transportation infrastructure  
        - 📋 Optimize service distribution across wards
        - 🏛️ Strengthen public service delivery
        - 💡 **Focus:** Service accessibility improvement
        """)
        interpretation = "🟡 Medium-Low Development Level"
        color = "orange"
    elif cluster_ranking == 2:  # Second highest
        st.info("""
        **Growth Strategies for Medium-High Development Group:**
        - 🏢 Enhance service quality and coverage
        - 🌐 Improve digital connectivity and access
        - 📈 Focus on sustainable development practices
        - 🔧 Upgrade and modernize existing infrastructure
        - 💡 **Focus:** Quality and sustainability improvements
        """)
        interpretation = "🟠 Medium-High Development Level"
        color = "blue"
    else:  # Highest development cluster (ranking == 3)
        st.success("""
        **Maintenance Strategies for High Development Group:**
        - ✅ Maintain current high service levels
        - 🔄 Regular infrastructure updates and modernization
        - 📊 Implement advanced monitoring systems
        - 🤝 Share best practices with developing regions
        - 💡 **Focus:** Excellence maintenance and knowledge sharing
        """)
        interpretation = "🟢 High Development Level" 
        color = "green"
    
    st.markdown(f"**Interpretation:** {interpretation}")
    
    # Add development context
    if index < cluster_avg_dev:
        st.warning(f"⚠️ Your development index ({index:.4f}) is below the cluster average ({cluster_avg_dev:.4f}). Consider prioritizing the recommendations above.")
    elif index > cluster_avg_dev:
        st.success(f"🎉 Your development index ({index:.4f}) is above the cluster average ({cluster_avg_dev:.4f}). You're performing well within this group!")
    else:
        st.info(f"📊 Your development index ({index:.4f}) is close to the cluster average ({cluster_avg_dev:.4f}).")

def show_recommendations(tier, index):
    """Show development recommendations"""
    st.markdown("### 💡 Development Recommendations")
    
    if tier == "Low":
        st.warning("""
        **Priority Actions for Low Development Regions:**
        - 🏫 Increase educational infrastructure (schools and access)
        - 🏥 Improve healthcare accessibility within 30 minutes
        - 🚒 Establish emergency services coverage
        - 📚 Build libraries and information centers
        """)
    elif tier == "Medium":
        st.info("""
        **Enhancement Strategies for Medium Development Regions:**
        - 🎓 Focus on higher education accessibility
        - 🛣️ Improve transportation infrastructure
        - 📋 Optimize service distribution across wards
        - 🏛️ Strengthen public services
        """)
    else:
        st.success("""
        **Maintenance Strategies for High Development Regions:**
        - ✅ Maintain current service levels
        - 🔄 Regular infrastructure updates
        - 📊 Monitor service quality
        - 🤝 Share best practices with other regions
        """)

def show_feature_analysis(importance_df, df):
    """Feature importance analysis page"""
    st.markdown('<h2 class="sub-header">📈 Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Most Important Features")
        
        # Interactive feature importance plot
        fig = px.bar(
            importance_df.head(15), 
            x='importance', 
            y='feature',
            orientation='h',
            title='Top 15 Feature Importance for Development Prediction',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Key Insights")
        
        top_3 = importance_df.head(3)
        for idx, row in top_3.iterrows():
            st.metric(
                f"#{idx+1} {row['feature']}", 
                f"{row['importance']:.1%}",
                help=f"Contributes {row['importance']:.1%} to model decisions"
            )
        
        st.markdown("### 📈 Feature Categories")
        
        # Categorize features
        education_features = importance_df[importance_df['feature'].str.contains('school|Education|education', case=False)]
        health_features = importance_df[importance_df['feature'].str.contains('health|doctor', case=False)]
        access_features = importance_df[importance_df['feature'].str.contains('access|30 mins', case=False)]
        
        st.write(f"🎓 Education: {len(education_features)} features")
        st.write(f"🏥 Health: {len(health_features)} features")
        st.write(f"🚗 Access: {len(access_features)} features")

def show_search_areas(df, model):
    """Search and explore specific areas"""
    st.markdown('<h2 class="sub-header">🔍 Search Regional Areas</h2>', unsafe_allow_html=True)
    
    # Search functionality
    st.markdown("### 🔎 Find Specific Areas")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input(
            "Enter area name to search:",
            placeholder="e.g., Kathmandu, Pokhara, Dharan...",
            help="Search is case-insensitive and supports partial matches"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Advanced search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Remove tier_filter since Development_Tier doesn't exist
            st.info("**Note:** Development tier filtering temporarily disabled due to model update.")
        
        with col2:
            min_index = st.number_input(
                "Minimum Development Index:",
                min_value=0.0,
                max_value=float(df['Development_Index'].max()),
                value=0.0,
                step=0.1,
                format="%.2f"
            )
        
        with col3:
            max_index = st.number_input(
                "Maximum Development Index:",
                min_value=0.0,
                max_value=float(df['Development_Index'].max()),
                value=float(df['Development_Index'].max()),
                step=0.1,
                format="%.2f"
            )
    
    # Filter data based on search criteria
    filtered_df = df.copy()
    
    # Apply index range filter
    filtered_df = filtered_df[
        (filtered_df['Development_Index'] >= min_index) & 
        (filtered_df['Development_Index'] <= max_index)
    ]
    
    # Apply text search
    if search_term:
        # Case-insensitive search
        mask = filtered_df['Area'].str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Display results
    if len(filtered_df) > 0:
        st.markdown(f"### 📋 Search Results ({len(filtered_df)} areas found)")
        
        # Summary statistics for search results
        if len(filtered_df) > 1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Areas Found", len(filtered_df))
            
            with col2:
                avg_index = filtered_df['Development_Index'].mean()
                st.metric("Avg Dev. Index", f"{avg_index:.4f}")
            
            with col3:
                max_index = filtered_df['Development_Index'].max()
                st.metric("Max Dev. Index", f"{max_index:.4f}")
            
            with col4:
                top_area = filtered_df.loc[filtered_df['Development_Index'].idxmax(), 'Area']
                st.metric("Highest Developed", top_area[:15] + "..." if len(top_area) > 15 else top_area)
        
        # Detailed results table
        st.markdown("### 📊 Detailed Area Information")
        
        # Select columns to display
        display_columns = [
            'Area', 'Development_Index', 'Total_Wards',
            'Health_Access', 'Education_Access', 'Infrastructure_Score',
            'access to doctor in 30 mins', 'access to firebrigade in 30 mins'
        ]
        
        # Ensure columns exist in dataframe
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # Create display dataframe
        display_df = filtered_df[available_columns].copy()
        display_df = display_df.sort_values('Development_Index', ascending=False)
        
        # Round numerical columns
        numeric_columns = display_df.select_dtypes(include=[np.number]).columns
        display_df[numeric_columns] = display_df[numeric_columns].round(4)
        
        # Display with pagination for large results
        if len(display_df) > 20:
            st.info(f"Showing top 20 results out of {len(display_df)} total matches. Use filters to narrow down results.")
            display_df = display_df.head(20)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Detailed view for single area selection
        if len(filtered_df) <= 10:  # Only show detailed view for small result sets
            st.markdown("### 🔍 Detailed Area Analysis")
            
            selected_area = st.selectbox(
                "Select an area for detailed analysis:",
                options=filtered_df['Area'].tolist(),
                key="area_selector"
            )
            
            if selected_area:
                show_area_details(filtered_df, selected_area, model)
        
        # Visualization for multiple results
        if len(filtered_df) > 1:
            st.markdown("### 📈 Search Results Visualization")
            
            tab1, tab2 = st.tabs(["📊 Development Comparison", "🎯 Feature Analysis"])
            
            with tab1:
                # Development index comparison (simplified without tier colors)
                fig = px.bar(
                    filtered_df.head(15),  # Limit to top 15 for readability
                    x='Development_Index',
                    y='Area',
                    title='Development Index Comparison',
                    orientation='h'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Feature comparison for selected areas
                if len(filtered_df) <= 5:
                    feature_cols = ['Health_Access', 'Education_Access', 'Infrastructure_Score']
                    available_features = [col for col in feature_cols if col in filtered_df.columns]
                    
                    if available_features:
                        feature_data = filtered_df[['Area'] + available_features].set_index('Area')
                        
                        fig = px.bar(
                            feature_data.T,
                            title='Feature Comparison Across Selected Areas',
                            labels={'index': 'Features', 'value': 'Score'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select fewer areas (≤5) to see detailed feature comparison.")
    
    else:
        st.warning("🔍 No areas found matching your search criteria. Try:")
        st.write("- Using partial names (e.g., 'Kath' for Kathmandu)")
        st.write("- Checking spelling")
        st.write("- Adjusting the development index range")
        st.write("- Changing the tier filter")
        
        # Show some example areas
        st.markdown("### 💡 Example Areas You Can Search:")
        sample_areas = df['Area'].sample(n=min(10, len(df))).tolist()
        cols = st.columns(2)
        for i, area in enumerate(sample_areas):
            with cols[i % 2]:
                st.write(f"• {area}")

def show_area_details(df, area_name, model):
    """Show detailed information for a specific area"""
    area_data = df[df['Area'] == area_name].iloc[0]
    
    st.markdown(f"#### 📍 {area_name}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Show cluster assignment instead of tier
        cluster_labels = model.clusterer.predict(st.session_state.X_scaled)
        area_idx = df[df['Area'] == area_name].index[0]
        cluster = cluster_labels[area_idx]
        st.metric("Cluster Group", f"Group {cluster + 1}")
    
    with col2:
        st.metric("Development Index", f"{area_data['Development_Index']:.4f}")
    
    with col3:
        st.metric("Total Wards", int(area_data['Total_Wards']))
    
    with col4:
        # Calculate rank
        rank = (df['Development_Index'] > area_data['Development_Index']).sum() + 1
        st.metric("Development Rank", f"#{rank} of {len(df)}")
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏥 Health Infrastructure:**")
        if 'Health_Access' in area_data:
            st.write(f"Health Access Score: {area_data['Health_Access']:.4f}")
        if 'access to doctor in 30 mins' in area_data:
            st.write(f"Doctor Access (30 min): {area_data['access to doctor in 30 mins']:.1%}")
        
        st.markdown("**🚨 Emergency Services:**")
        if 'access to firebrigade in 30 mins' in area_data:
            st.write(f"Fire Brigade Access: {area_data['access to firebrigade in 30 mins']:.1%}")
    
    with col2:
        st.markdown("**🏫 Education Infrastructure:**")
        if 'Education_Access' in area_data:
            st.write(f"Education Access Score: {area_data['Education_Access']:.4f}")
        if 'higher education in 30 mins' in area_data:
            st.write(f"Higher Education Access: {area_data['higher education in 30 mins']:.1%}")
        
        st.markdown("**🛣️ Infrastructure:**")
        if 'Infrastructure_Score' in area_data:
            st.write(f"Infrastructure Score: {area_data['Infrastructure_Score']:.4f}")
    
    # Comparison with averages
    st.markdown("**📊 Comparison with Regional Averages:**")
    
    comparison_metrics = ['Development_Index', 'Health_Access', 'Education_Access', 'Infrastructure_Score']
    available_metrics = [col for col in comparison_metrics if col in df.columns]
    
    if available_metrics:
        comparison_data = []
        for metric in available_metrics:
            area_value = area_data[metric]
            avg_value = df[metric].mean()
            difference = area_value - avg_value
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Area Value': f"{area_value:.4f}",
                'Regional Average': f"{avg_value:.4f}",
                'Difference': f"{difference:+.4f}",
                'Status': '🟢 Above Average' if difference > 0 else '🔴 Below Average'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

def show_regional_insights(df, model):
    """Regional insights and data exploration"""
    st.markdown('<h2 class="sub-header">🗺️ Regional Development Insights</h2>', unsafe_allow_html=True)
    
    # Development tier distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Development Index Distribution")
        
        fig = px.histogram(
            df, 
            x='Development_Index',
            nbins=30,
            title="Development Index Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title="Development Index",
            yaxis_title="Number of Regions"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Average Development Index", f"{df['Development_Index'].mean():.4f}")
        st.metric("Standard Deviation", f"{df['Development_Index'].std():.4f}")
    
    with col2:
        st.markdown("### � Cluster Analysis")
        
        # Get cluster assignments for visualization
        cluster_labels = model.clusterer.predict(st.session_state.X_scaled)
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        fig = px.pie(
            values=cluster_counts.values,
            names=[f"Group {i+1}" for i in cluster_counts.index],
            title="Regional Cluster Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(cluster_labels)) * 100
            st.write(f"**Group {cluster_id + 1}**: {count} regions ({percentage:.1f}%)")
    
    # Top and bottom performing regions
    st.markdown("### 🏆 Regional Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥇 Top 10 Developed Regions**")
        top_regions = df.nlargest(10, 'Development_Index')[['Area', 'Development_Index']]
        st.dataframe(top_regions, use_container_width=True)
    
    with col2:
        st.markdown("**📈 Bottom 10 Regions (Growth Potential)**")
        bottom_regions = df.nsmallest(10, 'Development_Index')[['Area', 'Development_Index']]
        st.dataframe(bottom_regions, use_container_width=True)

if __name__ == "__main__":
    main()
