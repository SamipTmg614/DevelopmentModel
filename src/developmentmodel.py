import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

FILEPATH = "./data/final/merged_cleaned_dataset.csv"

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df = df[df['Total_Wards'] != 0]

    for col in df.columns[2:]:
        if not col.startswith('access') and col != 'Total_Wards':
            df[col] = df[col] / df['Total_Wards']

    df['Health_Access'] = (
        df['1_2 healthpost'] * 1.5 + 
        df['3_4 healthpost'] * 3.5 +
        df['5_9 healthpost'] * 7 +
        df['10_plus healthpost'] * 10
    )

    df['Education_Access'] = (
        df['with 1_2 school'] * 1.5 + 
        df['with 3_4 school'] * 3.5 +
        df['with 5_9 school'] * 7 +
        df['10_plus schools'] * 10
    )

    df['Infrastructure_Score'] = (
        df['road 50 plus'] * 0.4 + 
        df['road less than 50'] * 0.3 +
        df['road less than 10'] * 0.2
    )

    df['Development_Index'] = (
        0.3 * df['Health_Access'] + 
        0.3 * df['Education_Access'] + 
        0.2 * df['Infrastructure_Score'] +
        0.1 * df['access to doctor in 30 mins'] +
        0.1 * df['access to firebrigade in 30 mins']
    )

    df['Development_Tier'] = pd.qcut(
        df['Development_Index'], 
        q=[0, 0.25, 0.75, 1], 
        labels=['Low', 'Medium', 'High']
    )

    features = df.drop(columns=['Area', 'Development_Index', 'Development_Tier'])
    X = features.select_dtypes(include=np.number)
    y_index = df['Development_Index']
    y_tier = df['Development_Tier']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_index, y_tier, df, X.columns.tolist()

class RegionalDevelopmentModel:
    def __init__(self, n_clusters=3):
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, X, y_index, y_tier):
        self.regressor.fit(X, y_index)
        self.classifier.fit(X, y_tier)
        self.clusterer.fit(X)

    def evaluate(self, X, y_index, y_tier):
        y_pred_index = self.regressor.predict(X)
        mse = mean_squared_error(y_index, y_pred_index)
        r2 = r2_score(y_index, y_pred_index)

        y_pred_tier = self.classifier.predict(X)
        clf_report = classification_report(y_tier, y_pred_tier)

        silhouette = silhouette_score(X, self.clusterer.predict(X))

        return {'regression': {'MSE': mse, 'R2': r2, 'predictions': y_pred_index}, 'classification_report': clf_report, 'silhouette_score': silhouette}
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the trained regressor"""
        importance = self.regressor.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    importance_df = model.get_feature_importance(feature_names)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance for Development Index Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_actual_vs_predicted(y_actual, y_predicted, fold_num=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_actual, y_predicted, alpha=0.6, color='blue', s=20)
    
    # Perfect prediction line
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = r2_score(y_actual, y_predicted)
    
    plt.xlabel('Actual Development Index')
    plt.ylabel('Predicted Development Index')
    title = f'Actual vs Predicted Development Index'
    if fold_num:
        title += f' (Fold {fold_num})'
    title += f'\nR² = {r2:.4f}'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_actual, y_predicted, fold_num=None):
    """Plot residuals"""
    residuals = y_actual - y_predicted
    
    plt.figure(figsize=(12, 5))
    
    # Residuals vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_predicted, residuals, alpha=0.6, color='green', s=20)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    title = 'Residuals vs Predicted'
    if fold_num:
        title += f' (Fold {fold_num})'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def find_optimal_clusters(X, max_clusters=10):
    """
    Use the elbow method to find the optimal number of clusters
    """
    print("\n=== Finding Optimal Number of Clusters (Elbow Method) ===")
    
    # Calculate WCSS (Within-Cluster Sum of Squares) for different k values
    wcss = []
    k_range = range(1, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        print(f"K={k}: WCSS = {kmeans.inertia_:.2f}")
    
    # Plot the elbow curve
    plt.figure(figsize=(12, 5))
    
    # Elbow curve
    plt.subplot(1, 2, 1)
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)
    
    # Calculate the rate of change to help identify the elbow
    rates = []
    for i in range(1, len(wcss)):
        rate = wcss[i-1] - wcss[i]
        rates.append(rate)
    
    # Plot rate of change
    plt.subplot(1, 2, 2)
    plt.plot(k_range[1:], rates, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Rate of WCSS Decrease')
    plt.title('Rate of WCSS Decrease')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find the elbow point using the "elbow method"
    # Look for the point where the rate of decrease starts to slow down
    optimal_k = find_elbow_point(wcss)
    
    print(f"\n📊 WCSS Analysis:")
    for i, k in enumerate(k_range):
        print(f"K={k}: WCSS = {wcss[i]:.2f}")
    
    print(f"\n🎯 Suggested Optimal K: {optimal_k}")
    print(f"💡 The elbow method suggests using {optimal_k} clusters")
    
    return optimal_k, wcss

def find_elbow_point(wcss):
    """
    Find the elbow point using the method of maximum curvature
    """
    # Convert to numpy array for easier calculation
    wcss = np.array(wcss)
    n_points = len(wcss)
    
    # Simple method: find the point with maximum distance to the line connecting first and last points
    if n_points < 3:
        return 2  # Default to 2 if we don't have enough points
    
    # Create points array
    points = np.array([[i, wcss[i]] for i in range(n_points)])
    
    # Calculate distances from each point to the line connecting first and last points
    first_point = points[0]
    last_point = points[-1]
    
    distances = []
    for i in range(1, n_points - 1):  # Skip first and last points
        point = points[i]
        # Calculate perpendicular distance from point to line
        distance = np.abs(np.cross(last_point - first_point, first_point - point)) / np.linalg.norm(last_point - first_point)
        distances.append(distance)
    
    # Find the point with maximum distance (the elbow)
    if distances:
        elbow_idx = np.argmax(distances) + 1  # +1 because we skipped the first point
        return elbow_idx + 1  # +1 to convert from 0-based index to K value
    else:
        return 3  # Default fallback

def evaluate_clustering_quality(X, k_range=range(2, 8)):
    """
    Evaluate clustering quality using multiple metrics
    """
    print("\n=== Clustering Quality Evaluation ===")
    
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate various clustering metrics
        sil_score = silhouette_score(X, cluster_labels)
        cal_score = calinski_harabasz_score(X, cluster_labels)
        db_score = davies_bouldin_score(X, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)
        
        print(f"K={k}: Silhouette={sil_score:.4f}, Calinski-Harabasz={cal_score:.2f}, Davies-Bouldin={db_score:.4f}")
    
    # Plot clustering quality metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Silhouette Score (higher is better)
    axes[0].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score vs K\n(Higher is Better)')
    axes[0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score (higher is better)
    axes[1].plot(k_range, calinski_scores, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Calinski-Harabasz Score')
    axes[1].set_title('Calinski-Harabasz Score vs K\n(Higher is Better)')
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin Score (lower is better)
    axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('Davies-Bouldin Score vs K\n(Lower is Better)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal K based on silhouette score (most commonly used)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_calinski = k_range[np.argmax(calinski_scores)]
    optimal_k_davies = k_range[np.argmin(davies_bouldin_scores)]
    
    print(f"\n🎯 Optimal K based on different metrics:")
    print(f"   Silhouette Score: K = {optimal_k_silhouette} (score: {max(silhouette_scores):.4f})")
    print(f"   Calinski-Harabasz: K = {optimal_k_calinski} (score: {max(calinski_scores):.2f})")
    print(f"   Davies-Bouldin: K = {optimal_k_davies} (score: {min(davies_bouldin_scores):.4f})")
    
    return optimal_k_silhouette, silhouette_scores, calinski_scores, davies_bouldin_scores

# Run K-Fold Cross Validation
if __name__ == "__main__":
    X_scaled, y_index, y_tier, df, feature_names = load_and_preprocess_data(FILEPATH)

    # Find optimal number of clusters using elbow method
    print("🔍 STEP 1: Finding Optimal Number of Clusters")
    optimal_k_elbow, wcss_values = find_optimal_clusters(X_scaled, max_clusters=10)
    
    # Evaluate clustering quality with multiple metrics
    print("\n🔍 STEP 2: Evaluating Clustering Quality")
    optimal_k_metrics, sil_scores, cal_scores, db_scores = evaluate_clustering_quality(X_scaled, range(2, 8))
    
    # Use the optimal K from elbow method (or you can choose based on metrics)
    print(f"\n🎯 DECISION: Using K = {optimal_k_elbow} clusters based on elbow method")
    
    # Update the model to use optimal K
    print(f"\n📊 STEP 3: Training Model with Optimal K = {optimal_k_elbow}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    reg_mse_list, reg_r2_list, silhouette_scores = [], [], []
    all_actual, all_predicted = [], []
    
    print("\n=== K-Fold Cross-Validation (5 folds) ===")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_index_train, y_index_test = y_index.iloc[train_idx], y_index.iloc[test_idx]
        y_tier_train, y_tier_test = y_tier.iloc[train_idx], y_tier.iloc[test_idx]

        # Use optimal K for clustering
        model = RegionalDevelopmentModel(n_clusters=optimal_k_elbow)
        model.train(X_train, y_index_train, y_tier_train)
        results = model.evaluate(X_test, y_index_test, y_tier_test)

        reg_mse_list.append(results['regression']['MSE'])
        reg_r2_list.append(results['regression']['R2'])
        silhouette_scores.append(results['silhouette_score'])
        
        # Store predictions for overall visualization
        all_actual.extend(y_index_test.values)
        all_predicted.extend(results['regression']['predictions'])

        print(f"\nFold {fold}:")
        print(f"  Regression MSE: {results['regression']['MSE']:.4f}")
        print(f"  Regression R²: {results['regression']['R2']:.4f}")
        print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
        print(f"  Classification Report:\n{results['classification_report']}")

    print("\n=== Cross-Validation Summary ===")
    print(f"Avg Regression MSE: {np.mean(reg_mse_list):.4f}")
    print(f"Avg Regression R²: {np.mean(reg_r2_list):.4f}")
    print(f"Avg Silhouette Score: {np.mean(silhouette_scores):.4f}")
    
    # Train final model on full dataset for feature importance
    print("\n=== Training Final Model for Analysis ===")
    final_model = RegionalDevelopmentModel(n_clusters=optimal_k_elbow)
    final_model.train(X_scaled, y_index, y_tier)
    
    # Feature Importance Analysis
    print("\n=== Feature Importance Analysis ===")
    importance_df = plot_feature_importance(final_model, feature_names, top_n=15)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Overall Actual vs Predicted Plot
    print("\n=== Visualization: Actual vs Predicted (All Folds Combined) ===")
    plot_actual_vs_predicted(all_actual, all_predicted)
    
    # Residuals Analysis
    print("\n=== Residuals Analysis (All Folds Combined) ===")
    plot_residuals(np.array(all_actual), np.array(all_predicted))
    
    # Final clustering analysis
    print(f"\n=== Final Clustering Analysis (K = {optimal_k_elbow}) ===")
    cluster_labels = final_model.clusterer.predict(X_scaled)
    final_silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"Final Model Silhouette Score: {final_silhouette:.4f}")
    
    # Show cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster + 1}: {count} regions ({percentage:.1f}%)")
    
    print(f"\n✅ Model trained successfully with optimal K = {optimal_k_elbow} clusters!")
