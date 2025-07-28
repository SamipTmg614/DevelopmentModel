#!/usr/bin/env python3

from developmentmodel import load_and_preprocess_data, RegionalDevelopmentModel
import numpy as np

def main():
    # Load data
    X_scaled, y_index, df, feature_names = load_and_preprocess_data('../data/final/merged_cleaned_dataset.csv')

    # Train model with 4 clusters
    model = RegionalDevelopmentModel(n_clusters=4)
    model.train(X_scaled, y_index)

    # Get predictions and clusters
    predictions = model.regressor.predict(X_scaled)
    clusters = model.clusterer.predict(X_scaled)

    print('=== CLUSTER ANALYSIS ===')
    print(f'Total regions: {len(predictions)}')
    print(f'Development Index range: {y_index.min():.4f} to {y_index.max():.4f}')
    print(f'Prediction range: {predictions.min():.4f} to {predictions.max():.4f}')
    print()

    # Analyze each cluster
    for cluster_id in range(4):
        mask = clusters == cluster_id
        cluster_dev_indices = predictions[mask]
        actual_dev_indices = y_index[mask]
        
        print(f'Cluster {cluster_id + 1}:')
        print(f'  Count: {mask.sum()} regions ({mask.sum()/len(predictions)*100:.1f}%)')
        print(f'  Predicted Dev Index: {cluster_dev_indices.mean():.4f} ± {cluster_dev_indices.std():.4f}')
        print(f'  Range: {cluster_dev_indices.min():.4f} to {cluster_dev_indices.max():.4f}')
        print(f'  Actual Dev Index: {actual_dev_indices.mean():.4f} ± {actual_dev_indices.std():.4f}')
        print()

    # Test the specific inputs from user
    print('=== TESTING USER INPUTS ===')

    # Calculate development indices manually
    def calc_dev_index(data):
        health_access = (data['1_2 healthpost'] * 1.5 + data['3_4 healthpost'] * 3.5 + 
                        data['5_9 healthpost'] * 7 + data['10_plus healthpost'] * 10)
        education_access = (data['with 1_2 school'] * 1.5 + data['with 3_4 school'] * 3.5 + 
                           data['with 5_9 school'] * 7 + data['10_plus schools'] * 10)
        infrastructure_score = 0.3  # Assuming average
        dev_index = (0.3 * health_access + 0.3 * education_access + 0.2 * infrastructure_score + 
                    0.1 * data['access to doctor in 30 mins'] + 0.1 * data['access to firebrigade in 30 mins'])
        return health_access, education_access, dev_index

    # Region 1 data (previous)
    region1_data = {
        'Total_Wards': 10,
        '1_2 healthpost': 3/10, '3_4 healthpost': 2/10, '5_9 healthpost': 1/10, '10_plus healthpost': 0/10,
        'with 1_2 school': 4/10, 'with 3_4 school': 3/10, 'with 5_9 school': 2/10, '10_plus schools': 1/10,
        'access to doctor in 30 mins': 0.5, 'access to firebrigade in 30 mins': 0.5
    }

    # Region 2 data (current)  
    region2_data = {
        'Total_Wards': 13,
        '1_2 healthpost': 7/13, '3_4 healthpost': 1/13, '5_9 healthpost': 0/13, '10_plus healthpost': 0/13,
        'with 1_2 school': 8/13, 'with 3_4 school': 3/13, 'with 5_9 school': 2/13, '10_plus schools': 0/13,
        'access to doctor in 30 mins': 0.5, 'access to firebrigade in 30 mins': 0.5
    }

    h1, e1, d1 = calc_dev_index(region1_data)
    h2, e2, d2 = calc_dev_index(region2_data)

    print(f'Region 1: Health={h1:.4f}, Education={e1:.4f}, DevIndex={d1:.4f}')
    print(f'Region 2: Health={h2:.4f}, Education={e2:.4f}, DevIndex={d2:.4f}')
    print(f'Difference: {abs(d1-d2):.4f}')
    
    # Check which clusters these would be assigned to
    print(f'\nBased on cluster analysis:')
    print(f'Region 1 (DevIndex={d1:.4f}) should be in a different cluster than Region 2 (DevIndex={d2:.4f})')
    
    # Find appropriate cluster ranges
    sorted_clusters = []
    for cluster_id in range(4):
        mask = clusters == cluster_id
        cluster_dev_indices = predictions[mask]
        sorted_clusters.append((cluster_id + 1, cluster_dev_indices.mean()))
    
    sorted_clusters.sort(key=lambda x: x[1])
    print(f'\nCluster ranking by average Development Index:')
    for i, (cluster_num, avg_dev) in enumerate(sorted_clusters):
        level = ['Low', 'Medium-Low', 'Medium-High', 'High'][i]
        print(f'  Cluster {cluster_num}: {avg_dev:.4f} - {level} Development')

if __name__ == "__main__":
    main()
