# Regional Development Analysis System

🏘️ An AI-powered system for predicting regional development using machine learning algorithms.

## 🎯 Project Overview

This project implements a comprehensive AI solution for regional development analysis using:
- **Classification** (Development Tiers: Low/Medium/High)
- **Regression** (Development Index Prediction)
- **Clustering** (Regional Grouping)

## 🔧 Technologies Used

- **Machine Learning**: Random Forest Regressor, Random Forest Classifier, K-Means Clustering
- **Frontend**: Streamlit (Interactive Web Interface)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📊 Model Performance

- **R² Score**: 0.9871 (98.71% variance explained)
- **Classification Accuracy**: 99%
- **Cross-Validation**: 5-fold validation with consistent performance

## 🚀 Live Demo

**Access the live application**: [Your Streamlit App URL will be here]

## 🏃‍♂️ Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd SemesterAssignment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run src/app.py
```

4. **Open browser** at `http://localhost:8501`

### Windows Quick Start
Double-click `run_gui.bat` for automatic setup and launch.

## 📁 Project Structure

```
SemesterAssignment/
├── src/
│   ├── app.py                 # Streamlit GUI application
│   ├── developmentmodel.py    # Core ML model implementation
│   ├── cleaneachdata.py       # Data cleaning utilities
│   └── mergecleaneddata.py    # Data merging utilities
├── data/
│   ├── final/
│   │   └── merged_cleaned_dataset.csv
│   ├── cleaned/               # Processed datasets
│   └── newdata/              # Raw Excel files
├── requirements.txt          # Python dependencies
├── run_gui.bat              # Windows launcher
└── README.md                # This file
```

## 🔮 Features

### Interactive Web Interface
- **Multi-page Navigation**: Home, Model Performance, Predictions, Feature Analysis, Regional Insights
- **Real-time Predictions**: Enter regional data and get instant development predictions
- **Interactive Visualizations**: Charts, plots, and statistical analysis
- **Professional Dashboard**: Clean, modern interface with detailed metrics

### AI Capabilities
- **Development Index Prediction**: Continuous score prediction
- **Tier Classification**: Automatic categorization (Low/Medium/High)
- **Regional Clustering**: Identify similar development patterns
- **Feature Importance**: Understand key development factors

### Data Analysis
- **Cross-validation**: 5-fold validation for model reliability
- **Performance Metrics**: R², MSE, Accuracy, Confusion Matrix
- **Feature Analysis**: Top contributing factors visualization
- **Regional Insights**: Development distribution and trends

## 📈 Key Insights

**Top Development Factors:**
1. **Education Access** (59.1%) - Most critical factor
2. **Doctor Access** (23.9%) - Healthcare proximity matters
3. **Emergency Services** (6.3%) - Fire brigade accessibility
4. **Library Access** (3.9%) - Information infrastructure
5. **Higher Education** (1.9%) - Advanced learning opportunities

## 🎓 Academic Context

This project was developed as part of an AI coursework assignment demonstrating:
- **Problem Definition**: Real-world regional development analysis
- **Data Processing**: Cleaning, normalization, feature engineering
- **Algorithm Implementation**: Multiple AI techniques
- **Model Evaluation**: Comprehensive performance analysis
- **Interface Development**: Professional user experience
- **Version Control**: GitHub-based development workflow

## 📊 Dataset

The analysis uses regional infrastructure data including:
- Health facilities and accessibility
- Educational institutions and coverage
- Transportation infrastructure
- Emergency services availability
- Public amenities and social infrastructure

## 🛠️ Technical Implementation

### Data Preprocessing
- Duplicate removal and data validation
- Per-capita normalization by total wards
- Feature engineering for composite scores
- Standardization for ML algorithms

### Model Architecture
```python
class RegionalDevelopmentModel:
    - RandomForestRegressor (Development Index)
    - RandomForestClassifier (Development Tiers)
    - KMeans Clustering (Regional Groups)
```

### Evaluation Methodology
- K-Fold Cross-Validation (5 folds)
- Multiple performance metrics
- Feature importance analysis
- Residual analysis for model validation

## 🚀 Deployment

### Streamlit Community Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Automatic deployment and hosting

### Local Development
- Windows: Use `run_gui.bat`
- Cross-platform: `streamlit run src/app.py`

## 📝 Usage Examples

### Single Region Prediction
1. Navigate to "Make Predictions" page
2. Enter regional infrastructure data
3. Click "Predict Development"
4. View results and recommendations

### Batch Analysis
1. Upload CSV file with regional data
2. Process multiple regions simultaneously
3. Download results for further analysis

### Performance Analysis
1. View model metrics and visualizations
2. Analyze actual vs predicted values
3. Examine feature importance rankings

## 🤝 Contributing

This project is part of academic coursework. For questions or improvements:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

## 📄 License

Academic project - please respect university policies on code sharing and collaboration.

## 📞 Contact

For questions about this implementation or academic collaboration, please reach out through the appropriate academic channels.

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**
