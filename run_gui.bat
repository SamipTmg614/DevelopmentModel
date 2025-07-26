@echo off
echo Starting Regional Development Analysis GUI...
echo.
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Launching Streamlit application...
echo Open your browser and go to: http://localhost:8501
echo.
streamlit run src/app.py
