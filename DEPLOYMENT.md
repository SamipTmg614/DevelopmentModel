# 🚀 Deployment Guide - Host Your Regional Development Analysis System

## Option 1: Streamlit Community Cloud (FREE - Recommended for Students)

### Step 1: Prepare Your Repository
```bash
# 1. Initialize git repository (if not already done)
git init

# 2. Add all files
git add .

# 3. Commit your changes
git commit -m "Initial commit - Regional Development Analysis System"

# 4. Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/regional-development-analysis.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud
1. **Go to**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository**: `YOUR_USERNAME/regional-development-analysis`
5. **Set main file path**: `src/app.py`
6. **Click "Deploy"**

**Your app will be live at**: `https://YOUR_USERNAME-regional-development-analysis-srcapp-xxxx.streamlit.app/`

---

## Option 2: Heroku (FREE Tier Available)

### Step 1: Install Heroku CLI
Download from: https://devcenter.heroku.com/articles/heroku-cli

### Step 2: Create Heroku Files
```bash
# Create Procfile
echo "web: streamlit run src/app.py --server.port $PORT --server.enableCORS false" > Procfile

# Create runtime.txt (optional)
echo "python-3.9.18" > runtime.txt
```

### Step 3: Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-regional-development-app

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open your app
heroku open
```

---

## Option 3: Railway (Modern Alternative)

### Step 1: Railway Setup
1. **Go to**: https://railway.app/
2. **Sign up** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**

### Step 2: Configure
- **Start Command**: `streamlit run src/app.py --server.port $PORT`
- **Auto-deploy**: Enabled

---

## Option 4: Render (Reliable Free Hosting)

### Step 1: Render Setup
1. **Go to**: https://render.com/
2. **Sign up** with GitHub
3. **Click "New +" → "Web Service"**
4. **Connect your repository**

### Step 2: Configure
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`

---

## Option 5: Local Network Hosting

### For Assignment Demonstrations
```bash
# Run on local network (accessible to others on same WiFi)
streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501
```

**Access via**: `http://YOUR_LOCAL_IP:8501`

---

## 🔧 Quick Fixes for Common Issues

### Issue: Import Errors
**Solution**: Update `src/app.py` line 11:
```python
# Change this:
from developmentmodel import RegionalDevelopmentModel, load_and_preprocess_data

# To this:
from src.developmentmodel import RegionalDevelopmentModel, load_and_preprocess_data
```

### Issue: File Path Problems
**Solution**: Update file paths in `app.py`:
```python
# Change this:
filepath = "./data/final/merged_cleaned_dataset.csv"

# To this:
filepath = "data/final/merged_cleaned_dataset.csv"
```

### Issue: Large File Sizes
**Solution**: If your CSV is too large (>100MB):
1. Use Git LFS: `git lfs track "*.csv"`
2. Or compress your data
3. Or use a database service

---

## 📝 Pre-Deployment Checklist

- [ ] All files committed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] File paths are relative (no `./` or absolute paths)
- [ ] Large files handled appropriately
- [ ] `.gitignore` excludes unnecessary files
- [ ] README.md updated with deployment URL
- [ ] App tested locally with `streamlit run src/app.py`

---

## 🎯 Recommended Workflow for Students

### Best Option: Streamlit Community Cloud
**Why?**
- ✅ Completely FREE
- ✅ Easy GitHub integration
- ✅ Automatic deployments
- ✅ Perfect for academic projects
- ✅ Professional URLs
- ✅ No credit card required

### Steps:
1. **Push to GitHub** (5 minutes)
2. **Deploy to Streamlit Cloud** (2 minutes)
3. **Share URL** in your assignment
4. **Automatic updates** when you push changes

---

## 🔗 Expected Final URLs

After deployment, your app will be accessible at:

- **Streamlit Cloud**: `https://username-repo-name-srcapp-xxx.streamlit.app/`
- **Heroku**: `https://your-app-name.herokuapp.com/`
- **Railway**: `https://your-app-name.up.railway.app/`
- **Render**: `https://your-app-name.onrender.com/`

---

## 🆘 Need Help?

If you encounter issues:
1. Check the deployment logs
2. Verify all files are in GitHub
3. Test locally first
4. Check file paths and imports
5. Ensure dependencies are in `requirements.txt`

**Your app showcases multiple AI domains and provides a professional interface perfect for academic evaluation!**
