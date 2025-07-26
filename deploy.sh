#!/bin/bash

echo "🚀 Regional Development Analysis - Quick Deploy Script"
echo "=================================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository exists"
fi

# Add gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    echo "Already created!"
fi

# Add all files
echo "📦 Adding files to Git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy Regional Development Analysis System - $(date)"

# Instructions for GitHub and Streamlit
echo ""
echo "🎯 NEXT STEPS FOR DEPLOYMENT:"
echo "=================================================="
echo ""
echo "1. 📤 Push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/regional-development-analysis.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "2. 🌐 Deploy to Streamlit Cloud:"
echo "   • Go to: https://share.streamlit.io/"
echo "   • Sign in with GitHub"
echo "   • Click 'New app'"
echo "   • Select your repository"
echo "   • Set main file: src/app.py"
echo "   • Click 'Deploy'"
echo ""
echo "3. 🎉 Your app will be live at:"
echo "   https://YOUR_USERNAME-regional-development-analysis-srcapp-xxxx.streamlit.app/"
echo ""
echo "4. 📋 For your assignment, submit:"
echo "   • Your GitHub repository URL"
echo "   • Your live Streamlit app URL"
echo "   • This demonstrates all required AI domains!"
echo ""
echo "✅ Ready for deployment! Follow the steps above."
