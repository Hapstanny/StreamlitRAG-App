#!/bin/bash

echo "RAG Chat Assistant - Quick Start"
echo "================================"

echo
echo "1. Installing dependencies..."
pip install -r requirements.txt

echo
echo "2. Running validation..."
python validate.py

echo
echo "3. Running setup (creating Azure Search index)..."
python setup.py

echo
echo "4. Starting Streamlit application..."
echo "Open your browser to the URL shown below:"
streamlit run app.py