# BDD100k-object-Detection
Data Analysis of BDD100k dataset and object detection training repo


## 🧾 Step-by-Step Instructions

### 🧩 Step 1: Download Dataset
Download the dataset from the link below and place it inside the `analysis/` directory.

**[Download Dataset](https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view?usp=sharing)**

Make sure the dataset file appears in: analysis folder

## 🧭 Step 2: Navigate to Analysis Folder
Open a terminal and move into the analysis directory:

```bash
cd analysis
```
## 🧭 Step 3: Build Docker Image
```bash
docker build -t analysis-app .
```
This will create a Docker image named analysis-app.

## 🧭 Step 4: Run Analysis
```bash
./run_analysis.sh
```
If you encounter a “permission denied” error, make the script executable first:
```bash
chmod +x run_analysis.sh
```
This script will automatically launch a Docker container and run the analysis pipeline.
After successful execution, the image analysis output will be generated in the same analysis/ folder.
