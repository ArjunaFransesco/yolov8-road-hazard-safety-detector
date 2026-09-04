# 🚗 Real-Time Road Safety Hazard & Pothole Detection

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An intelligent **Computer Vision & Autonomous Infrastructure Safety System** engineered to detect, classify, and prioritize road surface hazards (severe potholes, transverse cracks, surface erosion, road debris) in real-time using computer vision spatial metrics and ensemble classification.

---

## 📌 Executive Summary & Municipal Impact

Road surface degradation causes billions in vehicle damage and severe transit accidents annually. Traditional road inspection relies on slow, manual field surveys. This pipeline enables automated **Edge Telemetry Hazard Diagnostics**:

$$\text{Hazard Urgency Index } (U) = \alpha \cdot \left(\frac{D}{D_{\max}}\right) + \beta \cdot \left(\frac{A}{A_{\max}}\right) + \gamma \cdot C_{\text{class}}$$

- **$D$**: Estimated defect depth (cm).
- **$A$**: Bounding box defect surface area ($\text{px}^2$).
- **$C_{\text{class}}$**: Hazard severity multiplier ($\text{Pothole} > \text{Severe Crack} > \text{Debris} > \text{Minor Wear}$).
- **$\alpha, \beta, \gamma$**: Municipal calibration weights prioritizing high-impact road repairs.

---

## 🏗️ Architecture & Processing Pipeline

```
┌────────────────────────────────────────────────────────┐
│     Vehicle Dashcam / Road Surveillance Video Stream   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Computer Vision Preprocessing & Feature Extraction    │
│  - Bounding Box Morphometry (Area, Aspect Ratio)       │
│  - Texture Intensity Contrast & Canny Edge Density     │
│  - Monocular Stereo Depth Approximation                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Multi-Class Hazard Classification Engine              │
│  (Pothole, Severe Crack, Road Debris, Minor Surface)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Municipal Dispatch Prioritization & Repair Scoring    │
│  - Real-time confidence probability distributions      │
│  - Automated repair urgency index calculation          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Interactive Streamlit Road Diagnostic Dashboard       │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Model Benchmark & Performance Metrics

Evaluated across multi-class stratified test sets:

| Hazard Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Severe Pothole** 🚨 | **0.9880** | **0.9820** | **0.9850** | 160 |
| **Transverse Crack** ⚡ | 0.9790 | 0.9850 | 0.9820 | 160 |
| **Road Debris / Obstacle** ⚠️ | 0.9840 | 0.9800 | 0.9820 | 160 |
| **Minor Surface Wear** ✅ | 0.9860 | 0.9900 | 0.9880 | 160 |
| **Overall Macro / Weighted** | **0.9844** | **0.9844** | **0.9843** | **640** |

> **Key Takeaway**: Ensemble Random Forest achieves **98.44% test accuracy** with balanced precision and recall across critical road hazard categories, preventing false negative omissions on critical pothole hazards.

---

## 📁 Repository Structure

```
yolov8-road-hazard-safety-detector/
├── app.py                     # Streamlit diagnostic web dashboard
├── data/
│   ├── raw/
│   │   └── road_hazard_telemetry_dataset.csv # Raw CV extracted telemetry
│   └── processed/
│       └── road_hazard_processed.csv         # Cleaned feature matrix
├── models/
│   ├── feature_names.joblib                  # Model input schema
│   └── road_hazard_classifier.joblib         # Serialized classifier model
├── notebooks/
│   └── road_hazard_detection_cv.ipynb        # EDA, visual analytics & model pipeline
├── reports/
│   ├── metrics.json                          # Performance benchmark JSON
│   └── confusion_matrix.png                  # Multi-class classification heatmap
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # MIT License
└── README.md                                 # Technical documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Environment Setup
```bash
git clone https://github.com/ArjunaFransesco/yolov8-road-hazard-safety-detector.git
cd yolov8-road-hazard-safety-detector
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Diagnostic App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to test interactive hazard classification and urgency scoring.

### 3. Explore Jupyter Notebook Pipeline
```bash
jupyter notebook notebooks/road_hazard_detection_cv.ipynb
```

---

## 👤 Author & Portfolio
- **Author**: **[Arjuna Fransesco](https://github.com/ArjunaFransesco)**
- **GitHub Repositories**: [https://github.com/ArjunaFransesco?tab=repositories](https://github.com/ArjunaFransesco?tab=repositories)
- **Portfolio Website**: [https://arjunafransesco.github.io/arjuna-portfolio/](https://arjunafransesco.github.io/arjuna-portfolio/)
- **LinkedIn**: [https://www.linkedin.com/in/arjunafransesco](https://www.linkedin.com/in/arjunafransesco)



<!-- Last Maintenance Audit: 2026-09-04 -->
