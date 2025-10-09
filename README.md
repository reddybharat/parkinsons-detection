# Parkinson's Disease Detection from Hand-drawn Images

[**Try the app online!**](https://parkinsons-detection-815144880915.us-central1.run.app/)

---

## Clinical Background & Motivation
Parkinson's Disease (PD) is a progressive neurodegenerative disorder with hallmark symptoms such as tremor and impaired motor control. One widely used, non-invasive screening method involves analyzing patients' hand-drawn spirals, waves, or geometric figures—patients with PD often produce noticeably irregular or jagged patterns. Timely detection is crucial for early intervention and ongoing disease management.

This project leverages computer vision and machine learning techniques to automate the analysis of such hand-drawn images, intending to provide a reproducible, accessible, and scalable approach to assist clinicians or researchers.

**For more context:** You can read the underlying published paper on this work here:  
➡️ [Published Paper: "Analysis of Hand-Drawn Spirals for Parkinson’s Disease Detection" (Springer)](https://link.springer.com/chapter/10.1007/978-981-33-6912-2_22)

---

## Overview
This is a project focused on using classical machine learning methods to detect Parkinson's Disease from hand-drawn images (e.g., spirals, waves, cubes, triangles). The project is designed for end-to-end automation: from raw data handling and augmentation, through feature engineering and training, to model deployment via a web API and interactive UI. The architecture emphasizes modularity and reproducibility.

---

## Project Structure

```
project/
├── models/                 # Trained model .pkl files & metrics.json
├── notebooks/              # EDA and feature exploration (Jupyter)
├── src/
│   ├── api/                # FastAPI application and endpoints
│   ├── predictions/        # Model inference logic
│   ├── preprocessing/      # Image preprocessing and dataset loader
│   ├── training/           # Model training scripts (KNN, SVM, RF)
│   ├── ui/                 # Streamlit front-end
│   └── utils/              # Data augmentation and metrics tools
├── tests/                  # Test scripts (augmentation, training)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── temp_uploads/           # Temporary folder for uploaded images (API input)
```

### Why This Structure?
- **Separation of Concerns:**
  - *Preprocessing*: Ensures input data is uniformly transformed for ML.
  - *Feature Extraction*: Isolated for experiments/updating feature methods.
  - *Model Training*: Each model is trained, evaluated, and saved independently.
  - *API & UI*: Clean boundary between backend prediction and user-facing app.
- **Ease of Experimentation:** Notebooks (EDA/Features) let you explore without polluting production scripts.
- **Reproducibility:** Data flows from raw → processed → models → interactive inference.
- **Modularity:** Easy to swap preprocessing, features, or model architectures.

---

## Pipeline

### 1. Data Preparation & Augmentation
- **Data Collection:** Raw hand-drawn images are organized by shape (spiral, wave, cube, triangle, etc.), health status (`healthy` or `parkinson`), and split into training/testing sets. Images are stored under a well-defined folder structure (`data/raw/`).
- **Data Augmentation:** To address the limited size and diversity of medical datasets, each image is further augmented:
  - **Techniques:**
    - Random rotation (±25°)
    - Random Gaussian noise addition
    - Horizontal flipping
  - Augmented images are saved in parallel structure under `data/processed/`, substantially increasing training data size and diversity.

### 2. Data Preprocessing
These preprocessing steps are applied to each image before feature extraction (HOG) occurs:
- **Grayscale Conversion:** Each input image is converted from color to grayscale to simplify downstream analysis and reduce computational complexity.
- **Image Resizing:** Every image is resized to a uniform 200x200 pixel grid, ensuring consistent feature extraction regardless of the original resolution or aspect ratio.
- **Binarization:** Otsu's thresholding method is applied to yield a clean, high-contrast binary image (foreground vs. background), which is ideal for feature extraction from hand-drawn sketches.

### 3. Feature Extraction (HOG)
- **Histogram of Oriented Gradients (HOG):** Applied to the preprocessed, binarized images to extract robust descriptors that capture the stroke directionality and path irregularity—key features for identifying PD-related drawing issues.
- HOG vectors are then used as input to machine learning models.

### 4. Model Training
- Models including KNN, Linear SVM, and Random Forest are trained on the extracted HOG features and evaluated on the test set.
- Results and model artifacts are stored for inference.

### 5. Inference & Serving
- A FastAPI backend and Streamlit UI coordinate to serve predictions and enable user interaction with the trained models, both locally and via the online deployment.

---

## Dataset

### Source & Background
- **Dataset 1** is directly based on the open dataset described in [Zham et al., "Distinguishing Different Stages of Parkinson’s Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral," *Frontiers in Neurology* 2017](https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full).
- Focus: Hand-drawn Spirals and Waves, which are validated clinical tasks for assessing motor impairment severity in PD patients ([see clinical literature](https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full)).

#### Composition
- **Shapes:** `spiral`, `sw` (spiral-wave combines), and `wave`.
- **Classes:** Separate folders for `healthy` vs. `parkinson` subjects.

### Image Distribution: Before and After Augmentation
Images and class counts below illustrate balance and size before and after data augmentation:

**Before Augmentation**
| Shape  | Healthy | Parkinson | Total |
| ------ | ------- | --------- | ----- |
| spiral |   51    |    51     | 102   |
| sw     |   102   |   102     | 204   |
| wave   |   51    |    51     | 102   |
|        |**204**  | **204**   |**408**|

**After Augmentation**
| Shape  | Healthy | Parkinson | Total |
| ------ | ------- | --------- | ----- |
| spiral |   204   |   204     | 408   |
| sw     |   408   |   408     | 816   |
| wave   |   204   |   204     | 408   |
|        |**816**  | **816**   |**1632**|

- The augmentation pipeline (random rotation, noise, flip) expands the training data while maintaining perfect class balance.
- All experiments and results throughout this project are specifically on **Dataset 1** (Spiral & Wave shapes), as established in the above-cited study.

### Citation
If you use this dataset or derived work, please cite:  
Zham P, Kumar DK, Dabnichki P, Poosapadi Arjunan S and Raghav S (2017) Distinguishing Different Stages of Parkinson’s Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral. *Front. Neurol.* 8:435. [https://doi.org/10.3389/fneur.2017.00435](https://doi.org/10.3389/fneur.2017.00435)

---

## Key Code Customization Points
Below are some important code sections you may wish to customize, based on experiments or dataset changes:

### 1. HOG Feature Extraction Parameters
**File:** `src/preprocessing/hog_filter.py`
```python
features, _ = feature.hog(
    image,
    orientations=9,            # <--- change number of orientation bins
    pixels_per_cell=(10, 10),  # <--- change cell size
    cells_per_block=(2, 2),    # <--- change block size
    transform_sqrt=True,
    block_norm="L1",
    visualize=True
)
```

### 2. Data Augmentation Techniques
**File:** `src/utils/augmentation.py`
```python
available_transformations = {
    'rotate': random_rotation,      # Random ±25 degree
    'noise': random_noise,          # Add random noise
    'horizontal_flip': horizontal_flip
    # Add more transformations if desired
}
```
Change or add functions for more advanced augmentation as needed.

### 3. Model Hyperparameters
**Files:**
- `src/training/train_knn.py`: `KNeighborsClassifier(n_neighbors=2)`
- `src/training/train_svc_linear.py`: `SVC(kernel='linear', C=1.0)`
- `src/training/train_random_forest.py`: `RandomForestClassifier(n_estimators=100, random_state=42)`

**Example (Random Forest):**
```python
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Change n_estimators, max_depth, or other sklearn hyperparameters
```

---

## Example: End-to-end Usage
1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. **Prepare your dataset folder structure (required):**  Your images must be organized as shown below for the code to work. Place the entire structure under the `data` folder at the project root:
    ```
    data/
    └── raw/
        └── dataset_1/
            ├── spiral/
            │   ├── testing/
            │   │   ├── healthy/
            │   │   └── parkinson/
            │   └── training/
            │       ├── healthy/
            │       └── parkinson/
            ├── sw/
            └── wave/
    ```
   Each shape (spiral, sw, wave) contains its own training and testing folders, each with healthy/parkinson classes. This layout must be followed for the scripts to function correctly.

3. Augment & process images:
    ```bash
    python -m src.utils.augmentation
    ```
4. Train models:
    ```bash
    python -m src.training.train_knn
    python -m src.training.train_svc_linear
    python -m src.training.train_random_forest
    ```
5. Start FastAPI server:
    ```bash
    uvicorn src.api.main:app --reload
    ```
6. Launch the UI:
    ```bash
    streamlit run src/ui/app.py
    ```

---

## DISCLAIMER
- Intended for personal learning, architecture experimentation, and medical ML exploration.
- Not intended for production, medical, or clinical use without significant further validation. Use at your own risk.