# Team 37: Emotional Color Grading Pipeline

An automated pipeline that performs semantic color grading based on emotional keywords. This project uses a retrieval-augmented approach to generate 3D Look-Up Tables (LUTs) that transfer the color palette of emotionally tagged reference images to a target input image.

## 1. Dataset Generation

The foundation of our pipeline is a high-quality dataset derived from professional photography, processed to include semantic emotion labels.

### Step 1: Unsplash Dataset
We utilize the open [Unsplash Dataset](https://github.com/unsplash/datasets). Specifically, we parse the `photos.csv000` file (Lite version) which contains metadata for 25,000 high-quality images.

### Step 2: Downloading & Manifest Creation
Using our data download script (`download_dataset.py`), we:
1. Read the Unsplash `photos.csv000` file.
2. Generate a `dataset_manifest.csv` containing:
    - `photo_id`
    - `photo_description` (AI descriptions provided by Unsplash)
    - `image_path` (local path to the downloaded image)

### Step 3: Emotion Generation with Gemini 2 Flash
To create the ground truth for our emotional search, we process the descriptions from the manifest. 
- We pass the `photo_description` of each image to **Gemini 2 Flash**.
- This results in our final `labeled_dataset.csv`, which maps image paths to specific emotions.

## 2. Vector Search Database (VSearch)

We use a Vector Database to enable semantic retrieval of reference images.

### Architecture
- **Model:** We utilize **OpenAI CLIP (ViT-Base-Patch32)** to generate embeddings. CLIP is chosen for its ability to understand visual concepts and map images into a dense vector space.
- **Indexing:** We use **FAISS (Facebook AI Similarity Search)** to build an efficient searchable index of these embeddings.
- **Storage:** The embeddings are stored in a `vector_db.index` file, and the associated metadata (paths and emotions) is pickled in `metadata_map.pkl`.

### Build Process
The `VSearchEngine` (in `vsearch.py`) iterates through `labeled_dataset.csv`, computes the CLIP embedding for every valid image, and adds it to the FAISS index.

## 3. Search Mechanism

When the user requests a color grade for a specific emotion (e.g., "Happy"):

1. **Input Embedding:** The input image to be graded is passed through the CLIP model to generate a query vector.
2. **Similarity Search:** We query the FAISS index to find the nearest neighbors (images with similar semantic visual content).
3. **Emotion Filtering:** The system filters these neighbors to ensure they match the **target emotion**. For example, if the input is a landscape and the target is "Horror", the system looks for landscape images that are specifically tagged as "Horror".
4. **Retrieval:** The system returns the top $K$ matches (References), which are then used by the Neural LUT Generator to transfer color styles.

## 4. Environment Setup

### Prerequisites
- Python 3.11.14+
- CUDA-capable GPU (Recommended for faster CLIP embedding and LUT training)

### Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
````

### Directory Structure

The project is structured to separate code, data, and outputs.

```text
.
├── config.py             # Global configuration 
├── dataset.py            # PyTorch Dataset wrapper
├── dd.py                 # Data downloader & Manifest generator
├── engine_core.py        # Core logic combining Search + LUT Generation
├── lut_generator.py      # Neural LUT optimization logic 
├── lut_model.py          # Trilinear LUT Neural Network Module
├── main.py               # CLI entry point for testing
├── server.py             # FastAPI backend for the web interface
├── vsearch.py            # Vector Search Engine 
├── dataset/              # Root directory for data
│   ├── images/           # Downloaded Unsplash images
│   ├── labeled_dataset.csv # The final CSV with Emotion tags
│   ├── vector_db.index   # FAISS index file
│   └── metadata_map.pkl  # Metadata mapping for the index
├── misc/                 # Logs and cache
└── output/               # Generated color-graded images
```

## Usage

**To run the API Server:**

```bash
python server.py
```

This starts the FastAPI backend on `http://0.0.0.0:8000`.

Open the HTML file in `ui/index.html` to get the exeperience of colour grading.