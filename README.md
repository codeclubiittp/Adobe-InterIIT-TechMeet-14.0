# Team_37_Adobe_Technical_Report

## Contributors
[Anirudh Arrepu](https://github.com/AnirudhArrepu)
[Niranjan M](https://github.com/all-coder)
[Suriyaa MM](https://github.com/SuriyaaMM)
[Rishi Ravi](https://github.com/Rishi-Ravi)

## Market Research, Problem Understanding, Solution Description
- [Market Research](./MARKET_RESERACH_AND_UI/team37_Market_Research.pdf)
- [Figma Designs](./MARKET_RESERACH_AND_UI/team37_Adobe_UI_figma.png)
- [Design Rationale](./MARKET_RESERACH_AND_UI/team37_Design_Rationale.pdf)
- [Demo Video](./MARKET_RESERACH_AND_UI/team37_Demo_Video.mp4)
- [Feature Walkthrough Video](./MARKET_RESERACH_AND_UI/team37_feature_walkthrough.mp4)
- [Pure Figma](./MARKET_RESERACH_AND_UI/team37_Adobe_UI_Pure_Final.png)


## Proposed Features
- **Text Morph**
- **LightShift Remove**
- **Mood Lens**
- **HertzMark**

## Text Morph
Text Morph is a generative scene text editing engine. Instead of simple pixel overlay, it replaces text within natural images while strictly preserving the original typography, lighting, and perspective. The pipeline integrates state-of-the-art segmentation with a specialized diffusion model (GaMuSA) to ensure the new text is visually indistinguishable from the original scene.

---

### Architecture & Workflow
![text_morph_architecture](https://hackmd.io/_uploads/rJBWemkzWl.jpg)


---

**Pipeline/Workflow**
- **Segmentation & Localization**
    - **User Interaction:** The user provides point prompts on the text region.
    - **Masking:** We trigger **SAM 2.1 Base** to generate high-precision segmentation masks. **YOLO (MSCOCO)** is utilized to localize the bounding box context for the text region.
- **Generative Text Inpainting using GaMuSA**
The core engine is a diffusion-based text editor. It utilizes a **Gylph Adaptive Mutual Self Attention** to inject the new text glyphs while locking onto the original font style and background texture.
- **Latent Optimization**
The model performs controlled sampling. This controlled diffusion process ensures the generated text aligns perfectly with the surrounding pixel context.
- **High-Fidelity Reconstruction**
The generated latent representation is decoded and resized to match the original input resolution, blending the edited region seamlessly into the high-res image.

---

**Key Design Decisions**
- **yolo_mscoco (custom trained)**
We trained a custom [YOLO-v11s model](https://docs.ultralytics.com/models/yolo11/#key-features) on the `MSCOCO-Text` dataset to obtain fine-grained, text-level bounding boxes. SAM produced only coarse region proposals, whereas our task required precise, per-text localization. The custom detector delivers dense, piecewise text boxes suitable for downstream OCR and layout analysis.


---

### Weights & Biases=
| Model              | Params    |
|--------------------|------------|
| AutoencoderKL      | 83.65M     |
| ControlUNetModel   | 859.52M    |
| LabelEncoder       | 66.24M     |


---

### Compute Profile
![gamusa_latency_profile](https://hackmd.io/_uploads/HyuUBC0WWe.png)
- Since SAM 2 is deployed as a frozen foundational model (zero-shot), its computational profile is well-documented by Meta AI. We focused our profiling efforts on the custom components (GaMuSA and Color Grading) where our novel optimizations were applied.
[Compute Profile for SAM](https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints)
---

### Terms, Conditions & License
- Model
    - [TextCtrl](https://github.com/weichaozeng/TextCtrl) - License Free Usage
    - [SAM 2](https://github.com/facebookresearch/sam2) - [License](https://github.com/facebookresearch/sam2/blob/main/LICENSE)
    - [Yolo-V11S](https://github.com/ultralytics/ultralytics) - [License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- Dataset
    - [mscoco-text](https://www.kaggle.com/datasets/c7934597/cocotext-v20) - [License](https://creativecommons.org/licenses/by/4.0/legalcode)

### References
- [ANYTEXT: MULTILINGUAL VISUAL TEXT GENERA-
TION AND EDITING](https://arxiv.org/pdf/2311.03054)
- [TextCtrl: Diffusion-based Scene Text Editing with
Prior Guidance Control](https://arxiv.org/pdf/2410.10133v1)
---

## LightShift Remove
This proposed workflow addresses this challenge by combining state-of-the-art segmentation and generative inpainting with 3D geometric reconstruction. By leveraging SAM 2 for precise object isolation and Lama for seamless removal, the system clears the obstruction, it then utilizes Marigold to generate high-fidelity depth maps and surface normals from the modified image. This allows for a final shader computation step that physically re-lights the skin texture, ensuring that the area where the shadow once fell matches the lighting of the rest of the face perfectly.

### Architecture & Workflow
![relighting_architecture](https://hackmd.io/_uploads/S1VUKz0WWl.jpg)

---

**Pipeline/Workflow**

- **Segmentation for Region Masking using SAM 2**
  High-fidelity bounding boxes are generated from sparse point inputs, enabling precise localization of the target regions for subsequent processing.
- **LaMa Inpainting**
  The LaMa-Inpainting model, which leverages Fast Fourier Convolutions (FFCs), is employed to inpaint the masked regions while preserving global structural coherence.
- **Depth Estimation**
  Performs denoising on the inpainted RGB latent space to hallucinate affine-invariant depth maps.
  Recovers high-resolution depth topology, including subtle facial curvature (e.g., nose, cheekbones), typically lost during the 2D inpainting stage.
- **Surface Normal Reconstruction**
  Computes spatial gradients derived from the Marigold depth map.
  Produces a per-pixel normal map that encodes the orientation of the skin surface relative to the camera viewpoint.
- **Shader Computation & Relighting**
  Computes pixel intensity as a function of the reconstructed surface normals and the estimated light vector.
  Restores specular highlights and diffuse shading across the inpainted region, ensuring consistency with the global illumination characteristics of the portrait.

---

**Key Design Decisions**

**Transition to Marigold LCM for Surface Normals**

  - **Design** Adopted Marigold LCM (Latent Consistency Model) for direct surface-normal estimation.
  - **Rationale** The earlier approach, deriving normals from a predicted depth map was inherently unstable, producing geometric noise and unreliable surface orientation in complex scenes. Marigold reframes normal estimation as a generative task, yielding far smoother and more coherent geometry. The LCM variant preserves this quality while significantly reducing latency, aligning well with our mobile-first constraints.

**Transition to Custom-Trained LaMa Inpainting**

  - **Design** Employed the LaMa Inpainting architecture with Fast Fourier Convolutions (FFC), custom-trained on the Pipe dataset.
  - **Rationale** Although the ONNX baseline offered better runtime, the quality degradation was too severe for a creative tool. Our custom Fourier-based model captures long-range structure and global context far more effectively, resulting in cleaner, seamless inpainting.

---


### Use of Lama-Inpainting
To overcome the limitations inherent in the standard implementation, we extensively optimized the [Lama-Inpainting](https://github.com/advimman/lama) repository for integration into our AI workflow. This involved substantial code-level modifications, culminating in the creation of a dedicated GitHub repository to maintain and document our enhanced version.

---

### Weights & Biases
| Category            | Module               |  Params        | FLOPs         |
|---------------------|----------------------|----------------|---------------|
| Model               | Total                | 865.92M        | 761.498G      |
| Core Component      | time_embedding       | 2.05M          | 2.05M         |
| Attention (Total)   | All Attention Layers | 243.43M        | 229.398G      |
| ResNet (Total)      | All ResNet Layers    | 368.66M        | 369.240G      |

---

| Model Architecture         | Value     |
| -------------------------- | --------- |
| Parameters (Total)         | 865.92M   |
| Parameters (Trainable)     | 865.92M   |
| Weights Size               | 1651.62 MB   |
| Complexity                 | 761.50 GFLOPs   |


---

### Compute Profile

![relighting_latency_profile](https://hackmd.io/_uploads/BJ0ma-0W-g.png)


---

| Runtime Metric              | Value        |
|-----------------------------|--------------|
| Peak VRAM Usage             | 1,370.21 MB  |
| Total Latency               | 1,806.29 ms  |
| System Throughput           | 0.55 s       |

---

### Terms, Conditions & License
- Models
    - [Marigold-LCM](https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0) - [License](https://choosealicense.com/licenses/apache-2.0/)
    - [LaMa-Inpainting](https://github.com/advimman/lama/blob/main/LICENSE) - [License](https://github.com/advimman/lama/blob/main/LICENSE)
    - [Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2) - [License](https://huggingface.co/qualcomm/Depth-Anything-V2/blob/main/LICENSE)
- Datasets
    - [Pipe](https://huggingface.co/datasets/paint-by-inpaint/PIPE) - [License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-4.0.md)
---


## Mood Lens
Mood Lens is an emotion-based color grading engine. Rather than manually adjusting curves, the user provides an emotional intent (e.g., "Melancholy" or "Tenderness"). The system utilizes a Retrieval-Augmented Generation (RAG) pipeline to fetch style references and employs a Neural 3D LUT (Look-Up Table) to transfer the color palette while strictly preserving the original image's structure.

---

### Architecture & Workflow

![colour_correction_architecture_diagram](https://hackmd.io/_uploads/SJoLjyAWZx.jpg)


---

**Pipeline/Workflow**
- **Retrieval Augmented Generation**
    - **Embedding:** The input image is encoded via **CLIP ViT-B/32**. We embed only visual features, ignoring text labels to preserve geometric structure.
    - **Search & Filter:** We query FAISS for the top 300 visual neighbors, then apply a strict metadata filter on emotion to select the top 3 stylistic references.
- **Feature Extraction**
A pre-trained **VGG19** extracts Content Features for structure and Style Gram Matrices for texture/color.
- **Test-Time LUT Optimization**
A lightweight Trilinear 3D LUT is initialized as an Identity Matrix.
We run ~100 iterations of backpropagation to minimize Perceptual Loss on a downsampled proxy (256×256).
- **High-Res Inference**
The optimized LUT is applied to the original high-resolution image in sub-milliseconds, ensuring 100% detail preservation with zero quality loss.

---

**Key Design Decisions**

- **Image-Only Indexing Strategy**
    - **Strategy** We deliberately excluded emotion text labels from the embedding process.
    - **Rationale** Text embeddings dilute structural information (e.g., "Sad" retrieves crying faces). By embedding only the image, we restrict the search to geometry and luminance, ensuring that the retrieved reference shares the same spatial features as the user's photo.

- **Oversampling & Post-Filtering**
    - **Strategy** Instead of fetching the top 3 matches directly, we fetch the Top visual neighbours and filter for the target emotion.
    - **Rationale** Vector search is approximate. This two-step process guarantees (almost) candidates that are both visually compatible (structure) and thematically accurate (mood).

- **3D LUTs vs Generative Diffusion**
    - **Design** We built a differentiable TrilinearLUT module instead of using Stable Diffusion/ControlNet.
    - **Rationale**
        - Mobile Feasibility: ~100k parameters vs. ~860M for U-Nets.
        - Zero Hallucinations: A LUT is a pure colour mapping function. It is mathematically impossible for it to distort objects or add artefacts, making it safe for professional editing.

---

### Dataset
For the Vector Database construction we used the dataset from `unsplash` respecting the terms and condition.
[Unsplash Dataset Link](https://github.com/unsplash/datasets)


---

#### Weights & Biases
| **Model Architecture**    | **Value** |
| -------------------------- | --------- |
| Parameters (Total)         | 0.11M     |
| Parameters (Trainable)     | 0.11M     |
| Weights Size (Disk, MB)    | 0.41      |
| Complexity (FLOPs, M)      | 3.15      |

| **Module**   | **Number of Parameters** | **FLOPs** |
| ------------ | ------------------- | ---------- |
| TrilinearLUT | 107.81K             | 3.15M      |

---

### Compute Profile
![colour_correction_latency_profile](https://hackmd.io/_uploads/r1MHC1AbZl.png)

---

| Runtime Metrics              | Value       |
|------------------------------|-------------|
| VRAM                         | 1718.45MB   |
| Total Latency                | 7135.04ms   |
| System Throughput            | 0.14 img/s  |


---

### Terms, Conditions & Licenses
- Models
    - [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) - [License](https://github.com/openai/CLIP/blob/main/LICENSE)
- Datasets
    - [Unsplash-Dataset](https://github.com/unsplash/datasets) - [T&C](https://github.com/unsplash/datasets/blob/master/TERMS.md)

---

### References
- [Learning Image-adaptive 3D Lookup Tables for
High Performance Photo Enhancement in
Real-time](https://arxiv.org/pdf/2009.14468)
- [AdaInt: Learning Adaptive Intervals for 3D Lookup Tables
on Real-time Image Enhancement](https://arxiv.org/pdf/2204.13983)
- [NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement](https://arxiv.org/pdf/2306.11920)
- [CLIPstyler: Image Style Transfer with a Single Text Condition](https://arxiv.org/pdf/2112.00374)
---

## HertzMark
To ensure content provenance and detect AI-generated outputs, we implemented a robust watermarking system. Unlike spatial watermarks, this system operates in the Spectral Domain (Frequency Domain), making it resilient to compression and cropping.


### Architecture
![synth_id_architecture](https://hackmd.io/_uploads/H1HkK9AWbx.jpg)

---

#### Embedder Architecture
- **Embedder Pipeline**
The embedding process transforms the image into its frequency components to inject a learned signature into the **Mid Frequency** bands.

- **Spectral Transformation (FFT)**
The input image I is converted to the frequency domain using the Fast Fourier Transform. We shift the zero frequency component to the center to easily target specific frequency bands.
$$F(u,v)=FFT(I)$$

- **Key-Based Pattern Generation**
A dense neural network maps a unique **128-dim** cryptographic key to a spatial **64×64** watermark pattern. This is bilinearly upsampled to match the image resolution, ensuring every watermark is unique to its session/user.

- **Perceptual Masking**:
To prevent visual artifacts, we calculate a Texture Map using a convolution layer.
    - **Rationale**: The human eye is sensitive to noise in flat regions (e.g., clear sky) but insensitive in textured regions (e.g., foliage).
    - **Design**: The watermark intensity is scaled down in flat areas and scaled up in textured areas using this map.
   
- **Mid-Frequency Injection**
Low frequencies contain color/structure (modifying them ruins aesthetics). High frequencies contain noise (modifying them gets removed by JPEG compression). Mid-frequencies are the robust middle ground.
$$F_{new}= F + (\alpha  \times W_{pattern} \times M_{texture})$$

- **Reconstruction**
The modified frequency spectrum is converted back to the spatial domain via Inverse FFT to produce the final signed image.

---

#### Design Decisions
- **Frequency Domain Injection**
We chose spectral injection over pixel patching because it is Global. If a user crops the top-left corner of the image, the frequency information remains intact, allowing the detector to still identify the watermark.
- **Adaptive Alpha Scaling**
The parameter $\alpha=0.0005$ is extremely low. By combining this low baseline with the texture_map, we ensure the Peak Signal-to-Noise Ratio remains high, meaning the watermarked image is visually indistinguishable from the original.

---

### Compute Profile
- The watermarking injection operates as a near-instantaneous post-processing step. With an architectural footprint dominated by highly optimized Fast Fourier Transforms (FFT) and a compact dense projection head, the latency overhead is negligible (< 5ms).

### Terms, Conditions & Licenses
- Datasets
    - [Filmset](https://github.com/CXH-Research/FilmNet) - [License](https://github.com/CXH-Research/FilmNet/blob/main/LICENSE)

### References
- [A Robust Image Watermarking System Based
on Deep Neural Networks](https://arxiv.org/pdf/1908.11331)


## Appendix I - Installation Guide
### Software
| Field            | Value                       |
| ---------------- | --------------------------- |
| operating system | 24.04.1-Ubuntu              |
| miniconda        | 25.11.0                     |
| docker           | 29.0.0                      |
### Pre-requisites
- [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash=
# nvidia-container-toolkit installation
# -------------------------------------
# add the gpg key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# add the generic deb repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(dpkg --print-architecture) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
- Clone the repository
```bash=
# clone the repo
# --------------
git clone https://github.com/photosaverneedbackup-star/team37-adobe-14
cd team_37_adobe
git submodule init
```

---

### Text Morph
- Clone the repo
```bash=
# change the directory
# --------------
cd text_morph
```
#### Path 1 : Via Conda & Standalone server
```bash=
# create the conda environment
# ----------------------------
conda create -n text_morph python=3.8
conda activate text_morph
# install the requirements
# ------------------------
# NOTE: these versions are specifically written to avoid dependency issues,
# do not change them
pip install --upgrade torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
# other dependencies
pip install -r requirements.txt
pip install huggingface_hub

# download the dataset
python weights_download.py --path ./TextCtrl/weights
```
(incase the above fails)
- Download the model weights for `TextCtrl` from [Mega](https://mega.nz/file/LRUQxDJL#KHzmBYkCj82wPfbaBZK-t2Ma9ndUlvqqMqFdSFpP8L4)
- Move the `TextCtrl` weights to `text_morph/TextCtrl/weights`
```bash=
# start the server
# ----------------
python server.py

# use the html files provided to interact with the server
# or use CURL to send commands
```

#### Path 2 : Docker Environment
```bash=
# building the docker image
# -------------------------
sudo docker build -t text_morph .

# running the docker image
# ------------------------
sudo docker run -p 7000:7000 text_morph
```

---

### Lightshift Remove
- Clone the repo
```bash=
cd lightshift_remove
```
#### Path 1 : Via the Conda Environment
```bash=
# create the conda environment
# ----------------------------
conda create -n lightshift_remove python=3.11.14
conda activate lightshift_remove
pip install -r requirements.txt

# start the standalone server
# ---------------------------
python server.py
```

#### Path 2 : Docker Environment
```bash=
docker build -t lightshift_remove .
docker run -it --gpus all lightshift_remove
```

#### Training LaMa-Fourier
##### Download and setup the repo
```bash=
# create the conda environment
# ----------------------------
conda env create -f environment.yaml
conda activate lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

##### Training
- Download the dataset from [Pipe Dataset](https://huggingface.co/datasets/paint-by-inpaint/PIPE) and generate your validations masks from the images (or)
```bash=
python pipe_download.py
```
- Change the path of your dataset in `configs/training/orin.yaml` and the variable `data_root_dir` to the output directory of `pipe_download.py`

- Train the model
``` bash=
python bin/train.py -cn lama-fourier location=orin
```
##### Inference
- Place your test images in the folder `test_images/`
```bash=
python bin/predict.py model.path=$(pwd)/fourier indir=$(pwd)/test_images outdir=$(pwd)/test_images_output
```

---

### Mood Lens

#### Path 1 : Via Conda & Standalone server
```bash=
# create the conda environment
conda create -n mood_lens python=3.11.14
conda activate mood_lens
# install the dependencies
pip install -r requirements.txt
```

#### Preparing the Dataset
- Download the [Dataset](https://github.com/unsplash/datasets) [unsplash]
- Move the `photos.csv000` to this directory
```bash=
python mood_lens/download_dataset.py
```
- `dataset_manifest.csv` will be created

```bash=
# get the gemini api key for this step and replace the line at the top
# of the file to caption the dataset with emotion
#
# NOTE: run caption_dataset.ipynb
#
# main.py will automatically build the index for the first time & reuse it
# for subsequent runs!
cd .. #exit out of the pipeline directory
uvicorn mood_lens.server:app
```

#### Path 2 : Docker Environment
```bash=
sudo docker run -it -p 8000:8000 -e dataset_path="/app/dataset" -v $(pwd)/dataset:/app/dataset -v $(pwd)/output:/app/output -v $(pwd)/misc:/app/misc colour-correction
```
---

### HertzMark
```bash=
cd hertz_mark

# create the conda environment
# ----------------------------
conda env create -f environment.yaml
conda activate hertz_mark
```

- Download the dataset from [HertzMark](https://www.kaggle.com/datasets/xuhangc/filmset) [filmset from Kaggle]
- Move the dataset to `data/`
- To modify the dataset path, modify the `DATA_FOLDER` variable
```bash=
# for training
python train.py
```

- To infer, run these commands:
```python=
python embed_inferency.py #for embedding the input image with a watermark
python detect_inference.py #for detecting if the inout image was photoshopped by our pipeline
```

### Production Ready Docker
```bash
# uses the docker-compose.yml
docker compose up -d
```
### Website
```bash
# webprotoype
cd client
npm i
npm start
```

## Appendix I - Hardware

### CPU
---

### General Information

| Field         | Value                               |
| ------------- | ----------------------------------- |
| Architecture  | x86_64                              |
| CPU Modes     | 32-bit, 64-bit                      |
| Address Sizes | 46-bit physical, 48-bit virtual     |
| Byte Order    | Little Endian                       |
| Model Name    | 13th Gen Intel(R) Core(TM) i9-13900 |

---

### Topology

| Field            | Value    |
| ---------------- | -------- |
| Total CPUs       | 32       |
| Threads per Core | 2        |
| Max Frequency    | 5600 MHz |
| Min Frequency    | 800 MHz  |

---

### Cache Summary

| Cache Level | Size    | Instances |
| ----------- | ------- | --------- |
| L1d         | 896 KiB | 24        |
| L1i         | 1.3 MiB | 24        |
| L2          | 32 MiB  | 12        |
| L3          | 36 MiB  | 1         |

---

### GPU

---

### General Information

| Field          | Value                       |
| -------------- | --------------------------- |
| GPU Model      | NVIDIA GeForce RTX 3050 OEM |
| Driver Version | 570.195.03                  |
| CUDA Version   | 12.8                        |
| VRAM           | 8192 MiB                    |



## Appendix III - Abbreviations

- GaMuSa - Glyph Adaptive Mutual Self Attention
- CLIP - Contrastive Language-Image Pretraining
- SAM - Segment Anything
- LaMa - Large Mask
- YOLO - You Only Look Once

## NOTE
- All datasets linked are open-sourced and attached licenses.
- Terms have been attached in the repository for required Datasets.
- HertzMark has been implemented to ensure provenance.
- **Repository**:
https://github.com/photosaverneedbackup-star/team37-adobe-14
- **Drive Link**:
https://drive.google.com/drive/folders/1V2aOJvSMgiQv2A5QFmgA2ER-_sZHhoP1?usp=sharing
