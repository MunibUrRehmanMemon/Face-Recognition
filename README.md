# Face Recognition and Annotation Pipeline

This project implements a face recognition pipeline for annotating video files with detected face labels. It leverages the **InsightFace** library for face analysis and ONNX for runtime optimizations.

---

## Features

- **Face Embedding Generation**: Extracts face embeddings from labeled images and stores them in a CSV file.
- **Video Frame Encoding**: Extracts face embeddings from video frames.
- **Face Matching**: Matches embeddings from video frames against a labeled dataset.
- **Video Annotation**: Annotates video frames with recognized face labels and bounding boxes.

---

## Prerequisites

### Libraries
Install the required Python libraries:

```bash
pip install insightface
pip install onnxruntime
pip install opencv-python-headless numpy pandas
```

### Data Preparation
1. **Labeled Images**: Organize labeled images in directories where each folder name corresponds to the person's name.
   Example structure:
   ```
   /path/to/labeled/images
   ├── person1
   │   ├── image1.jpg
   │   └── image2.jpg
   └── person2
       ├── image1.jpg
       └── image2.jpg
   ```

2. **Input Video**: Provide the video file to process.

3. **Output Paths**: Define paths for saving the CSV file (for embeddings), pickle file (for encoded faces), and annotated video.

---

## How It Works

### 1. Generate Embeddings from Labeled Images
Extracts face embeddings from labeled images and saves them as a CSV file:

```python
output_csv = '/path/to/output/dataset.csv'
labeled_faces_dir = '/path/to/labeled/images'
save_embeddings_to_csv(labeled_faces_dir, output_csv)
```

### 2. Encode Faces from Video Frames
Extracts face embeddings from video frames and saves them in a pickle file:

```python
video_path = '/path/to/video.mp4'
output_pkl_file = '/path/to/encoded_faces.pkl'
encode_faces_from_new_video(video_path, output_pkl_file)
```

### 3. Match Faces and Annotate Video
Matches the embeddings from the video frames against the labeled dataset and generates an annotated video:

```python
csv_file = '/path/to/dataset.csv'
new_output_path = '/path/to/annotated_video.mp4'
match_faces_and_generate_video(video_path, new_output_path, encoded_faces, csv_file)
```

---

## Code Structure

### Functions

1. **`save_embeddings_to_csv(labeled_faces_dir, output_csv)`**
   - Extracts face embeddings from labeled images.
   - Saves embeddings to a CSV file.

2. **`load_embeddings_from_csv(csv_file)`**
   - Loads labeled face embeddings from a CSV file.

3. **`encode_faces_from_new_video(video_path, output_pkl_file)`**
   - Encodes faces from video frames and saves them as a pickle file.

4. **`match_faces_and_generate_video(video_path, output_path, encoded_faces, csv_file)`**
   - Matches embeddings and annotates video frames with face labels.

---

## Example Usage

### Step 1: Generate Embeddings from Images
```python
labeled_faces_dir = '/path/to/labeled/images'
output_csv = '/path/to/output/dataset.csv'
save_embeddings_to_csv(labeled_faces_dir, output_csv)
```

### Step 2: Encode Video Frames
```python
video_path = '/path/to/input_video.mp4'
output_pkl_file = '/path/to/encoded_faces.pkl'
encode_faces_from_new_video(video_path, output_pkl_file)
```

### Step 3: Annotate Video
```python
csv_file = '/path/to/dataset.csv'
new_output_path = '/path/to/annotated_video.mp4'
with open(output_pkl_file, 'rb') as f:
    encoded_faces = pickle.load(f)
match_faces_and_generate_video(video_path, new_output_path, encoded_faces, csv_file)
```

---

## Output
- **CSV File**: Stores labeled face embeddings.
- **Pickle File**: Encoded face embeddings from the video.
- **Annotated Video**: Video file annotated with face labels.

---

## Notes
- Ensure that labeled images are high-quality for accurate embeddings.
- Use the `InsightFace` library’s GPU acceleration for better performance.
- Tune the matching threshold (default: 1.0) for improved accuracy.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or contributions, feel free to create an issue or pull request on the GitHub repository.

