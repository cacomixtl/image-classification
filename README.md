# Stanford Cars Image Classification Project

## Description

This project implements a deep learning model to classify images from the Stanford Cars dataset into one of 196 distinct categories, typically defined by Make, Model, and Year (e.g., "Audi S4 Sedan 2012"). The primary goal was to explore the end-to-end process of building, training, and evaluating an image classifier using transfer learning within a Google Colab environment.

## Dataset

* **Name:** Stanford Cars Dataset
* **Source:** [Stanford AI Lab](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* **Contents:** Contains 16,185 images across 196 classes.
    * Training Set: 8,144 images
    * Test Set: 8,041 images
* **Structure Used:** This project assumes the images are organized into class-named subdirectories within `train` and `test` folders, like so:
    ```
    <Your Dataset Root>/
    ├── car_data/
    │   └── car_dataset/
    │       ├── train/
    │       │   ├── Acura Integra Type R 2001/
    │       │   │   ├── 00128.jpg
    │       │   │   └── ...
    │       │   └── ... (195 more class folders)
    │       └── test/
    │           ├── Acura Integra Type R 2001/
    │           │   ├── yyyyy.jpg
    │           │   └── ...
    │           └── ... (195 more class folders)
    ├── anno_test.csv
    ├── anno_train.csv
    └── names.csv
    ```
    * Annotation CSVs (`anno_*.csv`) map filenames to class IDs (1-196) and bounding boxes (bounding boxes were not used in this classification task).
    * `names.csv` maps class IDs to human-readable names.
    *(Note: The CSV files used in this project appeared to be headerless).*

## Methodology

1.  **Environment:** Google Colab with Python 3 and GPU acceleration.
2.  **Data Loading & Prep:**
    * Annotations loaded using Pandas, handling headerless CSVs.
    * Class names mapped to annotations.
    * File paths constructed dynamically based on the class-subdirectory structure, handling special characters (like `/` -> `-`).
    * Data split into training, validation (20% of original train), and test sets using Scikit-learn (`train_test_split` with stratification).
3.  **Preprocessing:**
    * Images resized to a fixed input size (224x224 pixels).
    * Pixel values normalized using the specific `preprocess_input` function for the chosen base model (EfficientNetB0).
4.  **Data Augmentation (Training Set Only):**
    * Applied using `tf.keras.layers` (RandomFlip, RandomRotation, RandomZoom, RandomBrightness) to increase robustness and reduce overfitting.
5.  **Data Pipelines:**
    * Efficient data loading pipelines created using `tf.data.Dataset.from_tensor_slices` and `.map()` for preprocessing and augmentation.
    * Data was shuffled (`.shuffle()`), batched (`.batch()`), and prefetched (`.prefetch()`) for optimal training performance.
6.  **Modeling (Transfer Learning):**
    * **Base Model:** `EfficientNetB0` pre-trained on ImageNet, with the top classification layer removed (`include_top=False`).
    * **Freezing:** Base model layers were initially frozen (`trainable = False`).
    * **Custom Head:** A new classification head was added on top, consisting of `GlobalAveragePooling2D`, `Dropout` (30%), and a `Dense` output layer with 196 units and `softmax` activation.
7.  **Training:**
    * **Optimizer:** Adam.
    * **Loss Function:** Sparse Categorical Crossentropy (suitable for integer labels).
    * **Metrics:** Accuracy.
    * **Callbacks:** `ModelCheckpoint` (saving best weights based on `val_accuracy`), `EarlyStopping` (monitoring `val_accuracy` with `restore_best_weights=True`).
    * **Initial Phase:** Trained only the custom head with the base model frozen.
    * **Fine-Tuning Phase:** Unfroze the top ~30 layers of `EfficientNetB0`, recompiled with a very low learning rate (`1e-5`), and continued training.

## Setup & Usage

### Environment

* Recommended: Google Colab
* Language: Python 3
* Hardware: GPU recommended for efficient training (CPU possible but very slow for training).

### Dependencies

The primary libraries used are:

* TensorFlow / Keras (`tensorflow`)
* Pandas (`pandas`)
* Scikit-learn (`sklearn`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* Seaborn (`seaborn`)
* OpenCV (`opencv-python`) - Optional, mainly for image dimension checks.

You can typically install these via pip:
`pip install tensorflow pandas scikit-learn numpy matplotlib seaborn opencv-python`

### Dataset Setup

1.  Download the Stanford Cars Dataset from the official source or other providers (ensure it includes the images and the `anno_*.csv`, `names.csv` structure mentioned above).
2.  Upload the dataset to your Google Drive.
3.  **Configure Path:** Open the main notebook (`.ipynb` file) and find the **Phase 1** setup cells. Modify the `RELATIVE_DATASET_PATH_IN_DRIVE` variable to match the path *within* your Google Drive where you placed the top-level dataset folder. For example, if your dataset is at `MyDrive/MyDatasets/Stanford_Cars_Dataset`, set `RELATIVE_DATASET_PATH_IN_DRIVE = 'MyDatasets/Stanford_Cars_Dataset'`.

### Running the Code

1.  Open the notebook in Google Colab.
2.  Ensure the runtime type is set to GPU (Runtime -> Change runtime type -> GPU).
3.  Mount your Google Drive when prompted.
4.  Run the cells sequentially, following the phases (Phase 1 through Phase 7).
    * Phase 1: Sets up paths and environment.
    * Phase 2: Loads data, generates paths, explores.
    * Phase 3: Creates validation split, defines preprocessing/augmentation, builds data pipelines.
    * Phase 4: Defines the model architecture.
    * Phase 5: Compiles and trains the model (initial & fine-tuning). Requires GPU for reasonable speed. Saves best weights to Drive.
    * Phase 6: Loads best weights and evaluates the model on the test set.
    * Phase 7: Documentation & Conclusion (this README corresponds to this).

## Results

* **Test Set Accuracy:** **~53.63%**
* **Test Set Loss:** ~1.8221

This performance was achieved after fine-tuning the EfficientNetB0 model. While the model demonstrates learning, there was evidence of overfitting (a significant gap between training accuracy (~78%) and validation/test accuracy (~53%)), suggesting potential areas for improvement.

## Discussion & Future Work

* **Overfitting:** The primary limitation observed was overfitting. While techniques like dropout and data augmentation were used, more could be explored.
* **Class Imbalance:** The dataset has some class imbalance, which might affect performance on less frequent classes.
* **Potential Improvements:**
    * More advanced data augmentation (e.g., RandAugment, MixUp).
    * Stronger regularization techniques (e.g., increased dropout, weight decay).
    * Hyperparameter tuning (learning rates, batch size, number of unfrozen layers, optimizer parameters).
    * Trying different pre-trained base models (e.g., ResNets, Vision Transformers).
    * Implementing class weighting or focal loss to address imbalance.
    * Experimenting with different input resolutions.

## License

*This project is licensed under the terms of the MIT license.

## Authors

* Marco Alonso López Franco / Gemini 2.5 Pro (experimental)
