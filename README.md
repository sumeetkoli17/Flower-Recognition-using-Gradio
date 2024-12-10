# Flower-Recognition-using-Gradio

This project is a flower recognition model built using TensorFlow and deployed using Gradio. The model classifies images of flowers into one of five categories: **Daisy**, **Dandelion**, **Roses**, **Sunflowers**, and **Tulips**. The dataset is pre-processed, trained using a Convolutional Neural Network (CNN), and then the model is deployed using Gradio for easy interaction.

## Dataset

The dataset used in this project is the **Flower Photos Dataset** provided by TensorFlow. It contains images of 5 different types of flowers, each located in a separate folder.

- **Dataset URL**: [Flower Photos Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

### Dataset Details:
- **Number of Classes**: 5
  - Daisy
  - Dandelion
  - Roses
  - Sunflowers
  - Tulips
- **Total Images**: 3,670 images in total (734 images per class)
- **Image Size**: Each image is resized to 180x180 pixels before training.

## Libraries Used

This project uses several libraries for data manipulation, model building, training, and deployment.

### 1. **NumPy**
   - **Description**: NumPy is a package for scientific computing in Python. It provides support for large multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays.

### 2. **Matplotlib**
   - **Description**: Matplotlib is a plotting library for Python and its numerical mathematics extension NumPy. It is used to create static, animated, and interactive visualizations, such as graphs and charts, to display the results of model training and predictions.

### 3. **TensorFlow**
   - **Description**: TensorFlow is an open-source machine learning framework developed by Google. It is widely used for building and training deep learning models, including neural networks for tasks such as image classification, natural language processing, and reinforcement learning.

### 4. **Gradio**
   - **Description**: Gradio is an open-source library that allows you to quickly create user interfaces for machine learning models. It helps you create interactive demos that can be shared with others, making it easier to test and showcase models in real-time.

### 5. **Pillow (PIL)**
   - **Description**: Pillow is a Python Imaging Library (PIL) fork that adds image processing capabilities to your Python interpreter. It supports opening, manipulating, and saving many different image file formats, which is essential for image pre-processing before feeding the data into the model.

## Project Workflow

1. **Data Preparation**:
   - Download and extract the flower photos dataset.
   - Preprocess images (resize, normalization) to prepare them for model training.

2. **Model Building**:
   - Use TensorFlow to create a Convolutional Neural Network (CNN) for flower classification.
   - Train the model using the flower photos dataset.

3. **Model Evaluation**:
   - Evaluate the model's accuracy on the validation dataset.
   - Save the model.

4. **Model Deployment**:
   - Deploy the trained model using Gradio, allowing users to interact with it by uploading flower images and receiving predictions.

## How to Run the Project

1. **Clone the Repository**:
   - Clone this repository to your local machine using the following command:
     ```
     git clone https://github.com/your-username/flower-recognition-using-gradio.git
     ```

2. **Install Dependencies**:
   - Make sure you have Python 3.7 or higher installed.
   - Install the required libraries using pip:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Application**:
   - After the dependencies are installed, run the Gradio app with the following command:
     ```
     python app.py
     ```

4. **Access the App**:
   - Once the app is running, open a browser and navigate to `http://127.0.0.1:7861` to interact with the flower recognition model.

