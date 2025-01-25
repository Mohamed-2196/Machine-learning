How to Open and Run the Code
1. Prerequisites
Python: Ensure Python 3.7 or higher is installed.

Libraries: Install the required libraries using pip:

bash
Copy
pip install pandas numpy opencv-python tensorflow
Dataset: Ensure the dataset is available in the correct directory structure:

Copy
/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/
├── train.csv
├── test.csv
└── Images/
    └── Images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
2. Running the Code

Open a terminal or command prompt and navigate to the directory where the script is saved.

Run the script:

bash
Copy
python main.py
3. Output
The script will:

Load and preprocess the training and test data.

Train a Convolutional Neural Network (CNN) model.

Generate predictions on the test data.

Save the predictions to a CSV file named predictions.csv.

4. Troubleshooting
Missing Dataset: Ensure the dataset is correctly placed in the /kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/ directory.

Library Errors: If any library is missing, install it using pip install <library_name>.

Image Loading Issues: If images fail to load, verify the paths and ensure the images exist in the specified directory.

5. Customization
Model Architecture: Modify the CNN architecture in the model = keras.Sequential([...]) section to experiment with different layers.

Training Parameters: Adjust the number of epochs, batch size, or optimizer in the model.fit() function.

6. Dependencies
The code uses the following libraries:

pandas for data manipulation.

numpy for numerical operations.

opencv-python for image processing.

tensorflow for building and training the neural network.

7. License
This code is provided as-is. Feel free to modify and use it for your purposes.