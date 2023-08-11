# Pytorch-Age-Classifier
Pytorch-Age-Classifier is a deep learning model implemented using PyTorch that classifies ages into seven different categories:

    Baby (0-2 years)
    Child (3-12 years)
    Teen (13-20 years)
    Young adult (21-34 years)
    Middle age adult (35-54 years)
    Senior adult (55-69 years)
    Third age person (70-129 years)

The model uses the UTKFace dataset and the inception_v3 architecture for classification.
Dataset Representation
UTKFace

This class represents the UTK Face dataset.

Usage:

python

dataset = UTKFace(root_dir='path_to_data')
sample = dataset[0]
print(sample['x'], sample['y'], sample['label'])

    root_dir - Directory with all the images.
    transform - (optional) Transformations to be applied to the images. Default is None.

Each sample from this dataset returns a dictionary containing:

    x: the image.
    y: the age label in terms of 0 to 6.
    label: a string representing the age category.

Model Architecture

The model uses the inception_v3 architecture. The last layer of the pretrained model is replaced with a fully connected layer with 7 output units, corresponding to the 7 age categories.
Training and Evaluation

The train_epoch function trains the model for one epoch, while the eval_epoch function evaluates the model on a given data loader.

To train the model:

python

train(model, train_data_loader, val_data_loader, train_writer, val_writer, num_epochs)

After each epoch, the model is evaluated on both training and validation datasets. The best model weights are saved when the validation accuracy improves.
Checkpoint

The save_check_point function allows for saving model weights and the training epoch.
Dependencies

To use this classifier, make sure to have the following libraries:

    torch
    torchvision
    numpy
    Pillow

Installation

    Clone the repository:

    bash

git clone https://github.com/[your_username]/Pytorch-Age-Classifier.git

Navigate to the directory:

bash

cd Pytorch-Age-Classifier

Install the required packages:

    pip install -r requirements.txt

License

This project is licensed under the MIT License.
Author

[Your Name]

Note: Before sharing or pushing your project to GitHub, make sure to add a .gitignore file to avoid uploading large dataset files or unnecessary files. Also, this is a basic README structure. Depending on the scope of your project, you might need to include more sections like 'Contribution', 'Acknowledgments', etc.
