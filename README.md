# Grass-Thatched House Classifier

This project is aimed at building an image classifier using Fastai to distinguish between images of grass-thatched houses and images of other types of houses.
The project is motivated by [Zindi competition](https://zindi.africa/competitions/arm-unicef-disaster-vulnerability-challenge)
## Dataset

The dataset consists of two classes:
- Grass-Thatched Houses
- Non-Grass-Thatched Houses

The dataset is organized into train and validation sets, each containing subfolders for each class.

## Project Structure


- **data/**: Contains the dataset split into train,test and validation sets.
- **models/**: Directory to save trained models.
- **notebooks/**: Jupyter notebooks for data exploration, model training, and inference.
- **src/**: Source code files for utility functions and data preprocessing.
- **requirements.txt**: List of Python packages required for the project.

## Usage

1. Clone the repository:
- git clone git@github.com:i4seeu/disaster_vun_challenge.git


2. Install dependencies:

- pip install -r requirements.txt


3. Explore the data using `data_exploration.ipynb` to gain insights into the dataset.

4. Train the model using `model_training.ipynb`.

5. Perform inference on new images using `inference.ipynb`.

## Contributing

Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

