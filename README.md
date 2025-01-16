## Sentiment Optimization Using PSO and GWO

# OVERVIEW
This project applies Particle Swarm Optimization (PSO) and Grey Wolf Optimization (GWO) to tune hyperparameters for an SVM model. Using the 20 Newsgroups dataset, it classifies text from two categories: rec.sport.baseball and sci.space. Features are extracted using TF-IDF, and model performance is evaluated with key metrics.

# KEY FEATURES
1. Text classification using Support Vector Machines (SVM).
2. Hyperparameter tuning with PSO and GWO algorithms.
3. TF-IDF Vectorization for text feature extraction.
4. Performance evaluation with metrics like accuracy and execution time.

# REQUIREMENTS
1. System Requirements
> Operating System: Windows, macOS, or Linux.
> Python Version: 3.8 or later
> Memory: At least 4 GB of RAM (8 GB recommended for large datasets).
> Storage: 500 MB of free disk space for outputs and dependencies.

2. Software Dependencies
> The following Python libraries are required:
> joblib==1.3.2
> numpy==1.24.3
> scikit-learn==1.2.2
> pathlib==1.0.1
> pandas==1.5.3
> logging==0.5.1.2

# INSTALLATION STEPS
1. Clone the Repository:
git clone https://github.com/diyaaistwal/sentiment-optimization.git
cd sentiment-optimization

2. Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

3. Install Dependencies:
pip install -r requirements.txt

4. Verify Installation:
python -c "import numpy, sklearn, pathlib; print('All dependencies installed successfully!')"

# PROJECT STRUCTURE
sentiment-optimization/
├── main.py                 # Main script to run the project
├── scripts/                # Folder containing all core script files
│   ├── optimization.py     # Optimization methods (PSO, GWO, etc.)
│   ├── pso.py              # Particle Swarm Optimization implementation
│   └── gwo.py              # Grey Wolf Optimization implementation
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── LICENSE                 # Project license

# USAGE
1. Run the Project
To execute the optimization process and evaluate the SVM model:
python main.py

2. Change Optimization Method
You can switch between PSO and GWO by modifying the optimization_method variable in main.py:
optimization_method = 'pso'  # Use 'gwo' for Grey Wolf Optimization

# CONTRIBUTING
Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

# LICENSE
This project is licensed under the MIT License. See the LICENSE file for details.

# ACKNOWLEDGEMENTS
1. Dataset: 20 Newsgroups Dataset
2. Optimization Algorithms: PSO and GWO implementations inspired by academic research.

