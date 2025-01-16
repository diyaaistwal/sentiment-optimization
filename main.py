import sys
import os
import numpy as np
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing the vectorizer
from pathlib import Path
import ssl
import time

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Add the scripts folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import functions from optimization.py
from scripts.optimization import optimize_svm_hyperparameters, optimize_svm_hyperparameters_pso, optimize_svm_hyperparameters_gwo, train_optimized_model, evaluate_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load dataset 
    logging.info("Loading dataset...")
    newsgroups = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball', 'sci.space'])

    X = newsgroups.data
    y = newsgroups.target

    # TF-IDF Vectorization
    logging.info("Converting text data to numerical features using TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english')  # Using stop words to ignore common words
    X = vectorizer.fit_transform(X)

    # Split 
    logging.info("Splitting dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Directory to save outputs
    save_dir = "outputs"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set max_iter for optimization
    max_iter = 10  

    try:
        
        optimization_method = 'pso'  

        logging.info(f"Starting optimization using {optimization_method.upper()} method...")
        start_time = time.time()  

        if optimization_method == 'pso':
            best_params = optimize_svm_hyperparameters_pso(X_train, y_train, max_iter=max_iter)
        elif optimization_method == 'gwo':
            best_params = optimize_svm_hyperparameters_gwo(X_train, y_train, max_iter=max_iter)
        else:
            best_params = optimize_svm_hyperparameters(X_train, y_train)

        logging.info(f"Optimized parameters: {best_params}")

        # Train model with optimized parameters
        model = train_optimized_model(best_params, X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

        end_time = time.time()  
        logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error occurred during execution: {e}")
