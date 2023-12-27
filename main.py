from src import data_loader, data_preparation, model_training, model_evaluation

def main():
    file_path = 'data/training.csv'

    # Load data
    df = data_loader.load_data(file_path)

    # Explore and prepare data
    data_preparation.explore_data(df)
    X, y = data_preparation.prepare_data(df)

    # Train the model
    y_test, y_pred = model_training.train_model(X, y)

    # Evaluate the model
    model_evaluation.evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()