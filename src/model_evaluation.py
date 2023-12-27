from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(y_test, y_pred):
    f1 = f1_score(y_test, y_pred)
    accu = accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accu}")