    y_score: np.ndarray = model.predict_proba(x_test)[:, 1]

    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred)
    recall: float = recall_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred)
    roc_auc: float = roc_auc_score(y_test, y_score)
    matrix: np.ndarray = confusion_matrix(y_test, y_pred)
    report: str = classification_report(y_test, y_