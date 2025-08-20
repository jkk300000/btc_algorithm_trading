from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model_performance(y_true, y_pred, verbose=True):
    """
    분류 모델의 성능을 평가합니다.
    Args:
        y_true: 실제값 (array-like)
        y_pred: 예측값 (array-like)
        verbose: True면 결과를 출력합니다.
    Returns:
        dict: 평가 지표 및 리포트
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, digits=4)
    
    if verbose:
        print("==== 모델 평가 결과 ====")
        print(f"정확도(Accuracy): {acc:.4f}")
        print(f"정밀도(Precision): {prec:.4f}")
        print(f"재현율(Recall): {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    } 