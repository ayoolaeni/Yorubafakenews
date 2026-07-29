from src.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = ["genuine", "genuine", "fake", "fake"]
    y_pred = ["genuine", "genuine", "fake", "fake"]
    result = compute_metrics(y_true, y_pred)
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["confusion_matrix_labels"] == ["genuine", "fake"]


def test_compute_metrics_with_errors():
    y_true = ["genuine", "genuine", "fake", "fake"]
    y_pred = ["genuine", "fake", "fake", "fake"]
    result = compute_metrics(y_true, y_pred)
    assert result["accuracy"] == 0.75
    # one genuine misclassified as fake -> confusion_matrix[0] = [genuine->genuine, genuine->fake]
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]