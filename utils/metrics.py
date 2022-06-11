from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def print_cross_val_summary(scores, metric_used='neg_mean_squared_error'):
    print(f"{scores.mean():.2f} {metric_used} with a standard deviation of {scores.std():.2f}")

def print_metrics(y, y_hat):
    print(f"MSE: {mean_squared_error(y, y_hat)}")
    print(f"RMSE: {mean_squared_error(y, y_hat, squared=False)}")
    print(f"MAE: {mean_absolute_error(y, y_hat)}")
    print(f"R2: {r2_score(y, y_hat)}")
