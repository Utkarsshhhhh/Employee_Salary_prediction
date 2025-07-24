import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(file_path):
    """Load dataset from an Excel file."""
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def display_data_info(data):
    """Display basic information about the dataset."""
    print(data.head(10))
    print(data.tail(10))
    print(data.info())
    print(data.shape)
    print(data.isna().sum())
    print(data.duplicated().sum())

def visualize_gender_distribution(data):
    """Visualize gender distribution in a pie chart."""
    data["Gender"].value_counts().sort_values(ascending=False).plot(kind="pie", autopct='%1.1f%%')
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

def visualize_job_rate_histogram(data):
    """Visualize job rate distribution in a histogram."""
    plt.hist(data["Job Rate"], bins=20, color='skyblue', edgecolor='black')
    plt.title("Job Rate Histogram")
    plt.xlabel("Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def visualize_avg_salary_by_dept(data):
    """Visualize average salary by department in a bar chart."""
    avg_salary_by_dept = data.groupby("Department")["Annual Salary"].mean().sort_values(ascending=False)
    avg_salary_by_dept.head(10).plot(kind="bar", color='skyblue')
    plt.title("Average Salary by Department")
    plt.xlabel("Department")
    plt.ylabel("Average Salary")
    plt.tight_layout()
    plt.show()

def visualize_avg_job_rate_by_country(data):
    """Visualize average job rate by country in a bar chart."""
    avg_job_rate_by_country = data.groupby("Country")["Job Rate"].mean().sort_values(ascending=False)
    avg_job_rate_by_country.plot(kind="bar", color='skyblue')
    plt.title("Average Job Rate by Country")
    plt.xlabel("Country")
    plt.ylabel("Job Rate Average")
    plt.tight_layout()
    plt.show()

def visualize_overtime_hours_histogram(data):
    """Visualize overtime hours distribution in a histogram."""
    plt.hist(data["Overtime Hours"], bins=20, color='skyblue', edgecolor='black')
    plt.title("Overtime Hours Histogram")
    plt.xlabel("Overtime Hours")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def train_and_evaluate_models(x_train, y_train, x_test, y_test):
    """Train and evaluate regression models."""
    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "MLP Regressor": MLPRegressor(solver='adam', hidden_layer_sizes=(5, 2), random_state=2, max_iter=500),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
        print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

    return results, models

def visualize_model_comparison(results):
    """Visualize model comparison based on Mean Absolute Error."""
    mae_values = [metrics['MAE'] for metrics in results.values()]
    plt.bar(results.keys(), mae_values, color='skyblue')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Comparison (Mean Absolute Error)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def main():
    # Load the dataset
    data = load_data("Data/Employees.xlsx")
    if data is None:
        return

    # Display data information
    display_data_info(data)

    # Visualizations
    visualize_gender_distribution(data)
    visualize_job_rate_histogram(data)
    visualize_avg_salary_by_dept(data)
    visualize_avg_job_rate_by_country(data)
    visualize_overtime_hours_histogram(data)

    # Feature selection
    x = data[["Years", "Job Rate"]]
    y = data["Annual Salary"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results, models = train_and_evaluate_models(x_train, y_train, x_test, y_test)

    # Get the best model
    best_model_name = min(results, key=lambda k: results[k]['MAE'])
    best_model = models[best_model_name]
    print(f"\n✅ Best model: {best_model_name} with MAE {results[best_model_name]['MAE']:.4f}")

    # Save the best model
    joblib.dump(best_model, "Best_model.pkl")
    print("✅ Saved best model as Best_model.pkl")

    # Visualize model comparison
    visualize_model_comparison(results)

if __name__ == "__main__":
    main()
