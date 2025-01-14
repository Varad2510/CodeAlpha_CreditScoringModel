# Credit Scoring Model

## Project Title
**Credit Scoring Model to Predict Creditworthiness**

## Project Description
This project involves developing a **Credit Scoring Model** to predict the creditworthiness of individuals. The model is based on historical financial data such as **Age, Annual Income, Loan Amount**, and **Credit History**. The goal is to assess whether a person is likely to default on a loan using classification algorithms like **Random Forest Classifier**.

The project uses machine learning techniques to predict the binary outcome where:
- **0**: Non-default
- **1**: Default

## Prerequisites and Dependencies
To run this project, you need the following Python libraries:

- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical computations.
- `scikit-learn` - For building and evaluating the machine learning model.
- `matplotlib` (optional) - For visualization (e.g., plotting confusion matrices).
- `seaborn` (optional) - For enhanced data visualization.

You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Dataset
This project uses a **synthetic dataset** that represents individual financial data. The dataset includes the following features:
- `Age`: Age of the individual.
- `Annual_Income`: Annual income of the individual.
- `Loan_Amount`: The amount of loan the individual is applying for.
- `Credit_History`: A score representing the individual's credit history.
- `Default`: Whether the individual has defaulted on a loan (1: Default, 0: No Default).

The dataset is created for demonstration purposes, but real-world projects would use actual historical financial data.

## Instructions on How to Run the Code
1. **Clone the GitHub repository** to your local machine:
   ```bash
   git clone https://github.com/Varad2510/CodeAlpha_CreditScoringModel.git
   ```

2. **Upload the dataset** to your environment (Google Colab, Jupyter Notebook, or local machine).

3. **Run the Python script** or **Jupyter notebook**. The model will:
   - Preprocess the dataset.
   - Split it into training and testing sets.
   - Train a **Random Forest Classifier** to predict credit default.
   - Display the **accuracy**, **classification report**, and **confusion matrix**.

4. To run the code, make sure to first load the necessary libraries and import your dataset as shown below:
   ```python
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report, confusion_matrix

   # Load dataset
   data = pd.read_csv('path_to_dataset.csv')
   ```

5. After running the code, you should see the following output in the console:
   - Model accuracy (percentage).
   - A classification report showing precision, recall, and F1-score.
   - A confusion matrix displaying the number of true positives, false positives, true negatives, and false negatives.

## Sample Output
After running the model, you should expect output similar to this:

### Model Accuracy:
```
Model Accuracy: 85.00%
```

### Classification Report:
```
              precision    recall  f1-score   support

           0       0.87      0.83      0.85       500
           1       0.83      0.87      0.85       500

    accuracy                           0.85      1000
   macro avg       0.85      0.85      0.85      1000
weighted avg       0.85      0.85      0.85      1000
```

### Confusion Matrix:
```
[[415  85]
 [ 65 435]]
```

## Attribution
- The dataset used in this project is synthetic and created for demonstration purposes. In real-world applications, you would use actual financial data from sources like **credit bureaus** or **banks**.
- If using this project for academic purposes or real-world deployment, ensure that the dataset is compliant with data privacy regulations.

## License
This project is open-source and available under the [MIT License](LICENSE).

