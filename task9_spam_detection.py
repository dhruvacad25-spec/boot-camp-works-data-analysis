

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset Sample:\n", df.head())
print("\nClass Distribution:\n", df['label'].value_counts())



df['msg_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))
df['digit_count'] = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
df['upper_count'] = df['message'].apply(lambda x: sum(c.isupper() for c in x))

print("\nEngineered Features:\n",
      df[['message', 'msg_length', 'word_count',
          'digit_count', 'upper_count', 'label_num']].head())



X = df[['msg_length', 'word_count', 'digit_count', 'upper_count']]  # Independent variables
y = df['label_num']                                                  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients (Impact of features):", model.coef_)
print("Model Intercept:", model.intercept_)


y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("R2 Score:", round(r2, 4))
print("RMSE:", round(rmse, 4))


comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted (sample):\n", comparison.head(10))


residuals = y_test - y_pred
sse = round((residuals ** 2).sum(), 2)

print("\nResiduals (sample):\n", residuals.head(10))
print("\nSum of Squared Errors (SSE):", sse)


new_message = "WIN a FREE prize NOW!!! Call 0800123456 to claim"
new_features = [[
    len(new_message),
    len(new_message.split()),
    sum(c.isdigit() for c in new_message),
    sum(c.isupper() for c in new_message)
]]
predicted_score = model.predict(new_features)
print("\nPredicted spam-score for new message:", predicted_score[0])



plt.figure(figsize=(7, 5))
plt.hist(y_pred, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Values')
plt.tight_layout()
plt.show()



plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, color='purple', alpha=0.4)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.tight_layout()
plt.show()
