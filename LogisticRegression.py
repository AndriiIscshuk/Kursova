import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23, 30, 40, 55, 60, 28],
    'income': [30000, 80000, 50000, 120000, 20000, 40000, 70000, 100000, 90000, 35000],
    'delinquencies': [0, 1, 0, 2, 3, 0, 1, 0, 2, 1],
    'cards': [1, 3, 2, 4, 0, 2, 3, 5, 2, 1],
    'bankruptcies': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    'loan_amount': [5000, 20000, 10000, 30000, 4000, 8000, 15000, 25000, 18000, 7000],
    'employment_years': [1, 10, 5, 15, 0, 3, 8, 20, 12, 2],
    'has_mortgage': [0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    'education_level': [1, 2, 2, 3, 0, 1, 2, 3, 2, 1],  # 0=none, 1=school, 2=college, 3=university
    'owns_car': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    'default': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
})

features = [
    'age', 'income', 'delinquencies', 'cards', 'bankruptcies',
    'loan_amount', 'employment_years', 'has_mortgage',
    'education_level', 'owns_car'
]
X = data[features]
y = data['default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔁 Навчання логістичної регресії
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

print("\nВведіть анкету клієнта:")
def ask(question, type_fn=int):
    return type_fn(input(question + ": "))

new_client_input = [
    ask("Вік", int),
    ask("Місячний дохід (грн)", float),
    ask("Кількість прострочень", int),
    ask("Кількість кредитних карт", int),
    ask("Кількість банкрутств", int),
    ask("Бажана сума кредиту (грн)", float),
    ask("Стаж роботи (роки)", float),
    ask("Іпотека (1 - є, 0 - нема)", int),
    ask("Освіта (0=немає, 1=школа, 2=коледж, 3=вища)", int),
    ask("Власне авто (1 - так, 0 - ні)", int)
]
new_df = pd.DataFrame([new_client_input], columns=features)
new_client_scaled = scaler.transform(new_df)

# 📐 Обчислення ймовірності дефолту через модель sklearn
sklearn_prob = model.predict_proba(new_client_scaled)[0][1]

# 📤 Вивід результатів
print("\n📊 Результати скорингу:")
print(f"Ймовірність дефолту (модель sklearn): {sklearn_prob:.4f}")

if sklearn_prob >= 0.7:
    risk = "🔴 Високий ризик (відмова)"
elif sklearn_prob >= 0.4:
    risk = "🟠 Середній ризик (перевірка вручну)"
else:
    risk = "🟢 Низький ризик (схвалення ймовірне)"

print(f"Рівень ризику: {risk}")
