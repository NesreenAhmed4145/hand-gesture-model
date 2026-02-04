import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. تحميل البيانات
data_file = 'hand_gesture_dataset_normalized.csv'

try:
    df = pd.read_csv(data_file)
    print(f"✅ Data Loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: CSV file not found.")
    exit()

# 2. تجهيز البيانات
X = df.drop('label', axis=1)
y = df['label']

# تقسيم الداتا: 80% تدريب - 20% اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. الموديلات المختارة للمقارنة (تم حذف Gradient Boosting)
pipelines = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression()), 
    'Random Forest': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42)),
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)) 
}

best_model = None
best_accuracy = 0.0
best_model_name = ""

print("\n🔄 Training & Comparing Models...")

for name, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🔹 {name}: {accuracy*100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\n🏆 Winner: {best_model_name} ({best_accuracy*100:.2f}%)")

# تقرير الأفضل
print(f"\n📊 Report for {best_model_name}:")
print(classification_report(y_test, best_model.predict(X_test)))

# 4. الحفظ
model_filename = 'hand_gesture_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Model saved as '{model_filename}'")