import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_DIR = 'dataset'
image_paths = glob.glob(os.path.join(DATASET_DIR, '*/*.jpg'))
print(f"Total images found: {len(image_paths)}")

images = []
labels = []
class_map = {'cats': 0, 'dogs': 1, 'snakes': 2}

for path in image_paths:
    try:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.flatten())
        folder = os.path.basename(os.path.dirname(path)).lower()
        if folder in class_map:
            labels.append(class_map[folder])
    except:
        continue

if not images:
    raise ValueError("No images loaded. Check your folder structure.")

X = np.array(images) / 255.0
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)

print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=25, random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

print("Training Neural Network...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    solver='adam',
    learning_rate_init=0.0001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

def show_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix - {name}")
    print(cm)

show_confusion("Naive Bayes", y_test, nb_preds)
show_confusion("Decision Tree", y_test, dt_preds)
show_confusion("Neural Network", y_test, mlp_preds)

evaluate("Naive Bayes", y_test, nb_preds)
evaluate("Decision Tree", y_test, dt_preds)
evaluate("Neural Network", y_test, mlp_preds)
