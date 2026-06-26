import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# Dataset Path
# =====================================================

dataset_path = "dataset"

# =====================================================
# Image Size
# =====================================================

IMAGE_SIZE = 64
LIMIT_PER_CLASS = 100

# =====================================================
# Calorie Dictionary (Approximate kcal per 100g)
# Keys are in lowercase
# =====================================================

calorie_dict = {
    "apple": 52,
    "banana": 89,
    "beetroot": 43,
    "bell pepper": 31,
    "cabbage": 25,
    "capsicum": 31,
    "carrot": 41,
    "cauliflower": 25,
    "chilli pepper": 40,
    "corn": 86,
    "cucumber": 15,
    "eggplant": 25,
    "garlic": 149,
    "ginger": 80,
    "grapes": 69,
    "jalepeno": 29,
    "kiwi": 61,
    "lemon": 29,
    "lettuce": 15,
    "mango": 60,
    "onion": 40,
    "orange": 47,
    "paprika": 282,
    "pear": 57,
    "peas": 81,
    "pineapple": 50,
    "pomegranate": 83,
    "potato": 77,
    "raddish": 16,
    "soy beans": 446,
    "spinach": 23,
    "sweetcorn": 86,
    "sweetpotato": 86,
    "tomato": 18,
    "turnip": 28,
    "watermelon": 30
}

# =====================================================
# Load Dataset
# =====================================================

data = []
labels = []

print("Loading dataset...\n")

folders = sorted(os.listdir(dataset_path))

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    count = 0

    for file in os.listdir(folder_path):

        if count >= LIMIT_PER_CLASS:
            break

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        try:

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data.append(img.flatten())

            # Store label in lowercase
            labels.append(folder.lower())

            count += 1

        except:
            continue

X = np.array(data)
y = np.array(labels)

print("Dataset Shape :", X.shape)
print("Number of Classes :", len(np.unique(y)))

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# Train SVM
# =====================================================

print("\nTraining SVM Model...\n")

model = SVC(kernel="linear")

model.fit(X_train, y_train)

# =====================================================
# Prediction
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# =====================================================
# Display Predictions
# =====================================================

plt.figure(figsize=(15, 8))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    img = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE)

    food = y_pred[i].lower()

    calories = calorie_dict.get(food, "Unknown")

    plt.imshow(img, cmap="gray")

    plt.title(f"{food.title()}\n{calories} kcal", fontsize=10)

    plt.axis("off")

plt.tight_layout()

plt.savefig("food_predictions.png", dpi=300)

plt.show()

print("\nPrediction image saved as food_predictions.png")