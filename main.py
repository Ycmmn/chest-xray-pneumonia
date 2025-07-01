# -------------------- 1. ایمپورت کتابخانه‌ها --------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K

# -------------------- 2. مسیر دیتاست --------------------
masir_data = r"C:\Users\Yasaman\Desktop"

# -------------------- 3. نرمال‌سازی و بارگذاری داده‌ها --------------------
train_data = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

x_train = train_data.flow_from_directory(
    os.path.join(masir_data, 'chest-xray-pneumonia-project', 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

x_val = train_data.flow_from_directory(
    os.path.join(masir_data, 'chest-xray-pneumonia-project', 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_data = ImageDataGenerator(rescale=1./255)

x_test = test_data.flow_from_directory(
    os.path.join(masir_data, 'chest-xray-pneumonia-project', 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# -------------------- 4. تعریف focal loss --------------------
def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

# -------------------- 5. بارگذاری مدل پایه ResNet50 --------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

# -------------------- 6. ساخت مدل نهایی --------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# -------------------- 7. کامپایل مدل --------------------
model.compile(
    optimizer='adam',
    loss=focal_loss(),
    metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
)

# -------------------- 8. وزن‌دهی به کلاس‌ها --------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(x_train.classes),
    y=x_train.classes
)
class_weights_dict = dict(enumerate(class_weights))

# -------------------- 9. Callbacks --------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# -------------------- 10. آموزش مدل --------------------
model.fit(
    x_train,
    validation_data=x_val,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr]
)

# -------------------- 11. ارزیابی نهایی --------------------
model.evaluate(x_test)

# -------------------- 12. پیش‌بینی --------------------
x_test.reset()
pred = model.predict(x_test)
y_true_labels = x_test.classes


# محاسبه بهترین threshold
precision_vals, recall_vals, thresholds = precision_recall_curve(y_true_labels, pred)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold:.2f}")

# اعمال threshold
y_pred_label = (pred > best_threshold).astype(int).flatten()

# -------------------- 13. گزارش عملکرد --------------------
print(classification_report(y_true_labels, y_pred_label, target_names=['Normal', 'Pneumonia']))

# -------------------- 14. confusion matrix --------------------
cm = confusion_matrix(y_true_labels, y_pred_label)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

model.save("my_model.h5")

