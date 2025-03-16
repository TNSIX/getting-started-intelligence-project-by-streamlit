import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore
import matplotlib.pyplot as plt
import os
import streamlit as st
import numpy as np
from tensorflow.keras.callbacks import Callback  # type: ignore
from PIL import Image
import pickle

# ดาวน์โหลด dataset
path = kagglehub.dataset_download("muratkokludataset/pistachio-image-dataset")
dataset_path = os.path.join(path, "Pistachio_Image_Dataset", "Pistachio_Image_Dataset")

# ตั้งค่าพารามิเตอร์
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
NUM_CLASSES = 2

# สร้าง ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# โหลดข้อมูลฝึก
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# โหลดข้อมูล validation
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Custom Callback เพื่ออัปเดต Streamlit
class StreamlitCallback(Callback):
    def __init__(self):
        super().__init__()
        self.progress_text = st.empty()

    def on_epoch_begin(self, epoch, logs=None):
        self.progress_text.write(f"Starting Epoch {epoch + 1}/{self.params['epochs']}...")

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy', 0)
        val_accuracy = logs.get('val_accuracy', 0)
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        self.progress_text.write(
            f"Epoch {epoch + 1}/{self.params['epochs']} completed - "
            f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}"
        )

# ฟังก์ชันสร้างและฝึกโมเดล
def create_and_train_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        epochs=20,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        callbacks=[StreamlitCallback()]
    )
    
    return model, history

# ฟังก์ชันทำนายรูปภาพ
def predict_image(model, img):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_indices = train_generator.class_indices
    class_name = list(class_indices.keys())[np.argmax(prediction)]
    confidence = np.max(prediction)
    return class_name, confidence

# ฟังก์ชันแสดงกราฟจาก history
def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.title("Pistachio Classification Demo with CNN")

# เพิ่มรูปภาพ 5 รูปในแนวนอน
st.write("<br>", unsafe_allow_html=True)
cols = st.columns(5)
kirmizi_path = os.path.join(dataset_path, "Kirmizi_Pistachio")
siirt_path = os.path.join(dataset_path, "Siirt_Pistachio")
kirmizi_images = [os.path.join(kirmizi_path, f) for f in os.listdir(kirmizi_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:3]
siirt_images = [os.path.join(siirt_path, f) for f in os.listdir(siirt_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:2]
sample_images = kirmizi_images + siirt_images

for i, col in enumerate(cols):
    if i < len(sample_images):
        img = Image.open(sample_images[i])
        col.image(img, caption=f"Sample {i+1}", use_container_width=True)

# ข้อมูล Dataset Information
st.write("<br>", unsafe_allow_html=True)
st.write("##### Dataset Information")
st.write("Dataset Path:", dataset_path)
st.write("Number of training samples:", train_generator.samples)
st.write("Number of validation samples:", validation_generator.samples)
st.write("Class indices:", train_generator.class_indices)

st.divider()
st.write("##### Upload an image to predict or select a sample")
# อัปโหลดรูปภาพเพื่อทำนาย
uploaded_file = st.file_uploader("Upload a pistachio image", type=["jpg", ".jpeg", ".png"])

# เพิ่มตัวเลือกภาพตัวอย่าง
sample_options = ["None"] + [f"Sample {i+1}" for i in range(len(sample_images))]
selected_sample = st.selectbox("Or select a sample image to predict:", sample_options)

# แสดงกราฟจากโฟลเดอร์ images
st.write("<br>", unsafe_allow_html=True)
st.write("##### Sample Model Performance Graphs")
accuracy_graph_path = "images/model_accuracy.png"
loss_graph_path = "images/model_loss.png"

# สร้าง 2 คอลัมน์
cols = st.columns(2)

# คอลัมน์แรก: Accuracy Graph
with cols[0]:
    if os.path.exists(accuracy_graph_path):
        st.image(accuracy_graph_path, caption="Sample Accuracy Graph", use_container_width=True)
    else:
        st.write("Accuracy graph not found in 'images' folder.")

# คอลัมน์ที่สอง: Loss Graph
with cols[1]:
    if os.path.exists(loss_graph_path):
        st.image(loss_graph_path, caption="Sample Loss Graph", use_container_width=True)
    else:
        st.write("Loss graph not found in 'images' folder.")

# ตรวจสอบว่ามีโมเดลที่ฝึกแล้วหรือไม่
if os.path.exists("pistachio_model.h5"):
    model = tf.keras.models.load_model("pistachio_model.h5")
    
    # กรณีอัปโหลดรูปภาพ
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        class_name, confidence = predict_image(model, img)
        st.write(f"Prediction: **{class_name}**")
        st.write(f"Confidence: **{confidence:.2%}**")
    
    # กรณีเลือกภาพตัวอย่าง
    elif selected_sample != "None":
        sample_idx = int(selected_sample.split(" ")[1]) - 1
        img = Image.open(sample_images[sample_idx])
        st.image(img, caption=f"Selected {selected_sample}", use_container_width=True)
        class_name, confidence = predict_image(model, img)
        st.write(f"Prediction: **{class_name}**")
        st.write(f"Confidence: **{confidence:.2%}**")
    
else:
    st.write("Please train the model first by clicking 'Train Model & Save' below!")

st.write("<br>", unsafe_allow_html=True)
st.write("##### Train or view model performance")
# ปุ่มสำหรับฝึกโมเดล
if st.button("Train Model & Save"):
    with st.spinner("Training model..."):
        model, history = create_and_train_model()
        
        # บันทึกโมเดล
        model.save("pistachio_model.h5")
        
        # บันทึก history
        with open("training_history.pkl", "wb") as f:
            pickle.dump(history.history, f)
        
        # แสดงกราฟทันทีหลังฝึก
        plot_history(history)

# ปุ่มสำหรับแสดงกราฟจาก history ที่บันทึกไว้
if st.button("View Training Graphs"):
    if os.path.exists("training_history.pkl"):
        with open("training_history.pkl", "rb") as f:
            history_data = pickle.load(f)
        class History:
            def __init__(self, data):
                self.history = data
        history = History(history_data)
        plot_history(history)
    else:
        st.write("No training history available. Please train the model first.")