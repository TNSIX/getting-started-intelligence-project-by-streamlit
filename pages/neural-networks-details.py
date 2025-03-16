import streamlit as st

st.title("Neural Network Details")

# Data Preparation
st.write("<br>", unsafe_allow_html=True)
st.write("<h3>Data Preparation <span>—</span> การเตรียมข้อมูล</h3>", unsafe_allow_html=True)
st.markdown("- Referenced from https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset")

st.divider()

# Theory of Algorithms
st.write("<h3>Theorem of Algorithms <span>—</span> ทฤษฎีของอัลกอริทึมที่พัฒนา</h3>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)
st.markdown(
    """
    ##### 🔸Convolutional Neural Network
    """
)
cnn_image = "https://miro.medium.com/v2/resize:fit:640/format:webp/1*kkyW7BR5FZJq4_oBTx3OPQ.png"
st.markdown(
    f"""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 20px;'>
        <img src='{cnn_image}' style='height: 100%; object-fit: cover;'>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    Convolutional neural networks are distinguished from other neural networks by their superior performance with image
    , speech or audio signal inputs. They have four main types of layers, which are:
    - Convolutional layer (with activation function)
    - Pooling layer
    - Flatten layer
    - Fully connected layer (Artificial Neural Network)
    """
)
st.write("<br>", unsafe_allow_html=True)
st.markdown(
    """
    ##### Convolutional Layer (with activation function)
    """
)
st.markdown("""
1. Number of Filters: Filters are like tools that find special features in an image, such as edges or lines. If you use 3 different filters, you get 3 feature maps, making the output 3 layers deep.
2. Stride: This is how far the filter moves across the image, like 1 or 2 pixels at a time. A bigger stride means a smaller output.
3. Zero-padding: If the filter doesn’t fit the image perfectly, zeros are added around the edges. There are 3 types:
    - Valid padding: No zeros added. If the size doesn’t match, the extra part gets cut off.
    - Same padding: Zeros are added so the output stays the same size as the input.
    - Full padding: More zeros are added, making the output bigger.
            
4. Activation Function: This is a function that decides whether a neuron should be activated or not. The most common one is the ReLU function, which replaces negative values with zero.

""")
st.write("<br>", unsafe_allow_html=True)
convolutional_layer_image = "https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/ed/92/iclh-diagram-convolutional-neural-networks.png"
st.markdown(
    f"""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 20px;'>
        <img src='{convolutional_layer_image}' style='height: 100%; object-fit: cover;'>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("<br>", unsafe_allow_html=True)
st.markdown(
    """
    ##### Pooling Layer
    """
)
st.markdown("""
Pooling layers, also known as downsampling, conducts dimensionality reduction, reducing the number of parameters in the input. There are two main types of pooling:
- Max pooling: As the filter moves across the input, it selects the pixel with the maximum value to send to the output array. As an aside, this approach tends to be used more often compared to average pooling.
- Average pooling: As the filter moves across the input, it calculates the average value within the receptive field to send to the output array.

While a lot of information is lost in the pooling layer, it also has a number of benefits to the CNN. They help to reduce complexity, improve efficiency, and limit risk of overfitting. 
""")
st.write("<br>", unsafe_allow_html=True)
max_pooling_layer_image = "https://miro.medium.com/max/700/1*FHPUtGrVP6fRmVHDn3A7Rw.png"
st.markdown(
    f"""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 20px;'>
        <img src='{max_pooling_layer_image}' style='height: 100%; object-fit: cover;'>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("<br>", unsafe_allow_html=True)
st.markdown(
    """
    ##### Flatten Layer
    """
)
st.markdown("""
The Flatten Layer is a step in a CNN where you take all the data from the previous layers "flatten" it into a single, long line of numbers (a 1D array). It’s like unrolling a rolled-up mat into a straight strip.
How it works:
- Imagine you have a 2x2x3 feature map (2 pixels wide, 2 pixels tall, 3 filters deep). That’s 12 numbers total (2 × 2 × 3).
- The Flatten Layer takes those 12 numbers and lines them up into a single row, like: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].
- Now it’s ready to feed into the next layer!
""")
st.write("<br>", unsafe_allow_html=True)
flatten_layer_image = "https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/73_blog_image_1.png"
st.markdown(
    f"""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 20px;'>
        <img src='{flatten_layer_image}' style='height: 100%; object-fit: cover;'>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("<br>", unsafe_allow_html=True)
st.markdown(
    """
    ##### Fully-Connected Layer (Artificial Neural Network)
    """
)
st.markdown("""
A Fully Connected Layer is a part of a Convolutional Neural Network (CNN), usually found near the end of the model. It acts like the "brain" that gathers all the information from previous layers (like convolutional or pooling layers) and makes a final decision or prediction, such as identifying whether an image is of a cat or a dog.

- Connects everything: In this layer, every "data point" (neuron) from the previous layer is connected to every point in this layer (like a web of fully linked nerves), hence the name "fully connected."
- Turns data into answers: It takes the processed data (e.g., image features) and calculates it with weights and biases to produce an output, such as the probability of "what this is."

Example: If you have an image and the model needs to decide if it’s a "cat" or a "dog," the fully connected layer looks at all the data and might say, "90% chance it’s a cat."
""")
st.write("<br>", unsafe_allow_html=True)
fully_connected_layer_image = "https://www.researchgate.net/publication/361040463/figure/fig3/AS:1162656631271425@1654210350919/Fully-connected-layer.ppm"
st.markdown(
    f"""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 20px;'>
        <img src='{fully_connected_layer_image}' style='height: 400px; object-fit: cover;'>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
# Model Training
st.write("<h3>Model Development <span>—</span> การพัฒนาโมเดล</h3>", unsafe_allow_html=True)
st.write("1. Import libraries and install kagglehub used to download dataset")
st.code("""
# ติดตั้ง kagglehub
!pip install kagglehub
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
""")

st.write("2. Dowload dataset from kaggle")
st.code("""
# ดาวน์โหลด dataset
path = kagglehub.dataset_download("muratkokludataset/pistachio-image-dataset")
print("Dataset downloaded to:", path)
""")
st.write("3. Configure path to pistacio image folders")
st.code("""
# กำหนด path ไปยังโฟลเดอร์รูปภาพ
dataset_path = os.path.join(path, "Pistachio_Image_Dataset", "Pistachio_Image_Dataset")
print("Image folders:", os.listdir(dataset_path))
""")
st.write("4. Configure parameters for the model")
st.code("""
# ตั้งค่าพารามิเตอร์
IMG_HEIGHT = 128  # ปรับขนาดภาพ
IMG_WIDTH = 128
BATCH_SIZE = 16
NUM_CLASSES = 2  # 2 คลาส: Kirmizi_Pistachio และ Siirt_Pistachio
""")
st.write("5. Create ImageDataGenerator")
st.code("""
# สร้าง ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
""")
st.write("6. Load training data")
st.code("""
# โหลดข้อมูลฝึก
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
""")
st.write("7. Load validation data")
st.code("""
# โหลดข้อมูล validation
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
""")
st.write("8. Check all data loaded")
st.code("""
# ตรวจสอบข้อมูล
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)
print("Class indices:", train_generator.class_indices)
""")
st.write("9. Create CNN model")
st.code("""
# สร้างโมเดล CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')  # 2 คลาส
])
model.summary()
""")
st.write("10. Compile the model")
st.code("""
# Compile โมเดล
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
""")
st.write("11. Train the model")
st.code("""
# ฝึกโมเดล
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // BATCH_SIZE)
)
""")
st.write("12. Plot the training history")
st.code("""
# แสดงผลลัพธ์
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
""")