import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from flask import Flask, request, jsonify
from flask_cors import CORS

import pathlib

from mysql.connector import Error

import mysql.connector

from werkzeug.utils import secure_filename
import traceback
import os


def load_and_train_model():
    global model
    global class_names
    global history
    global epochs
    
    data_dir = "data"
    data_dir = pathlib.Path(data_dir)

    batch_size = 32
    img_height = 180
    img_width = 180

    #Training 80%
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
     
    #Validation 20%
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
     
    class_names = train_ds.class_names
    print(class_names)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)
    
       #Augmentation/retraining of model on existing data samples
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs=30
    
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )
    
    # Plot results
    plotResults()
    
    return model

# Plotting results
def plotResults():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show()


# Function to create a connection to MySQL database
def create_connection():
    global connection
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='pass123db',
            database='andradb'
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None

# Function to insert a new class into the database
def insert_class(cursor, class_name):
    try:
        # Check if the class already exists in the table
        cursor.execute("SELECT ID FROM CLASS WHERE NAME = %s", (class_name,))
        existing_class_id = cursor.fetchone()
        
        if existing_class_id:
            # If the class already exists, return its ID
            print("Class already exists")
            return existing_class_id[0]
        else:
            # If the class doesn't exist, insert it into the table
            cursor.execute("INSERT INTO CLASS (NAME) VALUES (%s)", (class_name,))
            print("Class inserted successfully")
            return cursor.lastrowid
    except Error as e:
        print(f"Error inserting class: {e}")
        return None


# Function to insert a new image into the database
def insert_image(cursor, file_name, file_path, user_id):
    try:
        upload_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
        cursor.execute(f"INSERT INTO IMAGE (FILE_NAME, FILE_PATH, UPLOAD_TIMESTAMP, USER_ID) VALUES ('{file_name}', '{file_path}', '{upload_timestamp}', '{user_id}')")
        
        return cursor.lastrowid
    except Error as e:
        print(f"Error inserting image: {e}")
        return None
        
# Function to insert a new classification into the database
def insert_classification(cursor, image_id, class_id, user_id, score, class_name):
    try:
        upload_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
        cursor.execute(f"INSERT INTO CLASSIFICATION (IMAGE_ID, CLASS_ID, USER_ID, TIMESTAMP, SCORE, CLASS_NAME) VALUES ({image_id}, {class_id}, '{user_id}', '{upload_timestamp}', '{score}', '{class_name}')")
    except Error as e:
        print(f"Error inserting classification: {e}")

# Reading and pre-processing an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized_img = tf.image.resize(img, (180, 180))
    return resized_img
    
# Function to preprocess and classify an image
def classify_image(model, image_data, cursor, image_path, user_id):
    try:
        # Convert the image data to a NumPy array
        print(f"image data is: {image_data}")
        img_array = tf.keras.preprocessing.image.img_to_array(image_data)
        
        # Resize the image
        img_array = tf.image.resize(img_array, (180, 180))
        
        # Expand dimensions to create a batch
        img_array = tf.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class_name = class_names[np.argmax(score)]
        predicted_score = 100 * np.max(score)

        # Insert results into the database
        class_id = insert_class(cursor, predicted_class_name)
        image_id = insert_image(cursor, os.path.basename(image_path), image_path, user_id)
        print(f"imageee id is {image_id}")
        print(f"userrrrr id is {user_id}")
        insert_classification(cursor, image_id, class_id, user_id, predicted_score, predicted_class_name)

        return predicted_class_name

    except Exception as e:
        traceback.print_exc()
        print(f"Error during classification: {e}")
        return None


def query_class_characteristics(cursor, class_name):
    try:
        # SQL query to retrieve class characteristics
        query = """
        SELECT c.name AS component, m.name AS material, ch.percentage, 
        cc.additional_info AS additional_info, cc.spec as specification
        FROM class cl
        INNER JOIN class_characteristic cc ON cc.class_id = cl.id
        INNER JOIN characteristic ch ON cc.characteristic_id = ch.id
        INNER JOIN material m ON ch.material_id = m.id
        INNER JOIN component c ON ch.component_id = c.id
        WHERE cl.name = %s
        """

        # Execute the query with the class name as parameter
        cursor.execute(query, (class_name,))
        
        # Fetch all the results
        rows = cursor.fetchall()
        
        # Format the retrieved data into JSON format
        class_data = []
        for row in rows:
            material = row[0]
            component = row[1]
            percentage = row[2]
            additional_info = row[3]
            specification = row[4]
            class_data.append({"material": material, "component": component, "percentage": percentage, "additional_info": additional_info, "specification": specification})

        return class_data
    
    except Error as e:
        print(f"Error querying class characteristics: {e}")
        return None
        

def select_all_from_class_table(cursor):
    try:
        # Query to select all columns from the class table
        cursor.execute("SELECT * FROM CLASS")
        
        # Fetch all rows from the result
        rows = cursor.fetchall()
        
        # Display the rows
        for row in rows:
            print(row)
    
    except Error as e:
        print(f"Error selecting from CLASS table: {e}")
        

app = Flask(__name__)
#app.config['DEBUG'] = True  # Enable debug mode
CORS(app)
model = None
connection = None

# Load the trained model and create database connection on application startup
def startup():
    global model
    global connection
    model = load_and_train_model()
    connection = create_connection()

# Define a route for image classification
@app.route('/classify_image', methods=['POST'])
def classify_image_route():
    try:
        file = request.files['image']
        file_path = secure_filename(file.filename)  # Use the filename of the uploaded file
        file.save(file_path)
        
        user_id = request.form.get('user_id')

        global model
        global connection

        create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                image_data = tf.keras.utils.load_img(file_path, target_size=(256, 256))

                # Example usage with a variable containing the file path
                result = classify_image(model, image_data, cursor, file_path, user_id)
                
                if result:
                    print(f"Predicted class: {result}")

                    # Query for class characteristics
                    class_characteristics = query_class_characteristics(cursor, result)
                    if class_characteristics:
                        print("Class Characteristics:")
                        for char in class_characteristics:
                            print(char)
                    else:
                        print("No characteristics found for the class.")
                    
                    # Return all information from the database
                    all_data = {
                        'prediction': result,
                        'class_characteristics': class_characteristics
                    }
                    print(f"All data: {all_data}")
                    return jsonify(all_data)
            finally:
                connection.commit()
                cursor.close()
                connection.close()
                print("Connection closed")

    except Exception as e:
        # Print out any exceptions that occur during execution
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/test', methods=['GET'])
def hello():
    print('working')
    return jsonify({'test': 'test'})

# Run Flask app
if __name__ == '__main__':
    startup()
    app.run(host='0.0.0.0', port=5000)