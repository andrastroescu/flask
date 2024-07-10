#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
import cv2
import numpy as np
import traceback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from flask_cors import CORS
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify

import matplotlib.pyplot as plt
from mysql.connector import Error

import mysql.connector


# In[84]:


def load_and_train_model():
    global model
    model = Sequential()

    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    data = tf.keras.utils.image_dataset_from_directory('data')

    data_iterator = data.as_numpy_iterator()

    batch = data_iterator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

    data = data.map(lambda x,y: (x/255, y))

    data.as_numpy_iterator().next()

    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)

    train_size

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    train

    model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])

    return model


# In[85]:


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


# In[86]:


# Function to insert a new class into the database
def insert_class(cursor, class_name):
    try:
        cursor.execute(f"""
            INSERT INTO CLASS (NAME)
            SELECT * FROM (SELECT '{class_name}') AS tmp
            WHERE NOT EXISTS (
                SELECT NAME FROM CLASS WHERE NAME = '{class_name}'
            )
        """)
        if cursor.rowcount > 0:
            print("Class inserted successfully")
        else:
            print("Class already exists")
        return cursor.lastrowid
    except Error as e:
        print(f"Error inserting class: {e}")


# In[113]:


# Function to insert a new image into the database
def insert_image(cursor, file_name, file_path):
    try:
        cursor.execute(f"INSERT INTO IMAGE (FILE_NAME, FILE_PATH) VALUES ('{file_name}', '{file_path}')")
        
        return cursor.lastrowid
    except Error as e:
        print(f"Error inserting image: {e}")
        return None


# In[88]:


# Function to insert a new classification into the database
def insert_classification(cursor, image_id, class_id):
    try:
        cursor.execute(f"INSERT INTO CLASSIFICATION (IMAGE_ID, CLASS_ID) VALUES ({image_id}, {class_id})")
    except Error as e:
        print(f"Error inserting classification: {e}")


# In[89]:


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized_img = tf.image.resize(img, (256, 256))
    return resized_img


# In[97]:


# Function to preprocess and classify an image
def classify_image(model, image_path, cursor):
    try:
        # Read and preprocess the image
        img = preprocess_image(image_path)
        
        # Make predictions
        yhat = model.predict(np.expand_dims(img / 255, 0))
        predicted_class = np.argmax(yhat) + 1

        # Get class names
        class_names = {
            1: 'Aluminium_Cans',
            2: 'Cartons',
            3: 'Coffee_Cups',
            4: 'Newspapers'
            # Add more class indices and names as needed
        }

        predicted_class_name = class_names.get(predicted_class, f'UnknownClass_{predicted_class}')

        # Insert results into the database
        insert_class(cursor, predicted_class_name)
        image_id = insert_image(cursor, os.path.basename(image_path), image_path)
        print(f"image id is {image_id}")
        insert_classification(cursor, image_id, predicted_class)

        return predicted_class_name

    except Exception as e:
        traceback.print_exc()
        print(f"Error during classification: {e}")
        return None


# In[98]:

# Function to train the model
def train_model():
    # ... (your existing training code)
    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Function to evaluate the model
def evaluate_model(test_data):
    # ... (your existing evaluation code)
    from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    
    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    
    print(pre.result(), re.result(), acc.result())
    


# In[99]:


def query_class_characteristics(cursor, class_name):
    try:
        # SQL query to retrieve class characteristics
        query = """
        SELECT m.name AS material, c.name AS component, ch.percentage, 
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


# In[100]:


# Extra test for JSON 
# Get a database connection
#connection = create_connection()

#if connection:
#    try:
        # Your existing code here to initialize the cursor
        #cursor = connection.cursor()
        
       #class_characteristics = query_class_characteristics(cursor, "NEWSPAPER")
       # if class_characteristics:
            #print("Class Characteristics:")
           # for char in class_characteristics:
             #   print(char)
       # else:
            #print("No characteristics found for the class.")
            
    #finally:
       # cursor.close()
       # connection.close()
       # print("Connection closed")
#else:
   # print("Failed to establish a connection to the database.")


# In[101]:


# Function to display an image
def show_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        plt.imshow(img)
        plt.show()
    else:
        print(f"Error: Unable to read image from file '{file_path}'.")


# In[102]:





# In[107]:


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


# In[ ]:


app = Flask(__name__)
#app.config['DEBUG'] = True  # Enable debug mode
CORS(app)
model = None
connection = None


# In[ ]:


# Load the trained model and create database connection on application startup
def startup():
    global model
    global connection
    model = load_and_train_model()
    connection = create_connection()


# In[ ]:


# Define a route for image classification
@app.route('/classify_image', methods=['POST'])
def classify_image_route():
    try:
        file = request.files['image']
        file_path = secure_filename(file.filename)  # Use the filename of the uploaded file
        file.save(file_path)

        global model
        global connection

        create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Example usage with a variable containing the file path
                result = classify_image(model, file_path, cursor)
                
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
# In[ ]:


# Run Flask app

if __name__ == '__main__':
    startup()
    app.run(host='0.0.0.0', port=5000)

