import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

# Load the trained model from the folder
model_path = r'C:\Users\Dell\Desktop\NNFLC project\codes\best_model.h5'
model = load_model(model_path)

# Class indices (automatic mapping based on your training)
class_indices = {
    'Aphids': 0, 
    'Bacterial_Blight': 1, 
    'Leaf_Curl_Disease': 2, 
    'Powdery_Mildew': 3, 
    'Target_spot': 4, 
    'boll_rot': 5, 
    'healthy': 6, 
    'wilt': 7
}

# Control measures for diseases
control_measures = {
    "Aphids": (
        "Cultural Control: Use crop rotation, maintain healthy plants, and remove weeds.\n"
        "Biological Control: Introduce natural predators such as lady beetles, lacewings, and parasitic wasps.\n"
        "Chemical Control: Use insecticidal soaps or systemic insecticides for severe infestations. Rotate insecticides to prevent resistance."
    ),
    "Bacterial_Blight": (
        "Resistant Varieties: Plant cotton varieties resistant to bacterial blight.\n"
        "Sanitation: Remove and destroy infected plant debris from the field.\n"
        "Crop Rotation: Avoid continuous cotton planting in the same field.\n"
        "Seed Treatment: Use certified disease-free seeds or treat seeds with appropriate fungicides."
    ),
    "boll_rot": (
        "Cultural Practices: Improve field drainage to reduce moisture. Avoid dense planting to ensure good airflow.\n"
        "Sanitation: Remove infected bolls from the field.\n"
        "Chemical Control: Apply fungicides during wet conditions to reduce boll rot risks."
    ),
    "Leaf_Curl_Disease": (
        "Resistant Varieties: Use leaf curl virus-resistant cotton varieties.\n"
        "Vector Control: Control the whitefly population, which transmits the virus, using insecticides or biological methods.\n"
        "Cultural Practices: Practice good field sanitation by removing infected plants."
    ),
    "Powdery_Mildew": (
        "Fungicide Application: Use sulfur-based or other fungicides to manage outbreaks.\n"
        "Resistant Varieties: Plant varieties that show resistance to powdery mildew.\n"
        "Cultural Control: Increase air circulation by spacing plants properly."
    ),
    "Target_spot": (
        "Use Fungicides: Apply fungicides at the early stages of infection.\n"
        "Crop Rotation: Rotate with non-host crops to reduce disease carryover.\n"
        "Remove Infected Debris: Clear out plant residues after harvesting."
    ),
    "wilt": (
        "Resistant Varieties: Plant cotton varieties resistant to wilt.\n"
        "Crop Rotation: Rotate crops to break the disease cycle.\n"
        "Soil Treatment: Use soil fumigants or solarization techniques to reduce pathogen levels."
    ),
}

def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Rescale the image
    return img_array

def show_popup(disease_name, measures, bg_color):
    popup = tk.Toplevel()
    popup.title("Prediction Result")
    popup.configure(bg=bg_color)

    # Create a label with the message
    label_color = "black" if bg_color in ["orange", "green"] else "white"
    message = f"{disease_name} detected!\n\n{measures}"
    label = tk.Label(popup, text=message, fg=label_color, bg=bg_color, font=("Arial", 12), wraplength=350)  # Multiline support
    label.pack(padx=20, pady=20)

    # Set window size to fit content with padding
    popup.geometry("400x300")  # Increased size for better fitting
    popup.resizable(False, False)  # Prevent resizing

    # Center the popup window
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
    y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
    popup.geometry(f"+{x}+{y}")

    # Add a close button
    close_button = tk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=10)

def predict_disease(model, img_path, class_indices):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)

    # Print prediction probabilities for debugging
    print("Prediction probabilities:", prediction)

    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    disease_name = list(class_indices.keys())[predicted_class]  # Get the disease name

    print("Predicted class index:", predicted_class)
    print("Predicted disease name:", disease_name)
    print(f"Confidence level: {confidence:.2f}")

    # Set a higher threshold for predictions
    confidence_threshold = 0.7  # Increase threshold to reduce false positives
    if confidence < confidence_threshold:
        show_popup("Unable to predict. Please try a different image.", "red")
    elif confidence < 0.85:  # Adjust this threshold as needed
        show_popup("Prediction is uncertain. Please try a different image.", "orange")
    elif disease_name == "healthy":
        show_popup("Healthy", "", "green")  # Pass an empty string for measures
    else:
        measures = control_measures[disease_name]
        show_popup(disease_name, measures, "orange")

def upload_image():
    # Open file dialog to select an image
    img_path = filedialog.askopenfilename(title="Select an Image", 
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        predict_disease(model, img_path, class_indices)

# Create a Tkinter window for image upload
root = tk.Tk()
root.title("Cotton Disease Prediction")
root.geometry("300x150")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=50)

root.mainloop()
