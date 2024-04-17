import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_hybrid_model import *
import concurrent.futures

# Function to run hybrid model and display result
def run_hybrid_model(image_path):
    IMAGE_PATH = image_path
    image = cv2.imread(IMAGE_PATH)
    result = inference_detector(model_rtmdet, image)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rtmdet_thread = executor.submit(get_predictions, model_rtmdet, IMAGE_PATH)
        predictions_rtmdet = rtmdet_thread.result() 
        yolo_thread = executor.submit(get_predictions, model_yolo, IMAGE_PATH)
        predictions_yolo = yolo_thread.result()
		
    # Placeholder variables for final predictions    
    final_boxes = []
    final_confidences = []
    final_classes = []
    final_boxes, final_confidences, final_classes = vote(predictions_rtmdet=predictions_rtmdet, predictions_yolo=predictions_yolo)
    image_path = displayHybridPrediction(final_boxes, final_confidences, final_classes, IMAGE_PATH=IMAGE_PATH)
    return image_path

# image uploader function
def imageUploader():
	label.config(image="")
	label.image = ""
	fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
	path = tk.filedialog.askopenfilename(filetypes=fileTypes)

	# if file is selected
	if len(path):	
		loading_label = tk.Label(app, text="Loading....", fg="blue", bg="#CDF5FD", font=("Arial", 16, "bold"))
		loading_label.place(x=250, y=300)  # Adjust the position as needed
		
		app.update()  # Update the GUI to immediately display the loading label
		
		img = Image.open(path)
		path_ = run_hybrid_model(path)
		img = Image.open(path_)
		img = img.resize((512, 512))
		
		pic = ImageTk.PhotoImage(img)

		# re-sizing the app window in order to fit picture
		# and buttom
		app.geometry("640x640")
		label.config(image=pic)
		label.image = pic
		loading_label.destroy()

	# if no file is selected, then we are displaying below message
	else:
		print("No file is Choosen !! Please choose a file.")

# Function to handle video uploading
def videoUploader():
    file_types = [("Video files", "*.mp4;*.avi;*.mov")]
    path = filedialog.askopenfilename(filetypes=file_types)

    if path:  # If a file is selected
        loading_label = tk.Label(app, text="Loading....", fg="blue", bg="#CDF5FD", font=("Arial", 16, "bold"))
        loading_label.place(x=250, y=300)  # Adjust the position as needed
        
        app.update()  # Update the GUI to immediately display the loading label

        frame = Image.fromarray(frame)
        frame = frame.resize((512, 512))
        photo = ImageTk.PhotoImage(frame)

        # Display the first frame
        label.config(image=photo)
        label.image = photo

        loading_label.destroy()

    else:  # If no file is selected
        print("No file is chosen! Please choose a video file.")

# Close the application properly
def closeApp():
    app.destroy()
    sys.exit()
    
# Main method
if __name__ == "__main__":
    # Defining tkinter object
    app = tk.Tk()

    # Setting title and basic size to our App
    app.title("AquareGuardian")
    app.geometry("640x640")

    # Adding background color to our widgets
    app.option_add("*Label*Background", "#CDF5FD")
    app.option_add("*Button*Background", "lightblue")
    app['background']='#CDF5FD'

    label = tk.Label(app, bd=5)
    label.pack(pady=10)

    # Defining the image upload button
    image_upload_button = tk.Button(app, text="Locate Image", command=imageUploader, pady=6, fg="blue", bg="#86B6F6", 
                                    activeforeground="blue", activebackground="#86B6F6", font=("Arial", 13))
    image_upload_button.place(x=150, y=560)

    # Close button event handler
    exit_button = tk.Button(app, text='Exit', command=closeApp, fg="red", bg="pink", activeforeground="red", 
                            activebackground="pink", pady=6, padx=30, font=("Arial", 13))
    exit_button.place(x=410, y=560)

    app.mainloop()
