import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import argparse

# example：python camera_app.py --output_dir calibration_checkerboard

parser = argparse.ArgumentParser(description="Camera App")
parser.add_argument("--output_dir", type=str, default="photos", help="Directory to save photos")
args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

photo_count = len(os.listdir(output_dir))

def update_frame():
    ret, frame = cap.read()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
    lbl.after(10, update_frame)

def take_photo():
    global photo_count
    ret, frame = cap.read()
    if ret:
        photo_path = os.path.join(output_dir, f"photo_{photo_count}.jpg")
        cv2.imwrite(photo_path, frame)
        photo_count += 1
    else:
        messagebox.showerror("Error", "Cannot take photo")

cap = cv2.VideoCapture(1) # 调整系统摄像头编号
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot open camera")
    exit()

root = tk.Tk()
root.title("Camera App")

lbl = tk.Label(root)
lbl.pack()

btn = tk.Button(root, text="Take Photo", command=take_photo)
btn.pack(pady=20)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()