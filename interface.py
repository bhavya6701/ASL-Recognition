import customtkinter
import cv2
import numpy as np
from PIL import ImageDraw, Image
from customtkinter import CTkLabel, CTkCanvas, CTkButton, CTkFrame
from keras import models

classifier_set = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K',
                  12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'space', 21: 'T', 22: 'U',
                  23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}


def predict_digit(img):
    # convert to grayscale, and resize to 28x28 pixel
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (28, 28))

    # reshaping to support our model input and normalizing
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    # predicting the class
    res = model.predict([img_array])[0]
    return np.argmax(res), max(res)


# load the trained model from pickle file
model = models.load_model("asl_recognition_model.h5")


# class MainWindow:
#
#     def __init__(self, window):
#         self.x = self.y = 0
#
#         window.title("BAD AI")
#         frame = CTkFrame(window, border_width=2, border_color="cyan")
#         frame.grid(row=0, column=0, sticky="NESW")
#         frame.grid_rowconfigure(0, weight=1)
#         frame.grid_columnconfigure(0, weight=1)
#         frame.pack(fill='both', expand=1)
#         frame.place(in_=window, anchor="center", relx=.5, rely=.5)
#
#         self.heading = CTkLabel(frame, text="Digit & Alphabet Recognition", font=("Gabriola", 40))
#         self.canvas = CTkCanvas(frame, bg="white", height=400, width=400, cursor="cross")
#         self.note = CTkLabel(frame, text="Draw Digit/Alphabet to recognize!", font=("Corbel Light", 30))
#         self.rec_btn = CTkButton(frame, text="Recognize", font=("Corbel Light", 20), command=self.classify_handwriting)
#         self.clear_btn = CTkButton(frame, text="Clear", font=("Corbel Light", 20), command=self.clear_all)
#         self.predict = CTkLabel(frame, text="", font=("Corbel Light", 45))
#         self.accurate = CTkLabel(frame, text="", font=("Corbel Light", 45))
#
#         self.heading.grid(row=0, column=0, rowspan=2, columnspan=5, padx=20, pady=30)
#         self.note.grid(row=2, column=0, rowspan=1, columnspan=2, padx=20)
#         self.canvas.grid(row=3, column=0, rowspan=3, columnspan=2, padx=20, pady=20)
#         self.rec_btn.grid(row=2, column=3, padx=20)
#         self.clear_btn.grid(row=2, column=4, padx=20)
#         self.predict.grid(row=3, column=3, columnspan=2, padx=20)
#         self.accurate.grid(row=4, column=3, columnspan=2, padx=20)
#
#         self.canvas.bind("<B1-Motion>", self.draw_lines)
#         # Get Image from canvas
#         self.img = Image.new('RGB', (400, 400), (0, 0, 0))
#         self.img_draw = ImageDraw.Draw(self.img)
#
#     def clear_all(self):
#         self.canvas.delete("all")
#         self.img = Image.new('RGB', (400, 400), (0, 0, 0))
#         self.img_draw = ImageDraw.Draw(self.img)
#
#     def classify_handwriting(self):
#         prediction, acc = predict_digit(self.img)
#         self.predict.configure(text="Predicted Value: " + str(by_merge_map[prediction]) + "!")
#         self.accurate.configure(text="Accuracy: " + str(int(acc * 100)) + "%")
#
#     def draw_lines(self, event):
#         self.x = event.x
#         self.y = event.y
#         r = 15
#         self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
#         self.img_draw.ellipse((self.x - r, self.y - r, self.x + r, self.y + r), fill='white')
#
#
# def main():
#     # Run the application
#     customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
#     customtkinter.set_default_color_theme("blue")
#     tk_win = customtkinter.CTk()
#     win = MainWindow(tk_win)
#
#     width = 1366
#     height = 768
#     x = (tk_win.winfo_screenwidth() / 2) - (width / 2)
#     y = (tk_win.winfo_screenheight() / 2) - (height / 2)
#     tk_win.geometry('%dx%d+%d+%d' % (width, height, x, y))
#     tk_win.mainloop()


if __name__ == '__main__':
    main()
