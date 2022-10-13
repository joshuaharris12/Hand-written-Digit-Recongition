from os import stat
from symbol import arglist
from time import sleep
import tkinter as Tkinter
import utils
import nn
from PIL import Image, ImageTk
from threading import Thread

# Information that changes

# Global
# X_train, Y_train, X_val, Y_val = utils.load_training_dataset('mnist_train.csv')
# W1, b1, W2, b2 = None, None, None, None



# root = Tkinter.Tk()
# root.geometry('500x400')

# status = Tkinter.StringVar(root)
# status.set('Status: Not Trained')

# predicted_class = Tkinter.StringVar()
# actual_class = Tkinter.StringVar()
# epoch = Tkinter.IntVar()
# epoch.set(500)

# selected_example = Tkinter.IntVar()
# selected_example.set(0)


# prediction_btn = None
# app = Tkinter.Tk()
# app.geometry('500x500')

# status = Tkinter.StringVar(app)
# status.set("Not Trained")
# accuracy = Tkinter.StringVar(app)
# accuracy.set('Accuracy: 0%')

# X_train, Y_train, X_val, Y_val = utils.load_training_dataset('mnist_train.csv')

# def start_training():
#     status.set(f'Training started')
#     app.update()
#     # W1, b1, W2, b2 = nn.train(X_train,Y_train, 0.1, 500)
#     # accuracy = nn.compute_accuracy(X_val, Y_val, W1, b1, W2, b2)
#     # print(f'Accuracy: {accuracy}')
#     # # status = 'Accuracy: ' + str(accuracy)
#     # status.set('Training completed')

#     imgArr = X_val[11]
#     imgArr.shape = (28, 28)
#     print(imgArr.shape)
#     image = ImageTk.PhotoImage(image=Image.fromarray(imgArr))
#     # image_label = Tkinter.Label(app, image=image, text="hello ")
    

#     # test_prediction(X_val[233], Y_val[233], W1, b1, W2, b2)
#     # test_prediction(X_val[11], Y_val[11], W1, b1, W2, b2)
#     # test_prediction(X_val[789], Y_val[789], W1, b1, W2, b2)
#     # test_prediction(X_val[10], Y_val[10], W1, b1, W2, b2)




# train_button = Tkinter.Button(app, text="Train", command=start_training)
# train_button.grid(column=0, row=0)

# status_label = Tkinter.Label(app, textvariable=status)
# status_label.grid(column=1, row=0)

# accuracy_label = Tkinter.Label(app, textvariable=accuracy)
# accuracy_label.grid(column=0, row=1)


# imgArr = X_val[11] * 255
# imgArr.shape = (28, 28)
# image = ImageTk.PhotoImage(image=Image.fromarray(imgArr))
# image_label = Tkinter.Label(app, image=image, width=200, height=200)
# image_label.grid(column=0, row=2)

# prediction = Tkinter.StringVar(app)
# prediction.set('Prediction: Not Made')

# prediction_label = Tkinter.Label(app, textvariable=prediction)
# prediction_label.grid(column=1, row=3)

# def threaded_train():
#     W1, b1, W2, b2 = nn.train(X_train,Y_train, 0.1, epoch.get())
#     status.set('Status: Training Completed')
    



# def build_ui():
#     training_frame = Tkinter.Frame(root)
#     training_frame.grid(column=0, row=0)

#     epoch_lbl = Tkinter.Label(training_frame, text='Select Epoch: ')
#     epoch_lbl.grid(column=0, row=0)
#     epoch_spinbox = Tkinter.Spinbox(training_frame, from_=1, to=1000, textvariable=epoch, wrap=True)
#     epoch_spinbox.grid(column=1, row=0)

#     train_btn = Tkinter.Button(training_frame, text='Train', command=handle_train)
#     train_btn.grid(column=0, row=1)

#     status_lbl = Tkinter.Label(training_frame, textvariable=status)
#     status_lbl.grid(column=1, row=1)

#     validation_frame = Tkinter.Frame(root)
#     validation_frame.grid(column=0, row=4)

#     prediction_btn = Tkinter.Button(text='Predict', command=handle_prediction, state='disabled')
#     prediction_btn.grid(column=0, row=0)


#     root.mainloop()



# def handle_train():
#     status.set('Status: Training Started')
#     root.update()

#     train_thread = Thread(target=threaded_train, daemon=True)
#     train_thread.start()
#     prediction_btn['state'] = 'active'
#     root.update()

    

# def handle_prediction():
#     print(W1.shape)
#     nn.test_prediction(X_val[selected_example.get()], Y_val[selected_example.get()], W1, b1, W2, b2)
    



class Model:
    
    def __init__(self) -> None:
        X_train, Y_train, X_val, Y_val = utils.load_training_dataset('mnist_train.csv')

    def train(self):
        pass
        # perform training process
        # updateStatus(msg)



    def predict(self, idx):
        pass
        # predict for X_val[idx]
        

class GUI:

    def __init__(self, model) -> None:
        self.model = model
        self.root = Tkinter.Tk()
        self.root.geometry('500x500')
        self.trainingStatus = Tkinter.StringVar(self.root)
        self.trainingStatus.set('Status: Not Trained')
        self.epochs = Tkinter.IntVar(self.root)
        self.epochs.set(500)
        
    def __build(self):

        epoch_lbl = Tkinter.Label(self.root, text='Select Epoch: ')
        epoch_lbl.grid(column=0, row=0)
        epoch_spinbox = Tkinter.Spinbox(self.root, from_=1, to=1000, textvariable=self.epochs, wrap=True)
        epoch_spinbox.grid(column=1, row=0)

        train_btn = Tkinter.Button(self.root, text='Train', command=self.handle_train_click)
        train_btn.grid(column=0, row=1)

        status_lbl = Tkinter.Label(self.root, textvariable=self.trainingStatus)
        status_lbl.grid(column=1, row=1)


        prediction_btn = Tkinter.Button(self.root, text='Predict', command=self.handle_prediction_click, state='disabled')
        prediction_btn.grid(column=0, row=2)
            

    def start(self):
        self.root.mainloop()

    def handle_prediction_click():
        pass
    
    def handle_train_click():
        pass


def __main__():
    model = Model()
    gui = GUI(model)
    gui.start()


if __name__ == '__main__':
    __main__()
