import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
from tkinter import messagebox
from resources import *



class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def train_neural_network(inputs, targets):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


def predict(input_data):
    model.eval()
    with torch.no_grad():
        return model(input_data)


class TrimWindow:
    def __init__(self, master):
        self.master = master
        self.master.title('Trim Window')
        self.master.geometry('400x200')
        self.master.resizable(False, False)
        self.master.configure(background=DARK)
        self.create_widgets()

    def create_widgets(self):
       
        self.initialize_neural_network()

        label = tk.Label(self.master, text="Trim Window", font=('Arial', 12, 'bold'), fg=LIGHT_GRAY, bg=DARK)
        label.pack(pady=(20, 10))

       
        torch_button = tk.Button(self.master, text="Run PyTorch Task", command=self.run_pytorch_task)
        torch_button.pack(pady=10)

    def initialize_neural_network(self):
        global model, criterion, optimizer
        input_size = 10  
        output_size = 1  
        model = SimpleNN(input_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    def run_pytorch_task(self):
      
        input_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
        target = torch.tensor([3.0], dtype=torch.float32)

        train_neural_network(input_data, target)

        prediction = predict(input_data)
        messagebox.showinfo("PyTorch Task", f"PyTorch Prediction: {prediction.item()}")


if __name__ == "__main__":
    root = tk.Tk()
    TrimWindow(root)
    root.mainloop()
