import pennylane as qml
import pennylane.numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse
import os

### Variable ####################################
torch_pi = torch.Tensor([np.pi])
 
parser = argparse.ArgumentParser(description='Run Quantum Neural Network training.')
### Pennylane Quantum Device ####################################
dev_QNN = qml.device("default.qubit", wires = 2)
dev_QNN_noise = qml.device("default.mixed", wires = 2)
dev_QSP = qml.device("default.qubit", wires = 4)
dev_QSP_noise = qml.device("default.mixed", wires = 4)
dev_DQC1 = qml.device('default.qubit', wires = 3)
dev_DQC1_noise = qml.device("default.mixed", wires = 3)


################################################################################################################################################
### Quantum Circuit ####################################
@qml.qnode(dev_QNN, interface='torch')
def QNN_circuit(layers, x, theta):
    i = 0

    for _ in range(layers):
        for q in range(2):
            qml.RY(theta[i], wires = q)
            qml.RZ(theta[i+1] * x, wires = q)
            qml.RX(theta[i+2], wires = q)
            i += 3
        
        qml.CZ(wires = [0, 1])

    qml.RY(theta[-1], wires = 0)
    qml.RY(theta[-2], wires = 1)

    return qml.expval((qml.PauliZ(0)+qml.PauliZ(1))/2)

### Quantum Circuit - BitFlip ####################################
@qml.qnode(dev_QNN_noise, interface='torch')
def QNN_circuit_BitFlip(layers, x, phi):
    i = 0

    for _ in range(layers):
        for q in range(2):
            qml.RY(phi[i], wires = q)
            qml.RZ(phi[i+1] * x, wires = q)
            qml.RX(phi[i+2], wires = q)
            i += 3
        
        qml.CZ(wires = [0, 1])

    qml.RY(phi[-1], wires = 0)
    qml.RY(phi[-2], wires = 1)

    qml.BitFlip(0.001, wires = 0)
    qml.BitFlip(0.001, wires = 1)

    return qml.expval((qml.PauliZ(0)+qml.PauliZ(1))/2)


### Quantum Circuit - Depoloarizing ####################################
@qml.qnode(dev_QNN_noise, interface='torch')
def QNN_circuit_Depolarizing(layers, x, phi):
    i = 0

    for _ in range(layers):
        for q in range(2):
            qml.RY(phi[i], wires = q)
            qml.RZ(phi[i+1] * x, wires = q)
            qml.RX(phi[i+2], wires = q)
            i += 3
        
        qml.CZ(wires = [0, 1])

    qml.RY(phi[-1], wires = 0)
    qml.RY(phi[-2], wires = 1)

    qml.DepolarizingChannel(0.001, wires = 0)
    qml.DepolarizingChannel(0.001, wires = 1)

    return qml.expval((qml.PauliZ(0)+qml.PauliZ(1))/2)

################################################################################################################################################
### QSP Circuit ####################################
@qml.qnode(dev_QSP, interface = 'torch')
def QSP_circuit(layers, x, phi):
    theta = -2 * torch.arccos(x)
    
    qml.Hadamard(wires = 0)
    qml.Hadamard(wires = 1)
    for i in range(layers):
        qml.CRZ(phi[i], wires=[0,1])
        qml.CRX(theta, wires=[0,1], id = "SRO")
    qml.CRZ(phi[layers], wires=[0,1])  
    qml.Hadamard(wires = 1)
    qml.Hadamard(wires = 0)

    qml.Hadamard(wires = 2)
    qml.Hadamard(wires = 3)
    for i in range(layers-1):
        qml.CRZ(phi[layers+1+i], wires=[2,3])
        qml.CRX(theta, wires=[2,3], id = "SRO")
    qml.CRZ(phi[-1], wires=[2,3])  
    qml.Hadamard(wires = 3)
    qml.Hadamard(wires = 2)

    return qml.expval((qml.PauliZ(0)+qml.PauliZ(2))/2)

### QSP Circuit - BitFlip ####################################
@qml.qnode(dev_QSP_noise, interface = 'torch')
def QSP_circuit_BitFlip(layers, x, phi):
    theta = -2 * torch.arccos(x)
    
    qml.Hadamard(wires = 0)
    qml.Hadamard(wires = 1)
    for i in range(layers):
        qml.CRZ(phi[i], wires=[0,1])
        qml.CRX(theta, wires=[0,1], id = "SRO")
    qml.CRZ(phi[layers], wires=[0,1])  
    qml.Hadamard(wires = 1)
    qml.Hadamard(wires = 0)

    qml.Hadamard(wires = 2)
    qml.Hadamard(wires = 3)
    for i in range(layers-1):
        qml.CRZ(phi[layers+1+i], wires=[2,3])
        qml.CRX(theta, wires=[2,3], id = "SRO")
    qml.CRZ(phi[-1], wires=[2,3])  
    qml.Hadamard(wires = 3)
    qml.Hadamard(wires = 2)

    qml.BitFlip(0.001, wires = 0)
    qml.BitFlip(0.001, wires = 2)

    return qml.expval((qml.PauliZ(0)+qml.PauliZ(2))/2)

### QSP Circuit - Depolarizing ####################################
@qml.qnode(dev_QSP_noise, interface = 'torch')
def QSP_circuit_Depolarizing(layers, x, phi):
    theta = -2 * torch.arccos(x)
    
    qml.Hadamard(wires = 0)
    qml.Hadamard(wires = 1)
    for i in range(layers):
        qml.CRZ(phi[i], wires=[0,1])
        qml.CRX(theta, wires=[0,1], id = "SRO")
    qml.CRZ(phi[layers], wires=[0,1])  
    qml.Hadamard(wires = 1)
    qml.Hadamard(wires = 0)

    qml.Hadamard(wires = 2)
    qml.Hadamard(wires = 3)
    for i in range(layers-1):
        qml.CRZ(phi[layers+1+i], wires=[2,3])
        qml.CRX(theta, wires=[2,3], id = "SRO")
    qml.CRZ(phi[-1], wires=[2,3])  
    qml.Hadamard(wires = 3)
    qml.Hadamard(wires = 2)

    qml.DepolarizingChannel(0.001, wires = 0)
    qml.DepolarizingChannel(0.001, wires = 2)

    return qml.expval((qml.PauliZ(0)+qml.PauliZ(2))/2)

################################################################################################################################################
def S(x, theta, wires):
    qml.RX(theta[0]*x, wires = wires[0])
    qml.RX(theta[1]*x, wires = wires[1])

def W(phi, wires):
    i = 0
    for w in wires:
        qml.RX(phi[i], wires = w) 
        qml.RY(phi[i+1], wires = w)
        qml.RZ(phi[i+2], wires = w) 
        i += 3
    
    qml.CZ(wires = wires)

def DQC1_Unit(layers, x, thetas, phis, wires):
    i = 0

    for _ in range(layers):
        W(phis[i], wires)
        S(x, thetas[i], wires)
        i += 1
    W(phis[i], wires)

### DQC1 Circuit ####################################
@qml.qnode(dev_DQC1, interface = 'torch')
def DQC1_circuit(layers, x, thetas, phis):
    qml.Hadamard(0)

    qml.ctrl(DQC1_Unit, 0)(layers, x, thetas, phis, [1,2])

    return qml.expval(qml.PauliX(wires = 0))


### DQC1 Circuit - Bit Flip ####################################
@qml.qnode(dev_DQC1_noise, interface = 'torch')
def DQC1_circuit_BitFlip(layers, x, thetas, phis):
    qml.Hadamard(0)

    qml.ctrl(DQC1_Unit, 0)(layers, x, thetas, phis, [1,2])

    qml.BitFlip(0.001, wires = 0)

    return qml.expval(qml.PauliX(wires = 0))

### DQC1 Circuit - Depolarizing ####################################
@qml.qnode(dev_DQC1_noise, interface = 'torch')
def DQC1_circuit_Depolarizing(layers, x, thetas, phis):
    qml.Hadamard(0)

    qml.ctrl(DQC1_Unit, 0)(layers, x, thetas, phis, [1,2])

    qml.DepolarizingChannel(0.001, wires = 0)

    return qml.expval(qml.PauliX(wires = 0))

################################################################################################################################################
### VQC 모델(QNN - Ideal) ####################################
class QNN_Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(6 * layers + 2, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_paramters = 6 * layers + 2
        self.num_qubits = 2
        self.num_layers = layers

    def forward(self, data_point):
        output = QNN_circuit(self.num_layers, data_point, self.theta)
        return output
    
### VQC 모델(QNN - Gate Error) ####################################
class QNN_Model_GateError(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(6 * layers + 2, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_paramters = 6 * layers + 2
        self.num_qubits = 2
        self.num_layers = layers

    def forward(self, data_point):
        noise = torch.randn(len(self.theta))
        noise_theta = self.theta + 0.001 * noise
        output = QNN_circuit(self.num_layers, data_point, noise_theta)
        return output

### VQC 모델(QNN - BitFlip) ####################################
class QNN_Model_BitFlip(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(6 * layers + 2, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_paramters = 6 * layers + 2
        self.num_qubits = 2
        self.num_layers = layers

    def forward(self, data_point):
        outputs = torch.zeros_like(data_point)
        for i, x in enumerate(data_point):
            outputs[i] = QNN_circuit_BitFlip(self.num_layers, x, self.theta)
        return outputs
    

### VQC 모델(QNN - Depolarizing) ####################################
class QNN_Model_Depolarizing(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(6 * layers + 2, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_paramters = 6 * layers + 2
        self.num_qubits = 2
        self.num_layers = layers

    def forward(self, data_point):
        outputs = torch.zeros_like(data_point)
        for i, x in enumerate(data_point):
            outputs[i] = QNN_circuit_Depolarizing(self.num_layers, x, self.theta)
        return outputs

################################################################################################################################################
### VQC 모델(QSP - Ideal)  ####################################
class QSP_Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(2 * layers + 1, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_phi = 2 * layers + 1
        self.num_qubits = 4
        self.num_layers = layers

    def forward(self, data_point):
        """PennyLane forward implementation"""
        output = QSP_circuit(self.num_layers, data_point, self.theta)
        return output
    
### VQC 모델(QSP - Gate Error)  ####################################
class QSP_Model_GateError(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(2 * layers + 1, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_parameters = 2 * layers + 1
        self.num_qubits = 4
        self.num_layers = layers

    def forward(self, data_point):
        noise = torch.randn(len(self.theta))
        noise_theta = self.theta + 0.001 * noise
        output = QSP_circuit(self.num_layers, data_point, noise_theta)
        return output
    
### VQC 모델(QSP - BitFlip)  ####################################
class QSP_Model_BitFlip(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(2 * layers + 1, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_parameters = 2 * layers + 1
        self.num_qubits = 4
        self.num_layers = layers

    def forward(self, data_point):
        outputs = torch.zeros_like(data_point)
        for i, x in enumerate(data_point):
            outputs[i] = QSP_circuit_BitFlip(self.num_layers, x, self.theta)
        return outputs
    
### VQC 모델(QSP - Depolarizing)  ####################################
class QSP_Model_Depolarizing(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.theta = torch_pi * torch.rand(2 * layers + 1, requires_grad=True)
        self.theta = nn.Parameter(self.theta)
    
        self.num_training_parameters = 2 * layers + 1
        self.num_qubits = 4
        self.num_layers = layers

    def forward(self, data_point):
        outputs = torch.zeros_like(data_point)
        for i, x in enumerate(data_point):
            outputs[i] = QSP_circuit_Depolarizing(self.num_layers, x, self.theta)
        return outputs
    
############################################################################################################
### VQC 모델(DQC1 - Ideal)  ####################################
class DQC1_Model(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.phi = torch.rand((degree+1, 6), requires_grad=True)
        self.theta = torch.rand((degree, 2) ,requires_grad = True)
        self.phi = nn.Parameter(self.phi)
        self.theta = nn.Parameter(self.theta)
    
        self.num_phi = 6 * degree + 6 + 2 * degree
        self.num_qubits = 2
        self.num_layers = degree

    def forward(self, data_point):
        output = DQC1_circuit(self.num_layers, data_point, self.theta, self.phi)
        return output
    
### VQC 모델(DQC1 - Gate Error)  ####################################
class DQC1_Model_GateError(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.phi = torch.rand((degree+1, 6), requires_grad=True)
        self.theta = torch.rand((degree, 2) ,requires_grad = True)
        self.phi = nn.Parameter(self.phi)
        self.theta = nn.Parameter(self.theta)
    
        self.num_phi = 6 * degree + 6 + 2 * degree
        self.num_qubits = 2
        self.num_layers = degree

    def forward(self, data_point):
        noise1 =torch.randn_like(self.phi)
        noise2 = torch.randn_like(self.theta)
        noise_phi = self.phi + 0.001 * noise1
        noise_theta = self.theta + 0.001 * noise2
        output = DQC1_circuit(self.num_layers, data_point, noise_theta, noise_phi)
        return output
    
### VQC 모델(DQC1 - Bit Flip) ####################################
class DQC1_Model_BitFlip(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.phi = torch.rand((degree+1, 6), requires_grad=True)
        self.theta = torch.rand((degree, 2) ,requires_grad = True)
        self.phi = nn.Parameter(self.phi)
        self.theta = nn.Parameter(self.theta)
    
        self.num_phi = 6 * degree + 6 + 2 * degree
        self.num_qubits = 2
        self.num_layers = degree

    def forward(self, data_point):
        outputs = torch.zeros_like(data_point)
        for i, x in enumerate(data_point):
            outputs[i] = DQC1_circuit_BitFlip(self.num_layers, x, self.theta, self.phi)
        return outputs
    
### VQC 모델(DQC1 - Depolarizing) ####################################
class DQC1_Model_Depolarizing(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.phi = torch.rand((degree+1, 6), requires_grad=True)
        self.theta = torch.rand((degree, 2) ,requires_grad = True)
        self.phi = nn.Parameter(self.phi)
        self.theta = nn.Parameter(self.theta)
    
        self.num_phi = 6 * degree + 6 + 2 * degree
        self.num_qubits = 2
        self.num_layers = degree

    def forward(self, data_point):
        outputs = torch.zeros_like(data_point)
        for i, x in enumerate(data_point):
            outputs[i] = DQC1_circuit_Depolarizing(self.num_layers, x, self.theta, self.phi)
        return outputs


############################################################################################################
def load_model_environment(model, env, layers):
    model_classes = {
        "QNN": {
            "Ideal": QNN_Model,
            "GateError": QNN_Model_GateError,
            "BitFlip": QNN_Model_BitFlip,
            "Depolarizing": QNN_Model_Depolarizing,
        },
        "QSP": {
            "Ideal": QSP_Model,
            "GateError": QSP_Model_GateError,
            "BitFlip": QSP_Model_BitFlip,
            "Depolarizing": QSP_Model_Depolarizing,
        },
        "DQC1": {
            "Ideal": DQC1_Model,
            "GateError": DQC1_Model_GateError,
            "BitFlip": DQC1_Model_BitFlip,
            "Depolarizing": DQC1_Model_Depolarizing,
        }
    }

    try:
        model_class = model_classes[model][env]
        return model_class(layers)
    except KeyError:
        raise ValueError(f"Invalid model ({model}) or environment ({env}) specified.")
    
    ### Data Preprocessing (Normaliztion & Seperating to Train and Val Dataset)####
def Data_loader(file_path):
    Data = np.load(file_path)
    X = Data['x_data']
    y = Data['y_data']

    lower_bound = Data['x_init']
    upper_bound = Data['x_final']

    return X, y, lower_bound, upper_bound

def Data_Normalization(X, y, lower_bound, upper_bound):
    X_norm = (2*X - upper_bound - lower_bound) / (upper_bound - lower_bound)

    y_norm = y / np.max(np.abs(y))

    return X_norm, y_norm

def Making_Dataset(data_file_path):
    X, y, lower_bound, upper_bound = Data_loader(data_file_path)
    X_norm, y_norm = Data_Normalization(X, y, lower_bound, upper_bound)

    X_norm = torch.tensor(X_norm)
    y_norm = torch.tensor(y_norm)

    dataset = TensorDataset(X_norm, y_norm)
    
    return dataset, X_norm, y_norm

### Loss function ####################################
def MSE_loss(pred, target):
    loss = torch.mean((target - pred)**2)
    return loss

def Log_Cosh_loss(pred, target):
    loss = torch.mean(torch.log(torch.cosh(pred - target)))
    return loss

def Chisqr_loss(pred, target):
    loss = torch.mean(((target - pred)**2) /torch.abs(target))
    return loss

def KL_loss(pred, target):
    logsoft_target = F.log_softmax(target, dim = -1)
    logsoft_pred = F.log_softmax(pred, dim = -1)
    soft_target = F.softmax(target, dim = -1)
    loss =  torch.sum(soft_target * (logsoft_target - logsoft_pred))
    return loss

def MSE_KL_loss(pred, target):
    mse = MSE_loss(pred, target)
    kl = KL_loss(pred, target)
    return mse + kl

### Train ####################################
def train(model, dataset, n_epochs, loss_type, optimizer, batch_size, device = 'cpu'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # --- Early Stopping Parameters ---
    min_delta = 1e-5
    patience_limit = 15
    patient = 0  

    def moving_average(arr, window_size=5):
        if len(arr) < window_size:
            return np.mean(arr)
        return np.mean(arr[-window_size:])
    
    # --- 미분 연산 ---
    def derivative(y, t) : 
        return torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones(y.size()).to(device))[0]
    
    # --- Model device로 이동 ---
    model = model.to(device)

    # --- Training Data & Validation Data ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = int(batch_size / 2))

    # --- Loss function Setting ---
    loss_fn_dict = {
        'MSE': MSE_loss,
        'Log_Cosh': Log_Cosh_loss,
        'Chisqr': Chisqr_loss,
        'MSE_KL': MSE_KL_loss
    }
    loss_fn = loss_fn_dict[loss_type]

    # --- Training ---
    for epoch in tqdm(range(n_epochs)):
        model.train()

        # --- Training Process ---
        epoch_train_loss = 0.0
        for xb, yb in train_loader:

            xb = xb.requires_grad_()
            Qx = model(xb)
            dQdx = derivative(Qx, xb)
            loss = loss_fn(dQdx, yb)
            optimizer.zero_grad()
            loss.backward(retain_graph = False)

            optimizer.step()

            epoch_train_loss += loss.item() * xb.size(0)

        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- Validation Loss ---
        model.eval()
        epoch_val_loss = 0.0
        for xb, yb in val_loader:

            # --- Validation 계산 ---
            xb = xb.requires_grad_()
            Qx = model(xb)
            dQdx = derivative(Qx, xb)
            loss = loss_fn(dQdx, yb)
            epoch_val_loss += loss.item() * xb.size(0)

        val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- Best Model 저장 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # --- Early Stopping ---
        if len(val_losses) > 5:
            current_avg = moving_average(val_losses, window_size=2)
            previous_avg = moving_average(val_losses[:-1], window_size=2)

            if previous_avg - current_avg < min_delta:
                patient += 1
            else:
                patient = 0

            if patient >= patience_limit:
                print("Early stopping triggered.")
                break
                
    # --- Best Result ---
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print("--------------------------------------------")
    print(f"Training has finished and saved the  model with the best validation loss({best_val_loss :.4f})")

    return model, train_losses, val_losses

# --- Circuit & Model Configuration ---
parser.add_argument('--model_type', type=str, default='DQC1', choices=['QNN', 'QSP', 'DQC1'], help='Model type')
parser.add_argument('--env', type=str, default='Ideal', choices=['Ideal', 'GateError', 'BitFlip', 'Depolarizing'], help='Environment')
parser.add_argument('--layers', type=int, default=10, help='Number of layers in the circuit')
parser.add_argument('--test_func', type=str, default='CPF', choices=['CPF', 'Step', 'BW'], help='Test function')
parser.add_argument('--sample_type', type=str, default='Uni', choices=['Uni', 'Imp', 'HMC'], help='Sample type')
parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE', 'Log_Cosh', 'Chisqr', 'MSE_KL'], help='Loss function')
parser.add_argument('--num_epochs', type=int, default=int(1000), help = 'Number of Epochs')
parser.add_argument('--lr', type =float, default=1e-3, help = 'Learning Rate')
parser.add_argument('--model', type=str, default='.pt', help = 'Model State')

args = parser.parse_args()

# Use args in place of hardcoded values
model_type = args.model_type
env = args.env
layers = args.layers
test_func = args.test_func
sample_type = args.sample_type
loss_type = args.loss_type
n_epochs = args.num_epochs
lr = args.lr
model_state = args.model


### Data Prepertaion ####################################
file_path = f'Data File/{test_func}_Data({sample_type}, 5000).npz'
dataset, X_norm, y_norm = Making_Dataset(file_path)

save_dir = f"./Model/{model_type}/{env}/{test_func}/{sample_type}/{loss_type}/{lr}/Additional_Run"
os.makedirs(save_dir, exist_ok=True)

print("--------------------------------------------")
print("Directory Path :", save_dir)
print("--------------------------------------------")

model_path = f"./Model/{model_type}/{env}/{test_func}/{sample_type}/{loss_type}/{lr}/{model_state}"

print("--------------------------------------------")
print("Model Path :", model_path)
print("--------------------------------------------")

QNN_model = load_model_environment(model_type, env, layers)
QNN_model.load_state_dict(torch.load(model_path))

### Optimizer ####################################
optimizer = optim.Adam(QNN_model.parameters(), lr=lr)

if __name__ == "__main__":
    optimized_model, train_losses, val_losses = train(QNN_model, dataset, n_epochs, loss_type, optimizer, 256)

val_str = f"{val_losses[-1]:.3e}".replace('.', '_').replace('e-', '_e')

### Training Result Plot #######################
plt.figure(figsize=(5,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Training & Validation Loss over Epochs')
plt.savefig(f"{save_dir}/{model_type}_{env}_loss_curve_{val_str}.png", dpi=300)  # 저장

def derivative(y, t) : 
    return torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones(y.size()))[0]

optimized_model.eval()
X_norm = X_norm.requires_grad_()
Qx = optimized_model(X_norm)
dQdx = derivative(Qx, X_norm)

plt.figure(figsize=(5,4))
plt.scatter(X_norm.detach().numpy(), dQdx.detach().numpy(), s=1, label='Model derivative')
plt.scatter(X_norm.detach().numpy(), y_norm.detach().numpy(), s=1, label='Target function')
plt.legend()
# plt.title('Model Derivative vs Target')
plt.savefig(f"{save_dir}/{model_type}_{env}_derivative_{val_str}.png", dpi=300)  # 저장

torch.save(optimized_model.state_dict(), f"{save_dir}/{model_type}_{env}_L{layers}_addN{n_epochs}_{val_str}.pt")
np.savez(f'{save_dir}/Training_Graph_{val_str}.npz', train_graph = train_losses, val_graph = val_losses)