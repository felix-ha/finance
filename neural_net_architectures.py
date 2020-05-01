import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



class FC_2_Layers_Binary_Output(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, drop_p):
            super(FC_2_Layers_Binary_Output, self).__init__()
            self.input_size = input_size
            self.hidden_size_1  = hidden_size_1
            self.hidden_size_2  = hidden_size_2
            
            self.fc1 = nn.Linear(self.input_size,self.hidden_size_1)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(p=drop_p)
                    
            self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.relu2 = nn.ReLU()   
            self.drop2 = nn.Dropout(p=drop_p)
            
            self.fc3 = nn.Linear(self.hidden_size_2, 1)         
           
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu1(hidden1)
            drop1 = self.drop1(relu1)
            
            hidden2 = self.fc2(drop1)
            relu2 = self.relu2(hidden2)
            drop2 = self.drop1(relu2)
            
            output = self.fc3(drop2)
            output = self.sigmoid(output)
            return output
        
        
        
class RNN(nn.Module):
    def __init__(self, input_size, seq_len, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):

        # get RNN outputs, hidden is  unused
        r_out, hidden = self.rnn(x, hidden)

        
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out[:, self.seq_len-1]

        # get final output
        r_out = self.fc(r_out)
        
        output = self.sigmoid(r_out)
        

        return output
           
    
def training_SGD(model, X_train_T, y_train_T, X_val_T, y_val_T,
                 lr, n_epochs, batch_size):
    
    training_losses = np.empty(n_epochs)
    valid_losses = np.empty(n_epochs)
    
    train_ds = TensorDataset(X_train_T, y_train_T)
    train_dl = DataLoader(train_ds, batch_size=batch_size)  
    
    valid_ds = TensorDataset(X_val_T, y_val_T)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
    
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)   
   
    
    model.train()
    for epoch in range(n_epochs):
        training_loss = 0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X_batch)
            # Compute Loss
            loss = loss_func(y_pred.squeeze(), y_batch)
            training_loss += loss.item()
           
            # Backward pass
            loss.backward()
            optimizer.step()
            
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_dl:
                y_pred = model(X_batch)
                loss = loss_func(y_pred.squeeze(), y_batch) 
                valid_loss += loss.item()
                
            
        training_loss_epoch = training_loss / len(train_dl)
        valid_loss_epoch = valid_loss / len(valid_dl)
        
        training_losses[epoch] = training_loss_epoch
        valid_losses[epoch] = valid_loss_epoch
        
        
        print('Epoch {}: train loss: {:.4} valid loss: {:.4}'
              .format(epoch, training_loss_epoch, valid_loss_epoch))  
        
    return training_losses, valid_losses
      

  

        
def training_full_batch_SGD(model, X_train_T, y_train_T, lr,  n_epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)   
   
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train_T)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train_T)
       
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()
        
def training_full_batch_SGD_RNN(model, X_train_T, y_train_T, hidden_0, lr,  n_epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)   
   
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train_T, hidden_0)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train_T)
       
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()
        
        
def training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T, hidden_0,
                 lr, n_epochs, batch_size):
    
    training_losses = np.empty(n_epochs)
    valid_losses = np.empty(n_epochs)
    
    train_ds = TensorDataset(X_train_T, y_train_T)
    train_dl = DataLoader(train_ds, batch_size=batch_size)  
    
    valid_ds = TensorDataset(X_val_T, y_val_T)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
    
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)   
   
    
    model.train()
    for epoch in range(n_epochs):
        training_loss = 0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X_batch, hidden_0)
            # Compute Loss
            loss = loss_func(y_pred.squeeze(), y_batch)
            training_loss += loss.item()
           
            # Backward pass
            loss.backward()
            optimizer.step()
            
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_dl:
                y_pred = model(X_batch, hidden_0)
                loss = loss_func(y_pred.squeeze(), y_batch) 
                valid_loss += loss.item()
                
            
        training_loss_epoch = training_loss / len(train_dl)
        valid_loss_epoch = valid_loss / len(valid_dl)
        
        training_losses[epoch] = training_loss_epoch
        valid_losses[epoch] = valid_loss_epoch
        
        
        print('Epoch {}: train loss: {:.4} valid loss: {:.4}'
              .format(epoch, training_loss_epoch, valid_loss_epoch))  
        
    return training_losses, valid_losses
        