import torch.nn as nn
import torch.optim as optim


class FC_2_Layers_Binary_Output(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2):
            super(FC_2_Layers_Binary_Output, self).__init__()
            self.input_size = input_size
            self.hidden_size_1  = hidden_size_1
            self.hidden_size_2  = hidden_size_2
            
            self.fc1 = nn.Linear(self.input_size,self.hidden_size_1)
            self.relu1 = nn.ReLU()
                    
            self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.relu2 = nn.ReLU()          
            
            self.fc3 = nn.Linear(self.hidden_size_2, 1)         
           
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu1(hidden1)
            
            hidden2 = self.fc2(relu1)
            relu2 = self.relu2(hidden2)
            
            output = self.fc3(relu2)
            output = self.sigmoid(output)
            return output
           
        
def training_full_batch_SGD(model, X_train_T, y_train_T, lr,  n_epochs):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)   
   
    
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
        