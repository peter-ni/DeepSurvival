import numpy as np
import torch

from torch.utils.data import DataLoader


import common_utils

# Import data
from common_utils import cancer_pd
full_dataset = torch.from_numpy(cancer_pd.values)

train_size = int(0.7 * full_dataset.shape[0])
test_size = full_dataset.shape[0] - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size , test_size])
train_data_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_data_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)






# Model Architecture
model = torch.nn.Sequential(
    torch.nn.Linear(473, 25), # Hidden Layer 1
    torch.nn.SELU(),
    torch.nn.Dropout(p = 0.3),
    torch.nn.Linear(25, 25), # Hidden Layer 2
    torch.nn.SELU(),
    torch.nn.Linear(25, 1), # Output Layer
    
)


# Loss Function (works on batches)

from common_utils import weibull_pdf, one_minus_weibull_cdf, timeAUC, getSurvTime

def weibull_loss(eta, base_haz_params, targets):
    # Force baseline hazard parameters to be positive with softplus
    base_haz_params_trans = torch.nn.functional.softplus(base_haz_params)

    lam = base_haz_params_trans[0] * torch.exp(- eta / base_haz_params_trans[1])
    k = base_haz_params_trans[1]

    likelihood = torch.zeros(targets.shape[0])
    # clipping = 1e-6

    dead_idx = np.where(targets[:,1] == 1)[0]
    censor_idx = np.where(targets[:,1] == 0)[0]

    # Removed clipping, you would add back here if needed
    likelihood[dead_idx] = torch.log(weibull_pdf(t = targets[dead_idx,0], lam = lam[dead_idx], k = k) )
    likelihood[censor_idx] = torch.log(one_minus_weibull_cdf(t = targets[censor_idx,0], lam = lam[censor_idx], k = k))

    # print(likelihood[dead_idx])
    # print(likelihood[censor_idx])
    # print(eta.min(), eta.max())

    loss = - torch.sum(likelihood)
    return loss


# Loss Function Test
# eta = torch.from_numpy(np.random.normal(size = 100))
# base_haz_params = torch.from_numpy(np.array([1, 1]))
# targets = full_dataset[0:100, 0:2]

# test_loss = weibull_loss(eta = eta, base_haz_params = base_haz_params, targets = targets)
# print(test_loss)


# Hyperparameters
# ==================

# Parameter Initialization
def init_network_weights(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.normal_(m.weight, mean = 0, std = 0.001)
        # m.bias.data.fill_(1e-4)

model.apply(init_network_weights)
base_haz_params = torch.tensor([1, 1], dtype = torch.float32, requires_grad = True)


# Optimizer
import torch.optim as optim

start_learning_rate = 5e-5
opt_param_set = list(model.parameters()) + [base_haz_params]
optimizer = optim.Adam(params = opt_param_set, lr = start_learning_rate)


# Training Loop
# ==================

from datetime import datetime





def train_one_epoch(epoch_index, train_data_loader, l1_lambda = 0.01):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_data_loader):
        # Remove if doesnt work
        data = data[0]
        targets, inputs = data[:,1:3], data[:,3:]
        optimizer.zero_grad()

        eta = model(inputs.float())
        loss = weibull_loss(eta, base_haz_params, targets)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm


        running_loss += loss
        loss.backward()
        optimizer.step()
    
    epoch_loss = running_loss / (i + 1)
    return epoch_loss
    

# Epoch Training
# ==================
import pandas as pd
from datetime import datetime


#timestamp for only the hour and minutes
timestamp = datetime.now().strftime('%H_%M')

num_epochs = 500
epoch_index = 1
best_tloss = 1e6

loss_df = pd.DataFrame(columns = ['epoch', 'train_loss', 'test_loss'])



# CV Training
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


n_splits = 5
kf = KFold(n_splits = n_splits, shuffle = True)
full_dataset_np = full_dataset.numpy()
auc_scores = []
auc_times =  np.arange(1, 6.25, 0.25)

from common_utils import timeAUC, getSurvTime

for epoch in range(1, num_epochs):
    epoch_index += 1
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_auc = []

    for fold, (train_index, test_index) in enumerate(kf.split(full_dataset_np)):
        print(f"Epoch: {epoch} -- Training on Fold: {fold + 1}/{n_splits}")

        # Split dataset within fold
        train_dataset = TensorDataset(torch.from_numpy(full_dataset_np[train_index]))
        test_dataset = TensorDataset(torch.from_numpy(full_dataset_np[test_index]))
        train_data_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
        test_data_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

        model.apply(init_network_weights)
        base_haz_params = torch.tensor([1, 1], dtype = torch.float32, requires_grad = True)
        opt_param_set = list(model.parameters()) + [base_haz_params]
        optimizer = optim.Adam(params = opt_param_set, lr = start_learning_rate)

        # Train on current fold train set
        train_loss = train_one_epoch(epoch_index = epoch_index, train_data_loader = train_data_loader)
        epoch_train_losses.append(train_loss)

        # Validate on current fold test set
        running_tloss = 0.0
        running_tauc = np.empty(len(auc_times))
        batches_used = 0
        model.eval()

        # Get loss and time-dependent AUC for current fold test set
        with torch.no_grad():
            for _, tdata in enumerate(test_data_loader):
                tdata = tdata[0]
                ttargets, tinputs = tdata[:,1:3], tdata[:,3:]
                teta = model(tinputs.float())

                tpreds = getSurvTime(teta, base_haz_params)
                auc_input = torch.column_stack((ttargets[:,0], ttargets[:,1], tpreds)).detach().numpy()
                auc_input = pd.DataFrame(auc_input)
                current_batch_auc = timeAUC(input = auc_input, times = auc_times)
                

                if any(element is None for element in current_batch_auc):
                    1
                else:
                    running_tauc += current_batch_auc
                    batches_used += 1
                


                tloss = weibull_loss(teta, base_haz_params, ttargets)
                running_tloss += tloss
            
            fold_auc = running_tauc / batches_used
            test_loss = running_tloss / len(test_data_loader)
            epoch_test_losses.append(test_loss)
            epoch_auc.append(fold_auc)

        
        
    # Average train/test loss for current epoch across all folds
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
    avg_auc = sum(epoch_auc) / len(epoch_auc)
    loss_df = loss_df.append({'epoch': epoch_index, 'train_loss': avg_train_loss.item(), 'test_loss': avg_test_loss.item()}, ignore_index=True)
    print('LOSS -- Train: {} -- Test: {}'.format(avg_train_loss, avg_test_loss))







# Save Stuff for Model
model_path = 'saved_models/model_{}_epochs{}'.format(timestamp, epoch_index)
baseline_tensor_path = 'saved_models/baseline_tensor_{}_epochs{}.pt'.format(timestamp, epoch_index)
torch.save(model.state_dict(), model_path)
torch.save(base_haz_params, baseline_tensor_path)

loss_df_path = 'saved_models/loss_df_{}_epochs{}.csv'.format(timestamp, epoch_index)
loss_df.to_csv(loss_df_path, index = False)

tauc = epoch_auc[len(epoch_auc) - 1]
auc_df = np.column_stack((auc_times, tauc))
auc_df = pd.DataFrame(auc_df)
auc_df_path = 'saved_models/auc_df_{}_epochs{}.csv'.format(timestamp, epoch_index)
auc_df.to_csv(auc_df_path, index = False)

# Load Model
saved_mod_path = model_path
saved_mod = model
saved_mod.load_state_dict(torch.load(saved_mod_path))


saved_baseline_tensor_path = baseline_tensor_path
saved_baseline_tensor = torch.load(saved_baseline_tensor_path)

saved_mod.load_state_dict(torch.load(saved_mod_path))
saved_mod.eval()

# Make Predictions on Full Dataset with Best Model

predictions = []
full_data_loader = DataLoader(full_dataset, batch_size = 32, shuffle = True)

with torch.no_grad():
    for data in full_data_loader:
        inputs = data[:,3:]
        batch_pred = saved_mod(inputs.float())
        predictions.append(batch_pred)


predictions = torch.cat(predictions)

pred_surv_times = getSurvTime(predictions, saved_baseline_tensor).detach().numpy()


test_targets = pd.DataFrame(columns = ['time', 'event'])

for data in full_data_loader:
    targets = data[:,1:3].detach().numpy()

    temp_df = pd.DataFrame({'time': targets[:,0], 'event': targets[:,1]})
    test_targets = pd.concat([test_targets, temp_df], ignore_index=True)

target_out = np.column_stack((test_targets, pred_surv_times))
target_out = pd.DataFrame(target_out)
target_out_path = 'target_out_{}.csv'.format(timestamp)
target_out.to_csv(target_out_path, index = False)












