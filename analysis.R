library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# Best so far
best_auc_df = read_csv("Desktop/Projects/DeepSurvival/DeepSurvival/saved_models/auc_df_22_57_epochs450.csv") %>% as.data.frame()
colnames(best_auc_df) = c("Time", "AUC")
# loss_df = read_csv("Desktop/Projects/DeepSurvival/DeepSurvival/saved_models/loss_df_22_57_epochs450.csv") %>% as.data.frame()
best_auc_plt = ggplot(data = best_auc_df, aes(x = Time, y = AUC)) + geom_line() + ggtitle("Time-dependent AUC")
best_auc_plt

# Reg 1
# auc_df = read_csv("Desktop/Projects/DeepSurvival/DeepSurvival/saved_models/auc_df_01_05_epochs450.csv") %>% as.data.frame()
# loss_df = read_csv("Desktop/Projects/DeepSurvival/DeepSurvival/saved_models/loss_df_01_05_epochs450.csv") %>% as.data.frame()


auc_df = read_csv("Desktop/Projects/DeepSurvival/DeepSurvival/saved_models/auc_df_01_05_epochs450.csv") %>% as.data.frame()
loss_df = read_csv("Desktop/Projects/DeepSurvival/DeepSurvival/saved_models/loss_df_01_05_epochs450.csv") %>% as.data.frame()



colnames(auc_df) = c("Time", "AUC")
colnames(loss_df) = c("Epoch", "Train Loss", "Test Loss")

loss_df_long = loss_df %>% 
  pivot_longer(
    cols = c("Train Loss", "Test Loss"),
    names_to = "Type",
    values_to = "Loss"
  )



auc_plt = ggplot(data = auc_df, aes(x = Time, y = AUC)) + geom_line() + ggtitle("Time-dependent AUC")
loss_plt = ggplot(data = loss_df_long, aes(x = Epoch, y = Loss, color = Type)) + geom_line() + ggtitle("Train/Test Loss over Epochs") +xlim(c(0,500))

auc_plt
loss_plt








