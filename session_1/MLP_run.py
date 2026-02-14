from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import KNNImputer
# from sklearn.metrics import roc_auc_score, average_precision_score
# from category_encoders import TargetEncoder
from torchmetrics.functional.classification import binary_auroc as roc_auc, binary_average_precision as pr_auc
"""
Setting up the same seed as in the replication notebook.
"""
SEED = 3508706438

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")
# Load datasets
test_hef = pd.read_csv('../data/MIMIC/MIMIC_III_dataset_death/mimic_test_death.csv')
train_hef = pd.read_csv('../data/MIMIC/MIMIC_III_dataset_death/mimic_train.csv')
extra_diag = pd.read_csv('../data/MIMIC/MIMIC_III_dataset_death/extra_data/MIMIC_diagnoses.csv')
extra_diag['ICD9_CODE'] = extra_diag['ICD9_CODE'].astype(str)
extra_diag['ICD9_CHAPTER'] = extra_diag['ICD9_CODE'].str[:3]

extra_diag['IS_SEPSIS'] = extra_diag['ICD9_CODE'].str.startswith(('9959', '7855')).astype(int)
extra_diag['IS_HEART_FAIL'] = extra_diag['ICD9_CODE'].str.startswith('428').astype(int)
extra_diag['IS_CANCER'] = extra_diag['ICD9_CODE'].str.startswith(('196', '197', '198', '199')).astype(int)
extra_diag['IS_RENAL'] = extra_diag['ICD9_CODE'].str.startswith(('584', '585')).astype(int)

diag_grouped = extra_diag.groupby('HADM_ID').agg({
    'ICD9_CODE': [
        ('NUM_DIAGNOSES', 'count'),                                  
        ('DIAG_STRING', lambda x: ' '.join(x.dropna().astype(str)))],
    'ICD9_CHAPTER': [('UNIQUE_CHAPTERS', 'nunique')],
    'IS_SEPSIS': [('HAS_SEPSIS', 'max')],
    'IS_HEART_FAIL': [('HAS_HEART_FAIL', 'max')],
    'IS_CANCER': [('HAS_CANCER', 'max')],
    'IS_RENAL': [('HAS_RENAL', 'max')]
})

diag_grouped.columns = diag_grouped.columns.droplevel(0)
diag_grouped = diag_grouped.reset_index()

# Merge features
train_hef = train_hef.merge(diag_grouped, left_on='hadm_id', right_on='HADM_ID', how='left')
test_hef = test_hef.merge(diag_grouped, left_on='hadm_id', right_on='HADM_ID', how='left')

# Drop HADM_ID as it's not needed anymore
train_hef.drop('HADM_ID', axis=1, inplace=True)
test_hef.drop('HADM_ID', axis=1, inplace=True)
def engineer_features(df_input):
    df = df_input.copy()
    
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DOB'] = pd.to_datetime(df['DOB'])
    df['AGE'] = df['ADMITTIME'].dt.year - df['DOB'].dt.year
    df.loc[df['AGE'] > 89, 'AGE'] = 90
    df.loc[df['AGE'] < 0, 'AGE'] = df['AGE'].median()
    
    original_index = df.index
    df = df.sort_values(by=['subject_id', 'ADMITTIME'])
    df['PREV_ICU_STAYS'] = df.groupby('subject_id').cumcount()
    df['LAST_ADMIT'] = df.groupby('subject_id')['ADMITTIME'].shift(1)
    seconds_diff = (df['ADMITTIME'] - df['LAST_ADMIT']).dt.total_seconds()
    df['DAYS_SINCE_LAST'] = seconds_diff / (24 * 3600)
    df['DAYS_SINCE_LAST'] = df['DAYS_SINCE_LAST'].fillna(-1)
    df = df.reindex(original_index)

    cols_to_drop = ['ADMITTIME', 'DOB', 'LAST_ADMIT', 'DISCHTIME', 'DEATHTIME', 
                    'DOD', 'LOS', 'Diff', 'MeanBP_Min', 'MeanBP_Max', 
                    'MeanBP_Mean', 'subject_id' #, 'hadm_id'
                    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    return df

train_processed = engineer_features(train_hef)
test_processed = engineer_features(test_hef)
# Define Column Groups
num_cols = [
    'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean', 
    'SysBP_Min', 'SysBP_Max', 'SysBP_Mean', 
    'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean', 
    'RespRate_Min', 'RespRate_Max', 'RespRate_Mean', 
    'TempC_Min', 'TempC_Max', 'TempC_Mean', 
    'SpO2_Min', 'SpO2_Max', 'SpO2_Mean', 
    'Glucose_Min', 'Glucose_Max', 'Glucose_Mean', 
    'PREV_ICU_STAYS', 'AGE', 'DAYS_SINCE_LAST', 
    'NUM_DIAGNOSES', 'UNIQUE_CHAPTERS', 
    'HAS_SEPSIS', 'HAS_HEART_FAIL', 'HAS_CANCER', 'HAS_RENAL'] 

categorical_cols = [
    'ICD9_diagnosis', 'DIAGNOSIS', 'FIRST_CAREUNIT', 
    'GENDER', 'ADMISSION_TYPE', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']

text_col = 'DIAG_STRING'
class DynamicMLP(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, dropout_rate):
        super(DynamicMLP, self).__init__()
        layers = []
        curr_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, 1))
        
        # Note: We do NOT use Sigmoid here because we use BCEWithLogitsLoss for numerical stability
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
# Prepare for Cross Validation
X = train_processed.drop('HOSPITAL_EXPIRE_FLAG', axis=1).copy()
y = train_processed['HOSPITAL_EXPIRE_FLAG'].values.copy()

# StratifiedGroupKFold to prevent leakage in medical data groups
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

print("CV setup complete. Text vectorization moved to Optuna study for tuning.")
y_tr_t = torch.tensor(y, dtype=torch.float32)
print(y_tr_t.shape)

y_tr_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
print(y_tr_t.shape)
def objective(trial):
    # --- Hyperparameters ---
    max_features = trial.suggest_int("max_features", 200, 1000, step=100)
    n_layers = trial.suggest_int("n_layers", 1, 7)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 512, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # HEF class distribution: 0.89 neg, 0.11 pos.
    # To perfectly balance it, we need to multiply the weight of the positives by 8.1 approx.
    pos_weight_val = trial.suggest_float("pos_weight", 1.0, 8.5)
    
    # We are using large batch sizes because our data fits perfectly in memory.
    size_train_data = len(train_processed)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192, size_train_data])
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups=train_processed['hadm_id'])):
        X_tr_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        imputer = KNNImputer(n_neighbors=5)
        X_tr_df_num = imputer.fit_transform(X_tr_df[num_cols])
        X_val_df_num = imputer.transform(X_val_df[num_cols])
        
        # 1. Numerical scaling
        X_tr_num = torch.tensor(X_tr_df_num, dtype=torch.float32)
        X_val_num = torch.tensor(X_val_df_num, dtype=torch.float32)
        mean, std = X_tr_num.mean(0), X_tr_num.std(0)
        X_tr_num = (X_tr_num - mean) / (std + 1e-7)
        X_val_num = (X_val_num - mean) / (std + 1e-7)
        
        # 2. Text Vectorization
        vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\b\w+\b', max_features=max_features)
        X_tr_text = torch.tensor(vectorizer.fit_transform(X_tr_df[text_col]).todense(), dtype=torch.float32)
        X_val_text = torch.tensor(vectorizer.transform(X_val_df[text_col]).todense(), dtype=torch.float32)
        
        # # 2.1 Categorical: Target Encode
        # encoder = TargetEncoder(cols=categorical_cols)
        # X_tr_cat = torch.tensor(encoder.fit_transform(X_tr_df[categorical_cols], y_tr).values, dtype=torch.float32)
        # X_val_cat = torch.tensor(encoder.transform(X_val_df[categorical_cols]).values, dtype=torch.float32)
        
        # 3. Concatenate
        X_tr_final = torch.cat([X_tr_num, X_tr_text], dim=1)
        X_val_final = torch.cat([X_val_num, X_val_text], dim=1)
        # X_tr_final = torch.cat([X_tr_num, X_tr_cat, X_tr_text], dim=1) # Uncomment for categorical
        
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        train_loader = DataLoader(TensorDataset(X_tr_final, y_tr_t), batch_size=batch_size, shuffle=True)
        
        model = DynamicMLP(X_tr_final.shape[1], n_layers, hidden_dim, dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_fold_score = 0
        patience = 10
        no_impr = 0
        
        # Training Loop with Early Stopping
        for epoch in range(100):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                logits = model(X_val_final.to(device))
                probs = torch.sigmoid(logits).flatten()
                labels = y_val_t.to(device).int().flatten()
                
                cur_roc = roc_auc(probs, labels)
                cur_pr = pr_auc(probs, labels)
                
                score = (0.75 * cur_roc + 0.25 * cur_pr).item()
                
            if score > best_fold_score:
                best_fold_score = score
                no_impr = 0
            else:
                no_impr += 1
                
            if no_impr >= patience:
                break
        
        fold_scores.append(best_fold_score)
        
    return np.mean(fold_scores)

# Create Study
db_path = Path("../data/MIMIC/optuna_HEF/optuna_mimic.db").resolve()
date_minutes = datetime.now().strftime("%Y%m%d_%H:%M")
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=SEED),
    direction="maximize",
    storage=f"sqlite:///{db_path.as_posix()}",
    study_name=f"{date_minutes}_mimic_model_optimization",
    load_if_exists=True
    )
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\nBest trial params:", study.best_params)
print(f"\nBest score: {study.best_value:.2f}")
# 1. Prepare Full Training Data
X_train_df = train_processed.drop('HOSPITAL_EXPIRE_FLAG', axis=1).copy()
y_train_full = train_processed['HOSPITAL_EXPIRE_FLAG'].values.copy()
X_test_df = test_processed.copy()

best_params = study.best_params

imputer = KNNImputer(n_neighbors=5)
X_tr_df_num = imputer.fit_transform(X_train_df[num_cols])
X_te_df_num = imputer.transform(X_test_df[num_cols])

# Numerical
X_tr_num = torch.tensor(X_tr_df_num, dtype=torch.float32)
X_te_num = torch.tensor(X_te_df_num, dtype=torch.float32)
mean, std = X_tr_num.mean(0), X_tr_num.std(0)
X_tr_num = (X_tr_num - mean) / (std + 1e-7)
X_te_num = (X_te_num - mean) / (std + 1e-7)

# # Categorical
# encoder = TargetEncoder(cols=categorical_cols)
# X_tr_cat = torch.tensor(encoder.fit_transform(X_train_df[categorical_cols], y_train_full).values, dtype=torch.float32)
# X_te_cat = torch.tensor(encoder.transform(X_test_df[categorical_cols]).values, dtype=torch.float32)

# Text
vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\b\w+\b', max_features=best_params['max_features'])
X_tr_text = torch.tensor(vectorizer.fit_transform(X_train_df[text_col]).todense(), dtype=torch.float32)
X_te_text = torch.tensor(vectorizer.transform(X_test_df[text_col]).todense(), dtype=torch.float32)

# Concatenate
X_train_final = torch.cat([X_tr_num, X_tr_text], dim=1)
X_test_final = torch.cat([X_te_num, X_te_text], dim=1)

y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)

# 2. Final Training
final_model = DynamicMLP(
                    X_train_final.shape[1],
                    best_params['n_layers'], 
                    best_params['hidden_dim'],
                    best_params['dropout_rate']
                ).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([best_params['pos_weight']]).to(device))

train_loader = DataLoader(TensorDataset(X_train_final, y_train_tensor), 
                            batch_size=best_params['batch_size'], shuffle=True)

final_model.train()
for epoch in range(50):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(final_model(xb), yb)
        loss.backward()
        optimizer.step()

# 3. Predict
final_model.eval()
with torch.no_grad():
    logits = final_model(X_test_final.to(device))
    y_proba = torch.sigmoid(logits).cpu().numpy().flatten()

# Save
submission = pd.DataFrame({'icustay_id': test_hef['icustay_id'], 'prediction': y_proba})
submission.to_csv(f'../data/MIMIC/{date_minutes}_pytorch_mlp_optuna_submission.csv', index=False)
print(f"Saved prediction to ../data/MIMIC/{date_minutes}_pytorch_mlp_optuna_submission.csv")