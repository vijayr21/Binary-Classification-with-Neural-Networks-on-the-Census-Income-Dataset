# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

NAME : VIJAY R

REG NO : 212223240178
```
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/income (1).csv')

categorical_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
continuous_cols = ['age', 'education-num', 'hours-per-week']
label_col = 'label'

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.strip().str.lower())
    label_encoders[col] = le

# Prepare tensors
cat_data = np.stack([df[col].values for col in categorical_cols], axis=1)
con_data = np.stack([df[col].values for col in continuous_cols], axis=1)
y = torch.tensor(df[label_col].values, dtype=torch.long)

X_cat = torch.tensor(cat_data, dtype=torch.int64)
X_con = torch.tensor(con_data, dtype=torch.float)

X_cat_train, X_cat_test, X_con_train, X_con_test, y_train, y_test = train_test_split(
    X_cat, X_con, y, test_size=0.2, random_state=42
)

# Define model
class TabularModel(nn.Module):
    def __init__(self, emb_sizes, n_cont, out_sz, p=0.4):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        self.hidden = nn.Linear(sum([nf for ni, nf in emb_sizes]) + n_cont, 50)
        self.bn_hidden = nn.BatchNorm1d(50)
        self.out = nn.Linear(50, out_sz)
        self.dropout = nn.Dropout(p)

    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = F.relu(self.bn_hidden(self.hidden(x)))
        x = self.dropout(x)
        x = self.out(x)
        return x

cat_szs = [len(df[col].unique()) for col in categorical_cols]
emb_szs = [(size, min(50, (size + 1) // 2)) for size in cat_szs]

torch.manual_seed(42)
model = TabularModel(emb_szs, len(continuous_cols), 2, p=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 300
for epoch in range(epochs):
    model.train()
    y_pred = model(X_cat_train, X_con_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_cat_test, X_con_test)
            val_loss = criterion(val_pred, y_test)
            acc = (val_pred.argmax(1) == y_test).float().mean()
            print(f'Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc:.4f}')

# --- FIXED PREDICTION FUNCTION ---
def safe_transform(le, value):
    """Safely transform new category values unseen during training."""
    value = value.strip().lower()
    if value not in le.classes_:
        print(f"⚠️ Warning: '{value}' unseen. Defaulting to '{le.classes_[0]}'")
        return le.transform([le.classes_[0]])[0]
    return le.transform([value])[0]

def predict_new_input(model, label_encoders):
    model.eval()
    sex = input("Enter sex (Male/Female): ").strip().lower()
    education = input("Enter education (e.g., hs-grad, masters): ").strip().lower()
    marital_status = input("Enter marital status (e.g., married, never-married): ").strip().lower()
    workclass = input("Enter workclass (e.g., private, federal-gov): ").strip().lower()
    occupation = input("Enter occupation (e.g., exec-managerial, craft-repair): ").strip().lower()
    age = float(input("Enter age: "))
    education_num = float(input("Enter education-num: "))
    hours_per_week = float(input("Enter hours-per-week: "))

    cat_values = [
        safe_transform(label_encoders['sex'], sex),
        safe_transform(label_encoders['education'], education),
        safe_transform(label_encoders['marital-status'], marital_status),
        safe_transform(label_encoders['workclass'], workclass),
        safe_transform(label_encoders['occupation'], occupation)
    ]
    con_values = [age, education_num, hours_per_week]

    x_cat = torch.tensor(np.array(cat_values).reshape(1, -1), dtype=torch.int64)
    x_con = torch.tensor(np.array(con_values).reshape(1, -1), dtype=torch.float)

    with torch.no_grad():
        out = model(x_cat, x_con)
        pred = torch.argmax(out, 1).item()

    print("Predicted Income: >50K" if pred == 1 else "Predicted Income: <=50K")

# Run prediction
predict_new_input(model, label_encoders)
```
# output:
<img width="1015" height="217" alt="image" src="https://github.com/user-attachments/assets/05390589-0bd4-48f1-8956-8becb7566de5" />
