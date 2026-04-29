from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_correlation_heatmap(df, exclude_cols=['CompanyID', 'Target']):
    corr_matrix = df.drop(exclude_cols, axis=1).corr()
    
    plt.figure(figsize=(30, 20))
    ax = sns.heatmap(data=corr_matrix, cmap='YlGnBu', annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
    
    return corr_matrix


def get_high_correlations(corr_matrix, threshold=0.90):
    sol = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    sol.columns = ['Variabile_1', 'Variabile_2', 'Correlazione']

    mask = sol['Correlazione'].abs() >= threshold
    risultato = sol[mask].sort_values(by='Correlazione', ascending=False)
    
    return risultato

def to_tensors(X, y):
                y_np = y.values if hasattr(y, 'values') else y
                return (
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
                )
def get_probs(loader, model, device):
                all_probs, all_labels = [], []
                with torch.no_grad():
                    for X_batch, y_batch in loader:
                        X_batch = X_batch.to(device)
                        logits = model(X_batch)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_probs.extend(probs)
                        all_labels.extend(y_batch.numpy())
                return np.array(all_probs).flatten(), np.array(all_labels).flatten().astype(int)
