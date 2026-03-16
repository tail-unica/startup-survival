import pandas as pd
import numpy as np


from sklearn.metrics import  confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import itertools

import torch

country_mapping = {
    "NULL":0,
    "Albania": 1,
    "Andorra": 2,
    "Armenia": 3,
    "Austria": 4,
    "Azerbaijan": 5,
    "Belarus": 6,
    "Belgium": 7,
    "Bosnia and Herzegovina": 8,
    "Bulgaria": 9,
    "Croatia": 10,
    "Cyprus": 11,
    "Czech Republic": 12,
    "Denmark": 13,
    "Estonia": 14,
    "Finland": 15,
    "France": 16,
    "Georgia": 17,
    "Germany": 18,
    "Gibraltar": 19,
    "Greece": 20,
    "Hungary": 21,
    "Iceland": 22,
    "Ireland": 23,
    "Italy": 24,
    "Kazakhstan": 25,
    "Kosovo": 26,
    "Latvia": 27,
    "Liechtenstein": 28,
    "Lithuania": 29,
    "Luxembourg": 30,
    "Malta": 31,
    "Moldova": 32,
    "Monaco": 33,
    "Montenegro": 34,
    "Netherlands": 35,
    "North Macedonia": 36,
    "Norway": 37,
    "Poland": 38,
    "Portugal": 39,
    "Romania": 40,
    "Russia": 41,
    "San Marino": 42,
    "Serbia": 43,
    "Slovakia": 44,
    "Slovenia": 45,
    "Spain": 46,
    "Sweden": 47,
    "Switzerland": 48,
    "Turkey": 49,
    "Ukraine": 50,
    "United Kingdom": 51,
    "Vatican City": 52,
    "Other":53
}
    
group_mapping={
    "NULL":0,
    "Agriculture":1,
    "Apparel and Accessories":2,
    "Capital Markets/Institutions":3,
    "Chemicals and Gases":4,
    "Commercial Banks":5,
    "Commercial Products":6,
    "Commercial Services":7,
    "Commercial Transportation":8,
    "Communications and Networking":9,
    "Computer Hardware":10,
    "Construction (Non-Wood)":11,
    "Consumer Durables":12,
    "Consumer Non-Durables":13,
    "Containers and Packaging":14,
    "Energy Equipment":15,
    "Energy Services":16,
    "Exploration, Production and Refining":17,
    "Forestry":18,
    "Healthcare Devices and Supplies":19,
    "Healthcare Services":20,
    "Healthcare Technology Systems":21,
    "IT Services":22,
    "Insurance":23,
    "Media":24,
    "Metals, Minerals and Mining":25,
    "Other Business Products and Services":26,
    "Other Consumer Products and Services":27,
    "Other Energy":28,
    "Other Financial Services":29,
    "Other Healthcare":30,
    "Other Information Technology":31,
    "Other Materials":32,
    "Pharmaceuticals and Biotechnology":33,
    "Restaurants, Hotels and Leisure":34,
    "Retail":35,
    "Semiconductors":36,
    "Services (Non-Financial)":37,
    "Software":38,
    "Textiles":39,
    "Transportation":40,
    "Utilities":41
}




# Carica e unisce i parametri globali e specifici di un modello dal file config.yaml
def load_config(config,model_name, config_type='default'):
    try:
        full_params = config.get('global_settings', {}).copy()

        model_params = config['models'][model_name]['configurations'][config_type]

        full_params.update(model_params)

        return full_params
        
    except KeyError:
        print(f"Errore: Configurazione '{config_type}' per il modello '{model_name}' non trovata.")
        return None

# Legge un dataset in fomrato CSV dalla cartella 'dataset/processati' gestendo eventuali errori di percorso
def load_dataset(config,dataset_name):
    dt=None
    try:
        dt= pd.read_csv(config['paths'][dataset_name],low_memory=False)

    except: 
        print(f"Errore: Dataset'{dataset_name}' non trovata.")
        return None
    return dt

# Genera una heatmap avanzata della confusion matrix con conteggi, percentuali di riga/colonna e totali marginali
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Calcolo dei totali su righe e colonne
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    grand_total = cm.sum()

    fig, ax = plt.subplots(figsize=(6,5))
    
    # Disabilitazione dell'annotazione automatica e stampa della heatmap
    sns.heatmap(
        df_cm, 
        annot=False,       # <-- importante disabilitare
        fmt='d',
        cmap=sns.light_palette("gray", as_cmap=True), 
        cbar=False,
        linewidths=.5, 
        linecolor='gray',
        ax=ax,
        clip_on=False  
    )
    
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticklabels(class_names, rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xlim(0, n_classes + 1.5)

    # Aggiunta totali di riga (a destra)
    for i in range(n_classes):
        y_center = i + 0.5
        x_pos = n_classes + 0.1
        txt = f"{row_totals[i]}"
        ax.text(
            x_pos, y_center, txt,
            ha='left', va='center',
            fontsize=10, fontweight='bold', color='green'
        )
    
    # Aggiunta totali di colonna (in basso)
    for j in range(n_classes):
        x_center = j + 0.5
        y_pos = n_classes + 0.1
        txt = f"{col_totals[j]}"
        ax.text(
            x_center, y_pos, txt,
            ha='center', va='top',
            fontsize=10, fontweight='bold', color='blue'
        )
    
    # Totale globale (in basso a destra)
    ax.text(
        n_classes + 0.1, n_classes + 0.1,
        f"{grand_total}",
        ha='left', va='top',
        fontsize=10, color='black', fontweight='bold'
    )
    
    # Aggiunta in ogni cella del valore grezzo + 2 righe di percentuali
    for i in range(n_classes):
        for j in range(n_classes):
            cell_value = cm[i, j]
            # Calcolo delle due percentuali
            row_percentage = 100.0 * cell_value / row_totals[i] if row_totals[i] else 0
            col_percentage = 100.0 * cell_value / col_totals[j] if col_totals[j] else 0

            # ---- Valore grezzo (in alto)
            ax.text(
                j + 0.5, 
                i + 0.5,   # centrato verticalmente
                f"{cell_value}", 
                ha='center', va='center',
                fontsize=11, color='black', fontweight='bold'
            )
            # ---- Percentuale di riga (in rosso, sotto)
            ax.text(
                j + 0.5, 
                i + 0.7,   # un po' più giù
                f"{row_percentage:.1f}%", 
                ha='center', va='center', 
                fontsize=10, color='green',fontweight='medium'
            )
            # ---- Percentuale di colonna (in blu, ancora più sotto)
            ax.text(
                j + 0.5, 
                i + 0.9,   # ancora più giù
                f"{col_percentage:.1f}%", 
                ha='center', va='center', 
                fontsize=10, color='blue',fontweight='medium'
            )
    
    # Label assi (centrate sopra il quadrato NxN)
    center_x_fraction = (n_classes/2) / (n_classes + 1.5)
    center_y_fraction = (n_classes/2) / (n_classes+0.5)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    # Modifica delle coordinate delle label
    ax.xaxis.set_label_coords(center_x_fraction, 1.10)  # y > 1 per posizionarla sopra
    ax.yaxis.set_label_coords(-0.15, center_y_fraction)
    
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.85, right=1.2)  # ridotto top per lasciare spazio
    
    return plt

# Crea un grafico a barre orizzontali per visualizzare la Permutation Importance
def plotPermutationImportance(features, importance,max_features=0):
    # Ordinamento degli indici in base all'importanza (descending)
    if(max_features>0):
        idx = np.argsort(importance)[::-1]

        sorted_features = features[idx]
        sorted_importances = importance[idx]

        sorted_features = sorted_features[:max_features]
        sorted_importances = sorted_importances[:max_features]

        sorted_features = sorted_features[::-1]
        sorted_importances = sorted_importances[::-1]
    else:
        idx = np.argsort(importance)

        sorted_features = features[idx]
        sorted_importances = importance[idx]

    num_features = len(sorted_features)

    # Impostazione di una figura con dimensioni proporzionate al numero di caratteristiche
    # Ad esempio, 0.3 pollici di altezza per ogni caratteristica
    fig_height = max(4, 0.1 * num_features)  # Minimo 6 pollici di altezza
    fig, ax = plt.subplots(figsize=(5, fig_height))  # Larghezza 10 pollici, altezza calcolata

    # Creazione grafico a barre orizzontali
    ax.barh(sorted_features, sorted_importances, color='skyblue')

    # Etichette degli assi e il titolo
    ax.set_xlabel('Importanza', fontsize=10)
    ax.set_ylabel('Caratteristiche', fontsize=10)

    ax.set_yticklabels(sorted_features, fontsize=8)

    plt.tight_layout()

    return plt

# Visualizza le curve di sopravvivenza per una o più features
def plot_custom_survival_analysis(model, x_baseline, feature_names, var_configs, 
                                 manual_combos=None, max_time=15):
    """
    Genera grafici di sopravvivenza variando una o più feature.
    
    Args:
        model: Il modello RandomSurvivalForest addestrato.
        x_baseline: Vettore (array) con i valori medi/mediani di riferimento.
        feature_names: Lista o Index dei nomi delle colonne.
        var_configs: Dizionario { 'NomeVariabile': [lista_valori_da_testare] }.
        manual_combos: Lista di tuple con combinazioni specifiche. Se None, 
                       il codice genera tutte le combinazioni possibili (prodotto cartesiano).
        max_time: Limite asse X per il grafico.
    """
    feature_list = list(feature_names)
    var_names = list(var_configs.keys())
    
    # Se non vengono fornite combo manuali, calcoliamo il prodotto cartesiano
    if manual_combos is None:
        list_of_values = [var_configs[var] for var in var_names]
        combos_to_plot = list(itertools.product(*list_of_values))
    else:
        combos_to_plot = manual_combos

    plt.figure(figsize=(10, 7))
    
    for combo in combos_to_plot:
        x_test = x_baseline.copy()
        labels = []
        
        # Aggiornamento del vettore x_test per la combinazione corrente
        for var_name, val in zip(var_names, combo):
            if var_name in feature_list:
                col_idx = feature_list.index(var_name)
                x_test[col_idx] = val
                
                # Formattazione label (senza decimali se piccolo intero)
                fmt = ".0f" if (val <= 10 and val % 1 == 0) else ".2f"
                labels.append(f"{var_name}={val:{fmt}}")
        
        # Predizione
        surv_fun = model.predict_survival_function([x_test], return_array=False)[0]
        
        # Filtraggio temporale
        times = surv_fun.x
        surv_probs = surv_fun.y
        mask = times <= max_time
        
        plt.step(times[mask], surv_probs[mask], where='post', label=", ".join(labels))

    plt.xlabel("Tempo")
    plt.ylabel("Prob. di sopravvivenza")
    plt.legend(
        loc='lower left', 
        bbox_to_anchor=(0, -0.5), )
    plt.subplots_adjust(bottom=0.3)
    return plt

# Wrapper per integrare un modello PyTorch nell'ecosistema Scikit-Learn (compatibilità con matrici di confusione e metriche)
class PyTorchEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Non allena davvero il modello (perché l'hai già allenato fuori),
        ma imposta almeno self.classes_ per compatibilità con sklearn.
        """
        self.classes_ = np.unique(y)  # imposta le classi
        return self
    
    def predict(self, X):
        """
        Effettua la predizione usando PyTorch.
        """
        self.pytorch_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.pytorch_model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
    
models_map = {
    0: {
        "name": "decision_tree",
        "configs": {0: "default", 1: "bilanciato"}
    },
    1: {
        "name": "random_forest",
        "configs": {0: "default", 1: "ottimizzato"}
    },
    2: {
        "name": "neural_network",
        "configs": {0: "default", 1: "ottimizzato"}
    }
}

def validate_and_assign(m_idx, c_idx):
    # Controllo esistenza modello
    if m_idx not in models_map:
        raise ValueError(f"Indice modello {m_idx} non valido. Scegli tra: {list(models_map.keys())}")
    
    selected_model = models_map[m_idx]
    
    # Controllo esistenza configurazione per quel modello
    if c_idx not in selected_model["configs"]:
        raise ValueError(
            f"Indice config {c_idx} non valido per {selected_model['name']}. "
            f"Scegli tra: {list(selected_model['configs'].keys())}"
        )
    
    # Assegnazione variabili finali
    model_name = selected_model["name"]
    config_type = selected_model["configs"][c_idx]
    
    return model_name, config_type