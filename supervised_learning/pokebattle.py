from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
    
type_chart = {
    "Normal":   {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
    "Fighting": {"Normal": 2, "Ice": 2, "Rock": 2, "Dark": 2, "Steel": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Ghost": 0, "Fairy": 0.5},
    "Poison":   {"Grass": 2, "Fairy": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2, "Steel": 0.5, "Fairy": 0.5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
    "Dragon":   {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
    "Dark":     {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5, "Fairy": 2},
    "Fairy":    {"Fire": 0.5, "Fighting": 2, "Poison": 0.5, "Dragon": 2, "Dark": 2, "Steel": 0.5}
}


def type_effectiveness(attacker_type1, attacker_type2, defender_type1, defender_type2):
    """
    calcolo il moltiplicatore totale dei tipi di un Pokémon A contro un Pokémon B
    """

    mult1 = type_chart.get(attacker_type1, {}).get(defender_type1, 1)
    mult2 = type_chart.get(attacker_type1, {}).get(defender_type2, 1) 

    mult3 = type_chart.get(attacker_type2, {}).get(defender_type1, 1) 
    mult4 = type_chart.get(attacker_type2, {}).get(defender_type2, 1) 

    return mult1 * mult2 * mult3 * mult4

def calculator():
    battles = [] 
    for i, j in combinations(range(len(csv)), 2): 
        A = csv.loc[i] 
        B = csv.loc[j] 
        effA = type_effectiveness(A["Type1"], A["Type2"], B["Type1"], B["Type2"]) 
        powerA = A["Total"] * effA 
        effB = type_effectiveness(B["Type1"], B["Type2"], A["Type1"], A["Type2"]) 
        powerB = B["Total"] * effB 
        probA = powerA / (powerA + powerB) if (powerA + powerB) != 0 else 0.5
        battles.append({ "A_index": i, "B_index": j, "probA": probA }) 
    df_battles = pd.DataFrame(battles)
    return df_battles
    
pd.set_option('display.max_columns', None)
csv = pd.read_csv("pokemon/National_Pokedex.csv")
df_dummies = (pd.get_dummies(csv["Type1"]).fillna(0) + pd.get_dummies(csv["Type2"]).fillna(0)).astype(float) # one hot encoding
csv = pd.concat([csv, df_dummies], axis=1)

cols = list(csv.loc[:, "HP":"Spd"].columns) + list(df_dummies.columns)

#tip_advantage = csv["Total"] * type_effectiveness(1, 1, 1, 1)

X = []
y = []

for row in calculator().itertuples():
    A = csv.loc[row.A_index]
    B = csv.loc[row.B_index]

    # Feature = differenza delle statistiche
    features = list((A[cols] - B[cols]).values)
    effA = type_effectiveness(A["Type1"], A["Type2"], B["Type1"], B["Type2"])
    effB = type_effectiveness(B["Type1"], B["Type2"], A["Type1"], A["Type2"])
    features.append(effA-effB)
    X.append(features)
    y.append(1 if row.probA > 0.5 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)), # quante feature dopo compressione PCA (dimensionality reductor)
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(
    y_test, probs, n_bins=10, strategy="uniform"
)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, alpha=0.5)
plt.plot([0,1],[0,1], 'r--')  # ideal line y=x
plt.xlabel("True freq")
plt.ylabel("Predicted freq")
plt.title("curva")
plt.show()