from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pokemon_util as calc
from moves_sample import damaging_moves, all_moves, non_damaging_moves
from matplotlib.colors import ListedColormap

# abilities,against_bug,dark,dragon,electric,fairy,fight,fire,flying,ghost,grass,ground,ice,normal,poison,psychic,rock,steel,water,attack,base_egg_steps,base_happiness,base_total,capture_rate,classfication,defense,experience_growth,height_m,hp,japanese_name,name,percentage_male,pokedex_number,sp_attack,sp_defense,speed,type1,type2,weight_kg,generation,is_legendary

pd.set_option('display.max_columns', None)
csv = pd.read_csv("pokemon/pokemonlist.csv")
enc_abs = pd.get_dummies(csv["abilities"])
types_dummies = (pd.get_dummies(csv["type1"]).fillna(0) + pd.get_dummies(csv["type2"]).fillna(0)).astype(float)
csv = pd.concat([csv, enc_abs, types_dummies], axis= 1)

def squad_rand():

    unique_all_moves = list(dict.fromkeys(all_moves))
    unique_non_damaging = [m for m in non_damaging_moves if m in unique_all_moves]

    n_pokemon = 3
    indexes = np.random.choice(len(csv), n_pokemon, replace=False)

    main_cols = ["hp", "type1", "type2", "attack", "sp_attack", "speed"]
    squad = csv.loc[indexes, main_cols].copy()
    squad["mean_atk"] = (squad["attack"] + squad["sp_attack"]) / 2
    squad_abilities = enc_abs.loc[indexes].copy()
    squad_abilities = squad_abilities.loc[:, (squad_abilities == 1).any()]
    types_squad = types_dummies.loc[indexes].copy()

    squad = pd.concat(
        [squad.reset_index(drop=True),
         squad_abilities.reset_index(drop=True),
         types_squad.reset_index(drop=True)],
        axis=1
    )

    squad = squad.loc[:, ~squad.columns.duplicated()]
    max_moves = 4
    moves_matrix = pd.DataFrame(0, index=squad.index, columns=unique_all_moves)

    for i in squad.index:
        if np.random.rand() < 0.7:
            # senza mosse dannose
            chosen_moves = np.random.choice(unique_non_damaging, size=max_moves, replace=False)
        else:
            chosen_moves = np.random.choice(unique_all_moves, size=max_moves, replace=False)

        moves_matrix.loc[i, chosen_moves] = 1
    squad = pd.concat([squad, moves_matrix], axis=1)
    squad = squad.loc[:, ~squad.columns.duplicated()]

    return squad


def keep_or_not(A, B):
    eff = calc.type_effectiveness(A["type1"], A["type2"], B["type1"], B["type2"])

    if eff >= 2:
        return True          
    if eff == 1:
        return np.random.rand() < 0.5  
    if eff == 0.5:
        return np.random.rand() < 0.2   
    return False            


def turn_starter(A, B):
    return A["speed"] > B["speed"]

def turn_solver(idx1, idx2, sqd1, sqd2):

    pk1 = sqd1.loc[idx1]
    pk2 = sqd2.loc[idx2]

    one_condition = keep_or_not(pk1, pk2)
    two_condition = keep_or_not(pk2, pk1)

    turn_A = one_condition
    turn_B = two_condition
    
    if turn_A and turn_B:
        start_val = turn_starter(pk1, pk2)
    elif turn_A and not turn_B:
        start_val = True
    elif turn_B and not turn_A:
        start_val = False
    else:
        return False, True

    ability_cols_pk1 = [col for col in sqd1.columns if col in damaging_moves and col not in ["hp","attack","sp_attack","speed","type1","type2","mean_atk"]]
    ability_cols_pk2 = [col for col in sqd2.columns if col in damaging_moves and col not in ["hp","attack","sp_attack","speed","type1","type2","mean_atk"]]
    pk1_damaging = any((pk1[col] == 1).any() for col in ability_cols_pk1)
    pk2_damaging = any((pk2[col] == 1).any() for col in ability_cols_pk2)


    if not pk1_damaging and not pk2_damaging:
        return False, False

    if start_val:
        if pk1_damaging and turn_A:
            sqd2.loc[idx2, "hp"] -= pk1["mean_atk"] // 2
            if sqd2.loc[idx2, "hp"] <= 0:
                return True, False
        else:
            one_condition = False

        if pk2_damaging and turn_B:
            sqd1.loc[idx1, "hp"] -= pk2["mean_atk"] // 2
            if sqd1.loc[idx1, "hp"] <= 0:
                return False, True
        else:
            two_condition = False

    else:
        if pk2_damaging and turn_B:
            sqd1.loc[idx1, "hp"] -= pk2["mean_atk"] // 2
            if sqd1.loc[idx1, "hp"] <= 0:
                return False, True
        else:
            two_condition = False

        if pk1_damaging and turn_A:
            sqd2.loc[idx2, "hp"] -= pk1["mean_atk"] // 2
            if sqd2.loc[idx2, "hp"] <= 0:
                return True, False
        else:
            one_condition = False

    return one_condition, two_condition



def battle_sim(sqd1, sqd2):

    idx1 = sqd1[sqd1["hp"] > 0].index[0]
    idx2 = sqd2[sqd2["hp"] > 0].index[0]

    turns = 0
    prev_state = None
    stall_counter = 0

    while (sqd1["hp"] > 0).any() and (sqd2["hp"] > 0).any():

        current_state = (
            tuple(sqd1["hp"]),
            tuple(sqd2["hp"]),
        )

        if current_state == prev_state:
            stall_counter += 1
        else:
            stall_counter = 0

        prev_state = current_state

        if stall_counter > 50:
            return 0

        one_cond, two_cond = turn_solver(idx1, idx2, sqd1, sqd2)

        alive1 = sqd1[sqd1["hp"] > 0]
        alive2 = sqd2[sqd2["hp"] > 0]

        def squad_can_act(squad, enemy):
            for idx in squad.index:
                pk = squad.loc[idx]
                if calc.type_effectiveness(pk["type1"], pk["type2"], enemy["type1"], enemy["type2"]) >= 1:
                    return True
                ability_cols = [c for c in squad.columns if c in damaging_moves]
                if any(pk[c] == 1 for c in ability_cols):
                    return True
            return False

        can1 = squad_can_act(alive1, sqd2.loc[idx2])
        can2 = squad_can_act(alive2, sqd1.loc[idx1])

        if not can1 and not can2:
            return 0

        turns += 1
        if turns > 200:
            return 0

        if not one_cond:
            alive_pk1 = sqd1[sqd1["hp"] > 0]
            preferred_types = calc.type_chooser(sqd2.loc[idx2]["type1"], sqd2.loc[idx2]["type2"])
            preferred_types = [t for t in preferred_types if t in sqd1.columns]
            if preferred_types:
                candidates = alive_pk1[alive_pk1[preferred_types].any(axis=1)]
            else:
                candidates = pd.DataFrame()
            if not candidates.empty:
                idx1 = candidates.index[0]
            else:
                others = [i for i in alive_pk1.index if i != idx1]
                idx1 = others[0] if others else idx1

        if not two_cond:
            alive_pk2 = sqd2[sqd2["hp"] > 0]
            preferred_types = calc.type_chooser(sqd1.loc[idx1]["type1"], sqd1.loc[idx1]["type2"])
            preferred_types = [t for t in preferred_types if t in sqd2.columns]
            if preferred_types:
                candidates = alive_pk2[alive_pk2[preferred_types].any(axis=1)]
            else:
                candidates = pd.DataFrame()
            if not candidates.empty:
                idx2 = candidates.index[0]
            else:
                others = [i for i in alive_pk2.index if i != idx2]
                idx2 = others[0] if others else idx2

        if sqd1.loc[idx1, "hp"] <= 0:
            alive1 = sqd1[sqd1["hp"] > 0]
            if alive1.empty:
                return 2
            idx1 = alive1.index[0]

        if sqd2.loc[idx2, "hp"] <= 0:
            alive2 = sqd2[sqd2["hp"] > 0]
            if alive2.empty:
                return 1
            idx2 = alive2.index[0]

    return 1 if (sqd1["hp"] > 0).any() else 2


if __name__ == "__main__":

    data = []

    for _ in range(1000): # ci mette tipo 5 min
        sqd1 = squad_rand()
        sqd2 = squad_rand()
        winner = battle_sim(sqd1.copy(), sqd2.copy())
        # estrai feature aggregate
        row = {
            "sqd1_hp_mean": sqd1["hp"].mean(),
            "sqd1_attack_mean": sqd1["attack"].mean(),
            "sqd1_speed_mean": sqd1["speed"].mean(),
            "sqd2_hp_mean": sqd2["hp"].mean(),
            "sqd2_attack_mean": sqd2["attack"].mean(),
            "sqd2_speed_mean": sqd2["speed"].mean(),
            "diff_hp": sqd1["hp"].mean() - sqd2["hp"].mean(),
            "diff_attack": sqd1["attack"].mean() - sqd2["attack"].mean(),
            "diff_speed": sqd1["speed"].mean() - sqd2["speed"].mean(),
            "winner": winner
        }
        data.append(row)

    df = pd.DataFrame(data)
    X = df.drop("winner", axis=1)
    y = df["winner"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

    model = Pipeline([
        ("scaler", StandardScaler()),
        #("pca", PCA(n_components=10)), # quante feature dopo compressione PCA (dimensionality reductor)
        ("clf", LogisticRegression())
    ])

    model.fit(X_train, y_train)
    #probs = model.predict_proba(X_test)[:, 1]
    my_cmap = ListedColormap(["green", "blue", "red"]) 
    plt.figure(figsize=(8,6)) 
    sc = plt.scatter(
    df["diff_speed"],
    df["diff_attack"], 
    #df["diff_hp"], 
    c=df["winner"], 
    cmap=my_cmap, 
    edgecolor='k',
    alpha=0.8
    )
    print(df["winner"].value_counts())

    plt.colorbar(sc, ticks=[0, 1, 2], label="Vincitore (0 = stallo, 1 = sqd1, 2 = sqd2)")
    plt.xlabel("Differenza velocità media (sqd1 - sqd2)")
    plt.ylabel("Differenza attacco medio (sqd1 - sqd2)")
    plt.title("Differenze stats tra squadre vs vincitore")
    plt.grid(True)
    plt.show()

