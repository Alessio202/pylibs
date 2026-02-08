import os
import random
import numpy as np
import pygame
from pettingzoo import ParallelEnv
from gymnasium import spaces
import supersuit as ss
from stable_baselines3 import PPO

# ==========================================================
# ADVANCED MULTI-AGENT SOCCER ENV (PETTINGZOO)
# ==========================================================

class SoccerParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "soccer_pro_v2"}

    def __init__(self, render_mode=None):
        self.width, self.height = 1000, 600
        self.player_radius, self.ball_radius = 9, 6
        self.anchor_recovery_timer = 0
        self.post_kick_grace_period = 0 
        self.max_speed = 1.45  # Increased again for better ball chasing
        self.friction = 0.77  
        self.ball_speed = 2.5
        self.stun_timer = np.zeros(22)
        self.agents = ["blue", "red"]
        self.possible_agents = self.agents[:]
        self.is_goal_kick = False
        self.goal_kick_timer = 0
        self.goal_kick_team = None
        self.free_kick_active = False
        self.free_kick_timer = 0
        self.free_kick_taker_idx = None
        self.free_kick_team = None
        self.mandatory_wait = 0

        self.action_spaces = {
            "blue": spaces.MultiDiscrete([7]*11),
            "red": spaces.MultiDiscrete([7]*11)
        }

        self.observation_spaces = {
            "blue": spaces.Box(low=0, high=1, shape=(48,), dtype=np.float32),
            "red": spaces.Box(low=0, high=1, shape=(48,), dtype=np.float32)
        }

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.elo_blue = 1000
        self.elo_red = 1000

        # NUOVO: Definisci formazioni base (4-3-3)
        self._init_formations()

        self.reset()

    def _init_formations(self):
        """Define home positions for each player in 4-3-3 formation"""
        h = self.height
        w = self.width
        
        # Blue team (attacking right) - positions as (x, y)
        self.blue_formation_433 = {
            0: (80, h/2),           # GK
            1: (200, h*0.20),       # LB (Left Back)
            2: (200, h*0.40),       # CB (Center Back)
            3: (200, h*0.60),       # CB
            4: (200, h*0.80),       # RB (Right Back)
            5: (w*0.35, h*0.30),    # LM (Left Mid)
            6: (w*0.35, h*0.50),    # CM (Center Mid)
            7: (w*0.35, h*0.70),    # RM (Right Mid)
            8: (w*0.65, h*0.25),    # LW (Left Wing)
            9: (w*0.70, h*0.50),    # ST (Striker)
            10: (w*0.65, h*0.75),   # RW (Right Wing)
        }
        
        # Blue defensive 4-5-1 (when winning)
        self.blue_formation_451 = {
            0: (80, h/2),
            1: (200, h*0.20),
            2: (200, h*0.40),
            3: (200, h*0.60),
            4: (200, h*0.80),
            5: (w*0.40, h*0.20),    # Wide mids drop back
            6: (w*0.35, h*0.35),
            7: (w*0.35, h*0.50),
            8: (w*0.35, h*0.65),
            9: (w*0.40, h*0.80),
            10: (w*0.60, h*0.50),   # Lone striker
        }
        
        # Red team (attacking left) - mirror positions
        self.red_formation_433 = {i: (w - x, y) for i, (x, y) in self.blue_formation_433.items()}
        self.red_formation_451 = {i: (w - x, y) for i, (x, y) in self.blue_formation_451.items()}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.players_pos = np.zeros((22,2), dtype=np.float32)
        self.players_vel = np.zeros((22,2), dtype=np.float32)
        self.ball_pos = np.array([self.width/2, self.height/2], dtype=np.float32)
        self.ball_vel = np.zeros(2)
        self.ball_spin = 0.0
        self.stun_timer = np.zeros(22)
        self.is_goal_kick = False
        self.goal_kick_timer = 0
        self.goal_kick_team = None
        self.dribble_timer = np.zeros(22)  # Quanto tempo dura l'essere "instoppabile"
        self.slow_timer = np.zeros(22)     # Timer per gli avversari saltati
        self.blue_score = 0
        self.red_score = 0
        self.last_touch_team = None
        self.same_team_pass_count = 0

        self._reset_positions()
        return self._get_obs(), {}


    def step(self, actions):
        # --- 1. INIZIALIZZAZIONE DIZIONARI (Essenziale per evitare errori) ---
        rewards = {"blue": 0.0, "red": 0.0}
        terminations = {"blue": False, "red": False}
        truncations = {"blue": False, "red": False}
        infos = {"blue": {}, "red": {}}
        self.mandatory_wait -= 1

        # --- 2. GESTIONE PYGAME ---
        if self.render_mode == "human" and self.screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        
        if self.free_kick_active:
            self.ball_vel = np.zeros(2) # Blocca la palla durante l'attesa
            self.free_kick_timer -= 1
            team_acts = actions[self.free_kick_team]
            taker_rel_idx = self.free_kick_taker_idx % 11
            
            if self.free_kick_timer <= 0:
                auto_action = 2
                self._execute_free_kick_ml(auto_action)
                self.stun_timer[:] = 0 
                self.free_kick_active = False
                
                if self.render_mode == "human": self.render()
                return self._get_obs(), rewards, terminations, truncations, infos

        # --- 3. GESTIONE RINVIO DAL FONDO (PAUSA E LANCIO) ---
        if self.is_goal_kick:
            self.goal_kick_timer -= 1
            
            # BLOCCA I GIOCATORI: Nessun movimento durante l'attesa
            for i in range(22):
                self.players_vel[i] = np.zeros(2)

            if self.goal_kick_timer > 0:
                if self.render_mode == "human":
                    self.render() 
                return self._get_obs(), rewards, terminations, truncations, infos
            else:
                self.is_goal_kick = False
                self.post_kick_grace_period = 360
                is_corner = self.ball_pos[0] < 50 or self.ball_pos[0] > self.width - 50

                if is_corner:
                    # Profondità variabile nell'area
                    depth = random.uniform(120, 220)

                    if self.goal_kick_team == "blue":
                        target_x = self.width - depth
                    else:
                        target_x = depth

                    # Angolo variabile più ampio
                    target_y = self.height / 2 + random.uniform(-150, 150)

                    power_mult = random.uniform(3.8, 4.5)
                else:
                    target_x = self.width * 0.7 if self.goal_kick_team == "blue" else self.width * 0.3
                    target_y = self.height / 2 + random.uniform(-100, 100)
                    power_mult = 6.2 

                direction = np.array([target_x, target_y]) - self.ball_pos 
                dist = np.linalg.norm(direction)
                
                # LANCIO:
                self.ball_vel = (direction / (dist + 1e-6)) * (self.ball_speed * power_mult) 
                
                # EFFETTO A GIRO PER L'ANGOLO:
                if is_corner:
                    # Se batte da sopra, curva verso il basso e viceversa
                    side = -0.08 if self.ball_pos[1] < self.height/2 else 0.12
                    self.ball_spin = side
                else:
                    self.ball_spin = 0.0 

                self.last_touch_team = self.goal_kick_team
                
                # Fondamentale: azzeriamo le velocità post-lancio per una ripartenza pulita
                for i in range(22):
                    self.players_vel[i] = np.zeros(2)

                return self._get_obs(), rewards, terminations, truncations, infos

        # --- 4. LOGICA MOVIMENTO E FISICA ---
        prev_ball_x = self.ball_pos[0]

        for i in range(22):
            # --- GESTIONE DRIBBLING ---
            is_dribbling = self._handle_dribbling(i) # Questa funzione deve gestire dribble_timer e ball_pos
            
            # --- MOVIMENTO IA ---
            team_key = "blue" if i < 11 else "red"
            act_idx = i if i < 11 else i - 11
            self._move_player(i, actions[team_key][act_idx])

            # --- APPLICAZIONE VINCOLI (Le nostre funzioni!) ---
            self._apply_positional_anchor(i)
            self._apply_teammate_spacing(i)

            # --- GESTIONE RALLENTAMENTO (SLOW) ---
            if self.slow_timer[i] > 0:
                progress = 1.0 - (self.slow_timer[i] / 120.0)
                speed_factor = 0.1 + (0.9 * progress)
                self.players_vel[i] *= speed_factor
                self.slow_timer[i] -= 1

            # --- ESECUZIONE FISICA POSIZIONE ---
            # Se è stordito (stun), non si muove
            if self.stun_timer[i] > 0:
                self.players_vel[i] = np.zeros(2)
                self.stun_timer[i] -= 1
            
            self.players_pos[i] += self.players_vel[i]

            # --- CALCIO (Solo se non sta dribblando) ---
            if not is_dribbling and actions[team_key][act_idx] >= 5:
                self._kick(i, 0 if i < 11 else 1)

        # Fisica Palla (Effetto Magnus + Attrito)
        if np.linalg.norm(self.ball_vel) > 0.5:
            perp_vel = np.array([-self.ball_vel[1], self.ball_vel[0]])
            
            # RIDUZIONE DRASTICA: Da 0.06 a 0.025 per tiri più tesi e meno "a cerchio"
            # Moltiplichiamo per 0.8 se la velocità è molto alta per evitare curve assurde
            speed_damp = 0.8 if np.linalg.norm(self.ball_vel) > 10 else 1.0
            
            self.ball_vel += (perp_vel * self.ball_spin) * (0.025 * speed_damp) 
            
            # Lo spin deve esaurirsi molto più velocemente (0.98 -> 0.94)
            self.ball_spin *= 0.94
        
        # Nel metodo step(), se la palla è in volo da punizione:
        if hasattr(self, 'ball_curve_effect') and np.linalg.norm(self.ball_vel) > 5:
            # Applica forza perpendicolare alla velocità (effetto a giro)
            perp_vel = np.array([-self.ball_vel[1], self.ball_vel[0]])
            self.ball_vel += perp_vel * self.ball_curve_effect

        self.ball_pos += self.ball_vel
        self.ball_vel *= 0.97
        
        # --- 5. LOGICA GOAL ---
        # --- 5. LOGICA GOAL (MODIFICATA) ---
        # --- 5. LOGICA GOAL ---
        if 250 < self.ball_pos[1] < 350:
            goal_scored = False
            
            # Goal Squadra Rossa (Palla oltre il bordo sinistro)
            if self.ball_pos[0] < 0:
                self.red_score += 1
                rewards["red"] += 120
                rewards["blue"] -= 120
                goal_scored = True
                
            # Goal Squadra Blu (Palla oltre il bordo destro)
            elif self.ball_pos[0] > self.width:
                self.blue_score += 1
                rewards["blue"] += 120
                rewards["red"] -= 120
                goal_scored = True

            if goal_scored:
                # 1. Reset Palla al centro
                self.anchor_recovery_timer = 600  # 10 secondi a 60fps
                self.ball_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
                self.ball_vel = np.zeros(2)
                self.ball_spin = 0.0
                
                # 2. Reset stati fisici dei giocatori per evitare che rimangano bloccati/grigi
                self.dribble_timer.fill(0)
                self.slow_timer.fill(0)
                self.stun_timer.fill(0)
                self._reset_positions()
                
                # 3. Azzera le velocità correnti
                # In questo modo il _move_player e l'anchor del frame successivo 
                # li faranno camminare verso le posizioni di kickoff senza inerzie vecchie
                for i in range(22):
                    self.players_vel[i] = np.zeros(2)
                
                # 4. Opzionale: un piccolo periodo di grazia per rientrare in formazione
                self.post_kick_grace_period = 300 # 1 secondo di riposizionamento "morbido"

                return self._get_obs(), rewards, terminations, truncations, infos

        # Controllo bordi e rinvio dal fondo (se non è goal)
        self._handle_ball_bounds(rewards)

        # --- 6. COLLISIONI E REWARDS ---
        for i in range(22):
            self._collision(i, rewards)
        
        self._handle_tackles()

        # Reward Avanzamento
        delta_x = self.ball_pos[0] - prev_ball_x
        rewards["blue"] += delta_x * 0.03
        rewards["red"] -= delta_x * 0.03

        # Reward Proximity (Spaziatura Team)
        for team_name, indices in {"blue": range(11), "red": range(11, 22)}.items():
            dists = [(i, np.linalg.norm(self.ball_pos - self.players_pos[i])) for i in indices]
            closest_idx, min_dist = min(dists, key=lambda x: x[1])
            
            if min_dist < 300:
                rewards[team_name] += (300 - min_dist) * 0.05
            
            for i, dist in dists:
                if i != closest_idx and dist < 80:
                    rewards[team_name] -= 0.8

        # --- 7. RENDERING ---
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), rewards, terminations, truncations, infos

    def _move_player(self, idx, act):
        if self.free_kick_active:
            is_opp_gk = (self.free_kick_team == "blue" and idx == 11) or (self.free_kick_team == "red" and idx == 0)
            if is_opp_gk:
                # Forza la posizione X sulla linea di porta 
                self.players_pos[idx][0] = 950 if idx == 11 else 50
                # Insegui la palla solo in verticale (Y) lungo la porta
                target_y = np.clip(self.ball_pos[1], 250, 350) 
                # ... resto del codice per il movimento verticale ...
                move_speed = 2.5 # Velocità laterale sulla linea
                
                if self.players_pos[idx][1] < target_y - 5:
                    self.players_vel[idx][1] = move_speed
                elif self.players_pos[idx][1] > target_y + 5:
                    self.players_vel[idx][1] = -move_speed
                else:
                    self.players_vel[idx][1] = 0
                
                # Resetta velocità X per sicurezza
                self.players_vel[idx][0] = 0
                return # Esci, non processare altre azioni
        
        if self.stun_timer[idx] > 0:
            self.stun_timer[idx] -= 1
            self.players_vel[idx] *= 0.5 # Rallenta fino a fermarsi
            self.players_pos[idx] += self.players_vel[idx]
            return # Salta l'input dell'azione
        vel = np.zeros(2)
        if act == 1: vel[1] = -self.max_speed
        elif act == 2: vel[1] = self.max_speed
        elif act == 3: vel[0] = -self.max_speed
        elif act == 4: vel[0] = self.max_speed
        elif act >= 5:
            dir_to_ball = self.ball_pos - self.players_pos[idx]
            dist = np.linalg.norm(dir_to_ball)
            if dist > 0:
                vel = (dir_to_ball / dist) * self.max_speed

        self.players_vel[idx] = self.players_vel[idx]*self.friction + vel*(1-self.friction)
        self.players_pos[idx] += self.players_vel[idx]

        # NUOVO ORDINE: Prima ancoriamo alla posizione, POI evitiamo collisioni
        self._apply_positional_anchor(idx)
        self._apply_teammate_spacing(idx)
        
        # Hard bounds - never leave the pitch
        old_pos = self.players_pos[idx].copy()
        self.players_pos[idx][0] = np.clip(self.players_pos[idx][0], 0, self.width)
        self.players_pos[idx][1] = np.clip(self.players_pos[idx][1], 0, self.height)

        if self.players_pos[idx][0] != old_pos[0]:
            self.players_vel[idx][0] = 0
        if self.players_pos[idx][1] != old_pos[1]:
            self.players_vel[idx][1] = 0

    def _handle_dribbling(self, idx):
        # Se il giocatore è stordito, non può dribblare
        if self.stun_timer[idx] > 0:
            return False

        # Se il giocatore è già in dribbling, prosegui
        if self.dribble_timer[idx] > 0:
            self.dribble_timer[idx] -= 1
            # Durante il dribbling, la palla è "incollata"
            self.ball_pos = self.players_pos[idx].copy() + (self.players_vel[idx] * 0.5)
            self.ball_vel = self.players_vel[idx].copy()
            return True

        # Chi ha la palla?
        dist_to_ball = np.linalg.norm(self.players_pos[idx] - self.ball_pos)
        if dist_to_ball < 25: 
            rel = idx % 11
            dribble_chance = 0.006 if rel >= 8 else 0.001 
            
            opp_start = 11 if idx < 11 else 0
            for i in range(opp_start, opp_start + 11):
                # Un giocatore stordito non può essere "saltato" di nuovo con logica dribbling
                if self.stun_timer[i] > 0: continue
                
                dist_opp = np.linalg.norm(self.players_pos[idx] - self.players_pos[i])
                
                if dist_opp < 60 and random.random() < dribble_chance:
                    self.dribble_timer[idx] = 60 
                    self.slow_timer[i] = 120 
                    return True
        return False
    
    def _handle_tackles(self):
        for i in range(11): # Team Blue
            for j in range(11, 22): # Team Red
                # Se uno dei due sta dribblando, è immune ai tackle
                if self.dribble_timer[i] > 0 or self.dribble_timer[j] > 0:
                    continue
                
                # Se uno dei due è già a terra, non c'è un nuovo scontro
                if self.stun_timer[i] > 0 or self.stun_timer[j] > 0:
                    continue
                
                dist_players = np.linalg.norm(self.players_pos[i] - self.players_pos[j])
                
                if dist_players < (self.player_radius * 2.5):
                    dist_ball_i = np.linalg.norm(self.players_pos[i] - self.ball_pos)
                    dist_ball_j = np.linalg.norm(self.players_pos[j] - self.ball_pos)
                    
                    if dist_ball_i < 30 and dist_ball_j < 30:
                        if random.random() < 0.05: 
                            # 1. Determiniamo chi subisce il fallo (la vittima)
                            victim = random.choice([i, j, "both"])
                            
                            # 2. Logica Punizione (Solo se non è un "both")
                            if victim != "both":
                                v_pos_x = self.players_pos[victim][0]
                                # Controlliamo se la vittima è nella metà campo avversaria
                                # Blue attacca a destra (x > 500), Red a sinistra (x < 500)
                                is_foul_blue = (victim < 11 and 625 < v_pos_x < 900)
                                is_foul_red = (victim >= 11 and 100 < v_pos_x < 375)

                                if (is_foul_blue or is_foul_red) and random.random() < 0.99:
                                    # Attiva il setup della punizione che abbiamo visto prima
                                    self._setup_free_kick(victim_idx=victim)
                                    return # Interrompiamo il frame per gestire la punizione

                            # 3. Effetto caduta normale (se non è punizione o se è scontro fortuito)
                            if victim == i or victim == "both":
                                self.stun_timer[i] = 60 
                                self.players_vel[i] = np.zeros(2)
                            if victim == j or victim == "both":
                                self.stun_timer[j] = 60
                                self.players_vel[j] = np.zeros(2)
                            
                            self.ball_vel += np.random.uniform(-3, 3, 2)

    def _setup_free_kick(self, victim_idx):
        self.free_kick_active = True
        self.free_kick_timer = 180 # 3 secondi di attesa (60fps)
        self.free_kick_team = "blue" if victim_idx < 11 else "red"
        self.free_kick_taker_idx = victim_idx
        self.mandatory_wait = 120
        
        # Attiva il timer di recupero per far tornare gli altri in posizione lentamente
        self.anchor_recovery_timer = 600 # 10 secondi di rientro graduale
        
        self.ball_vel = np.zeros(2)
        self.ball_spin = 0
        
        my_start = 0 if self.free_kick_team == "blue" else 11
        opp_start = 11 if self.free_kick_team == "blue" else 0
        opp_goal_x = 950 if self.free_kick_team == "blue" else 50
        target_goal_x = 1000 if self.free_kick_team == "blue" else 0

        vec_to_goal = np.array([target_goal_x, 300]) - self.ball_pos
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)

        # --- 1. BARRIERA SOLIDA (4 Giocatori) ---
        wall_dist = 110 
        wall_pos_base = self.ball_pos + vec_to_goal * wall_dist
        
        for i in range(1, 5): # Usa indici 1, 2, 3, 4 (difensori)
            idx = opp_start + i
            perp_vec = np.array([-vec_to_goal[1], vec_to_goal[0]]) 
            # Offset stretto (18) per non lasciare buchi nel muro
            offset = perp_vec * ((i-2.5) * 18) 
            
            self.players_pos[idx] = wall_pos_base + offset
            self.players_vel[idx] = np.zeros(2)
            self.stun_timer[idx] = self.free_kick_timer + 20 

        # --- 2. SCHIERAMENTO SQUADRE ---
        for i in range(11):
            # Squadra in attacco
            a_idx = my_start + i
            self.players_vel[a_idx] = np.zeros(2)
            if a_idx == self.free_kick_taker_idx:
                self.players_pos[a_idx] = self.ball_pos - vec_to_goal * 15
                self.ball_pos = self.players_pos[a_idx].copy()
                self.ball_vel = np.zeros(2)
            
            # Squadra in difesa
            d_idx = opp_start + i
            if i == 0: # PORTIERE: sulla linea, centro porta
                self.players_pos[d_idx] = np.array([opp_goal_x, 300])
            elif i >= 5: # Quelli non in barriera, marcature casuali
                self.players_pos[d_idx] = np.array([
                    opp_goal_x + (random.uniform(-120, -40) if self.free_kick_team=="blue" else random.uniform(40, 120)),
                    random.uniform(150, 450)
                ])
            self.stun_timer[d_idx] = self.free_kick_timer
    
    def _execute_free_kick_ml(self, action_val):
        # 1. Coordinate porta avversaria
        target_x = 1000 if self.free_kick_team == "blue" else 0
        
        # 2. Mappatura azione (0-4 per i movimenti, ma usiamo l'input dell'IA)
        # Se l'IA usa Discrete(5), action_val va da 0 a 4. 
        # Centriamo il tiro: 0=Palo basso, 2=Centro, 4=Palo alto
        target_y = 265 + (action_val * 17.5) 
        
        # 3. Calcolo Vettore Direzione
        diff = np.array([target_x, target_y]) - self.ball_pos
        dist = np.linalg.norm(diff)
        direction = diff / (dist + 1e-6)
        
        # 4. APPLICAZIONE FISICA ISTANTANEA
        # Aumentiamo la velocità a 16.0 per renderlo un vero "proiettile"
        self.ball_vel = direction * 16.0 
        
        # 5. EFFETTO A GIRO (Magnus Effect)
        # Se action_val < 2 curva in un senso, > 2 nell'altro
        self.ball_curve_effect = (action_val - 2) * 0.02
        self.ball_spin = self.ball_curve_effect * 5.0
        
        # 6. STATI SPECIALI
        self.ball_is_free_kick = True
        self.free_kick_active = False # Fondamentale per sbloccare il ball_vel = 0 nello step()
        
        # 7. SBLOCCO GIOCATORI
        # Azzeriamo tutti i timer di blocco così possono correre sulla ribattuta
        self.stun_timer.fill(0) 
        
        # Feedback visivo: un piccolo "fumo" o reset delle collisioni
        self.last_touch_team = self.free_kick_team
        self.anchor_recovery_timer = 600  # 10 secondi a 60fps

        

    def _apply_positional_anchor(self, idx):
        # 1. Gestione Grace Period
        if idx == 0 and self.post_kick_grace_period > 0:
            self.post_kick_grace_period -= 1

        # 2. Marcatura speciale Goal Kick (invariata, serve per i corner)
        if self.is_goal_kick:
            if self.ball_pos[0] < 50 or self.ball_pos[0] > self.width - 50:
                if (idx < 11 and self.goal_kick_team == "red") or (idx >= 11 and self.goal_kick_team == "blue"):
                    opp_start = 11 if idx < 11 else 0
                    opp_pos = self.players_pos[opp_start + 9] 
                    self.players_pos[idx] = opp_pos + np.array([10 if idx < 11 else -10, 0])
                    self.players_vel[idx] = np.zeros(2)
                    return

        # -----------------------------------------------------------
        # LOGICA "ELASTICO DINAMICO"
        # -----------------------------------------------------------
        
        team_start = 0 if idx < 11 else 11
        team_end = team_start + 11
        
        # Calcoliamo distanza dalla palla
        dist_to_ball = np.linalg.norm(self.ball_pos - self.players_pos[idx])

        # Controllo se sono il più vicino della mia squadra
        is_closest = True
        current_min_dist = dist_to_ball
        for i in range(team_start, team_end):
            if i == idx: continue
            d = np.linalg.norm(self.ball_pos - self.players_pos[i])
            if d < current_min_dist:
                is_closest = False
                break
        
        # --- DEFINIZIONE DELLA FORZA DI RICHIAMO (ANCHOR) ---
        
        # 1. Se sono il più vicino -> NESSUN ANCHOR (Libertà totale)
        if is_closest:
            return 

        # 2. Se non sono il più vicino, calcolo dove dovrei essere
        team = 0 if idx < 11 else 1
        rel = idx % 11
        
        if self.post_kick_grace_period > 0 and rel != 0:
            self.players_vel[idx] *= 0.5
            return

        if team == 0:
            formation = self.blue_formation_451 if self.blue_score > self.red_score else self.blue_formation_433
        else:
            formation = self.red_formation_451 if self.red_score > self.blue_score else self.red_formation_433
        
        home_pos = np.array(formation[rel])
        to_home = home_pos - self.players_pos[idx]
        dist_from_home = np.linalg.norm(to_home)

        # 3. Base Anchor per ruolo (quanto è rigido il ruolo?)
        if rel <= 4:   # Difesa (più rigidi)
            base_anchor = 0.0035 
        elif rel <= 7: # Centrocampo (flessibili)
            base_anchor = 0.0020
        else:          # Attacco (molto liberi)
            base_anchor = 0.0015

        # 4. MODULAZIONE BASATA SULLA DISTANZA PALLA (Il "Realismo")
        # Se la palla è vicina a me (< 250px), l'anchor diventa debolissimo.
        # Questo permette al giocatore di avvicinarsi all'azione (supporto) senza essere "incollato".
        if dist_to_ball < 250:
            anchor_strength = base_anchor * 0.2 # 15% della forza: sento la formazione ma vado verso la palla
        elif dist_to_ball < 400:
            anchor_strength = base_anchor * 0.3 # 50% della forza: zona media
        else:
            anchor_strength = base_anchor * 0.4       # 100% della forza: sono lontano, resto in posizione

        # 5. GESTIONE RIENTRO LENTO (Dopo i goal/kickoff)
        if self.anchor_recovery_timer > 0:
            # Sale da 0.05 a 1.0 lentamente
            ramp_factor = max(0.05, 1.0 - (self.anchor_recovery_timer / 600.0))
            if idx == 0: self.anchor_recovery_timer -= 1
        else:
            ramp_factor = 1.0

        # 6. APPLICAZIONE FISICA
        # Applichiamo l'anchor solo se siamo fuori posto
        if dist_from_home > 25: 
            # La forza finale combina: ruolo, distanza palla, e tempo di recupero
            final_force = to_home * anchor_strength * ramp_factor
            
            self.players_vel[idx] += final_force
            
            # Limitatore di velocità solo per il rientro "passivo" (quando ramp_factor è basso)
            # Se ramp_factor è 1 (gioco normale), non limitiamo la velocità così possono scattare
            if ramp_factor < 0.9: 
                max_speed_limit = self.max_speed * (0.2 + 0.8 * ramp_factor)
                curr_vel = np.linalg.norm(self.players_vel[idx])
                if curr_vel > max_speed_limit:
                    self.players_vel[idx] = (self.players_vel[idx] / curr_vel) * max_speed_limit

    def _apply_teammate_spacing(self, idx):
        grace_timer = getattr(self, 'post_kick_grace_period', 0)
        
        # Inizializza un dizionario per i contatori di contatto se non esiste
        if not hasattr(self, 'contact_counters'):
            self.contact_counters = np.zeros((22, 22))

        # 1. PARAMETRI DI BASE
        if self.is_goal_kick:
            current_spacing_mult = 0.15 # Ridotto per stabilità
            current_min_dist = 30
        else:
            current_spacing_mult = 0.4   # Più dolce
            current_min_dist = 90

        team_start = 0 if idx < 11 else 11
        my_pos = self.players_pos[idx]
        dist_to_ball = np.linalg.norm(self.ball_pos - my_pos)

        for i in range(team_start, team_start + 11):
            if i == idx: continue
            
            diff = my_pos - self.players_pos[i]
            dist = np.linalg.norm(diff)
            min_spacing = current_min_dist
            
            if dist < min_spacing:
            # 1. CONTATORE DI CONTATTO (Più rapido)
                # 2 secondi (120 frame) per raggiungere il regime massimo
                max_ramp_frames = 120.0
                self.contact_counters[idx][i] = min(max_ramp_frames, self.contact_counters[idx][i] + 1)
                
                # 2. CURVA LOGARITMICA (Forza subito presente, ma controllata)
                # Usiamo la radice quadrata del progresso: sale subito, poi si stabilizza
                progress = self.contact_counters[idx][i] / max_ramp_frames
                time_ramp = progress ** 0.5  # Sale velocemente all'inizio per sbloccarli
                
                # 3. CALCOLO DELLA SPINTA
                # Aumentiamo la forza base (0.6) ma riduciamo il moltiplicatore globale
                local_mult = current_spacing_mult * time_ramp
                
                # Intensità basata sulla penetrazione (più sono vicini, più spinge)
                # Aggiungiamo un +0.1 fisso per garantire che si stacchino sempre
                intensity = ((min_spacing - dist) / min_spacing) + 0.1
                
                # Calcolo del vettore forza
                force = (diff / (dist + 1e-6)) * 0.4 * intensity * local_mult
                
                # 4. IL "FRENO DI SICUREZZA" (Cruciale per evitare esplosioni)
                # Impediamo che lo spacing aggiunga troppa velocità in un colpo solo
                max_spacing_velocity = 0.4  # Valore basso = movimento fluido
                force_mag = np.linalg.norm(force)
                if force_mag > max_spacing_velocity:
                    force = (force / force_mag) * max_spacing_velocity
                
                # Applichiamo la forza
                self.players_vel[idx] += force
                
                # 5. SMORZAMENTO EXTRA (Damping)
                # Se stanno collidendo, riduciamo leggermente la velocità attuale 
                # per evitare che l'energia cinetica esploda
                self.players_vel[idx] *= 0.98

            else:
                # Reset rapido quando si staccano
                self.contact_counters[idx][i] = max(0, self.contact_counters[idx][i] - 5)
    
    def _goal_kick_reset(self, defending_team):
        self._reset_positions()
        self.is_goal_kick = True
        self.goal_kick_timer = 90  # Circa 1.5 secondi di pausa a 60fps
        self.goal_kick_team = defending_team
        
        # Posiziona il portiere esattamente sulla palla per l'estetica
        gk_idx = 0 if defending_team == "blue" else 11
        if defending_team == "blue":
            self.ball_pos = np.array([60.0, self.height / 2], dtype=np.float32)
        else:
            self.ball_pos = np.array([self.width - 60.0, self.height / 2], dtype=np.float32)
        self.players_pos[gk_idx] = self.ball_pos.copy()
        self.ball_vel = np.zeros(2)
        self.ball_spin = 0.0
        self.anchor_recovery_timer = 600  # 10 secondi a 60fps

    def _corner_kick_reset(self, defending_team):
        self.is_goal_kick = True
        self.goal_kick_timer = 40
        self.goal_kick_team = "red" if defending_team == "blue" else "blue"
        
        # Palla dentro il campo (fondamentale per evitare il loop)
        corner_x = 25 if defending_team == "blue" else self.width - 25
        corner_y = 25 if self.ball_pos[1] < self.height/2 else self.height - 25
        self.ball_pos = np.array([corner_x, corner_y], dtype=np.float32)
        self.ball_vel = np.zeros(2)

        # Squadra che ATTACCA (quasi tutti in area)
        att_team_start = 0 if self.goal_kick_team == "blue" else 11
        # Prendiamo gli indici dal 5 al 10 (centrocampisti e attaccanti)
        for i in range(att_team_start + 5, att_team_start + 11):
            target_area_x = 150 if defending_team == "blue" else self.width - 150
            self.players_pos[i] = np.array([target_area_x + random.uniform(-40, 40), 
                                           self.height/2 + random.uniform(-120, 120)])

        # Squadra che DIFENDE (quasi tutti in area a marcare)
        def_team_start = 11 if self.goal_kick_team == "blue" else 0
        for i in range(def_team_start, def_team_start + 10): # Tutti tranne uno
            target_area_x = 100 if defending_team == "blue" else self.width - 100
            self.players_pos[i] = np.array([target_area_x + random.uniform(-50, 50), 
                                           self.height/2 + random.uniform(-130, 130)])

        # Il battitore sulla palla
        hitter_idx = (att_team_start + 10) 
        self.players_pos[hitter_idx] = self.ball_pos.copy()
        self.anchor_recovery_timer = 600  # 10 secondi a 60fps
    
    def _handle_ball_bounds(self, rewards): # <--- Aggiungi rewards qui
        elasticity = 0.7
        penalty_out = -0.5 
        penalty_corner = -1.0 

        # 1. FALLO LATERALE (Rimbalzo e penalità)
        out_top_bottom = False
        if self.ball_pos[1] < self.ball_radius:
            self.ball_pos[1] = self.ball_radius
            self.ball_vel[1] *= -elasticity
            out_top_bottom = True
        elif self.ball_pos[1] > self.height - self.ball_radius:
            self.ball_pos[1] = self.height - self.ball_radius
            self.ball_vel[1] *= -elasticity
            out_top_bottom = True

        if out_top_bottom and self.last_touch_team is not None:
            rewards[self.last_touch_team] += penalty_out

        # 2. LINEE DI FONDO (Corner / Rinvio)
        is_in_goal_y = 250 < self.ball_pos[1] < 350
        if (self.ball_pos[0] < 0 or self.ball_pos[0] > self.width) and not is_in_goal_y:
            defending_team = "blue" if self.ball_pos[0] < 0 else "red"
            attacking_team = "red" if defending_team == "blue" else "blue"
            
            if self.last_touch_team == defending_team:
                # Penalità per chi concede l'angolo
                rewards[defending_team] += penalty_corner
                self._corner_kick_reset(defending_team)
            elif self.last_touch_team == attacking_team:
                # Penalità per chi calcia fuori sul fondo
                rewards[attacking_team] += penalty_out
                self._goal_kick_reset(defending_team)

    def _kick(self, idx, team):
        dist = np.linalg.norm(self.ball_pos - self.players_pos[idx])
        # Distanza di contatto ridotta per realismo
        if dist < self.player_radius + self.ball_radius + 5:
            current_team = "blue" if team == 0 else "red"
            if self.last_touch_team == current_team: 
                self.same_team_pass_count += 1
            else: 
                self.same_team_pass_count = 0
            self.last_touch_team = current_team

            rel = idx % 11
            teammates = [i for i in range(11)] if team == 0 else [i + 11 for i in range(11)]
            
            # DECISIONE: Tiro o Passaggio?
            is_shooting = False
            if rel >= 8 and random.random() < 0.7:
                # Bersaglio: Porta avversaria
                target_x = self.width + 20 if team == 0 else -20
                target = np.array([target_x, self.height / 2])
                is_shooting = True
            else:
                # Bersaglio: Compagno casuale
                mate = random.choice(teammates)
                target = self.players_pos[mate]

            # CALCOLO DIREZIONE E VELOCITÀ
            direction = target - self.players_pos[idx]
            dist_to_target = np.linalg.norm(direction) + 1e-6
            self.ball_vel = (direction / dist_to_target) * self.ball_speed

            # --- LOGICA EFFETTO A GIRO INTELLIGENTE ---
            self.ball_spin = 0  # Default: tiro dritto
            
            if is_shooting:
                # Se la porta è angolata male (giocatore alto o basso rispetto al centro)
                # Calcola quanto il giocatore è lontano dall'asse centrale della porta
                offset_y = self.players_pos[idx][1] - (self.height / 2)
                if abs(offset_y) > 100:
                    # Applica spin per far rientrare la palla verso il centro
                    # Se sono in alto (offset > 0), devo curvare verso il basso (spin negativo)
                    self.ball_spin = -0.07 if offset_y > 0 else 0.07
            else:
                # Se è un passaggio e il compagno è in diagonale
                dx = abs(target[0] - self.players_pos[idx][0])
                dy = abs(target[1] - self.players_pos[idx][1])
                # Se il rapporto dy/dx è alto, sono in diagonale "stretta"
                if dy > 0.8 * dx:
                    # Spin casuale ma forte per simulare il filtrante a giro
                    self.ball_spin = random.choice([-0.08, 0.08])

            self.ball_spin += random.uniform(-0.01, 0.01)

    def _collision(self, idx, rewards):
        # Se la palla va molto veloce (sopra 5.5), è "in aria" e nessuno la tocca
        diff = self.ball_pos - self.players_pos[idx]
        dist = np.linalg.norm(diff)
        
        # Riduciamo l'area di interazione per evitare tocchi "fantasma"
        if dist < self.player_radius + self.ball_radius + 2:
            team = "blue" if idx < 11 else "red"
            rewards[team] += 5.0
            
            # Se la palla è molto veloce (gialla), il giocatore la "controlla" 
            # invece di farla schizzare via a caso
            if np.linalg.norm(self.ball_vel) > 5.5:
                self.ball_vel *= 0.5  # Ammortizza il colpo
                self.ball_spin = 0    # Annulla l'effetto strano
            
            # Applica una spinta direzionale basata sul movimento del giocatore
            self.ball_vel += diff / (dist + 1e-6) * 2.0
            self.last_touch_team = team


    def _get_obs(self):
        obs_dict = {}
        for team_name, start_idx in {"blue": 0, "red": 11}.items():
            team_obs = []
            for i in range(start_idx, start_idx + 11):
                rel_ball = (self.ball_pos - self.players_pos[i]) / 100.0
                team_obs.append(self.players_pos[i] / self.width)
                team_obs.append(rel_ball)
            
            ball_info = np.concatenate([self.ball_pos / self.width, self.ball_vel / self.max_speed])
            obs_dict[team_name] = np.concatenate(team_obs + [ball_info]).astype(np.float32)[:48]
            
        return obs_dict

    def _reset_positions(self):
        """Reset to formation positions with small random offset"""
        # Blue team
        is_blue_winning = self.blue_score > self.red_score
        blue_formation = self.blue_formation_451 if is_blue_winning else self.blue_formation_433
        for i in range(11):
            base_pos = np.array(blue_formation[i])
            offset = np.random.uniform(-20, 20, 2)  # Small random offset
            self.players_pos[i] = base_pos + offset
        
        # Red team
        is_red_winning = self.red_score > self.blue_score
        red_formation = self.red_formation_451 if is_red_winning else self.red_formation_433
        for i in range(11):
            base_pos = np.array(red_formation[i])
            offset = np.random.uniform(-20, 20, 2)
            self.players_pos[i + 11] = base_pos + offset
        
        self.ball_pos = np.array([self.width/2, self.height/2])
        self.ball_vel = np.zeros(2)

    def _update_elo(self, winner):
        K = 16
        eb = 1/(1+10**((self.elo_red-self.elo_blue)/400))
        er = 1/(1+10**((self.elo_blue-self.elo_red)/400))
        if winner == "blue":
            self.elo_blue += K*(1-eb); self.elo_red += K*(0-er)
        else:
            self.elo_red += K*(1-er); self.elo_blue += K*(0-eb)

    def render(self):
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Soccer AI Pro v2")
            self.font = pygame.font.SysFont("Arial", 24, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 18, bold=True)

        if self.free_kick_active or (hasattr(self, 'ball_curve_effect') and np.linalg.norm(self.ball_vel) > 8):
            ball_color = (255, 255, 0) # GIALLO: Tiro punizione in corso

        self.screen.fill((34,139,34))
        white = (255, 255, 255)
        
        pygame.draw.rect(self.screen, white, (10, 10, self.width-20, self.height-20), 3)
        pygame.draw.line(self.screen, white, (self.width/2, 10), (self.width/2, self.height-10), 3)
        pygame.draw.circle(self.screen, white, (int(self.width/2), int(self.height/2)), 70, 3)
        pygame.draw.rect(self.screen, white, (10, 150, 150, 300), 3)
        pygame.draw.rect(self.screen, white, (self.width-160, 150, 150, 300), 3)
        pygame.draw.rect(self.screen, (200, 200, 200), (-5, 250, 15, 100))
        pygame.draw.rect(self.screen, (200, 200, 200), (self.width-10, 250, 15, 100))

        blue_mod = "4-5-1" if self.blue_score > self.red_score else "4-3-3"
        red_mod = "4-5-1" if self.red_score > self.blue_score else "4-3-3"
        
        score_text = self.font.render(f"BLUE {self.blue_score} - {self.red_score} RED", True, white)
        self.screen.blit(score_text, (self.width/2 - score_text.get_width()/2, 20))
        
        blue_info = self.small_font.render(f"BLU: {blue_mod} | ELO: {int(self.elo_blue)}", True, (100, 200, 255))
        red_info = self.small_font.render(f"RED: {red_mod} | ELO: {int(self.elo_red)}", True, (255, 100, 100))
        self.screen.blit(blue_info, (50, 20))
        self.screen.blit(red_info, (self.width - red_info.get_width() - 50, 20))
        if self.free_kick_active:
            text = self.font.render("FREE KICK", True, (255,255,0))
            self.screen.blit(text, (self.width/2 - 60, 60))

        for i in range(22):
            # 1. Definizione colori "via di mezzo" (bilanciati)
            if i == 0 or i == 11:
                color = (240, 240, 240)  # Bianco sporco/Ghiaccio per i GK (meno accecante)
            elif i < 11:
                color = (40, 80, 200)    # Blu reale (meno elettrico del pure blue)
            else:
                color = (200, 40, 40)    # Rosso cardinale (meno forte del pure red)
            
            if not self.free_kick_active:
                if self.stun_timer[i] > 0:
                    color = (100, 100, 100)  # Grigio medio
                elif self.slow_timer[i] > 0:
                    color = (160, 160, 160)  # Grigio chiaro

            # 3. Disegno del giocatore con bordi BIANCHI
            # Bordo Bianco (fisso)
            pygame.draw.circle(self.screen, (255, 255, 255), self.players_pos[i].astype(int), 11)
            # Cerchio interno (dinamico)
            pygame.draw.circle(self.screen, color, self.players_pos[i].astype(int), 9)

        # Nel metodo render()
        ball_speed_norm = np.linalg.norm(self.ball_vel)
        
        # Logica colore palla prioritaria
        is_being_dribbled = any(self.dribble_timer > 0)
        
        if is_being_dribbled:
            ball_color = (0, 255, 127)   # VERDE: Dribbling in corso
        elif ball_speed_norm > 5.5 or self.is_goal_kick:
            ball_color = (255, 255, 0)   # GIALLO: Volo o Rinvio
        else:
            ball_color = (255, 255, 255) # BIANCO: Normale
            
        pygame.draw.circle(self.screen, (0,0,0), self.ball_pos.astype(int), 7)
        pygame.draw.circle(self.screen, ball_color, self.ball_pos.astype(int), 6)
        
        pygame.display.flip()
        self.clock.tick(60)

# ==========================================================
# TRAINING LOOP
# ==========================================================

def train():
    env_raw = SoccerParallelEnv(render_mode="human")
    env = ss.pettingzoo_env_to_vec_env_v1(env_raw)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    save_path = "ppo_soccer_pro"
    interrupt_path = "ppo_soccer_INTERRUPTED"

    if os.path.exists(f"{save_path}.zip"):
        print(f"Caricamento modello esistente da {save_path}...")
        model = PPO.load(save_path, env=env)
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            ent_coef=0.1,
            n_steps=64,
            batch_size=64,
            n_epochs=4
        )

    print("\nTraining Self-Play avviato. Premi CTRL+C per salvare e uscire.")
    
    try:
        model.learn(total_timesteps=200000, progress_bar=True, )
        print(f"Training completato. Salvataggio in {save_path}...")
        model.save(save_path)

    except KeyboardInterrupt:
        print(f"\n!!! INTERRUZIONE RILEVATA !!!")
        print(f"Sto salvando il modello di emergenza in: {interrupt_path}.zip")
        model.save(interrupt_path)
        print("Salvataggio completato con successo.")
        
    except Exception as e:
        print(f"\nErrore imprevisto: {e}")
        model.save("ppo_soccer_CRASH_BACKUP")

if __name__ == "__main__":
    train()
