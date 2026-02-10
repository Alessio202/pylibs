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
        self.ball_curve_effect = 0.0
        self.is_penalty = False
        self.anchor_recovery_timer = 0
        self.post_kick_grace_period = 0 
        self.max_speed = 2.0  # Increased again for better ball chasing
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
        self.match_timer = 60 * 60  # 60 secondi a 60 FPS (1 minuto) 
        self.is_penalty_shootout = False
        self.penalty_turn = 0 # 0 per blu, 1 per rossi
        self.penalties_taken = 0
        self.penalty_shootout_phase = 0  # 0=in preparazione, 1=tirando, 2=risultato


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

        self.penalty_results = []  # Lista per tracciare i risultati dei rigori
        self.contact_counters = np.zeros((22, 22))  # Inizializza qui
        
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

    def _setup_penalty(self, victim_idx, shooting_team):
        """Setup per un calcio di rigore (NON per penalty shootout)"""
        print(f"Setup penalty per {shooting_team}, tiratore {victim_idx}")
        
        self.free_kick_active = True
        self.free_kick_timer = 180
        self.free_kick_team = shooting_team
        self.free_kick_taker_idx = victim_idx
        self.ball_curve_effect = 0.0
        self.is_penalty = True
        self.mandatory_wait = 60
        
        # Reset fisici
        self.ball_vel = np.zeros(2)
        self.ball_spin = 0
        
        # Posizioni per rigore BLU (attaccano a destra)
        if shooting_team == "blue":
            self.ball_pos = np.array([880.0, 300.0])
            self.players_pos[victim_idx] = np.array([850.0, 300.0])
            self.players_pos[11] = np.array([995.0, 300.0])  # Portiere rosso
            
            # TUTTI gli altri giocatori BLU dietro la linea di centrocampo (sinistra)
            for i in range(0, 11):
                if i != victim_idx and i != 0:  # Escludi tiratore e portiere blu
                    x_pos = random.uniform(100, 300)  # Metà campo sinistra
                    y_pos = 100 + (i * 40)
                    self.players_pos[i] = np.array([x_pos, y_pos])
            
            # Portiere blu in porta (ma lontano)
            self.players_pos[0] = np.array([80.0, 300.0])
            
            # TUTTI gli altri giocatori ROSSI dietro la linea di centrocampo (destra)
            for i in range(12, 22):
                x_pos = random.uniform(700, 900)  # Metà campo destra (ma dietro)
                y_pos = 100 + ((i-11) * 40)
                self.players_pos[i] = np.array([x_pos, y_pos])
        
        else:  # ROSSI (attaccano a sinistra)
            self.ball_pos = np.array([120.0, 300.0])
            self.players_pos[victim_idx] = np.array([150.0, 300.0])
            self.players_pos[0] = np.array([5.0, 300.0])  # Portiere blu
            
            # TUTTI gli altri giocatori ROSSI dietro la linea di centrocampo (destra)
            for i in range(11, 22):
                if i != victim_idx and i != 11:  # Escludi tiratore e portiere rosso
                    x_pos = random.uniform(700, 900)  # Metà campo destra
                    y_pos = 100 + ((i-11) * 40)
                    self.players_pos[i] = np.array([x_pos, y_pos])
            
            # Portiere rosso in porta (ma lontano)
            self.players_pos[11] = np.array([995.0, 300.0])
            
            # TUTTI gli altri giocatori BLU dietro la linea di centrocampo (sinistra)
            for i in range(1, 11):
                x_pos = random.uniform(100, 300)  # Metà campo sinistra (ma dietro)
                y_pos = 100 + (i * 40)
                self.players_pos[i] = np.array([x_pos, y_pos])
        
        # Velocità a zero per tutti
        for i in range(22):
            self.players_vel[i] = np.zeros(2)
        
        # Solo tiratore e portiere difendente NON stunnati
        defending_gk = 11 if shooting_team == "blue" else 0
        for i in range(22):
            if i == victim_idx or i == defending_gk:
                self.stun_timer[i] = 0
            else:
                self.stun_timer[i] = self.free_kick_timer
        
    def _setup_penalty_shootout_kick(self):
        """Setup per un singolo rigore nel penalty shootout"""
        self.penalty_shootout_phase = 0  # Fase preparazione
        
        # Determina chi tira (sempre verso la porta di DESTRA per entrambe le squadre)
        shooting_team = "blue" if self.penalties_taken % 2 == 0 else "red"
        
        # Usa sempre la porta di DESTRA (porta blu)
        self.ball_pos = np.array([880.0, 300.0])  # Dischetto destro
        
        if shooting_team == "blue":
            # Blu tira verso la propria porta (destra)
            taker_idx = 9  # Attaccante blu
            defending_gk = 11  # Portiere rosso
            self.players_pos[taker_idx] = np.array([850.0, 300.0])
            self.players_pos[defending_gk] = np.array([995.0, 300.0])
        else:
            # Rossi tirano verso la porta di destra (quindi attaccano in direzione opposta)
            taker_idx = 20  # Attaccante rosso
            defending_gk = 11  # Portiere rosso (stesso portiere, ma ora deve difendere)
            # Posiziona il tiratore rosso dall'altra parte del campo
            self.players_pos[taker_idx] = np.array([850.0, 300.0])
            self.players_pos[defending_gk] = np.array([995.0, 300.0])
        
        # Posiziona tutti gli altri giocatori a bordo campo
        for i in range(22):
            if i not in [taker_idx, defending_gk]:
                # Posiziona lungo i bordi laterali
                if i < 11:  # Blu
                    self.players_pos[i] = np.array([50.0, 50.0 + (i * 25)])
                else:  # Rossi
                    self.players_pos[i] = np.array([950.0, 50.0 + ((i-11) * 25)])
        
        # Setup stato free kick per il rigore
        self.free_kick_active = True
        self.free_kick_timer = 60  # 1 secondo di preparazione
        self.free_kick_team = shooting_team
        self.free_kick_taker_idx = taker_idx
        self.is_penalty = True
        
        # Reset fisici
        self.ball_vel = np.zeros(2)
        self.ball_spin = 0
    
    def _start_penalty_shootout(self):
        self.is_penalty_shootout = True
        self.penalty_shootout_phase = 0
        self.penalties_taken = 0
        self.penalty_results = []  # Reset risultati
        self._setup_penalty_shootout_kick()

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
        self.anchor_disabled = np.zeros(22, dtype=bool) 
        self.attacking_in_box = []
        return self._get_obs(), {}

    def _handle_penalty_shootout_result(self, rewards, terminations):
        """Gestisce il risultato finale del penalty shootout"""
        # Conta i goal per squadra
        blue_goals = sum(1 for team, scored in self.penalty_results if team == "blue" and scored)
        red_goals = sum(1 for team, scored in self.penalty_results if team == "red" and scored)
        
        # Determina se c'è un vincitore dopo 10 rigori
        if self.penalties_taken >= 10:
            # Se siamo in sudden death (dopo 10 rigori) e c'è differenza di goal
            if blue_goals != red_goals:
                terminations = {"blue": True, "red": True}
                if blue_goals > red_goals:
                    rewards["blue"] += 100
                    rewards["red"] -= 100
                    self._update_elo("blue")
                else:
                    rewards["red"] += 100
                    rewards["blue"] -= 100
                    self._update_elo("red")
                return True, rewards, terminations
            
            # Se dopo 10 rigori è ancora pari, continua con sudden death
            elif self.penalties_taken >= 20:  # Limite massimo per evitare loop infinito
                terminations = {"blue": True, "red": True}
                # Pareggio dopo 20 rigori
                rewards["blue"] -= 50
                rewards["red"] -= 50
                print(f"PAREGGIO AI RIGORI DOPO 20 TENTATIVI! {blue_goals}-{red_goals}")
                return True, rewards, terminations
        
        return False, rewards, terminations

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
        
        # --- 3. GESTIONE TEMPO PARTITA ---
        if not self.is_penalty_shootout:
            self.match_timer -= 1
            if self.match_timer <= 0:
                if self.blue_score == self.red_score:
                    self._start_penalty_shootout()
                else:
                    # Termina la partita se non c'è pareggio
                    terminations = {"blue": True, "red": True}
                    # Aggiorna ELO e ricompense finali
                    if self.blue_score > self.red_score:
                        self._update_elo("blue")
                        rewards["blue"] += 50
                        rewards["red"] -= 50
                    else:
                        self._update_elo("red")
                        rewards["red"] += 50
                        rewards["blue"] -= 50
                    return self._get_obs(), rewards, terminations, truncations, infos
        
        # --- 4. GESTIONE PENALTY SHOOTOUT ---
        if self.is_penalty_shootout:
            # Controlla se il penalty shootout è finito
            shootout_ended, rewards, terminations = self._handle_penalty_shootout_result(rewards, terminations)
            if shootout_ended:
                return self._get_obs(), rewards, terminations, truncations, infos
            
            # Gestione normale del penalty shootout
            if not self.free_kick_active and self.penalty_shootout_phase == 2:
                # Aspetta un po' prima di passare al prossimo rigore
                if not hasattr(self, 'result_wait_timer'):
                    self.result_wait_timer = 30  # Aspetta 0.5 secondi
                else:
                    self.result_wait_timer -= 1
                    if self.result_wait_timer <= 0:
                        self.penalty_shootout_phase = 0
                        delattr(self, 'result_wait_timer')
                        self._setup_penalty_shootout_kick()
            elif not self.free_kick_active and self.penalty_shootout_phase == 0:
                # Inizia il countdown per il tiro
                self.penalty_shootout_phase = 1
                self.free_kick_timer = 60  # 1 secondo per prepararsi
        
        # --- 5. GESTIONE FREE KICK / PENALTY (INCLUSE ESECUZIONI) ---
        if self.free_kick_active:
            self.ball_vel = np.zeros(2)
            self.ball_spin = 0
            self.free_kick_timer -= 1

            # durante il conto alla rovescia la palla è ferma
            if self.free_kick_timer > 0:
                if self.render_mode == "human":
                    self.render()
                return self._get_obs(), rewards, terminations, truncations, infos

            # ESECUZIONE TIRO (Quando il timer scade)
            if self.is_penalty:
                # Usa l'azione del tiratore per determinare la direzione
                shooting_team = self.free_kick_team
                taker_idx = self.free_kick_taker_idx
                taker_action = actions[shooting_team][taker_idx % 11] if shooting_team in actions else 4
                
                # Mappa l'azione a una direzione (0-6 -> 0-6 per il tiro)
                # Per i rigori: 0=alto sinistra, 1=alto centro, 2=alto destra
                #               3=basso sinistra, 4=basso centro, 5=basso destra, 6=al centro
                mapped_action = min(taker_action, 6)
                
                # Esegui il tiro
                self._execute_free_kick_ml(mapped_action)
                
                if self.is_penalty_shootout:
                    self.penalty_shootout_phase = 2  # Fase risultato
                    self.penalties_taken += 1
            else:
                # Free kick normale (non rigore)
                self.ball_spin = random.uniform(-0.1, 0.1)
                self._execute_free_kick_ml(4)

            # Sblocca tutti i giocatori dopo il tiro
            for i in range(22):
                self.stun_timer[i] = 0
            
            # Pulisci stati
            self.is_penalty = False
            self.free_kick_active = False

            if self.render_mode == "human":
                self.render()
            return self._get_obs(), rewards, terminations, truncations, infos

        # --- 6. GESTIONE GOAL KICK ---
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
                self.post_kick_grace_period = 60
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

        # --- 7. UNICO LOOP MOVIMENTO E FISICA ---
        prev_ball_x = self.ball_pos[0]

        for i in range(22):
            team_key = "blue" if i < 11 else "red"
            act_idx = i if i < 11 else i - 11
                
            # 1. Movimento IA
            self._move_player(i, actions[team_key][act_idx])
            self._apply_positional_anchor(i)
            self._apply_teammate_spacing(i)
            self.players_pos[i] += self.players_vel[i]
            
            # 2. Muro Invisibile (Solo durante i calci piazzati)
            # 2. Muro Invisibile (Solo durante i calci piazzati)
            if self.free_kick_active and self.is_penalty:
                # Per i rigori, impedisci a TUTTI i giocatori di avvicinarsi all'area
                if i != self.free_kick_taker_idx and i != 0 and i != 11:
                    if self.free_kick_team == "blue":
                        # Se sei blu e stai dalla parte sbagliata (destra), torna indietro
                        if self.players_pos[i][0] > 500:  # Linea di centrocampo
                            self.players_pos[i][0] = 500
                            self.players_vel[i] = np.zeros(2)
                    else:
                        # Se sei rosso e stai dalla parte sbagliata (sinistra), torna indietro
                        if self.players_pos[i][0] < 500:  # Linea di centrocampo
                            self.players_pos[i][0] = 500
                            self.players_vel[i] = np.zeros(2)

            # 3. Gestione Slow e Stun
            if self.slow_timer[i] > 0:
                self.players_vel[i] *= 0.8
                self.slow_timer[i] -= 1

            if self.stun_timer[i] > 0:
                self.players_vel[i] = np.zeros(2)
                self.stun_timer[i] -= 1
            
            self._handle_dribbling(i)
            is_dribbling = (self.dribble_timer[i] > 0)
            if not self.free_kick_active and not is_dribbling and actions[team_key][act_idx] >= 5:
                self._kick(i, 0 if i < 11 else 1)

        # --- 8. FISICA PALLA (Effetto Magnus + Attrito) ---
        if np.linalg.norm(self.ball_vel) > 0.6:
            perp_vel = np.array([-self.ball_vel[1], self.ball_vel[0]])
            
            # Usa solo ball_spin
            self.ball_vel += (perp_vel * self.ball_spin) * 0.015 
            self.ball_spin *= 0.95
        
        if hasattr(self, 'ball_curve_effect') and np.linalg.norm(self.ball_vel) > 5:
            perp_vel = np.array([-self.ball_vel[1], self.ball_vel[0]])
            self.ball_vel += perp_vel * self.ball_curve_effect

        self.ball_pos += self.ball_vel
        self.ball_vel *= 0.97
        
        # --- 9. LOGICA GOAL (AGGIORNAMENTO PER PENALTY SHOOTOUT) ---
        if 250 < self.ball_pos[1] < 350:
            goal_scored = False
            
            # Goal Squadra Rossa (Palla oltre il bordo sinistro)
            if self.ball_pos[0] < 0:
                self.red_score += 1
                rewards["red"] += 120
                rewards["blue"] -= 120
                goal_scored = True
                
                # Se è penalty shootout, registra il risultato
                if self.is_penalty_shootout and self.penalty_shootout_phase == 2:
                    self.penalty_results.append(("red", True))
                    print(f"RIGORE ROSSO: GOAL!")
                
            # Goal Squadra Blu (Palla oltre il bordo destro)
            elif self.ball_pos[0] > self.width:
                self.blue_score += 1
                rewards["blue"] += 120
                rewards["red"] -= 120
                goal_scored = True
                
                # Se è penalty shootout, registra il risultato
                if self.is_penalty_shootout and self.penalty_shootout_phase == 2:
                    self.penalty_results.append(("blue", True))
                    print(f"RIGORE BLU: GOAL!")

            if goal_scored:
                # Se NON è penalty shootout, fai il reset normale
                if not self.is_penalty_shootout:
                    # Reset Palla al centro
                    self.anchor_recovery_timer = 0
                    self.ball_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
                    self.ball_vel = np.zeros(2)
                    self.ball_spin = 0.0
                    
                    # Reset stati fisici dei giocatori
                    self.dribble_timer.fill(0)
                    self.slow_timer.fill(0)
                    self.stun_timer.fill(0)
                    self._reset_positions()
                    
                    # Azzera le velocità correnti
                    for i in range(22):
                        self.players_vel[i] = np.zeros(2)
                    
                    # Periodo di grazia per rientrare in formazione
                    self.post_kick_grace_period = 300

                return self._get_obs(), rewards, terminations, truncations, infos
        
        # Se siamo in penalty shootout e la palla è uscita senza goal, registra fallimento
        if self.is_penalty_shootout and self.penalty_shootout_phase == 2:
            if self.ball_pos[0] < 0 or self.ball_pos[0] > self.width or \
               self.ball_pos[1] < 0 or self.ball_pos[1] > self.height:
                # Palla uscita senza goal
                shooting_team = self.free_kick_team
                self.penalty_results.append((shooting_team, False))
                print(f"RIGORE {shooting_team.upper()}: FALLITO!")

        # Controllo bordi e rinvio dal fondo (se non è goal)
        self._handle_ball_bounds(rewards)

        # --- 10. COLLISIONI E REWARDS ---
        for i in range(22):
            self._collision(i, rewards)
        
        # --- 11. LOGICA PENALITÀ RETROPASSAGGIO IN AREA ---
        for i in range(22):
            dist_to_ball = np.linalg.norm(self.ball_pos - self.players_pos[i])
            if dist_to_ball < self.player_radius + self.ball_radius + 2:
                
                p_pos = self.players_pos[i]
                ball_vel_x = self.ball_vel[0]
                
                # SQUADRA BLU (Attacca verso destra)
                if i < 11:
                    if p_pos[0] > 840 and 150 < p_pos[1] < 450:
                        if ball_vel_x < -1.0:
                            rewards["blue"] -= 7.5
                        elif ball_vel_x > 2.0:
                            rewards["blue"] += 2.0
                            
                # SQUADRA ROSSA (Attacca verso sinistra)
                else:
                    if p_pos[0] < 160 and 150 < p_pos[1] < 450:
                        if ball_vel_x > 1.0:
                            rewards["red"] -= 7.5
                        elif ball_vel_x < -2.0:
                            rewards["red"] += 2.0

        self._handle_tackles(rewards)

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

        # --- 12. RENDERING ---
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), rewards, terminations, truncations, infos

    def _move_player(self, idx, act):

        if self.free_kick_active:
            defending_gk = 11 if self.free_kick_team == "blue" else 0
            if idx == defending_gk:
                goal_x = 985 if idx == 11 else 15
                x_margin = 20
                # clamp della posizione X corrente entro il margine
                self.players_pos[idx][0] = np.clip(self.players_pos[idx][0], goal_x - x_margin, goal_x + x_margin)

                # inseguimento verticale della palla (entro i pali)
                target_y = np.clip(self.ball_pos[1], 250, 350)
                move_speed = 2.2
                if self.players_pos[idx][1] < target_y - 2:
                    self.players_vel[idx][1] = move_speed
                elif self.players_pos[idx][1] > target_y + 2:
                    self.players_vel[idx][1] = -move_speed
                else:
                    self.players_vel[idx][1] = 0

                # Anticipazione laterale: muoviti verso la palla (non invertire il segno)
                if hasattr(self, 'free_kick_timer'):
                    if self.free_kick_timer < 30:
                        predicted_dir = np.sign(self.ball_pos[0] - goal_x)  # >0 = palla a destra del goal_x
                        self.players_vel[idx][0] = predicted_dir * 1.2
                    else:
                        self.players_vel[idx][0] = 0

                # Logica specifica per rigore (is_penalty)
                if getattr(self, 'is_penalty', False):
                    if self.free_kick_timer <= 12:
                        shooter_pos = self.players_pos[self.free_kick_taker_idx]
                        aim_y = shooter_pos[1] + (self.ball_pos[1] - shooter_pos[1]) * 0.2
                        dive_speed = 6.0
                        self.players_vel[idx][1] = dive_speed if self.players_pos[idx][1] < aim_y else -dive_speed
                        self.players_vel[idx][0] = np.sign(self.ball_pos[0] - goal_x) * 2.0

                return

        if self.free_kick_active and self.is_penalty:
            # Durante i rigori, solo tiratore e portiere possono muoversi
            if idx != self.free_kick_taker_idx and idx != defending_gk:
                # Impedisci movimento a tutti gli altri giocatori
                self.players_vel[idx] = np.zeros(2)
                return


        # --- LOGICA MOVIMENTO NORMALE ---
        # Rimosso il 'self.players_pos += self.players_vel' da qui!
        # Rimosso anchor e spacing (sono già nello step)
        
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

        # Applica accelerazione e attrito alla velocità
        self.players_vel[idx] = self.players_vel[idx] * self.friction + vel * (1 - self.friction)

        
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
        if self.post_kick_grace_period > 0:
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
    
    def _handle_tackles(self, rewards):
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
                                v_pos = self.players_pos[victim]
                                # Controlliamo se la vittima è nella metà campo avversaria
                                # Blue attacca a destra (x > 500), Red a sinistra (x < 500)
                                #if victim < 11 and v_pos[0] > 700: # test
                                if victim < 11 and v_pos[0] > 900 and 200 < v_pos[1] < 400:
                                    rewards["blue"] += 15.0
                                    self._setup_penalty(victim_idx=victim, shooting_team="blue")
                                    return 

                                # Rigore per i ROSSI (Vittima è rossa, fallo in area blu)
                                #elif victim >= 11 and v_pos[0] < 300: # test
                                elif victim >= 11 and v_pos[0] < 100 and 200 < v_pos[1] < 400:
                                    rewards["red"] += 15.0
                                    self._setup_penalty(victim_idx=victim, shooting_team="red")
                                    return
                                is_foul_blue = (victim < 11 and 625 < v_pos[0] < 900)
                                is_foul_red = (victim >= 11 and 100 < v_pos[0] < 375)
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

        # se la palla è troppo vicina al dischetto rigore, spostala leggermente
        if abs(self.ball_pos[0] - 880) < 5 or abs(self.ball_pos[0] - 120) < 5:
            self.ball_pos[0] += 10

        self.free_kick_active = True
        self.free_kick_timer = 180  # 3 secondi a 60fps
        self.free_kick_team = "blue" if victim_idx < 11 else "red"
        self.free_kick_taker_idx = victim_idx
        self.mandatory_wait = 120

        # stato per disabilitare anchor a singoli giocatori
        if not hasattr(self, 'anchor_disabled'):
            self.anchor_disabled = np.zeros(22, dtype=bool)
        self.anchor_disabled[:] = False

        # attaccanti in area (lista per eventuale debug)
        self.attacking_in_box = []

        # niente rallentamento esagerato di rientro
        self.anchor_recovery_timer = 0

        self.ball_vel = np.zeros(2)
        self.ball_spin = 0

        my_start = 0 if self.free_kick_team == "blue" else 11
        opp_start = 11 if self.free_kick_team == "blue" else 0
        opp_goal_x = 950 if self.free_kick_team == "blue" else 50
        target_goal_x = 1000 if self.free_kick_team == "blue" else 0

        vec_to_goal = np.array([target_goal_x, 300]) - self.ball_pos
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)

        # ---------- BARRIERA SOLIDA (4 difensori) ----------
        self.free_kick_wall = [opp_start + i for i in range(1, 5)]
        wall_dist = 110
        perp_vec = np.array([-vec_to_goal[1], vec_to_goal[0]])
        wall_pos_base = self.ball_pos + vec_to_goal * wall_dist
        wall_spacing = 18.0

        for k, idx in enumerate(self.free_kick_wall):
            offset = perp_vec * ((k - (len(self.free_kick_wall)-1)/2.0) * wall_spacing)
            self.players_pos[idx] = wall_pos_base + offset
            self.players_vel[idx] = np.zeros(2)
            # stunna SOLO la barriera
            self.stun_timer[idx] = self.free_kick_timer + 10

        # ---------- PORTIERE DIFENDENTE: NON STUNNATO ----------
        gk_idx = 11 if self.free_kick_team == "blue" else 0
        goal_x = 995.0 if gk_idx == 11 else 5.0
        self.players_pos[gk_idx] = np.array([goal_x, 300.0])
        self.players_vel[gk_idx] = np.zeros(2)
        self.stun_timer[gk_idx] = 0  # portiere libero

        # ---------- ATTACCANTI NELL'AREA AVVERSARIA ----------
        for i in range(1, 11):
            a_idx = my_start + i
            if self.free_kick_team == "blue":
                self.players_pos[a_idx] = np.array([
                    880.0 - random.uniform(0, 60),
                    300.0 + random.uniform(-80, 80)
                ])
            else:
                self.players_pos[a_idx] = np.array([
                    120.0 + random.uniform(0, 60),
                    300.0 + random.uniform(-80, 80)
                ])
            self.players_vel[a_idx] = np.zeros(2)
            self.anchor_disabled[a_idx] = True
            self.attacking_in_box.append(a_idx)

        # ---------- TIRATORE ----------
        self.players_pos[self.free_kick_taker_idx] = self.ball_pos - vec_to_goal * 15
        self.ball_pos = self.players_pos[self.free_kick_taker_idx].copy()
        self.players_vel[self.free_kick_taker_idx] = np.zeros(2)
        self.stun_timer[self.free_kick_taker_idx] = 0

        # ---------- FLAG "PENALTY" VISIVO ----------
        self.is_penalty = (
            (self.free_kick_team == "blue" and self.ball_pos[0] > 840) or
            (self.free_kick_team == "red" and self.ball_pos[0] < 160)
        )



    def _execute_free_kick_ml(self, action_val):
        # 1. Coordinate porta avversaria
        if self.is_penalty_shootout:
            # Nel penalty shootout, tutti tirano verso la porta di destra
            target_x = 1000
        else:
            target_x = 1000 if self.free_kick_team == "blue" else 0
        
        self.is_penalty = False  # Reset dopo il tiro
        self.free_kick_active = False
        
        # 2. Mappatura azione (0-6 per i movimenti)
        # Per i rigori: 0=alto sinistra, 1=alto centro, 2=alto destra
        #               3=basso sinistra, 4=basso centro, 5=basso destra, 6=al centro
        
        if action_val == 0:  # Alto sinistra
            target_y = 280
            self.ball_spin = 0.05  # Leggera curva
        elif action_val == 1:  # Alto centro
            target_y = 295
            self.ball_spin = 0.0
        elif action_val == 2:  # Alto destra
            target_y = 310
            self.ball_spin = -0.05  # Leggera curva
        elif action_val == 3:  # Basso sinistra
            target_y = 320
            self.ball_spin = 0.05
        elif action_val == 4:  # Basso centro
            target_y = 310
            self.ball_spin = 0.0
        elif action_val == 5:  # Basso destra
            target_y = 300
            self.ball_spin = -0.05
        else:  # Centro (azione 6)
            target_y = 300
            self.ball_spin = 0.0
        
        # 3. Calcolo Vettore Direzione
        diff = np.array([target_x, target_y]) - self.ball_pos
        dist = np.linalg.norm(diff)
        direction = diff / (dist + 1e-6)
        
        # 4. APPLICAZIONE FISICA ISTANTANEA
        # Velocità molto alta per il rigore
        self.ball_vel = direction * 18.0 
        self.post_kick_grace_period = 20
        
        # 5. STATI SPECIALI
        self.ball_is_free_kick = True
        self.free_kick_active = False  # Fondamentale per sbloccare il ball_vel = 0 nello step()
        
        # Feedback
        self.last_touch_team = self.free_kick_team
        self.anchor_recovery_timer = 600  # 10 secondi a 60fps
        
        if self.is_penalty_shootout:
            self.penalty_shootout_phase = 2  # Passa alla fase risultato
            self.penalties_taken += 1

        

    def _apply_positional_anchor(self, idx):
    # Gestione Grace Period
        if idx == 0 and self.post_kick_grace_period > 0:
            self.post_kick_grace_period -= 1

        # Se il free kick è attivo e l'anchor è disabilitato per questo giocatore, non forzarlo
        if getattr(self, 'anchor_disabled', None) is not None and self.anchor_disabled[idx]:
            return

        if self.free_kick_active:
            return
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
        
        if self.free_kick_active:
            return

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
        if self.post_kick_grace_period > 0:
            return False
        
        dist = np.linalg.norm(self.ball_pos - self.players_pos[idx])
        
        # Distanza di contatto aumentata per rendere più facile tirare
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
            
            # LOGICA TIRO MIGLIORATA:
            pos = self.players_pos[idx]
            
            # Per la squadra BLU (attacca a destra)
            if team == 0:
                # Condizioni per tirare per i Blu
                in_attacking_half = pos[0] > self.width * 0.6
                close_to_goal = pos[0] > self.width - 200
                good_angle = 200 < pos[1] < 400
                
                # Probabilità di tiro basata sulla posizione
                if close_to_goal and good_angle:
                    shoot_prob = 0.9
                elif in_attacking_half and good_angle and rel >= 8:
                    shoot_prob = 0.8
                elif in_attacking_half:
                    shoot_prob = 0.6
                else:
                    shoot_prob = 0.3
            
            # Per la squadra ROSSA (attacca a sinistra)
            else:
                # Condizioni per tirare per i Rossi
                in_attacking_half = pos[0] < self.width * 0.4
                close_to_goal = pos[0] < 200
                good_angle = 200 < pos[1] < 400
                
                # Probabilità di tiro basata sulla posizione
                if close_to_goal and good_angle:
                    shoot_prob = 0.9
                elif in_attacking_half and good_angle and rel >= 8:
                    shoot_prob = 0.8
                elif in_attacking_half:
                    shoot_prob = 0.6
                else:
                    shoot_prob = 0.3
            
            # Decidi se tirare
            is_shooting = random.random() < shoot_prob
            
            if is_shooting:
                # Tiro in porta
                if team == 0:  # Blu tira a destra
                    target_x = self.width + 50
                    target_y = 300 + random.uniform(-50, 50)
                else:  # Rossi tirano a sinistra
                    target_x = -50
                    target_y = 300 + random.uniform(-50, 50)
                
                target = np.array([target_x, target_y])
                
                # Aumenta la potenza del tiro
                power_mult = random.uniform(1.5, 2.0)
            else:
                # Passaggio a un compagno
                possible_mates = [m for m in teammates if m != idx]
                if possible_mates:
                    # Preferisci compagni più avanzati
                    mate_weights = []
                    for m in possible_mates:
                        mate_pos = self.players_pos[m]
                        if team == 0:
                            # Per i Blu, preferisci compagni più a destra
                            weight = max(0.1, mate_pos[0] / self.width)
                        else:
                            # Per i Rossi, preferisci compagni più a sinistra
                            weight = max(0.1, (self.width - mate_pos[0]) / self.width)
                        mate_weights.append(weight)
                    
                    # Normalizza i pesi - CORREZIONE CRITICA
                    total_weight = sum(mate_weights)
                    if total_weight > 0:
                        # Calcola le probabilità
                        probabilities = [w/total_weight for w in mate_weights]
                        
                        # ASSICURA CHE LA SOMMA SIA ESATTAMENTE 1
                        # Questo corregge gli errori di floating-point
                        probabilities = np.array(probabilities, dtype=np.float64)
                        probabilities /= probabilities.sum()  # Rinominalizza
                        
                        # Se ancora non somma a 1 (per arrotondamento), corregge
                        if abs(probabilities.sum() - 1.0) > 1e-10:
                            probabilities[-1] += 1.0 - probabilities.sum()
                        
                        mate = np.random.choice(possible_mates, p=probabilities)
                    else:
                        # Se tutti i pesi sono 0, scegli a caso
                        mate = random.choice(possible_mates)
                    
                    target = self.players_pos[mate]
                    power_mult = 1.0
                else:
                    # Se non ci sono compagni, tira a caso
                    target = np.array([random.uniform(0, self.width), 
                                    random.uniform(0, self.height)])
                    power_mult = 0.8

            # CALCOLO DIREZIONE E VELOCITÀ
            direction = target - self.players_pos[idx]
            dist_to_target = np.linalg.norm(direction) + 1e-6
            
            # Aumenta la velocità della palla per i tiri
            base_speed = self.ball_speed * 1.5 if is_shooting else self.ball_speed * 1.2
            self.ball_vel = (direction / dist_to_target) * base_speed * power_mult

            # --- LOGICA EFFETTO A GIRO INTELLIGENTE ---
            self.ball_spin = 0
            
            if is_shooting:
                # Se il tiro è angolato, aggiungi effetto
                angle = np.arctan2(direction[1], direction[0])
                if abs(angle) > 0.3:
                    # Curva verso l'interno per i tiri da fuori
                    self.ball_spin = -0.05 if angle > 0 else 0.05
            else:
                # Per i passaggi lunghi, aggiungi un po' di effetto
                if dist_to_target > 200:
                    self.ball_spin = random.uniform(-0.03, 0.03)

            self.ball_spin += random.uniform(-0.01, 0.01)

    def _collision(self, idx, rewards):
        # Se la palla va molto veloce (sopra 5.5), è "in aria" e nessuno la tocca
        diff = self.ball_pos - self.players_pos[idx]
        dist = np.linalg.norm(diff)
        ball_speed = np.linalg.norm(self.ball_vel)
        if ball_speed >= 5.5:
            if idx != 0 and idx != 11:
                return
        # Riduciamo l'area di interazione per evitare tocchi "fantasma"
        if dist < self.player_radius + self.ball_radius + 2:
            team = "blue" if idx < 11 else "red"
            rewards[team] += 5.0
            
            # Se la palla è molto veloce (gialla), il giocatore la "controlla" 
            # invece di farla schizzare via a caso
            if np.linalg.norm(self.ball_vel) > 5.5:
                self.ball_vel *= 0.2  # Ammortizza il colpo
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
            self.timer_font = pygame.font.SysFont("Arial", 30, bold=True)
            self.big_font = pygame.font.SysFont("Arial", 36, bold=True)
            self.number_font = pygame.font.SysFont("Arial", 14, bold=True)  # Font per numeri

        # Disegna il campo
        self.screen.fill((34, 139, 34))  # Verde campo
        white = (255, 255, 255)
        
        # Linee del campo
        pygame.draw.rect(self.screen, white, (10, 10, self.width-20, self.height-20), 3)
        pygame.draw.line(self.screen, white, (self.width/2, 10), (self.width/2, self.height-10), 3)
        pygame.draw.circle(self.screen, white, (int(self.width/2), int(self.height/2)), 70, 3)
        pygame.draw.rect(self.screen, white, (10, 150, 150, 300), 3)
        pygame.draw.rect(self.screen, white, (self.width-160, 150, 150, 300), 3)
        pygame.draw.rect(self.screen, (200, 200, 200), (-5, 250, 15, 100))
        pygame.draw.rect(self.screen, (200, 200, 200), (self.width-10, 250, 15, 100))

        # Calcola formazioni
        blue_mod = "4-5-1" if self.blue_score > self.red_score else "4-3-3"
        red_mod = "4-5-1" if self.red_score > self.blue_score else "4-3-3"
        
        # --- GESTIONE DELLE SCRITTE ---
        y_offset = 20
        
        # 1. Timer principale
        seconds_left = max(0, self.match_timer // 60)
        timer_text = self.timer_font.render(f"Tempo: {seconds_left}s", True, white)
        self.screen.blit(timer_text, (self.width // 2 - timer_text.get_width() // 2, y_offset))
        y_offset += 40
        
        # 2. Penalty shootout (se attivo)
        if self.is_penalty_shootout:
            penalty_text = self.big_font.render("CALCI DI RIGORE", True, (255, 255, 0))
            self.screen.blit(penalty_text, (self.width // 2 - penalty_text.get_width() // 2, y_offset))
            y_offset += 50
            
            shootout_info = self.font.render(f"Tentativo: {self.penalties_taken}/10", True, white)
            self.screen.blit(shootout_info, (self.width // 2 - shootout_info.get_width() // 2, y_offset))
            y_offset += 40
            
            # 3. VISUALIZZAZIONE RISULTATI RIGORI (quadratini)
            box_size = 15
            box_spacing = 5
            start_x = self.width // 2 - (10 * (box_size + box_spacing)) // 2
            start_y = 10
            
            for i in range(10):
                box_x = start_x + i * (box_size + box_spacing)
                
                if i < len(self.penalty_results):
                    team, scored = self.penalty_results[i]
                    if scored:
                        color = (0, 255, 0)  # Verde per goal
                    else:
                        color = (255, 0, 0)  # Rosso per fallimento
                    
                    # Colore di sfondo per squadra
                    if team == "blue":
                        bg_color = (40, 80, 200, 100)
                    else:
                        bg_color = (200, 40, 40, 100)
                    
                    # Disegna sfondo squadra
                    s = pygame.Surface((box_size, box_size), pygame.SRCALPHA)
                    s.fill(bg_color)
                    self.screen.blit(s, (box_x, start_y))
                    
                    # Disegna quadratino risultato
                    pygame.draw.rect(self.screen, color, (box_x, start_y, box_size, box_size))
                    pygame.draw.rect(self.screen, white, (box_x, start_y, box_size, box_size), 1)
                    
                    # Numero del tentativo
                    num_text = self.number_font.render(str(i+1), True, white)
                    self.screen.blit(num_text, (box_x + 3, start_y + 1))
                else:
                    # Quadratino vuoto per tentativi futuri
                    pygame.draw.rect(self.screen, (100, 100, 100), (box_x, start_y, box_size, box_size))
                    pygame.draw.rect(self.screen, white, (box_x, start_y, box_size, box_size), 1)
            
            y_offset = start_y + box_size + 20
        
        # 4. Free kick / Penalty
        if self.free_kick_active and not self.is_penalty_shootout:
            if self.is_penalty:
                penalty_text = self.big_font.render("RIGORE", True, (255, 0, 0))
                self.screen.blit(penalty_text, (self.width // 2 - penalty_text.get_width() // 2, y_offset))
            else:
                freekick_text = self.big_font.render("FREE KICK", True, (255, 255, 0))
                self.screen.blit(freekick_text, (self.width // 2 - freekick_text.get_width() // 2, y_offset))
            y_offset += 50
        
        # 5. Formazioni (in angoli)
        blue_info = self.small_font.render(f"BLU: {blue_mod}", True, (100, 200, 255))
        red_info = self.small_font.render(f"RED: {red_mod}", True, (255, 100, 100))
        self.screen.blit(blue_info, (50, 20))
        self.screen.blit(red_info, (self.width - red_info.get_width() - 50, 20))
        
        # 6. Punteggio (in basso)
        score_text = self.font.render(f"BLUE {self.blue_score} - {self.red_score} RED", True, white)
        self.screen.blit(score_text, (self.width // 2 - score_text.get_width() // 2, self.height - 40))
        
        # --- DISEGNO GIOCATORI CON NUMERI ---
        for i in range(22):
            # Colore in base a ruolo e stato
            if i == 0 or i == 11:
                color = (240, 240, 240)  # Portieri
            elif i < 11:
                color = (40, 80, 200)    # Blu
            else:
                color = (200, 40, 40)    # Rossi
            
            # Gestione stati (stun, slow)
            if not self.free_kick_active:
                if self.stun_timer[i] > 0:
                    color = (100, 100, 100)  # Stunnato
                elif self.slow_timer[i] > 0:
                    color = (160, 160, 160)  # Rallentato

            # Disegna giocatore con bordo bianco
            pos = self.players_pos[i].astype(int)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 11)
            pygame.draw.circle(self.screen, color, pos, 9)
            
            # Aggiungi numero del giocatore
            if i < 11:
                player_num = i  # 0-10 per i blu
            else:
                player_num = i - 11  # 0-10 per i rossi
            
            # I portieri sono sempre 0
            if i == 0 or i == 11:
                player_num = 0
            
            num_text = self.number_font.render(str(player_num), True, (0, 0, 0))
            text_rect = num_text.get_rect(center=pos)
            self.screen.blit(num_text, text_rect)
        
        # --- DISEGNO PALLA ---
        ball_speed_norm = np.linalg.norm(self.ball_vel)
        is_being_dribbled = any(self.dribble_timer > 0)
        
        if is_being_dribbled:
            ball_color = (0, 255, 127)   # VERDE: Dribbling
        elif ball_speed_norm > 5.5 or self.is_goal_kick:
            ball_color = (255, 255, 0)   # GIALLO: Volo/Rinvio
        elif self.free_kick_active:
            ball_color = (255, 255, 0)   # GIALLO: Calcio piazzato
        else:
            ball_color = (255, 255, 255) # BIANCO: Normale
        
        ball_pos_int = self.ball_pos.astype(int)
        pygame.draw.circle(self.screen, (0, 0, 0), ball_pos_int, 7)
        pygame.draw.circle(self.screen, ball_color, ball_pos_int, 6)
        
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
