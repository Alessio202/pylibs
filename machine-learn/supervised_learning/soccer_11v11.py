import math
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# CALLBACK PER SALVATAGGIO AUTOMATICO E GESTIONE EVENTI
# ==========================================
class SaveOnInterruptCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveOnInterruptCallback, self).__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.training_env.get_attr("render_mode")[0] == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nChiusura rilevata. Salvataggio in corso...")
                    return False 
        return True

    def _on_training_end(self) -> None:
        print(f"\n💾 Modello salvato su: {self.save_path}")
        self.model.save(self.save_path)

# ==========================================
# ENVIRONMENT 11v11 SOCCER PRO
# ==========================================
class Soccer11v11Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.width, self.height = 1000, 600
        self.player_radius, self.ball_radius = 9, 6
        self.max_speed, self.friction = 5.0, 0.85 
        self.ball_speed_base = 7.5
        self.dt = 1.0

        self.action_space = spaces.MultiDiscrete([7] * 22)
        self.observation_space = spaces.Box(low=0, high=1, shape=(46,), dtype=np.float32)

        self.players_pos = np.zeros((22, 2), dtype=np.float32)
        self.players_vel = np.zeros((22, 2), dtype=np.float32)
        self.ball_pos, self.ball_vel = np.zeros(2), np.zeros(2)

        self.render_mode = render_mode
        self.screen, self.font = None, None
        self.blue_score, self.red_score = 0, 0
        
        # Porta e Aree
        self.goal_width = 100 
        self.goal_y_min = (self.height - self.goal_width) // 2
        self.goal_y_max = (self.height + self.goal_width) // 2

        self._reset_positions()

    def _team_of(self, idx): return 0 if idx < 11 else 1

    def _get_role(self, idx):
        rel = idx % 11
        if rel == 0: return "GK"
        if rel <= 4: return "DF"
        if rel <= 7: return "MF"
        return "FW"

    def _get_formation(self, team):
        # Logica: Se perdi di 2 o più, passi al 4-3-3 (offensivo), altrimenti 4-4-2 (equilibrato)
        diff = (self.red_score - self.blue_score) if team == 0 else (self.blue_score - self.red_score)
        return "4-3-3" if diff >= 2 else "4-4-2"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions()
        return self._get_obs(), {}

    def _reset_positions(self):
        self.ball_pos = np.array([self.width/2, self.height/2], dtype=np.float32)
        self.ball_vel[:] = 0
        self.players_vel[:] = 0 
        
        for t in range(2):
            off = 0 if t == 0 else 11
            form = self._get_formation(t)
            # Portiere
            self.players_pos[off] = [50 if t==0 else self.width-50, self.height/2]
            # Difesa (sempre 4)
            for i in range(1, 5):
                self.players_pos[off+i] = [self.width*(0.15 if t==0 else 0.85), self.height*(i/5)]
            
            if form == "4-3-3":
                for i in range(5, 8): # 3 Centrocampisti
                    self.players_pos[off+i] = [self.width*(0.35 if t==0 else 0.65), self.height*((i-4)/4)]
                for i in range(8, 11): # 3 Attaccanti
                    self.players_pos[off+i] = [self.width*(0.60 if t==0 else 0.40), self.height*((i-7)/4)]
            else: # 4-4-2
                for i in range(5, 9): # 4 Centrocampisti
                    self.players_pos[off+i] = [self.width*(0.40 if t==0 else 0.60), self.height*((i-4)/5)]
                for i in range(9, 11): # 2 Attaccanti
                    self.players_pos[off+i] = [self.width*(0.65 if t==0 else 0.35), self.height*((i-8)/3)]

    def _get_obs(self):
        return np.concatenate([self.players_pos.flatten()/self.width, self.ball_pos/self.width]).astype(np.float32)

    def step(self, action):
        reward = 0.0
        for i in range(22): self._move_player_smooth(i, action[i])
        for i in range(22):
            if action[i] >= 5: self._try_kick(i, 0.6 if action[i]==5 else 1.3)

        self.ball_pos += self.ball_vel * self.dt
        self.ball_vel *= 0.96 
        for i in range(22): self._player_ball_collision(i)

        # Reward: penalità distanza palla per evitare pigrizia
        for i in range(22):
            reward -= np.linalg.norm(self.players_pos[i] - self.ball_pos) * 0.00001

        goal = False
        if self.goal_y_min < self.ball_pos[1] < self.goal_y_max:
            if self.ball_pos[0] < 0:
                self.red_score += 1; reward -= 20; goal = True
            elif self.ball_pos[0] > self.width:
                self.blue_score += 1; reward += 20; goal = True

        if goal: self._reset_positions()
        if self.render_mode == "human": self.render()
        return self._get_obs(), reward, False, False, {}

    def _move_player_smooth(self, idx, act):
        t_vel = np.zeros(2)
        if act == 1: t_vel[1] = -self.max_speed
        elif act == 2: t_vel[1] = self.max_speed
        elif act == 3: t_vel[0] = -self.max_speed
        elif act == 4: t_vel[0] = self.max_speed
        
        self.players_vel[idx] = self.players_vel[idx]*self.friction + t_vel*(1-self.friction)
        self.players_pos[idx] += self.players_vel[idx]
        
        # VINCOLI RUOLI TATTICI
        role, team = self._get_role(idx), self._team_of(idx)
        px, py = self.players_pos[idx]
        
        if role == "GK":
            px = np.clip(px, 5, 120) if team == 0 else np.clip(px, 880, 995)
        elif role == "DF":
            px = np.clip(px, 5, 500) if team == 0 else np.clip(px, 500, 995)
        elif role == "FW":
            px = np.clip(px, 400, 995) if team == 0 else np.clip(px, 5, 600)

        self.players_pos[idx] = [np.clip(px, 5, self.width-5), np.clip(py, 5, self.height-5)]

    def _try_kick(self, idx, strength):
        dist = np.linalg.norm(self.ball_pos - self.players_pos[idx])
        if dist < self.player_radius + self.ball_radius + 5:
            target_x = self.width + 100 if self._team_of(idx) == 0 else -100
            dir_v = np.array([target_x - self.players_pos[idx,0], self.height/2 - self.players_pos[idx,1]])
            self.ball_vel = (dir_v / (np.linalg.norm(dir_v)+1e-6)) * self.ball_speed_base * strength

    def _player_ball_collision(self, idx):
        diff = self.ball_pos - self.players_pos[idx]
        dist = np.linalg.norm(diff)
        if dist < self.player_radius + self.ball_radius:
            self.ball_vel += (diff / (dist + 1e-6)) * 1.5
            self.ball_pos = self.players_pos[idx] + (diff / (dist + 1e-6)) * (self.player_radius + self.ball_radius)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont("Arial", 18, bold=True)
        
        self.screen.fill((34, 139, 34)) # Erba
        # Disegno Campo
        pygame.draw.rect(self.screen, (255,255,255), (0,0,self.width,self.height), 3) # Bordi
        pygame.draw.line(self.screen, (255,255,255), (self.width/2, 0), (self.width/2, self.height), 2) # Metà
        pygame.draw.circle(self.screen, (255,255,255), (self.width//2, self.height//2), 70, 2) # Cerchio
        # Aree di rigore
        pygame.draw.rect(self.screen, (255,255,255), (0, 120, 150, 360), 2)
        pygame.draw.rect(self.screen, (255,255,255), (850, 120, 150, 360), 2)
        # Porte
        pygame.draw.rect(self.screen, (200,200,200), (-5, self.goal_y_min, 15, self.goal_width))
        pygame.draw.rect(self.screen, (200,200,200), (self.width-10, self.goal_y_min, 15, self.goal_width))
        
        for i in range(22):
            color = (0, 100, 255) if i < 11 else (220, 20, 20)
            pygame.draw.circle(self.screen, (255,255,255), self.players_pos[i].astype(int), self.player_radius+1)
            pygame.draw.circle(self.screen, color, self.players_pos[i].astype(int), self.player_radius)
        
        # Palla con bordo
        pygame.draw.circle(self.screen, (0,0,0), self.ball_pos.astype(int), self.ball_radius+1)
        pygame.draw.circle(self.screen, (255,255,255), self.ball_pos.astype(int), self.ball_radius)
        
        txt = f"BLUE: {self.blue_score} ({self._get_formation(0)}) | RED: {self.red_score} ({self._get_formation(1)})"
        surf = self.font.render(txt, True, (255, 255, 255))
        self.screen.blit(surf, (self.width//2 - surf.get_width()//2, 10))
        pygame.display.flip()

def train():
    model_path = "soccer_11v11_model"
    # render_mode="human" ti permette di vedere il training live
    env = Soccer11v11Env(render_mode="human")
    env = Monitor(env)
    
    # Parametri ottimizzati per velocità e fluidità (n_steps bassi = aggiornamenti frequenti)
    ppo_config = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "learning_rate": 0.0005, # Leggermente più alto per imparare più in fretta
        "n_steps": 512,          # Aggiorna ogni 512 passi (riduce drasticamente la pausa)
        "batch_size": 128,       # Elaborazione veloce
        "n_epochs": 4,           # Meno ripetizioni per ogni aggiornamento (accelera il ciclo)
        "ent_coef": 0.05,        # Incoraggia l'esplorazione
    }

    try:
        # Carica il modello esistente e forza l'uso dei nuovi parametri
        model = PPO.load(model_path, env=env, custom_objects={
            "n_steps": ppo_config["n_steps"],
            "batch_size": ppo_config["batch_size"],
            "learning_rate": ppo_config["learning_rate"]
        })
        print("🔄 Modello caricato con n_steps=512. Ripresa addestramento...")
    except:
        # Crea un nuovo modello se non esiste
        model = PPO(**ppo_config)
        print("🆕 Nuovo modello PPO creato con n_steps=512.")

    callback = SaveOnInterruptCallback(save_path=model_path)
    
    try:
        # Avvio addestramento con barra di progresso
        model.learn(
            total_timesteps=1_000_000, 
            callback=callback,
            progress_bar=True
        )
    except (KeyboardInterrupt, pygame.error):
        print("\n🛑 Training interrotto dall'utente.")
    finally:
        print("💾 Salvataggio di sicurezza...")
        model.save(model_path)
        pygame.quit()

if __name__ == "__main__":
    train()
