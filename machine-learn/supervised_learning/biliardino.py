import pygame
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# --- COSTANTI ---
WIDTH, HEIGHT = 1000, 600
BALL_RADIUS = 15
PLAYER_WIDTH = 20
PLAYER_HEIGHT = 40
ROD_WIDTH = 12
MODEL_FILE_BLUE = "biliardino_blue.pth"
MODEL_FILE_RED = "biliardino_red.pth"
MAX_BALL_SPEED = 12
AI_DECISION_DELAY = 100 # Ridotto per rendere il gioco più fluido e reattivo

# Colori
GREEN_FIELD = (34, 139, 34)
STRIPE_GREEN = (30, 130, 30)
WHITE = (255, 255, 255)
RED = (220, 40, 40)
BLUE = (40, 80, 220)
SILVER = (180, 180, 180)
GOLD = (255, 215, 0)
BLACK = (30, 30, 30)
SKIN = (240, 200, 160)

pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Biliardino AI vs AI - Evoluzione Neurale")
CLOCK = pygame.time.Clock()

# --- MODELLO AI (PyTorch) ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Brain:
    def __init__(self, team_name, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 12 # Ball (x,y,vx,vy) + 8 posizioni stecche
        self.output_dim = 4 # 0: su, 1: giù, 2: ruota+, 3: ruota-
        self.model = DQN(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.5
        self.model_path = model_path
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.epsilon = 0.1
                print(f"Modello {team_name} caricato.")
            except:
                print(f"Nuovo modello per {team_name}.")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return self.model(state_t).argmax().item()

    def train(self, state, action, reward, next_state):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        
        current_q = self.model(state)[action]
        with torch.no_grad():
            next_q = self.model(next_state).max()
        target_q = reward + 0.98 * next_q
        
        loss = self.criterion(current_q.view(-1), target_q.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > 0.05: self.epsilon *= 0.99999

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

# --- CLASSI DI GIOCO ---
class Rod:
    def __init__(self, x, n_players, team):
        self.x = x
        self.n = n_players
        self.team = team
        self.y = HEIGHT // 2
        self.angle = 0
        if self.n == 5: self.spacing = 110 
        elif self.n == 2: self.spacing = 160 
        elif self.n == 3: self.spacing = 180
        else: self.spacing = 0

    def get_player_positions(self):
        if self.n == 1: return [self.y]
        total_h = self.spacing * (self.n - 1)
        return [self.y - total_h/2 + i * self.spacing for i in range(self.n)]

    def apply_action(self, action):
        # Azioni: 0: Up, 1: Down, 2: Kick Fwd, 3: Reset/Back
        if action == 0: self.y -= 8
        elif action == 1: self.y += 8
        elif action == 2: self.angle = min(1.4, self.angle + 0.3)
        elif action == 3: self.angle = max(-0.5, self.angle - 0.3)
        
        limit = (self.spacing * (self.n - 1) / 2) + 50 if self.n > 1 else 60
        self.y = max(limit, min(HEIGHT - limit, self.y))

    def draw(self, surf):
        color = RED if self.team == "red" else BLUE
        pygame.draw.line(surf, SILVER, (self.x, 20), (self.x, HEIGHT-20), ROD_WIDTH)
        for py in self.get_player_positions():
            foot_ext = math.sin(self.angle) * 35
            rect = pygame.Rect(0, 0, PLAYER_WIDTH, PLAYER_HEIGHT)
            rect.center = (self.x + foot_ext, py)
            pygame.draw.rect(surf, color, rect, border_radius=5)
            pygame.draw.rect(surf, BLACK, rect, 2, border_radius=5)
            pygame.draw.circle(surf, SKIN, (int(self.x), int(py)), 10)

class Ball:
    def __init__(self):
        self.reset()
    def reset(self):
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.vx, self.vy = random.choice([-6, 6]), random.uniform(-3, 3)
    def move(self, rods):
        self.x += self.vx
        self.y += self.vy
        speed = math.hypot(self.vx, self.vy)
        if speed > MAX_BALL_SPEED:
            self.vx *= (MAX_BALL_SPEED / speed)
            self.vy *= (MAX_BALL_SPEED / speed)

        if self.y <= 35 or self.y >= HEIGHT - 35:
            self.vy *= -0.9
            self.y = max(35, min(HEIGHT-35, self.y))
        
        if self.x <= 20 or self.x >= WIDTH - 20:
            if HEIGHT//2 - 80 < self.y < HEIGHT//2 + 80:
                # Ritorna chi ha segnato: 1 per Rosso, 2 per Blu
                scorer = 1 if self.x > WIDTH/2 else 2
                self.reset()
                return scorer 
            else:
                self.vx *= -0.8
                self.x = max(20, min(WIDTH-20, self.x))

        for r in rods:
            if abs(self.x - r.x) < 50:
                for py in r.get_player_positions():
                    fx = r.x + math.sin(r.angle)*35
                    if math.hypot(self.x - fx, self.y - py) < BALL_RADIUS + 15:
                        self.vx = -self.vx * 1.1 + math.sin(r.angle)*5
                        self.vy += (self.y - py) * 0.2
                        self.x += self.vx
                        return r.team # Ritorna il team che ha toccato la palla
        return None

def draw_field():
    SCREEN.fill(GREEN_FIELD)
    for i in range(0, WIDTH, 100):
        pygame.draw.rect(SCREEN, STRIPE_GREEN, (i, 20, 50, HEIGHT-40))
    pygame.draw.rect(SCREEN, WHITE, (20, 20, WIDTH-40, HEIGHT-40), 3)
    pygame.draw.line(SCREEN, WHITE, (WIDTH//2, 20), (WIDTH//2, HEIGHT-20), 2)
    pygame.draw.circle(SCREEN, WHITE, (WIDTH//2, HEIGHT//2), 70, 2)
    pygame.draw.rect(SCREEN, BLACK, (5, HEIGHT//2-80, 15, 160))
    pygame.draw.rect(SCREEN, BLACK, (WIDTH-20, HEIGHT//2-80, 15, 160))

# --- SETUP ---
LAYOUT = [(80,1,"red"), (180,2,"red"), (300,3,"blue"), (420,5,"red"),
          (580,5,"blue"), (700,3,"red"), (820,2,"blue"), (920,1,"blue")]
rods = [Rod(x, n, team) for x, n, team in LAYOUT]

brain_red = Brain("RED_TEAM", MODEL_FILE_RED)
brain_blue = Brain("BLUE_TEAM", MODEL_FILE_BLUE)

ball = Ball()
running = True

# Struttura per memorizzare stato e azione precedente di ogni stecca per il training
last_rod_data = {i: {"state": None, "action": 0} for i in range(len(rods))}

while running:
    draw_field()
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 1. Preparazione Stato Comune
    current_state = [ball.x/WIDTH, ball.y/HEIGHT, ball.vx/10, ball.vy/10] + [r.y/HEIGHT for r in rods]

    # 2. IA Decision Making (Ogni stecca agisce indipendentemente)
    for i, r in enumerate(rods):
        brain = brain_red if r.team == "red" else brain_blue
        
        # Sceglie azione
        action = brain.select_action(current_state)
        r.apply_action(action)
        
        # Salva per training successivo
        last_rod_data[i]["state"] = current_state
        last_rod_data[i]["action"] = action

    # 3. Fisica
    collision_result = ball.move(rods)

    # 4. Reward e Training
    next_state = [ball.x/WIDTH, ball.y/HEIGHT, ball.vx/10, ball.vy/10] + [r.y/HEIGHT for r in rods]
    
    for i, r in enumerate(rods):
        reward = 0
        # Reward per il tocco palla
        if collision_result == r.team: reward = 2
        # Reward per il Goal
        if collision_result == 1: # Goal Rosso
            reward = 10 if r.team == "red" else -10
        elif collision_result == 2: # Goal Blu
            reward = 10 if r.team == "blue" else -10
        
        # Train
        brain = brain_red if r.team == "red" else brain_blue
        if last_rod_data[i]["state"] is not None:
            brain.train(last_rod_data[i]["state"], last_rod_data[i]["action"], reward, next_state)

    # 5. Rendering Ball
    pygame.draw.circle(SCREEN, WHITE, (int(ball.x), int(ball.y)), BALL_RADIUS)
    pygame.draw.circle(SCREEN, BLACK, (int(ball.x), int(ball.y)), BALL_RADIUS, 2)
    
    for r in rods: r.draw(SCREEN)
    
    pygame.display.flip()
    CLOCK.tick(60)

brain_red.save()
brain_blue.save()
pygame.quit()
