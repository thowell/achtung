import random
import sys
import time
import pygame
import pygame.gfxdraw

from math import *
from pygame.locals import *

# game size
WINDOW_HEIGHT = 100
WINDOW_WIDTH = 100
WINDOW_BORDER = 10

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0,255,255)
ORANGE = (255, 69, 0)
COLORS = [MAGENTA,GREEN,CYAN,ORANGE]
COLORS_str = ['magenta','green','cyan','orange']

# keys
LEFT_KEYS = [pygame.K_LEFT, pygame.K_a, pygame.K_v, pygame.K_k]
LEFT_KEYS_str = ['<-', 'a', 'v', 'k']
RIGHT_KEYS = [pygame.K_RIGHT, pygame.K_s, pygame.K_b, pygame.K_l]
RIGHT_KEYS_str = ['->','s','b','l']

class Player():
    def __init__(self):
        self.active = True
        self.color = None
        self.score = 0
        self.radius = 2
        self.x = 0
        self.y = 0
        self.angle = 0

    def gen(self, game):
        self.x = random.randrange(WINDOW_BORDER, game.window_width - WINDOW_BORDER)
        self.y = random.randrange(WINDOW_BORDER, game.window_height - WINDOW_BORDER)
        self.angle = random.randrange(0, 360)

    def move(self):
        self.x += int(self.radius * 2 * cos(radians(self.angle)))
        self.y += int(self.radius * 2 * sin(radians(self.angle)))

    def draw(self, game):
        pygame.gfxdraw.aacircle(game.display, self.x,
                                self.y, self.radius, self.color)
        pygame.gfxdraw.filled_circle(
            game.display, self.x, self.y, self.radius, self.color)

    def collision(self, game):
        if (self.x > game.window_width-game.window_buffer or self.x < game.window_buffer or
            self.y > game.window_height-game.window_buffer or self.y < game.window_buffer or
                game.display.get_at((self.x, self.y)) != BLACK):
            self.active = False
            return True
        else:
            return False

    def angle_reset(self):
        if self.angle < 0:
            self.angle += 360
        elif self.angle >= 360:
            self.angle -= 360

class Achtung():
    def __init__(self,n):
        # pygame
        self.speed = 10 
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.window_buffer = 1
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.display = pygame.Surface(self.screen.get_size())
        self.render_game = True
        self.cache_frames = False

        # game
        self.game_over = True
        self.first_step = True
        self.n = n
        self.players = self.init_players(n)
        self.players_active = len(self.players)
        self.rnd = 1
        self.frame = 1
        self.games = 1
        self.action = "left"

    def render(self):
        self.screen.blit(self.display, (0, 0))
    
    def state(self):
        return pygame.surfarray.array3d(self.display)

    def init_players(self,n):
        # generate players
        players = [Player() for i in range(n)]

        for i in range(n):
            players[i].gen(self)
            players[i].color = COLORS[i]
            players[i].left_key = LEFT_KEYS[i]
            players[i].right_key = RIGHT_KEYS[i]

        return players

    def reset_colors(self):
        for i in range(self.n):
            self.players[i].color = COLORS[i]

    def reset(self):
        self.game_over = False
        self.first_step = True
        self.players_active = self.n
        self.frame = 0

        for i in range(self.n):
            self.players[i].gen(self)
            self.players[i].active = True

        # pygame.time.wait(1000)
        self.display.fill(BLACK)
    
        return self.state()
  
    def check_first_step(self):
        if self.first_step:
            print('Round %i' % (self.rnd))
            self.first_step = False

    def holes(self):
        hole = random.randrange(1, 20)
        for i in range(self.n):
            if hole == i+5 and self.players[i].active:
                self.players[i].move()
                self.players[i].color = BLACK

    def update_players(self,actions):
         # reset players' colors
        self.reset_colors()

        # random holes
        self.holes()

        for i in range(self.n):
            # actions
            if actions[i] == 1:
                self.players[i].angle -= 10
            elif actions[i] == 2:
                self.players[i].angle += 10
            else:
                None

            # update
            if self.players[i].active and ((self.players_active > 1 and self.n > 1) or (self.players_active > 0 and self.n == 1)):
                self.players[i].angle_reset()

                # checking if someone fails
                if self.players[i].collision(self):
                    self.players_active -= 1

                self.players[i].draw(self)
                self.players[i].move()
    
    def round_over(self):
        if (self.players_active == 1 and self.n > 1) or (self.players_active == 0 and self.n == 1):
            self.game_over = True
            self.rnd += 1
            for (i,p) in enumerate(self.players):
                if p.active == True:
                    print(" " + COLORS_str[i] + " wins")

    def rewards(self):
        return [1.0*self.players[i].active for i in range(self.n)]
        
    def step(self,actions):

        # check first step
        self.check_first_step()

        # update players
        self.update_players(actions)

        # update screen
        if self.render_game:  self.render()
        pygame.display.update()

        # check round over
        self.round_over()

        # game frames
        self.fps_clock.tick(self.speed)
        self.frame += 1

        # cache frames
        if self.cache_frames:
            filename = "images/{}_{}.JPG"
            pygame.image.save(self.display, filename.format(self.rnd,self.frame))

        return self.state(), self.rewards(), self.game_over, {}

def number_players(argv):
    # input number of players
    if len(argv) > 1:
        _n = int(argv[1])
        if _n < 5 and _n > 0:
            n = _n
        else:
            print('Invalid number of players, setting to: 2')
            n = 2
    else:
        n = 2

    print('  %i players' % (n))
    for i in range(n):
        print('     [%s] (%s,%s)' %
              (COLORS_str[i], LEFT_KEYS_str[i], RIGHT_KEYS_str[i]))

    return n

def keyboard_input(game,actions):
    for event in pygame.event.get():
            if event.type == QUIT:
                shutdown()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    shutdown()

    keys = pygame.key.get_pressed()
    for i in range(game.n):
        if keys[game.players[i].left_key]:
            actions[i] = 1
        if keys[game.players[i].right_key]:
            actions[i] = 2

def agent(n):
    return [0 for i in range(n)]
    

def main(argv):
    print('Achtung Die Kurve!')

    # get number of players
    n = number_players(argv)

    # setup
    done = True
    pygame.init()
    game = Achtung(n)

    # game
    while True:
        if done:
            obs = game.reset()

        # agent
        actions = agent(game.n)

        # keyboard input override
        keyboard_input(game,actions)

        # step
        obs, rewards, done, info = game.step(actions)
        
        reward = rewards[0]
        action = actions[0]
        # print("reward: ", reward)

if __name__ == '__main__':
    main(sys.argv)

