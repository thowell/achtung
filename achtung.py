import numpy as np
import random
import sys
import time
import pygame
import pygame.gfxdraw
import os

from math import *
from pygame.locals import *

import gym 
from gym import spaces

import matplotlib.pyplot as plt 

pygame.init()

# game size
WINDOW_HEIGHT = 320
WINDOW_WIDTH = 320
WINDOW_BORDER = 10

# observation
DOWNSCALE = 4
OBS_HEIGHT = int(np.floor(WINDOW_HEIGHT / DOWNSCALE))
OBS_WIDTH = int(np.floor(WINDOW_WIDTH / DOWNSCALE))

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

class Achtung(gym.Env):
    def __init__(self,n=1):
        print('Achtung Die Kurve!')
        pygame.display.set_caption('Achtung Die Kurve!')
        
        # pygame
        self.speed = 12
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.window_buffer = 1
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.display = pygame.Surface(self.screen.get_size())
        self.render_game = False #True
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
        self.verbose = True
        self.current_player = 0
        self.state_cache = np.resize(np.dot(np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8), [0.299, 0.587, 0.114])[::DOWNSCALE, ::DOWNSCALE], (OBS_WIDTH, OBS_HEIGHT, 1))

        # gym
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(OBS_HEIGHT, OBS_WIDTH, 1), dtype=np.uint8)

        if self.render_game == False:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.reset()

    def render(self):
        self.screen.blit(self.display, (0, 0))
    
    def state(self):
        if self.current_player == 0:
            self.state_cache = np.resize(np.dot(np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8), [0.299, 0.587, 0.114])[::DOWNSCALE, ::DOWNSCALE], (OBS_WIDTH, OBS_HEIGHT, 1))

        return self.state_cache

    def init_players(self,n):
        # generate players
        players = [Player() for i in range(n)]

        for i in range(n):
            players[i].gen(self)
            players[i].color = COLORS[i]
            players[i].left_key = LEFT_KEYS[i]
            players[i].right_key = RIGHT_KEYS[i]

        return players

    def reset_color(self):
        self.players[self.current_player].color = COLORS[self.current_player]

    def reset(self, mode='human'):
        self.game_over = False
        self.first_step = True
        self.players_active = self.n
        self.current_player = 0
        self.frame = 0

        self.display.fill(BLACK)

        for i in range(self.n):
            self.players[i].active = True
            self.players[i].gen(self)
            self.players[i].draw(self)
            self.players[i].color = COLORS[i]

    
        return self.state()

    def close(self):
        shutdown()

    def check_first_step(self):
        if self.first_step:
            # print('Round %i' % (self.rnd))
            self.first_step = False

    def hole(self):
        hole = random.randrange(1, 20)
        i = self.current_player
        if hole == i+5 and self.players[i].active:
            self.players[i].move()
            self.players[i].color = BLACK

    def update_player(self,action):
        # current player
        i = self.current_player
        player = self.players[i]

         # reset players' colors
        self.reset_color()

        # random hole
        self.hole()

        # action
        if action == 0:
            player.angle -= 10
        elif action == 1:
            player.angle += 10
        elif action == 2:
            None
        else:
            None

        # update
        if player.active and ((self.players_active > 1 and self.n > 1) or (self.players_active > 0 and self.n == 1)):
            player.angle_reset()

            # checking if someone fails
            if player.collision(self):
                self.players_active -= 1

            player.draw(self)
            player.move()

    def round_over(self):
        if (self.players_active == 1 and self.n > 1) or (self.players_active == 0 and self.n == 1):
            self.game_over = True
            self.rnd += 1
    
    def reward(self):
        if self.game_over == False:
            return 1.0 # nominal reward
        else:
            if self.players[self.current_player].active:
                return 10.0 # winning reward
            else:
                return -1.0 # losing reward
    
    def to_play(self):
        return self.current_player

    def legal_actions(self):
        return range(3)
        
    def step(self,action):

         # current state
        state = self.state()

        # check first step
        self.check_first_step()

        # update current player
        self.update_player(action)

        # check round over
        self.round_over()

        # get reward
        reward = self.reward()

        # check for done
        if self.game_over and self.current_player == self.n-1:
            done = True
        else:
            done = False

        # game frames
        if self.current_player == self.n-1:  
            if self.render_game: 
                self.render()
                self.fps_clock.tick(self.speed)
            self.frame += 1
        pygame.display.update()

        # cache frames
        if self.cache_frames:
            filename = "images/{}_{}.JPG"
            pygame.image.save(self.display, filename.format(self.rnd,self.frame))

        # update current player
        self.current_player += 1
        if self.current_player >= self.n:
            self.current_player = 0

        return state, reward, done, {}

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
            (game.frame != 0 and game.display.get_at((self.x, self.y)) != BLACK)):
            self.active = False
            return True
        else:
            return False

    def angle_reset(self):
        if self.angle < 0:
            self.angle += 360
        elif self.angle >= 360:
            self.angle -= 360

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

def keyboard_input(game):
    for event in pygame.event.get():
        if event.type == QUIT:
            shutdown()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                shutdown()

    action = 2

    i = game.current_player

    keys = pygame.key.get_pressed()

    if keys[game.players[i].left_key]:
        action = 0
    if keys[game.players[i].right_key]:
        action = 1

    return action

def main(argv):
    # get number of players
    n = number_players(argv)

    # setup
    done = True
    game = Achtung(n)
    game.render_game = True
    game.cache_frames = False
    
    obs = game.reset()

    # print(obs.shape)
    # print(OBS_HEIGHT)
    # print(OBS_WIDTH)
    # game
    while True:
        # print(np.max(obs))
        # print(np.min(obs))
        # plt.imshow(np.resize(obs, (OBS_HEIGHT, OBS_WIDTH)), cmap="gray") 
        # plt.show()

        if done:
            for (i,p) in enumerate(game.players):
                if p.active == True:
                    print(" " + COLORS_str[i] + " wins")
            obs = game.reset()

        for i in range(game.n):
            # keyboard input
            action = keyboard_input(game)

            # step
            obs, reward, done, info = game.step(action)
            
            print("player: %s" % COLORS_str[i])
            print("     reward: ", reward)
            print("     action: ", action)
            print("     frame: ", game.frame)

if __name__ == '__main__':
    main(sys.argv)

