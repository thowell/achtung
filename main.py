import random
import sys
import time
import pygame
import pygame.gfxdraw

from math import *
from pygame.locals import *

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

def main(argv):
    print('Achtung Die Kurve!')

    # input number of players
    if len(argv) > 1:
        _n = int(argv[1])
        if _n < 5 and _n > 0:
            n = _n
        else:
            n = 2
    else:
        n = 2

    print('  %i players' % (n))
    for i in range(n):
        print('     [%s] (%s,%s)' % (COLORS_str[i],LEFT_KEYS_str[i],RIGHT_KEYS_str[i]))

    # setup
    pygame.init()
    env = Environment()
    players = init_players(env,n)

    while True:
        rungame(env,players)

class Environment():
    def __init__(self):
        self.speed = 10 
        self.window_width = 500 
        self.window_height = 500 
        self.window_buffer = 1
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((500,500))
        self.display = pygame.Surface(self.screen.get_size())

class Player():
    def __init__(self):
        self.active = True
        self.color = None
        self.score = 0
        self.radius = 3
        self.x = 0
        self.y = 0
        self.angle = 0

    def gen(self,env):
        self.x = random.randrange(50, env.window_width - 50)
        self.y = random.randrange(50, env.window_height - 50)
        self.angle = random.randrange(0, 360)

    def move(self):
        self.x += int(self.radius * 2 * cos(radians(self.angle)))
        self.y += int(self.radius * 2 * sin(radians(self.angle)))

    def draw(self,env):
        pygame.gfxdraw.aacircle(env.display, self.x, self.y, self.radius, self.color)
        pygame.gfxdraw.filled_circle(env.display, self.x, self.y, self.radius, self.color)

    def collision(self,env):
        if (self.x > env.window_width-env.window_buffer or self.x < env.window_buffer or
            self.y > env.window_height-env.window_buffer or self.y < env.window_buffer or
            env.display.get_at((self.x, self.y)) != BLACK):
            self.active = False
            return True
        else:
            return False

    def angle_reset(self):
        if self.angle < 0:
            self.angle += 360
        elif self.angle >= 360:
            self.angle -= 360


def init_players(env,n):
    # generate players
    players = [Player() for i in range(n)]

    for i in range(n):
        players[i].gen(env)
        players[i].color = COLORS[i]
        players[i].left_key = LEFT_KEYS[i]
        players[i].right_key = RIGHT_KEYS[i]

    return players

def reset_colors(players):
    n = len(players)
    for i in range(n):
        players[i].color = COLORS[i]
        
def rungame(env,players):
    env.display.fill(BLACK)

    first = True
    run = True
    n = len(players)
    players_active = n

    max_score = 100
    rnd = 1

    while run:
        if first : print('Round %i' % (rnd))
        # reset players' colors
        reset_colors(players)

        # generating random holes
        hole = random.randrange(1, 20)
        for i in range(n):
            if hole == i+5 and players[i].active:
                players[i].move()
                players[i].color = BLACK
        
        # update players
        for i in range(n):  
            if players[i].active and ((players_active > 1 and n > 1) or (players_active > 0 and n == 1)):
                players[i].angle_reset()

                # checking if someone fails
                if players[i].collision(env):
                    players_active -= 1

                players[i].draw(env)
                players[i].move()

        for event in pygame.event.get():
            if event.type == QUIT:
                shutdown()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    shutdown()

        # input from keyboard
        keys = pygame.key.get_pressed()
        for i in range(n):
            if keys[players[i].left_key]:
                players[i].angle -= 10
            if keys[players[i].right_key]:
                players[i].angle += 10

        env.screen.blit(env.display, (0, 0))
        pygame.display.update()

        # check round over
        if (players_active == 1 and n > 1) or (players_active == 0 and n == 1):
            for i in range(n):
                if players[i].active:
                    players[i].score += 1
                    print(' [%s] wins' %(COLORS_str[i]))

            pygame.time.wait(1000)
            env.display.fill(BLACK)

            first = True
            players_active = n

            for i in range(n):
                players[i].gen(env)
                players[i].active = True

            rnd += 1
            continue

        if first:
            pygame.time.wait(1500)
            first = False

        env.fps_clock.tick(env.speed)

if __name__ == '__main__':
    main(sys.argv)

