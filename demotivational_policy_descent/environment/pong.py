import math
import random
import numpy as np
import pygame
import os
from pygame.locals import *


class Ball():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.last_touch = 0  # Remember who touched the ball last
        self.color = (255, 255, 255)
        self.rect = pygame.Rect(self.x, self.y, w, h)
        self.MAX_BOUNCE_ANGLE = 75
        self.reset_ball()

    def move(self):
        self.x += self.vector[0]
        self.y += self.vector[1]
        # Collisison with left and right wall
        if self.x <= 0 - self.w:
            return 2, True
        if self.x >= 210 - self.w:
            return 1, True
        # Collisison with top and bottom wall
        if (self.y - abs(self.vector[1]) <= 35 and self.vector[1] < 0) or (self.y + abs(self.vector[1]) >= 235 and self.vector[1] > 0):
            self.vector = (self.vector[0], -1 * self.vector[1])

        self.rect = pygame.Rect(self.x - self.w/2, self.y-self.h/2, self.w, self.h)

        return 0, False

    def reflect(self, offcenter, direction, player):
        # print("offcenter: ", offcenter)
        normalized_offcenter = offcenter / 10 * direction
        # print("norm offcenter: ",normalized_offcenter)
        bounce_angle = normalized_offcenter * self.MAX_BOUNCE_ANGLE
        # print("bounce_angle: ", bounce_angle)
        if player == 1:
            self.vector = (3 * (math.cos(math.radians(bounce_angle)) + 1)*self.speed_mul, 8 * -math.sin(math.radians(bounce_angle))*self.speed_mul)
        else:
            self.vector = (-3 * (math.cos(math.radians(bounce_angle)) + 1)*self.speed_mul, 8 * math.sin(math.radians(bounce_angle))*self.speed_mul)
        # print("vector: ", self.vector)
        self.speed_mul += .005

    def reset_ball(self):
        self.x = 105
        self.y = 135
        self.last_touch = 0
        self.rect = pygame.Rect(self.x - self.w/2, self.y-self.h/2, self.w, self.h)
        # Reset ball in random direction
        bounce_angle = np.random.random() * 40
        side = random.choice([True, False])
        if side == True:
            d = 1
        else:
            d = -1
        up_down = random.choice([True, False])
        if up_down == True:
            u = 1
        else:
            u = -1
        self.vector = (d * 3 * (math.cos(math.radians(bounce_angle)) + 1), u * 6 * math.sin(math.radians(bounce_angle)))
        self.speed_mul = 0.8


class Player():
    def __init__(self, player_number):
        self.player_number = player_number
        self.score = 0
        self.x = 0
        self.y = 0
        self.w = 5
        self.h = 20
        self.rect = None
        self.color = (0, 0, 0)
        self.reset()
        self.paddle_speed = 3
        self.name = ""

    def move_up(self):
        if self.y - self.paddle_speed >= 35 + self.h / 2:
            self.y -= self.paddle_speed
        self.rect = pygame.Rect(self.x, self.y - self.h / 2, self.w, self.h)

    def move_down(self):
        if self.y + self.paddle_speed <= 235 - self.h / 2:
            self.y += self.paddle_speed
        self.rect = pygame.Rect(self.x, self.y - self.h / 2, self.w, self.h)

    def reset(self):
        if self.player_number == 1:
            self.x = 10
            self.color = (0, 255, 0)
        else:
            self.x = 195
            self.color = (255, 0, 0)
        self.y = 135
        self.rect = pygame.Rect(self.x, self.y - self.h / 2, self.w, self.h)


class Pong():
    MOVE_UP, MOVE_DOWN, STAY = 1, 2, 0
    def __init__(self, headless=False):
        # [initiate game environment here]
        pygame.init()
        self.SCREEN_RESOLUTION = (210, 235)
        self.GAME_AREA_RESOLUTION = (210, 200)
        self.GAME_AREA_CENTER = (105, 135)
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.display.set_mode(self.SCREEN_RESOLUTION, 0, 24)
        else:
            self.screen = pygame.display.set_mode(self.SCREEN_RESOLUTION)
        pygame.display.set_caption('Pong')
        pygame.font.init()
        self.myfont = pygame.font.SysFont("Arial", 15)
        # Define colors
        self.BACKGROUND_COLOR = (0, 0, 0)
        # Fill background
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.BACKGROUND_COLOR)
        # Draw init stuff
        self.screen.blit(self.background, (0, 0))
        # Define game clock
        self.clock = pygame.time.Clock()
        # Define all game objectsBall
        self.player1 = Player(1)
        self.player2 = Player(2)
        self.ball = Ball(self.GAME_AREA_CENTER[0], self.GAME_AREA_CENTER[1] - 2, 4, 4)
        # Other stuff
        self.p1_move_up = False
        self.p1_move_down = False
        self.p2_move_up = False
        self.p2_move_down = False

    def set_names(self, p1, p2):
        self.player1.name = p1
        self.player2.name = p2
    def step(self, actions):
        """
        This functions is a modification of the openai gym step function for two players. The render option can be set directly in this functions

        TODO:
        catch error if wrong player id is entered

        ARGUMENS:
        player: integer; player=1: left player, player=2: right player
        action: the action the player took
        r: render the game or not

        RETURN:
        observation: return the current state of the game area mirrored so that every player plays from the perspective of the left player. This is done to speed up the training
        reward: the reward the given player got
        done: episode over
        info: debug output
        """
        info = {}
        r = True
        # Draw background if we want to render
        if r: self.background.fill(self.BACKGROUND_COLOR)
        if r: self.screen.blit(self.background, (0, 0))

        # Handle Player1's action
        if actions[0] == self.MOVE_UP:
            self.player1.move_up()
        elif actions[0] == self.MOVE_DOWN:
            self.player1.move_down()

        # Handle Player2's action
        if actions[1] == self.MOVE_UP:
            self.player2.move_up()
        elif actions[1] == self.MOVE_DOWN:
            self.player2.move_down()

        # ==================Logic============================
        # Check if the ball collided with one of the players
        if self.player1.rect.colliderect(self.ball.rect) and self.ball.last_touch is not 1:
            self.__reflect(self.player1.y, 1)
        if self.player2.rect.colliderect(self.ball.rect) and self.ball.last_touch is not 2:
            self.__reflect(self.player2.y, 2)

        # Move ball and check if game is over
        winner, done = self.ball.move()
        player1_reward = 0
        player2_reward = 0
        if done:
            if winner == 1:
                player1_reward = 10
                player2_reward = -10
                self.player1.score += 1
            if winner == 2:
                player1_reward = -10
                player2_reward = 10
                self.player2.score += 1
        # ==================Draw=============================
        if r: self.__draw_scores()
        if r: self.__render_ball()
        if r: self.__render_player1()
        if r: self.__render_player2()
        # =================Return========================
        observation_left = self.__get_observation(1)
        observation_right = self.__get_observation(2)

        return (observation_left, observation_right), (player1_reward, player2_reward), done, info

    def reset(self):
        # paint over the old frame
        self.background.fill(self.BACKGROUND_COLOR)
        self.screen.blit(self.background, (0, 0))
        # TODO Reset player scores
        self.ball.reset_ball()
        self.player1.reset()
        self.player2.reset()

        # Draw the changes so they are in the frame
        self.__render_player1()
        self.__render_player2()
        self.__draw_scores()
        self.__render_ball()
        return self.__get_observation(1), self.__get_observation(2)

    def __reflect(self, player_y, player):
        offcenter = abs(player_y - self.ball.y)
        if player_y > self.ball.y:
            if player == 1:
                direction = 1
            else:
                direction = -1
        else:
            if player == 1:
                direction = -1
            else:
                direction = 1
        self.ball.reflect(offcenter, direction, player)
        self.ball.last_touch = player  # Which player touched the ball last

    def __render_ball(self):
        # Draw ball
        pygame.draw.rect(self.screen, self.ball.color, self.ball.rect)

    def __render_player1(self):
        pygame.draw.rect(self.screen, self.player1.color, self.player1.rect)

    def __render_player2(self):
        pygame.draw.rect(self.screen, self.player2.color, self.player2.rect)

    def __draw_scores(self):
        # Draw the scoreboard
        scoreboard_background = pygame.Rect(0, 0, self.GAME_AREA_RESOLUTION[0], 35)
        scoreboard_border = pygame.Rect(1, 0, self.GAME_AREA_RESOLUTION[0] - 2, 35 - 2)
        scoreboard_separator = pygame.Rect(self.GAME_AREA_RESOLUTION[0] / 2 - 4, 0, 5, 35)
        pygame.draw.rect(self.screen, (150, 150, 150), scoreboard_background)
        pygame.draw.rect(self.screen, (0, 0, 0), scoreboard_border, 4)
        pygame.draw.rect(self.screen, (0, 0, 0), scoreboard_separator)
        #label_player_1 = self.myfont.render("{}".format(self.player1.name), 1, (0,255,0))
        #self.screen.blit(label_player_1, (5,5))
        label_score_p1 = self.myfont.render("{}: {}".format(self.player1.name,self.player1.score), 1, (0, 255, 0))
        self.screen.blit(label_score_p1, (5, 5))
        label_score_p2 = self.myfont.render("{}: {}".format(self.player2.name, self.player2.score), 1, (255, 0, 0))
        self.screen.blit(label_score_p2, (5 + self.SCREEN_RESOLUTION[0] / 2 + 4, 5))
        self.__render_ball()

    def __get_observation(self, player):
        """
        This function computes the observation depending on the player. player gets the normal observation and player 2 the inverted
        """
        if player == 1:
            observation_red = pygame.surfarray.pixels_red(self.screen)[:,35:]
            observation_green = pygame.surfarray.pixels_green(self.screen)[:,35:]
            observation_blue = pygame.surfarray.pixels_blue(self.screen)[:,35:]
            observation = np.stack((observation_red,observation_green, observation_blue),axis=2)
            observation = np.rot90(observation)
        # Player 2 gets a frame with inverted colors and inverted positions so that both sides look the same to the agent
        if player == 2:
            observation_green = pygame.surfarray.pixels_red(self.screen)[:,35:]
            observation_red = pygame.surfarray.pixels_green(self.screen)[:,35:]
            observation_blue = pygame.surfarray.pixels_blue(self.screen)[:,35:]
            observation = np.stack((observation_red,observation_green, observation_blue),axis=2)
            observation = np.rot90(observation)
            observation = np.flip(observation, 1)
        return observation

    def render(self):
        # Make sure game doesn't run at more than 30 frames per second
        # If we dont render we dont want this limitation
        self.clock.tick(60)
        pygame.event.get()

        # This method is used for everything regarding the rendering. Here pygame would paint to the window
        pygame.display.flip()

    def end(self):
        pygame.quit()