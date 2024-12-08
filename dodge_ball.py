import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ball Dodge")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Player settings
player_size = 50
player_pos = [WIDTH // 2, HEIGHT - player_size - 10]
player_speed = 10

# Ball settings
ball_size = 30
ball_list = []
ball_speed = 10
spawn_timer = 1000  # Time between ball spawns (in milliseconds)
last_spawn_time = pygame.time.get_ticks()

# Game clock
clock = pygame.time.Clock()

# Game state
score = 0
game_over = False

# Font
font = pygame.font.Font(None, 36)

def spawn_ball():
    """Creates a new ball at a random position."""
    x_pos = random.randint(0, WIDTH - ball_size)
    return [x_pos, 0]

def draw_balls(ball_list):
    """Draws all the balls."""
    for ball in ball_list:
        pygame.draw.circle(screen, RED, (ball[0] + ball_size // 2, ball[1] + ball_size // 2), ball_size // 2)

def update_ball_positions(ball_list, ball_speed):
    """Updates the position of all balls."""
    global score
    for ball in ball_list:
        ball[1] += ball_speed
        if ball[1] > HEIGHT:
            ball_list.remove(ball)
            score += 1

def check_collision(player_pos, ball_list):
    """Checks for collisions between the player and the balls."""
    player_x, player_y = player_pos
    for ball in ball_list:
        ball_x, ball_y = ball
        if (ball_x < player_x < ball_x + ball_size or ball_x < player_x + player_size < ball_x + ball_size) and \
           (ball_y < player_y < ball_y + ball_size or ball_y < player_y + player_size < ball_y + ball_size):
            return True
    return False

# Main game loop
while not game_over:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Player movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_pos[0] > 0:
        player_pos[0] -= player_speed
    if keys[pygame.K_RIGHT] and player_pos[0] < WIDTH - player_size:
        player_pos[0] += player_speed

    # Ball spawning
    current_time = pygame.time.get_ticks()
    if current_time - last_spawn_time > spawn_timer:
        ball_list.append(spawn_ball())
        last_spawn_time = current_time

    # Update ball positions
    update_ball_positions(ball_list, ball_speed)

    # Check for collisions
    if check_collision(player_pos, ball_list):
        game_over = True

    # Draw player
    pygame.draw.rect(screen, BLUE, (player_pos[0], player_pos[1], player_size, player_size))

    # Draw balls
    draw_balls(ball_list)

    # Display score
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(30)

# Game Over screen
screen.fill(WHITE)
game_over_text = font.render("Game Over! Press any key to exit.", True, RED)
screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))
pygame.display.flip()

# Wait for the player to press a key
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            pygame.quit()
            sys.exit()
