import numpy as np
import random
import pygame

# Define the Dodge Ball Environment
class DodgeBallEnv:
    def __init__(self):
        # Screen dimensions
        self.width = 800
        self.height = 600
        self.player_size = 50
        self.ball_size = 30
        self.ball_speed = 10
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.player_pos = [self.width // 2, self.height - self.player_size - 10]
        self.ball_pos = [random.randint(0, self.width - self.ball_size), 0]
        self.score = 0
        return self.get_state()

    def step(self, action):
        """Execute one step in the environment."""
        # Move player based on action
        if action == 1:  # Move Left
            self.player_pos[0] -= 20
        elif action == 2:  # Move Right
            self.player_pos[0] += 20
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.width - self.player_size)

        # Update ball position
        self.ball_pos[1] += self.ball_speed

        # Check if the ball is out of bounds
        if self.ball_pos[1] > self.height:
            self.ball_pos = [random.randint(0, self.width - self.ball_size), 0]
            self.score += 1

        # Check collision
        done = self.check_collision()
        reward = -10 if done else 1
        return self.get_state(), reward, done

    def check_collision(self):
        """Check if the ball hits the player."""
        px, py = self.player_pos
        bx, by = self.ball_pos
        return (bx < px + self.player_size and bx + self.ball_size > px and
                by < py + self.player_size and by + self.ball_size > py)

    def get_state(self):
        """Get the current state as a tuple."""
        player_x = self.player_pos[0] // 50
        ball_x = self.ball_pos[0] // 50
        ball_y = self.ball_pos[1] // 50
        return (player_x, ball_x, ball_y)

    def render(self):
        """Render the environment."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 255), (*self.player_pos, self.player_size, self.player_size))
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (self.ball_pos[0] + self.ball_size // 2, self.ball_pos[1] + self.ball_size // 2),
                           self.ball_size // 2)
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(30)

# Q-learning Agent
class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.q_table = {}
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay

    def get_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        return np.argmax(self.q_table.get(state, [0] * self.action_space))

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning formula."""
        current_q = self.q_table.get(state, [0] * self.action_space)[action]
        next_max_q = max(self.q_table.get(next_state, [0] * self.action_space))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_space
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

# Train and Test the Q-learning Agent
def train_agent():
    env = DodgeBallEnv()
    agent = QLearningAgent(state_space=(16, 16, 12), action_space=3)
    episodes = 500
    scores = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        scores.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return agent

def test_agent(agent):
    env = DodgeBallEnv()
    state = env.reset()
    done = False
    total_score = 0

    while not done:
        env.render()
        action = np.argmax(agent.q_table.get(state, [0] * 3))
        state, reward, done = env.step(action)
        total_score += reward

    print(f"Final Score: {total_score}")
    env.close()

if __name__ == "__main__":
    print("Training the agent...")
    trained_agent = train_agent()
    print("Testing the trained agent...")
    test_agent(trained_agent)