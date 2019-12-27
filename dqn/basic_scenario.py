import random

import click
from collections import deque
import numpy as np
from skimage import transform
import tensorflow as tf
from vizdoom import *

# Network parameters
INPUT_SIZE = np.array([84, 84, 1])
LEARNING_RATE = 0.0002

# Training parameters
TOTAL_EPISODES = 500
MAX_STEPS = 100
BATCH_SIZE = 64
MEMORY_SIZE = 1000000

# Exploring parameters
EPSILON_START = 1.0
EPSILON_STOP = 0.01

# Reinforcement problem parameters
GAMMA = 0.99  # Discounted reward


class Memory:
    """Replay buffer object"""

    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


def preprocess_frame(frame):
    """Takes an RGB frame from the game and returns a normalized grayscale and resized image"""
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, INPUT_SIZE)
    return preprocessed_frame


def create_environment(visible=False):
    """Creates a basic game environment"""
    game = DoomGame()
    # Load the correct configuration
    game.load_config("../scenarios/basic.cfg")
    # Load the scenario
    game.set_doom_scenario_path("../scenarios/basic.wad")
    # Set the desired game configuration
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_window_visible(visible)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.init()
    # Possible actions for this environment
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions


def create_model():
    """Creates Deep Q Network for the agent"""
    conv_dqn = tf.keras.Sequential()
    conv_dqn.add(
        tf.keras.layers.Conv2D(
            input_shape=INPUT_SIZE,
            activation="relu",
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding="VALID",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    conv_dqn.add(
        tf.keras.layers.Conv2D(
            filters=64,
            activation="relu",
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="VALID",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    conv_dqn.add(
        tf.keras.layers.Conv2D(
            filters=128,
            activation="relu",
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="VALID",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    conv_dqn.add(tf.keras.layers.Flatten())
    conv_dqn.add(tf.keras.layers.Dense(units=512, activation="relu"))
    conv_dqn.add(tf.keras.layers.Dense(units=3))
    conv_dqn.compile(loss="MSE", optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    return conv_dqn


def predict_action(state, actions, model, epsilon_start, epsilon_stop, episode):
    """Linear decaying e-greedy action selection"""
    explore_th = np.random.rand()
    explore_probability = (
        epsilon_start + (epsilon_stop - epsilon_start) / TOTAL_EPISODES * episode
    )
    if explore_probability > explore_th:
        action = random.choice(actions)
    else:
        expanded = state.reshape((1, *state.shape))
        Qs = model.predict(expanded)
        choice = np.argmax(Qs)
        action = actions[int(choice)]
    return action


def preload_buffer():
    """Loads the replay buffer with random experience to start training"""
    game, possible_actions = create_environment()
    memory = Memory()
    game.new_episode()
    for _ in range(BATCH_SIZE):
        state = game.get_state().screen_buffer
        state = preprocess_frame(state)
        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = False
        if game.is_episode_finished():
            done = True
            next_state = np.zeros(INPUT_SIZE)
            game.new_episode()
        else:
            next_state = game.get_state().screen_buffer
            next_state = preprocess_frame(next_state)
        memory.add((state, action, reward, next_state, done))
    game.close()
    return memory


@click.group()
def main():
    pass


@main.command()
def test_environment():
    """Creates a test environment to check if everything is ok"""
    game, actions = create_environment(visible=True)
    game.new_episode()
    while not game.is_episode_finished():
        action = random.choice(actions)
        game.make_action(action)
    print("Total reward:", game.get_total_reward())


@main.command()
@click.option("--model-path", default=None)
def train_model(model_path):
    """Train a Deep Q Learning Agent to play the basic scenario of VizDoom"""
    if not model_path:
        conv_dqn = create_model()
    else:
        print("Training existent model")
        conv_dqn = tf.keras.models.load_model(model_path)
    # Pre-loading the replay buffer
    memory = preload_buffer()
    game, actions = create_environment()
    for episode in range(TOTAL_EPISODES):
        print(f"EPSIODE {episode}")
        step = 0
        done = False
        game.new_episode()
        while step < MAX_STEPS and not done:
            # Playing part
            state = game.get_state().screen_buffer
            state = preprocess_frame(state)
            action = predict_action(
                state, actions, conv_dqn, EPSILON_START, EPSILON_STOP, episode
            )
            reward = game.make_action(action)
            done = game.is_episode_finished()
            if done:
                next_state = np.zeros(INPUT_SIZE)
                memory.add((state, action, reward, next_state, done))
                print(f"Episode: {episode}", f"Total reward: {game.get_total_reward()}")
            else:
                next_state = game.get_state().screen_buffer
                next_state = preprocess_frame(next_state)
                memory.add((state, action, reward, next_state, done))
            step += 1
            # Learning part
            batch = memory.sample(BATCH_SIZE)
            learning_states = np.array([sample[0] for sample in batch])
            learning_actions = np.array([sample[1] for sample in batch])
            learning_rewards = np.array([sample[2] for sample in batch])
            learning_next_states = np.array([sample[3] for sample in batch])
            learning_dones = np.array([sample[4] for sample in batch])
            Qs_next_state = conv_dqn.predict(learning_next_states)
            # Trick for computing the loss only for the desired action
            target_Qs = conv_dqn.predict(learning_states)
            for i in range(len(batch)):
                learning_action = learning_actions[i]
                learning_action = np.argwhere(learning_action)[0][0]
                terminal = learning_dones[i]
                if terminal:
                    target = learning_rewards[i]
                else:
                    target = learning_rewards[i] + GAMMA * np.max(Qs_next_state[i])
                # Trick for computing the loss only for the desired action
                target_Qs[i, learning_action] = target
            conv_dqn.fit(learning_states, target_Qs, epochs=1)
    game.close()
    conv_dqn.save("basic_dqn.h5")


@main.command()
@click.option("--model-path", default="basic_dqn.h5")
def test_model(model_path):
    """Test a trained agent in VizDoom's basic scenario"""
    conv_dqn = tf.keras.models.load_model(model_path)
    game, actions = create_environment(visible=True)
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state().screen_buffer
        state = preprocess_frame(state)
        action = predict_action(state, actions, conv_dqn, 0, 0, 0)
        print(action)
        reward = game.make_action(action)
        print(reward)
    print(f"Score: {game.get_total_reward()}")
    game.close()


if __name__ == "__main__":
    main()
