import keras
from keras import ops
import numpy as np
import tensorflow as tf

from rich import print
from pkg.model import create_q_model
from pkg.env import Game2048Env


NUM_ACTIONS = 4
# Configuration parameters for the whole setup
SEED = 42
# Discount factor for past rewards
GAMMA = 0.99
# Epsilon greedy parameter
EPSILON = 1.0
# Minimum epsilon greedy parameter
EPSILON_MIN = 0.1
# Maximum epsilon greedy parameter
EPSILON_MAX = 1.0
# Rate at which to reduce chance of random action being taken
EPSILON_INTERVAL = EPSILON_MAX - EPSILON_MIN
# Size of batch taken from replay buffer
BATCH_SIZE = 32
MAX_STEPS_PER_EPISODE = 1_000
# Limit training episodes, will run until solved if smaller than 1
MAX_EPISODES = 100
# Number of frames to take random action and observe output
EPSILON_RANDOM_FRAMES = 10
# Number of frames for exploration
EPSILON_GREEDY_FRAMES = 20
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
MAX_MEMORY_LENGTH = 100000
# Train the model after 4 actions
UPDATE_AFTER_ACTION = 1
# How often to update the target network
UPDATE_TARGET_NETWORK = 4
TARGET_REWARD = 2048


def main():
    env = Game2048Env()
    # The first model makes the predictions for Q-values which are used to make a action.
    model = create_q_model()
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_model()

    loss_function = keras.losses.Huber()
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    optimizer.build(model.trainable_variables)

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    epsilon = EPSILON

    while True:
        observation, _ = env.reset()
        state = np.array(observation)
        episode_reward = 0

        for _ in range(1, MAX_STEPS_PER_EPISODE):
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < EPSILON_RANDOM_FRAMES or EPSILON > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(NUM_ACTIONS)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = ops.convert_to_tensor(state)
                state_tensor = ops.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = ops.argmax(action_probs[0]).numpy()  # type: ignore

            # Decay probability of taking random action
            epsilon -= EPSILON_INTERVAL / EPSILON_GREEDY_FRAMES
            epsilon = max(epsilon, EPSILON_MIN)

            # TODO: collect batch of observations using AsyncVectorEnv
            state_next, reward, done, _, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % UPDATE_AFTER_ACTION == 0 and len(done_history) > BATCH_SIZE:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=BATCH_SIZE)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = np.asarray([rewards_history[i] for i in indices])
                action_sample = [action_history[i] for i in indices]
                done_sample = ops.convert_to_tensor([float(done_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample, verbose=0)  # type: ignore
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + GAMMA * ops.amax(future_rewards, axis=1)  # type: ignore
                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample  # type: ignore

                # Create a mask so we only calculate loss on the updated Q-values
                masks = ops.one_hot(action_sample, NUM_ACTIONS)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))  # type: ignore

            if frame_count % UPDATE_TARGET_NETWORK == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            if len(rewards_history) > MAX_MEMORY_LENGTH:
                # Limit the state and reward history
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > TARGET_REWARD:
            # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break

        if MAX_EPISODES > 0 and episode_count >= MAX_EPISODES:
            # Maximum number of episodes reached
            print("Stopped at episode {}!".format(episode_count))
            break

    model.export("models/", include_optimizer=False)


if __name__ == "__main__":
    main()
