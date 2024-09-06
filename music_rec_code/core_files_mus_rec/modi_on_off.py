import numpy as np
import pandas as pd
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Constants
NUM_FEATURES = 21  # Number of song features
BATCH_SIZE = 32
MEMORY_SIZE = 10000
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Exploration rate decay
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# Load the song data from CSV
def load_songs(file_path):
    df = pd.read_csv(file_path)
    song_titles = df.iloc[:, 0].values
    song_features = df.iloc[:, 1:].values
    return song_titles, song_features


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, num_features):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_features * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Action Head Deep Q-Network
class AH_DQN:
    def __init__(self, num_songs, num_features):
        self.num_songs = num_songs
        self.num_features = num_features
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = DQN(num_features).to(DEVICE)
        self.target_model = DQN(num_features).to(DEVICE)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.song_features = None

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_virtual_user):
        if is_virtual_user or np.random.rand() <= EPSILON:
            return random.randrange(self.num_songs)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        song_features = torch.tensor(self.song_features, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            act_values = self.model(state.repeat(self.num_songs, 1), song_features).squeeze().cpu().numpy()
        return np.argmax(act_values)

    def replay(self, batch_size):
        global EPSILON  # Add this line to manage the global variable
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        q_values = []
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
            song_features = torch.tensor(self.song_features, dtype=torch.float32).to(DEVICE)
            target = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
            if not done:
                with torch.no_grad():
                    target += GAMMA * self.target_model(next_state.repeat(self.num_songs, 1), song_features).max()
            target_f = self.model(state.repeat(self.num_songs, 1), song_features)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state.repeat(self.num_songs, 1), song_features), target_f)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            q_values.append(target.item())
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
        return np.mean(losses), np.mean(q_values)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# World Model (Virtual Users)
class WorldModel:
    def __init__(self, song_features, num_features):
        self.num_songs = song_features.shape[0]
        self.num_features = num_features
        self.song_features = song_features
        self.user_preferences = np.random.rand(self.num_features)

    def get_reward(self, song_features):
        return np.dot(self.user_preferences, song_features)


# Actual User
class ActualUser:
    def __init__(self, num_features):
        self.num_features = num_features
        self.user_preferences = None

    def set_preferences(self):
        print("Select song features that you like:")
        features = ["1980s", "1990s", "2000s", "2010s", "2020s", "Pop", "Rock", "Country", "Folk", "Dance",
                    "Grunge", "Love", "Metal", "Classic", "Funk", "Electric", "Acoustic", "Indie", "Jazz",
                    "SoundTrack", "Rap"]
        for i, feature in enumerate(features):
            print(f"{i + 1}. {feature}")
        liked_features = []
        while True:
            choice = input("Enter the number associated with the feature (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            try:
                num = int(choice)
                if 1 <= num <= len(features):
                    liked_features.append(features[num - 1])
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please try again.")
        self.user_preferences = np.zeros(self.num_features)
        for feature in liked_features:
            self.user_preferences[features.index(feature)] = 1.0 / len(liked_features)


# Training
def train(num_episodes, song_features):
    global EPSILON  # Add this line to manage the global variable
    world_model = WorldModel(song_features, NUM_FEATURES)
    agent = AH_DQN(song_features.shape[0], NUM_FEATURES)
    agent.song_features = world_model.song_features

    episode_rewards = []
    episode_losses = []
    episode_q_values = []
    epsilons = []
    actions = []
    preference_alignments = []

    for episode in range(num_episodes):
        state = world_model.user_preferences
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state, is_virtual_user=True)
            actions.append(action)
            song_features = world_model.song_features[action]
            reward = world_model.get_reward(song_features)
            preference_alignments.append(reward)
            next_state = state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            done = True
            if done:
                agent.update_target_model()
                if len(agent.memory) > BATCH_SIZE:
                    loss, avg_q_value = agent.replay(BATCH_SIZE)
                    episode_losses.append(loss)
                    episode_q_values.append(avg_q_value)
                else:
                    episode_losses.append(0)
                    episode_q_values.append(0)
                episode_rewards.append(total_reward)
                if EPSILON > EPSILON_MIN:
                    EPSILON *= EPSILON_DECAY
                    epsilons.append(EPSILON)
                print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward}, Loss: {episode_losses[-1]}, Q-value: {episode_q_values[-1]}")
                break

    agent.save("ah_dqn_weights.pth")
    print("Training completed and weights saved.")

    # Plotting
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_episodes + 1), episode_rewards, label='Episode vs Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Total Reward')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_episodes + 1), episode_losses, label='Episode vs Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_episodes + 1), episode_q_values, label='Episode vs Q-value')
    plt.xlabel('Episodes')
    plt.ylabel('Average Q-value')
    plt.title('Episode vs Q-value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Additional Plots
    plt.figure()
    plt.plot(epsilons)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.show()

    plt.figure()
    plt.hist(episode_rewards, bins=50)
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.show()

    plt.figure()
    plt.hist(episode_losses, bins=50)
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    plt.show()

    plt.figure()
    plt.hist(actions, bins=range(agent.num_songs + 1))
    plt.xlabel('Song Index')
    plt.ylabel('Frequency')
    plt.title('Action Distribution')
    plt.show()

    plt.figure()
    plt.plot(preference_alignments)
    plt.xlabel('Episodes')
    plt.ylabel('Preference Alignment')
    plt.title('Preference Alignment Over Time')
    plt.show()


# Playlist Generation
def generate_playlist(agent, actual_user):
    playlist = []
    state = actual_user.user_preferences
    for _ in range(10):  # Generate a playlist of 10 songs
        action = agent.act(state, is_virtual_user=False)
        playlist.append(action)
    return playlist


# Main
if __name__ == "__main__":
    # Load songs from CSV
    song_titles, song_features = load_songs("D:\\My_Acad\\VS_Code\\Sem_6_code\\RL\\music_RL\\songs.csv")

    mode = input("Enter mode (OFFLINE/ONLINE): ").strip().upper()

    if mode == "OFFLINE":
        # Training
        num_episodes = 50000
        train(num_episodes, song_features)
    elif mode == "ONLINE":
        # Load the trained agent
        agent = AH_DQN(song_features.shape[0], NUM_FEATURES)
        agent.load("ah_dqn_weights.pth")  # Load trained weights

        # Actual User
        actual_user = ActualUser(NUM_FEATURES)
        actual_user.set_preferences()
        playlist = generate_playlist(agent, actual_user)
        print("Playlist for Actual User:")
        for song in playlist:
            print(f"Song {song_titles[song]}")
    else:
        print("Invalid mode selected. Please choose either OFFLINE or ONLINE.")
