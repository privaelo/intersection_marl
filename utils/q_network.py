"""Small MLP Q-network shared by Phase 2 (one per agent) and Phase 3
(shared-parameter track)."""
import random
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim=105, n_actions=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def select_action(network, state, epsilon, n_actions=3):
    """Epsilon-greedy: random action with prob epsilon, else argmax Q-value.
    state: flat 1D numpy array."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

def train_step(q_net, target_net, optimizer, buffer, batch_size, gamma=0.99):
    """One gradient step of the Bellman update. Returns the loss (float),
    or None if the buffer doesn't have enough transitions yet."""
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = torch.as_tensor(states)
    actions = torch.as_tensor(actions)
    rewards = torch.as_tensor(rewards)
    next_states = torch.as_tensor(next_states)
    dones = torch.as_tensor(dones)

    # Q-value the network currently assigns to the action that was actually taken
    q_values = q_net(states)                                        # (batch, n_actions)
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)    # (batch,)

    # bootstrap target from the *target* network, not q_net itself —
    # this is what keeps the moving target from chasing its own tail
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1).values
        target = rewards + gamma * max_next_q * (1 - dones)

    # Huber rather than MSE: collision_reward is -5 against arrivals of +1, so a
    # squared TD error of ~25 from one crash would dominate the whole batch
    loss = nn.functional.smooth_l1_loss(q_taken, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def update_target_network(q_net, target_net):
    """Hard sync: copy q_net's weights into target_net exactly."""
    target_net.load_state_dict(q_net.state_dict())