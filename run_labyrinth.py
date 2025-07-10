import numpy as np
from labyrinth_env import LabyrinthEnv

# Value iteration algorithm
def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Performs Value Iteration to find the optimal value function and policy for the LabyrinthEnv.

    Args:
        env (LabyrinthEnv): An instance of the LabyrinthEnv.
        gamma (float): Discount factor.
        theta (float): Convergence threshold.

    Returns:
        V (np.ndarray): The optimal value function for each state.
        policy (np.ndarray): The optimal policy (action to take for each state).
    """
    n_states = env.n_states
    n_actions = env.n_actions

    # Initialize value function to zeros
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    # Get the transition probabilities from the environment
    P_a = env.get_transition_mat()

    # Define the reward function (R) based on the environment's reward logic.
    # If env.reward_map is provided, use that. Otherwise, use env.water_port.
    R = np.zeros(n_states)
    if env.reward_map is None:
        R[env.water_port] = 1  # Reward for reaching the specified water_port
    else:
        R = env.reward_map # Use the provided custom reward map

    iteration = 0
    while True:
        delta = 0
        V_new = np.copy(V) # Create a copy for batch updates

        for s in range(n_states):
            q_values = np.zeros(n_actions)

            for a in range(n_actions):
                # Calculate the expected Q-value for taking action 'a' from state 's'
                # Q(s,a) = Sum_s' [ P(s'|s,a) * (R(s') + gamma * V(s')) ]
                # Since transitions are deterministic (P(s'|s,a) is 1 for one s' and 0 otherwise),
                # this simplifies to: Q(s,a) = R(next_s) + gamma * V(next_s)
                
                # Find the next state for the deterministic transition
                # np.where(P_a[s, :, a] == 1) finds indices where probability is 1
                next_state_indices = np.where(P_a[s, :, a] == 1)[0]
                
                # There should always be exactly one next state for a deterministic environment
                if len(next_state_indices) == 1:
                    next_s_prime = next_state_indices[0]
                    q_values[a] = R[next_s_prime] + gamma * V[next_s_prime]
                else:
                    # This case should ideally not be hit with a correctly defined P_a for deterministic env.
                    # It implies no defined transition, or multiple. For safety, we can assume staying in place
                    # or assign a very low value.
                    # For this LabyrinthEnv, it implies a bug in P_a if it returns 0 or >1 transitions.
                    # Let's default to staying and getting current state reward if this happens unexpectedly.
                    q_values[a] = R[s] + gamma * V[s] # Fallback: stay in state 's'


            # Update the value function for state 's' using the Bellman Optimality Equation
            V_new[s] = np.max(q_values)

            # Calculate the maximum change across all states
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = np.copy(V_new) # Update V for the next iteration (batch update)
        iteration += 1

        # Check for convergence
        if delta < theta:
            break

    # Policy extraction
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            next_state_indices = np.where(P_a[s, :, a] == 1)[0]
            if len(next_state_indices) == 1:
                next_s_prime = next_state_indices[0]
                q_values[a] = R[next_s_prime] + gamma * V[next_s_prime]
            else:
                q_values[a] = R[s] + gamma * V[s] # Fallback
        
        policy[s] = np.argmax(q_values) # The action that yields the max Q-value

    print(f"Value Iteration converged in {iteration} iterations.")
    return V, policy

# Main execution
if __name__ == "__main__":
    print("--- Running Value Iteration for a single, fixed rewarding node ---")
    # Define environment parameters
    n_states = 63
    reward_state_fixed = 62 # Example: The last state is the reward state

    env_fixed_reward = LabyrinthEnv(n_states=n_states, reward_state=reward_state_fixed)

    gamma = 0.99  # Discount factor
    theta = 1e-9  # Convergence threshold

    # Run Value Iteration
    optimal_V_fixed, optimal_policy_fixed = value_iteration(env_fixed_reward, gamma, theta)

    print("\nOptimal Value Function (V) for fixed reward state:")
    print(np.round(optimal_V_fixed, 3)) # Round for better readability
    print("\nOptimal Policy (0: left, 1: right, 2: reverse, 3: stay) for fixed reward state:")
    print(optimal_policy_fixed)
    print("\n--- Testing the Optimal Policy for fixed reward state ---")
    start_state = env_fixed_reward.home_state
    current_state = env_fixed_reward.reset(start_state)
    print(f"Starting at state: {current_state}")

    path = [current_state]
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < env_fixed_reward.max_episode_length:
        action = optimal_policy_fixed[current_state]
        _, _, next_state, reward, done = env_fixed_reward.step(action)
        path.append(next_state)
        total_reward += reward
        print(f"  Step {step_count}: State {current_state}, Action {action} -> Next State {next_state}, Reward {reward}")
        current_state = next_state
        step_count += 1
        if done:
            print("  Episode ended.")
            break # Exit loop if done is True

    print(f"\nPath taken: {path}")
    print(f"Total reward received: {total_reward}")
    print(f"Number of steps: {step_count}")

    print("\n\n--- Running Value Iteration for four randomly switching rewarding nodes (using Expected Rewards) ---")

    # Define four possible reward locations (these are just examples)
    n_states_stochastic = 63
    possible_reward_nodes = [15, 30, 45, 60] # Example states within 0-62
    reward_value_at_target = 1 # Reward received if you land on the active target

    # Probability of each node being the active reward node for a trial
    prob_each_node_is_active = 1.0 / len(possible_reward_nodes)

    # Construct the R_expected array
    R_expected = np.zeros(n_states_stochastic)
    for s_idx in range(n_states_stochastic):
        if s_idx in possible_reward_nodes:
            R_expected[s_idx] = reward_value_at_target * prob_each_node_is_active
    
    print("\nCalculated Expected Reward (R_expected) for each state:")
    print(np.round(R_expected, 3))
    print(f"Example: Expected reward at state {possible_reward_nodes[0]}: {R_expected[possible_reward_nodes[0]]:.3f}")
    print(f"Example: Expected reward at a non-reward state (e.g., 0): {R_expected[0]:.3f}")

    env_stochastic_reward = LabyrinthEnv(n_states=n_states_stochastic, reward_state=0, reward_map=R_expected)

    # Run Value Iteration with the environment configured for expected rewards
    optimal_V_stochastic, optimal_policy_stochastic = value_iteration(env_stochastic_reward, gamma, theta)

    print("\nOptimal Value Function (V) for stochastic reward states:")
    print(np.round(optimal_V_stochastic, 3))
    print("\nOptimal Policy (0: left, 1: right, 2: reverse, 3: stay) for stochastic reward states:")
    print(optimal_policy_stochastic)

    # Test the policy for stochastic reward case (this will show path based on expected values)
    print("\n--- Testing the Optimal Policy for stochastic reward states ---")
    start_state_stochastic = env_stochastic_reward.home_state
    current_state_stochastic = env_stochastic_reward.reset(start_state_stochastic)
    print(f"Starting at state: {current_state_stochastic}")

    path_stochastic = [current_state_stochastic]
    total_reward_stochastic = 0
    done_stochastic = False
    step_count_stochastic = 0

    while not done_stochastic and step_count_stochastic < env_stochastic_reward.max_episode_length:
        action = optimal_policy_stochastic[current_state_stochastic]
        _, _, next_state, reward_from_env_step, done_stochastic = env_stochastic_reward.step(action)
        # Note: reward_from_env_step here will be the *expected* reward for landing in next_state
        # because we passed the R_expected map to the env.
        path_stochastic.append(next_state)
        total_reward_stochastic += reward_from_env_step
        print(f"  Step {step_count_stochastic}: State {current_state_stochastic}, Action {action} -> Next State {next_state}, Expected Reward (from map) {reward_from_env_step:.3f}")
        current_state_stochastic = next_state
        step_count_stochastic += 1
        if done_stochastic:
            print("  Episode ended.")
            break

    print(f"\nPath taken: {path_stochastic}")
    print(f"Total Expected Reward (sum of rewards from expected map): {total_reward_stochastic:.3f}")
    print(f"Number of steps: {step_count_stochastic}")
