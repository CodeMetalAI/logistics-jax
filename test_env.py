import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict
from container_simple import ShippingEnv, State

def test_environment_initialization():
    """Test basic environment setup and state initialization."""
    # Create environment with small number of ports and agents for testing
    env = ShippingEnv(
        num_ports=5,
        num_agents=2,
        max_unloads_per_day=50,  # Small enough to see partial unloading
        max_cargo=200
    )
    
    # Reset environment
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    print("\nTesting Environment Initialization:")
    print("Number of ports:", env.num_ports)
    print("Number of agents:", env.num_agents)
    print("Ship positions:", state.ship_positions)
    print("Ship cargo:", state.ship_cargo)
    print("First agent observation shape:", obs["agent_0"].shape)
    
    # Basic assertions
    assert state.ship_positions.shape == (env.num_agents,), "Incorrect ship positions shape"
    assert state.ship_cargo.shape == (env.num_agents, env.num_ports), "Incorrect ship cargo shape"
    assert state.port_containers.shape == (env.num_ports, env.num_ports), "Incorrect port containers shape"
    
    print("\nEnvironment initialization test passed!")
    
    # Test step function
    actions = {"agent_0": 0, "agent_1": 0}  # Both agents stay at port 0
    step_key = jax.random.PRNGKey(1)
    new_obs, new_state, rewards, dones, info = env.step(step_key, state, actions)
    
    print("\nAfter Step:")
    print("Rewards:", rewards)
    print("New ship positions:", new_state.ship_positions)
    print("New ship cargo:", new_state.ship_cargo)
    
    # Verify rewards
    assert rewards["agent_0"] > 0, f"Expected positive reward for agent 0, got {rewards['agent_0']}"

def test_port_container_distribution():
    """Verify containers are initialized correctly based on port volume tiers."""
    key = jax.random.PRNGKey(1)
    env = ShippingEnv(num_ports=100, num_agents=100)
    _, state = env.reset(key)
    
    high_tier_ports = (state.port_volume_tiers == 0).sum()
    med_tier_ports = (state.port_volume_tiers == 1).sum()
    low_tier_ports = (state.port_volume_tiers == 2).sum()

    assert high_tier_ports == env.num_high_tier, "Incorrect number of high-volume ports"
    assert med_tier_ports == env.num_med_tier, "Incorrect number of medium-volume ports"
    assert low_tier_ports == env.num_low_tier, "Incorrect number of low-volume ports"
    
    print("Port container distribution test passed.")

def test_observation_shape():
    """Check that each agent's observation has the correct shape and components."""
    key = jax.random.PRNGKey(2)
    env = ShippingEnv(num_ports=100, num_agents=10)
    obs, _ = env.reset(key)
    
    for agent_id, observation in obs.items():
        assert observation.shape == env.observation_space.shape, f"{agent_id} has incorrect observation shape"
    
    print("Observation shape test passed.")

def test_step_function():
    """Test step function for correct state update and reward distribution."""
    key = jax.random.PRNGKey(3)
    env = ShippingEnv(num_ports=100, num_agents=10)
    obs, state = env.reset(key)
    
    actions = {f"agent_{i}": jax.random.randint(key, (), 0, env.num_ports) for i in range(env.num_agents)}
    key, step_key = jax.random.split(key)
    
    new_obs, new_state, rewards, dones, infos = env.step(step_key, state, actions)
    
    assert isinstance(new_obs, Dict), "new_obs should be a dictionary"
    assert isinstance(new_state, State), "new_state should be of type State"
    assert isinstance(rewards, Dict), "rewards should be a dictionary"
    assert isinstance(dones, Dict), "dones should be a dictionary"
    
    print("Step function test passed.")

def test_rewards_for_container_delivery():
    """Test that agents receive rewards for delivering containers."""
    env = ShippingEnv(num_ports=5, num_agents=2)  # Smaller environment for testing
    
    # Reset with debug info
    reset_key = jax.random.PRNGKey(0)
    obs, state = env.reset(reset_key)
    
    print("\nInitial State:")
    print("Ship Positions:", state.ship_positions)
    print("Ship Cargo:", state.ship_cargo)
    
    # Create action for agent 0 to stay at port 0
    actions = {
        "agent_0": 0,  # Stay at port 0
        "agent_1": 0
    }
    
    # Step environment
    step_key = jax.random.PRNGKey(1)
    new_obs, new_state, rewards, dones, info = env.step(step_key, state, actions)
    
    print("\nAfter Step:")
    print("Ship Positions:", new_state.ship_positions)
    print("Ship Cargo:", new_state.ship_cargo)
    print("Rewards:", rewards)
    
    # Check rewards
    assert rewards["agent_0"] > 0, "Agent 0 should receive reward for delivered containers"

def test_done_flag():
    """Check that the done flag is correctly set when reaching max steps."""
    key = jax.random.PRNGKey(5)
    env = ShippingEnv(num_ports=10, num_agents=2, max_steps=5)
    obs, state = env.reset(key)
    
    actions = {f"agent_{i}": 1 for i in range(env.num_agents)}
    done_reached = False
    
    for _ in range(5):  # Step until max_steps
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step(step_key, state, actions)
        if dones["__all__"]:
            done_reached = True
            break
    
    assert done_reached, "Done flag should be set after max steps"
    
    print("Done flag test passed.")

def test_cargo_operations():
    """Test proper cargo loading and unloading."""
    env = ShippingEnv(
        num_ports=3,  # Small number for easy testing
        num_agents=1,
        max_cargo=200,
        max_unloads_per_day=50
    )
    
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    print("\nInitial State:")
    print("Ship cargo:", state.ship_cargo)
    print("Port containers:", state.port_containers)
    
    # Step 1: Ship stays at port 0 to unload
    new_obs, state, rewards, dones, info = env.step(key, state, {"agent_0": 0})
    
    print("\nAfter Unloading:")
    print("Ship cargo:", state.ship_cargo)
    print("Port containers:", state.port_containers)
    print("Rewards:", rewards)
    assert rewards["agent_0"] <= env.max_unloads_per_day, "Exceeded unloading limit"
    
    # Step 2: Ship moves to port 1 to load
    key, subkey = jax.random.split(key)
    new_obs, state, rewards, dones, info = env.step(subkey, state, {"agent_0": 1})
    
    print("\nAfter Moving and Loading:")
    print("Ship cargo:", state.ship_cargo)
    print("Port containers:", state.port_containers)
    print("Ship position:", state.ship_positions)
    
    return "Cargo operations test passed"


# Run tests
test_environment_initialization()
test_port_container_distribution()
test_observation_shape()
test_step_function()
test_cargo_operations()
test_rewards_for_container_delivery()
test_done_flag()
