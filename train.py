import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import wandb
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jaxmarl.wrappers.baselines import LogWrapper
from container_simple import ShippingEnv

class ShippingMetrics(NamedTuple):
    total_containers_delivered: jnp.ndarray
    containers_per_port_tier: Dict[str, jnp.ndarray]
    ships_at_port: jnp.ndarray
    ships_at_sea: jnp.ndarray
    avg_journey_time: jnp.ndarray
    cargo_utilization: jnp.ndarray

class ShippingNetwork(nn.Module):
    """Network for the shipping environment."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.tanh if self.activation == "tanh" else nn.relu
        
        # Process flat observation
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        
        # Actor head
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: ShippingMetrics

def make_env(config):
    """Create a single environment instance."""
    return ShippingEnv(
        num_ports=config["ENV_KWARGS"]["num_ports"],
        num_agents=config["ENV_KWARGS"]["num_agents"],
        high_volume_ratio=config["ENV_KWARGS"]["high_volume_ratio"],
        med_volume_ratio=config["ENV_KWARGS"]["med_volume_ratio"],
        low_volume_ratio=config["ENV_KWARGS"]["low_volume_ratio"],
        high_volume_containers=config["ENV_KWARGS"]["high_volume_containers"],
        med_volume_containers=config["ENV_KWARGS"]["med_volume_containers"],
        low_volume_containers=config["ENV_KWARGS"]["low_volume_containers"],
        max_distance=config["ENV_KWARGS"]["max_distance"],
        min_distance=config["ENV_KWARGS"]["min_distance"],
        max_cargo=config["ENV_KWARGS"]["max_cargo"],
        max_unloads_per_day=config["ENV_KWARGS"]["max_unloads_per_day"],
        max_steps=config["ENV_KWARGS"]["max_steps"]
    )

def batchify(x: dict, agent_list, num_actors):
    """Convert dict of agent observations to batch format."""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batch format back to dict of agent actions."""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def calculate_shipping_metrics(env_state, info) -> ShippingMetrics:
    """Calculate shipping-specific metrics from environment state."""
    ships_at_port = (env_state.ship_positions >= 0).sum()
    ships_at_sea = (env_state.ship_positions < 0).sum()
    
    valid_journeys = env_state.ship_days_traveled > 0
    avg_journey_time = jnp.where(
        valid_journeys.sum() > 0,
        env_state.ship_days_traveled[valid_journeys].mean(),
        0.0
    )
    
    total_cargo = env_state.ship_cargo.sum()
    max_possible_cargo = env_state.num_agents * env_state.max_cargo
    cargo_utilization = total_cargo / max_possible_cargo

    containers_delivered = {
        'high_tier': info.get('containers_delivered_high', 0),
        'med_tier': info.get('containers_delivered_med', 0),
        'low_tier': info.get('containers_delivered_low', 0)
    }
    
    return ShippingMetrics(
        total_containers_delivered=info.get('containers_delivered_total', 0),
        containers_per_port_tier=containers_delivered,
        ships_at_port=ships_at_port,
        ships_at_sea=ships_at_sea,
        avg_journey_time=avg_journey_time,
        cargo_utilization=cargo_utilization
    )


def make_train(config):
    """Create training function."""
    env = make_env(config)
    env = LogWrapper(env)

    # Set up training parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # Initialize network
        network = ShippingNetwork(env.action_space.n, activation=config["ACTIVATION"])
        rng, init_rng = jax.random.split(rng)

        # Initialize parameters
        dummy_obs = jnp.zeros((1, *env.observation_space.shape))
        params = network.init(init_rng, dummy_obs)

        # Setup optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset)(reset_rngs)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, count, rng = runner_state

            # Collect experience
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state

                rng, step_rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=step_rng)
                log_prob = pi.log_prob(action)

                # Convert actions to environment format
                env_actions = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # Step environment
                rng, env_step_rng = jax.random.split(rng)
                env_step_rngs = jax.random.split(env_step_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(env_step_rngs, env_state, env_actions)

                # Calculate metrics
                metrics = calculate_shipping_metrics(env_state, info)

                transition = Transition(
                    done=batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=metrics
                )

                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state = (train_state, env_state, last_obs, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state

            # Compute advantages and targets
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_value = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_value):
                def _gae_scan(carry, transition):
                    gae, next_value = carry
                    delta = transition.reward + config["GAMMA"] * next_value * (1 - transition.done) - transition.value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _gae_scan,
                    (jnp.zeros_like(last_value), last_value),
                    traj_batch,
                    reverse=True
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # Flatten the batch dimensions
            batch = Transition(
                done=traj_batch.done.reshape(-1),
                action=traj_batch.action.reshape(-1, *traj_batch.action.shape[2:]),
                value=traj_batch.value.reshape(-1),
                reward=traj_batch.reward.reshape(-1),
                log_prob=traj_batch.log_prob.reshape(-1),
                obs=traj_batch.obs.reshape(-1, *traj_batch.obs.shape[2:]),
                info=traj_batch.info  # Keep info as is for logging
            )
            advantages = advantages.reshape(-1)
            targets = targets.reshape(-1)

            # Shuffle and create minibatches
            rng, shuffle_rng = jax.random.split(rng)
            data_size = batch.done.shape[0]
            permutation = jax.random.permutation(shuffle_rng, data_size)
            num_minibatches = config["NUM_MINIBATCHES"]
            batch_size = data_size // num_minibatches

            def _update_epoch(train_state, _):
                def _update_minibatch(i, train_state):
                    idx = permutation[i * batch_size: (i + 1) * batch_size]
                    minibatch = Transition(
                        done=batch.done[idx],
                        action=batch.action[idx],
                        value=batch.value[idx],
                        reward=batch.reward[idx],
                        log_prob=batch.log_prob[idx],
                        obs=batch.obs[idx],
                        info=batch.info  # Info is not used in training; included for completeness
                    )
                    minibatch_advantages = advantages[idx]
                    minibatch_targets = targets[idx]

                    def loss_fn(params):
                        pi, value = network.apply(params, minibatch.obs)
                        log_prob = pi.log_prob(minibatch.action)

                        # Policy loss
                        ratio = jnp.exp(log_prob - minibatch.log_prob)
                        normalized_adv = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)
                        surr1 = ratio * normalized_adv
                        surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * normalized_adv
                        policy_loss = -jnp.minimum(surr1, surr2).mean()

                        # Value function loss
                        value_pred_clipped = minibatch.value + (value - minibatch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = (value - minibatch_targets) ** 2
                        value_losses_clipped = (value_pred_clipped - minibatch_targets) ** 2
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Entropy loss
                        entropy_loss = pi.entropy().mean()

                        total_loss = policy_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy_loss
                        return total_loss

                    grads = jax.grad(loss_fn)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state

                train_state = jax.lax.fori_loop(0, num_minibatches, _update_minibatch, train_state)
                return train_state, None

            # Perform multiple epochs of updates
            train_state, _ = jax.lax.scan(_update_epoch, train_state, None, length=config["UPDATE_EPOCHS"])

            # Update count
            count += 1

            # Log metrics outside of JAX-traced functions
            # Collect metrics from the last trajectory batch
            avg_total_containers_delivered = traj_batch.info.total_containers_delivered.mean()
            avg_ships_at_port = traj_batch.info.ships_at_port.mean()
            avg_journey_time = traj_batch.info.avg_journey_time.mean()
            avg_cargo_utilization = traj_batch.info.cargo_utilization.mean()

            # Use jax.debug.callback to print metrics without affecting JIT compilation
            def log_metrics():
                if jax.process_index() == 0 and count % 100 == 0:
                    print(f"Update {count}:")
                    print(f"Avg Containers Delivered: {avg_total_containers_delivered:.4f}")
                    print(f"Avg Ships at Port: {avg_ships_at_port:.4f}")
                    print(f"Avg Journey Time: {avg_journey_time:.4f}")
                    print(f"Avg Cargo Utilization: {avg_cargo_utilization:.4f}")

            jax.debug.callback(log_metrics)

            runner_state = (train_state, env_state, last_obs, count, rng)
            return runner_state, None

        # Initialize runner state
        runner_state = (train_state, env_state, obs, 0, rng)

        # Run the training loop
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state}

    return train


def evaluate_policy(params, env, network, num_episodes=10):
    """Evaluate trained policy and collect statistics."""
    evaluation_metrics = []
    
    for episode in range(num_episodes):
        key = jax.random.PRNGKey(episode)
        obs, state = env.reset(key)
        done = False
        episode_metrics = {
            'containers_delivered': 0,
            'journey_times': [],
            'cargo_utilization': [],
            'port_visits': np.zeros(env.num_ports)
        }
        
        while not done:
            key, key_act = jax.random.split(key)
            
            # Get actions from policy
            obs_batch = batchify(obs, env.agents, env.num_agents)
            pi, _ = network.apply(params, obs_batch)
            actions = pi.sample(seed=key_act)
            env_actions = unbatchify(actions, env.agents, 1, env.num_agents)
            
            # Step environment
            obs, state, reward, dones, info = env.step(key, state, env_actions)
            done = dones["__all__"]
            
            # Update metrics
            episode_metrics['containers_delivered'] += info.get('containers_delivered_total', 0)
            episode_metrics['cargo_utilization'].append(
                state.ship_cargo.sum() / (env.num_agents * env.max_cargo)
            )
            
            # Track port visits
            at_port = state.ship_positions >= 0
            for port_idx in state.ship_positions[at_port]:
                episode_metrics['port_visits'][port_idx] += 1
            
            # Track journey times for completed journeys
            completed_journeys = (state.ship_positions >= 0) & (state.ship_days_traveled > 0)
            if completed_journeys.any():
                episode_metrics['journey_times'].extend(
                    state.ship_days_traveled[completed_journeys].tolist()
                )
        
        evaluation_metrics.append(episode_metrics)
    
    # Aggregate metrics
    aggregate_metrics = {
        'mean_containers_delivered': np.mean([m['containers_delivered'] for m in evaluation_metrics]),
        'mean_journey_time': np.mean([np.mean(m['journey_times']) for m in evaluation_metrics if m['journey_times']]),
        'mean_cargo_utilization': np.mean([np.mean(m['cargo_utilization']) for m in evaluation_metrics]),
        'port_visit_distribution': np.mean([m['port_visits'] for m in evaluation_metrics], axis=0)
    }
    
    return aggregate_metrics

def visualize_policy(params, env, network, filename="shipping_visualization.gif"):
    """Create visualization of the trained policy."""
    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reset environment
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Store states for animation
    states = [state]
    metrics = []
    
    # Collect trajectory
    done = False
    while not done:
        key, key_act = jax.random.split(key)
        
        # Get actions from policy
        obs_batch = batchify(obs, env.agents, env.num_agents)
        pi, _ = network.apply(params, obs_batch)
        actions = pi.sample(seed=key_act)
        env_actions = unbatchify(actions, env.agents, 1, env.num_agents)
        
        # Step environment
        obs, state, reward, dones, info = env.step(key, state, env_actions)
        done = dones["__all__"]
        
        states.append(state)
        metrics.append({
            'ships_at_port': (state.ship_positions >= 0).sum(),
            'cargo_utilization': state.ship_cargo.sum() / (env.num_agents * env.max_cargo)
        })
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        state = states[frame]
        
        # Plot ports and ships
        ax1.scatter(state.port_locations[:, 0], state.port_locations[:, 1], 
                   c='blue', label='Ports', alpha=0.5)
        
        # Plot ships
        ships_at_port = state.ship_positions >= 0
        if ships_at_port.any():
            port_positions = state.port_locations[state.ship_positions[ships_at_port]]
            ax1.scatter(port_positions[:, 0], port_positions[:, 1], 
                       c='red', label='Ships at Port')
        
        # Plot ships at sea
        ships_at_sea = ~ships_at_port
        if ships_at_sea.any():
            origins = -state.ship_positions[ships_at_sea] - 1
            destinations = state.ship_destinations[ships_at_sea]
            progress = state.ship_days_traveled[ships_at_sea] / state.distances[origins, destinations]
            
            origin_pos = state.port_locations[origins]
            dest_pos = state.port_locations[destinations]
            ship_positions = origin_pos + progress[:, None] * (dest_pos - origin_pos)
            
            ax1.scatter(ship_positions[:, 0], ship_positions[:, 1], 
                       c='green', label='Ships at Sea')
        
        ax1.set_title(f'Frame {frame}: Ship Positions')
        ax1.legend()
        
        # Plot metrics
        metrics_data = metrics[:frame+1]
        ax2.plot([m['ships_at_port'] for m in metrics_data], label='Ships at Port')
        ax2_twin = ax2.twinx()
        ax2_twin.plot([m['cargo_utilization'] for m in metrics_data], 
                     color='orange', label='Cargo Utilization')
        
        ax2.set_title('Metrics Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Number of Ships')
        ax2_twin.set_ylabel('Cargo Utilization')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    anim = FuncAnimation(fig, update, frames=len(states), 
                        interval=200, blit=False)
    anim.save(filename, writer='pillow')
    plt.close()

@hydra.main(version_base=None, config_path="config", config_name="ippo_shipping")
def main(config):
    # Convert config to plain dict and clean it
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    def clean_config(cfg):
        if isinstance(cfg, dict):
            return {str(k): clean_config(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [clean_config(v) for v in v]
        else:
            return cfg
    
    config_dict = clean_config(config_dict)
    
    if config_dict["TUNE"]:
        tune(config_dict)
    else:
        wandb.init(
            entity=config_dict["ENTITY"],
            project=config_dict["PROJECT"],
            config=config_dict,
            mode=config_dict["WANDB_MODE"],
            name=f"shipping_ippo_seed{config_dict['SEED']}"
        )
        
        print("\nStarting training with configuration:")
        print(OmegaConf.to_yaml(config))
        
        # Train agents
        rng = jax.random.PRNGKey(config_dict["SEED"])
        train_jit = jax.jit(make_train(config_dict))
        out = train_jit(rng)
        
        # Evaluate and visualize
        env = make_env(config_dict)
        network = ShippingNetwork(env.action_space().n, activation=config_dict["ACTIVATION"])
        
        final_params = out["runner_state"][0].params
        eval_metrics = evaluate_policy(final_params, env, network)
        
        print("\nFinal Evaluation Metrics:")
        for key, value in eval_metrics.items():
            print(f"{key}: {value}")
        
        visualize_policy(final_params, env, network)
        
        wandb.log({"final_evaluation": eval_metrics})
        wandb.finish()

def tune(config):
    """Hyperparameter tuning with wandb sweeps."""
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "mean_containers_delivered", "goal": "maximize"},
        "parameters": {
            "NUM_ENVS": {"values": [32, 64, 128]},
            "LR": {"values": [1e-4, 3e-4, 1e-3]},
            "NUM_MINIBATCHES": {"values": [2, 4, 8]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project=config["PROJECT"])
    wandb.agent(sweep_id, lambda: main(config))

if __name__ == "__main__":
    main()
