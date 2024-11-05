from typing import Tuple, Dict
import jax
import jax.numpy as jnp
from flax import struct
import chex
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete
import numpy as np

@struct.dataclass
class State:
    """Environment state with explicit types."""
    # Port information
    port_locations: chex.Array  # float32[num_ports, 2]
    port_containers: chex.Array  # int32[num_ports, num_ports]
    port_volume_tiers: chex.Array  # int32[num_ports]
    
    # Ship/Agent information
    ship_positions: chex.Array  # int32[num_agents]
    ship_destinations: chex.Array  # int32[num_agents]
    ship_cargo: chex.Array  # int32[num_agents, num_ports]
    ship_days_traveled: chex.Array  # int32[num_agents]
    
    # Distance matrix
    distances: chex.Array  # int32[num_ports, num_ports]
    
    # Time tracking
    time: int
    terminal: bool

    def replace(self, **updates):
        """Override replace to ensure type consistency."""
        return self.__class__(
            port_locations=updates.get('port_locations', self.port_locations),
            port_containers=updates.get('port_containers', self.port_containers).astype(jnp.int32),
            port_volume_tiers=updates.get('port_volume_tiers', self.port_volume_tiers).astype(jnp.int32),
            ship_positions=updates.get('ship_positions', self.ship_positions).astype(jnp.int32),
            ship_destinations=updates.get('ship_destinations', self.ship_destinations).astype(jnp.int32),
            ship_cargo=updates.get('ship_cargo', self.ship_cargo).astype(jnp.int32),
            ship_days_traveled=updates.get('ship_days_traveled', self.ship_days_traveled).astype(jnp.int32),
            distances=updates.get('distances', self.distances).astype(jnp.int32),
            time=updates.get('time', self.time),
            terminal=updates.get('terminal', self.terminal)
        )

class ShippingEnv(MultiAgentEnv):
    def __init__(self,
                num_ports: int = 100,
                num_agents: int = 100,
                high_volume_ratio: float = 0.15,
                med_volume_ratio: float = 0.65,
                low_volume_ratio: float = 0.20,
                high_volume_containers: int = 500000,
                med_volume_containers: int = 50000,
                low_volume_containers: int = 5000,
                max_distance: int = 30,
                min_distance: int = 1,
                max_cargo: int = 10000,
                max_unloads_per_day: int = 2000,
                max_steps: int = 60):
        
        super().__init__(num_agents=num_agents)
        
        # Add agents property
        self.agents = [f"agent_{i}" for i in range(num_agents)]
            

        # Validate and store configuration
        assert abs(high_volume_ratio + med_volume_ratio + low_volume_ratio - 1.0) < 1e-6, \
            "Volume ratios must sum to 1"
        
        self.num_ports = num_ports
        self.high_volume_ratio = high_volume_ratio
        self.med_volume_ratio = med_volume_ratio
        self.low_volume_ratio = low_volume_ratio
        
        # Calculate number of ports per tier (ensure they sum to num_ports)
        self.num_high_tier = int(num_ports * high_volume_ratio)
        self.num_med_tier = int(num_ports * med_volume_ratio)
        self.num_low_tier = num_ports - self.num_high_tier - self.num_med_tier
        
        print(f"\nInitializing Environment:")
        print(f"Total ports: {self.num_ports}")
        print(f"High tier ports: {self.num_high_tier}")
        print(f"Med tier ports: {self.num_med_tier}")
        print(f"Low tier ports: {self.num_low_tier}")
        print(f"Sum of tier ports: {self.num_high_tier + self.num_med_tier + self.num_low_tier}")
        
        self.high_volume_containers = high_volume_containers
        self.med_volume_containers = med_volume_containers
        self.low_volume_containers = low_volume_containers
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.max_cargo = max_cargo
        self.max_unloads_per_day = max_unloads_per_day
        self.max_steps = max_steps
        
        # Set up action and observation spaces
        self.action_space = Discrete(num_ports)
        
        # Observation space dimensions
        obs_dim = 1 + num_ports + num_ports + 1 + 1 + num_ports + num_ports
        self.observation_space = Box(low=0, high=float('inf'), shape=(obs_dim,))

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment with proper port initialization."""
        keys = jax.random.split(key, 4)
        
        # Generate port locations
        port_locations = self._generate_port_locations(keys[0])
        
        # Calculate distance matrix
        distances = self._calculate_distances(port_locations)
        
        # Initialize port tiers
        high_tier = jnp.zeros(self.num_high_tier)
        med_tier = jnp.ones(self.num_med_tier)
        low_tier = 2 * jnp.ones(self.num_low_tier)
        port_volume_tiers = jnp.concatenate([high_tier, med_tier, low_tier]).astype(jnp.int32)
        
        # Initialize port containers with proper distribution
        port_containers = self._initialize_port_containers(keys[1], port_volume_tiers)
        
        # Initialize ships
        ship_positions = jnp.zeros(self.num_agents, dtype=jnp.int32)
        ship_destinations = -jnp.ones(self.num_agents, dtype=jnp.int32)
        ship_days_traveled = jnp.zeros(self.num_agents, dtype=jnp.int32)
        
        # Initialize ship cargo - testing cargo for first ship
        ship_cargo = jnp.zeros((self.num_agents, self.num_ports), dtype=jnp.int32)
        ship_cargo = ship_cargo.at[0, 0].set(100)
        
        state = State(
            port_locations=port_locations,
            port_containers=port_containers,
            port_volume_tiers=port_volume_tiers,
            ship_positions=ship_positions,
            ship_destinations=ship_destinations,
            ship_cargo=ship_cargo,
            ship_days_traveled=ship_days_traveled,
            distances=distances.astype(jnp.int32),
            time=0,
            terminal=False
        )
        
        obs = self._get_obs(state)
        
        # Print reset debug info using a JAX-compatible approach
        def debug_print(port_volume_tiers):
            # Count tiers manually instead of using bincount
            high_count = jnp.sum(port_volume_tiers == 0)
            med_count = jnp.sum(port_volume_tiers == 1)
            low_count = jnp.sum(port_volume_tiers == 2)
            
            print("\nReset Debug Info:")
            print(f"Port volume tier counts - High: {high_count}, Med: {med_count}, Low: {low_count}")
            print(f"Total containers in system: {port_containers.sum()}")
            print(f"Average containers per port: {port_containers.sum() / self.num_ports}")
        
        jax.debug.callback(debug_print, port_volume_tiers)
        
        return obs, state
    
    def _update_state(self, state: State, actions: chex.Array) -> Tuple[State, chex.Array]:
        """Update state with integrated movement and cargo operations."""
        # Identify ships at port and update destinations
        at_port = state.ship_positions >= 0
        new_destinations = jnp.where(
            at_port,
            actions,
            state.ship_destinations
        )
        
        # Process cargo first
        state_after_cargo, rewards = self._process_port_operations(state, at_port)
        
        # Move ships
        new_positions, new_days = self._move_ships(
            state_after_cargo.ship_positions,
            state_after_cargo.ship_destinations,
            new_destinations,
            state_after_cargo.ship_days_traveled,
            state_after_cargo.distances
        )
        
        # Create final state
        new_state = state_after_cargo.replace(
            ship_positions=new_positions,
            ship_destinations=new_destinations,
            ship_days_traveled=new_days
        )
        
        print("\nState Update Summary:")
        print(f"At port: {at_port.sum()} ships")
        print(f"Position changes: {(new_positions != state.ship_positions).sum()} ships")
        print(f"Total rewards: {rewards.sum()}")
        
        return new_state, rewards
    
    def step(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one step of the environment."""
        action_array = jnp.array([actions[f"agent_{i}"] for i in range(self.num_agents)])
        
        # Update state and calculate rewards
        new_state, rewards = self._update_state(state, action_array)
        
        # Update time and check for terminal condition
        new_state = new_state.replace(
            time=state.time + 1,
            terminal=(state.time + 1 >= self.max_steps)
        )
        
        # Get observations
        obs = self._get_obs(new_state)
        
        # Package rewards and dones for each agent
        rewards_dict = {f"agent_{i}": rewards[i] for i in range(self.num_agents)}
        dones = {f"agent_{i}": new_state.terminal for i in range(self.num_agents)}
        dones["__all__"] = new_state.terminal
        
        return obs, new_state, rewards_dict, dones, {}

    def _generate_port_locations(self, key: chex.PRNGKey) -> chex.Array:
        """Generate port locations ensuring minimum distance between ports."""
        grid_size = int(np.ceil(np.sqrt(self.num_ports)))
        spacing = self.max_distance / grid_size

        x = jnp.linspace(0, self.max_distance, grid_size)
        y = jnp.linspace(0, self.max_distance, grid_size)
        xx, yy = jnp.meshgrid(x, y)

        # Add random jitter to port locations
        key1, key2 = jax.random.split(key)
        xx = xx + jax.random.uniform(key1, xx.shape) * spacing * 0.5
        yy = yy + jax.random.uniform(key2, yy.shape) * spacing * 0.5
        # Take first num_ports points
        locations = jnp.stack([xx.flatten(), yy.flatten()], axis=1)[:self.num_ports]
        
        return locations

    def _initialize_port_containers(self, key: chex.PRNGKey, volume_tiers: chex.Array) -> chex.Array:
        """Initialize port containers with overflow protection."""
        # Scale down the base volumes to prevent overflow
        scale_factor = 0.001  # Reduce container counts to prevent overflow
        
        base_volumes = jnp.where(
            volume_tiers == 0, int(self.high_volume_containers * scale_factor),
            jnp.where(volume_tiers == 1, int(self.med_volume_containers * scale_factor),
                    int(self.low_volume_containers * scale_factor))
        )
        
        # Create distribution weights
        dist_weights = jnp.where(
            volume_tiers[None, :] == 0, 0.6,
            jnp.where(volume_tiers[None, :] == 1, 0.3, 0.1)
        )
        
        # Zero diagonal for no self-shipping
        dist_weights = dist_weights.at[jnp.arange(self.num_ports), jnp.arange(self.num_ports)].set(0)
        dist_weights = dist_weights / dist_weights.sum(axis=1, keepdims=True)
        
        # Generate container distribution
        container_rates = jnp.expand_dims(base_volumes, 1) * dist_weights
        containers = jax.random.poisson(key, container_rates).astype(jnp.int32)
        
        return containers

    def _calculate_distances(self, locations: chex.Array) -> chex.Array:
        """Calculate realistic shipping distances between ports."""
        # Calculate Manhattan distance to better represent shipping routes
        x_diff = jnp.abs(locations[:, None, 0] - locations[None, :, 0])
        y_diff = jnp.abs(locations[:, None, 1] - locations[None, :, 1])
        distances = x_diff + y_diff
        
        # Scale to desired range and ensure minimum distance
        distances = jnp.maximum(
            (distances * (self.max_distance / distances.max())).astype(jnp.int32),
            self.min_distance
        )
        
        return distances

    def _move_ships(self, positions: chex.Array, current_destinations: chex.Array,
                new_destinations: chex.Array, days_traveled: chex.Array,
                distances: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Update ship positions with fixed movement logic."""
        
        # Identify ships at sea and in port
        at_sea = positions < 0
        at_port = ~at_sea
        
        # For ships at sea: continue journey or arrive
        origin_ports = jnp.where(at_sea, -positions - 1, 0)
        
        # Calculate journey lengths for ships at sea
        journey_lengths = jnp.where(
            at_sea,
            distances[origin_ports, current_destinations],
            0
        )
        
        # Update days traveled for ships at sea
        new_days = jnp.where(at_sea, days_traveled + 1, 0)
        
        # Check which ships have arrived
        arrived = (new_days >= journey_lengths) & at_sea
        
        # For ships in port: start new journey only if destination is different from current port
        current_ports = jnp.where(at_port, positions, 0)
        starting_journey = at_port & (new_destinations != current_ports) & (new_destinations >= 0)
        
        # Update positions
        new_positions = jnp.where(
            arrived,  # Ships that arrived
            current_destinations,  # Move to destination
            jnp.where(
                starting_journey,  # Ships starting new journey
                -current_ports - 1,  # Store origin port
                positions  # Keep current position
            )
        )
        
        # Reset days for new journeys and arrived ships
        new_days = jnp.where(
            arrived | starting_journey,
            0,
            new_days
        )
        
        print("\nShip Movement Debug:")
        print(f"Ships at sea: {at_sea.sum()}")
        print(f"Ships arrived: {arrived.sum()}")
        print(f"Ships starting journey: {starting_journey.sum()}")
        print(f"Current ports: {current_ports}")
        print(f"New destinations: {new_destinations}")
        print(f"Journey lengths: {journey_lengths}")
        
        return new_positions, new_days

    def _process_port_operations(self, state: State, at_port: chex.Array) -> Tuple[State, chex.Array]:
        """Process cargo operations with unloading limits and loading."""
        # Initialize arrays
        rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)
        new_ship_cargo = state.ship_cargo.copy()
        new_port_containers = state.port_containers.copy()
        
        def process_ship(i, carry):
            ship_cargo, port_containers, ship_rewards = carry
            
            if at_port[i]:
                port_idx = state.ship_positions[i]
                
                # 1. Unload cargo first - respect max_unloads_per_day
                cargo_at_current_port = ship_cargo[i, port_idx]
                unload_amount = jnp.minimum(
                    cargo_at_current_port,
                    self.max_unloads_per_day
                ).astype(jnp.int32)
                
                # Update cargo and containers
                ship_cargo = ship_cargo.at[i, port_idx].add(-unload_amount)
                port_containers = port_containers.at[port_idx, port_idx].add(unload_amount)
                ship_rewards = ship_rewards.at[i].set(jnp.float32(unload_amount))
                
                # 2. Load new cargo if there's available space
                available_space = self.max_cargo - ship_cargo[i].sum()
                if available_space > 0 and port_idx < self.num_ports:
                    # Get available containers at current port for other destinations
                    available_containers = port_containers[port_idx]
                    
                    # Don't load containers for current port
                    available_containers = available_containers.at[port_idx].set(0)
                    
                    # Calculate load amounts per destination
                    load_amounts = jnp.minimum(
                        available_containers,
                        jnp.minimum(
                            self.max_unloads_per_day,
                            available_space // self.num_ports
                        )
                    ).astype(jnp.int32)
                    
                    # Update cargo and containers
                    ship_cargo = ship_cargo.at[i].add(load_amounts)
                    port_containers = port_containers.at[port_idx].add(-load_amounts)
            
            return ship_cargo, port_containers, ship_rewards
        
        # Process ships sequentially
        for i in range(self.num_agents):
            new_ship_cargo, new_port_containers, rewards = process_ship(
                i, (new_ship_cargo, new_port_containers, rewards)
            )
        
        # Create new state
        new_state = state.replace(
            ship_cargo=new_ship_cargo.astype(jnp.int32),
            port_containers=new_port_containers.astype(jnp.int32)
        )
        
        # Debug info
        print("\nPort Operations Debug:")
        print(f"At port: {at_port}")
        print(f"Current ports: {state.ship_positions}")
        print(f"Original cargo: {state.ship_cargo}")
        print(f"New cargo: {new_ship_cargo}")
        print(f"Unloaded amounts: {rewards}")
        
        return new_state, rewards

    def _get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations for all agents."""
        observations = {}
        for i in range(self.num_agents):
            at_port = state.ship_positions[i] >= 0
            current_port = jnp.where(
                at_port,
                state.ship_positions[i],
                -state.ship_positions[i] - 1
            )
            departure_port = jax.nn.one_hot(current_port, self.num_ports)
            destination_port = jax.nn.one_hot(
                jnp.maximum(state.ship_destinations[i], 0),
                self.num_ports
            )
            days_to_dest = jnp.where(
                at_port,
                0,
                state.distances[current_port, state.ship_destinations[i]] - state.ship_days_traveled[i]
            )
            
            # Convert boolean to float using where instead of float()
            at_port_float = jnp.where(at_port, 1.0, 0.0)
            
            obs = jnp.concatenate([
                jnp.array([at_port_float]),  # Changed this line
                departure_port,
                destination_port,
                jnp.array([state.ship_days_traveled[i]], dtype=jnp.float32),
                jnp.array([days_to_dest], dtype=jnp.float32),
                state.ship_cargo[i].astype(jnp.float32),
                jnp.where(at_port, 
                        state.port_containers[current_port].astype(jnp.float32), 
                        jnp.zeros(self.num_ports, dtype=jnp.float32))
            ])
            observations[f"agent_{i}"] = obs
        return observations
