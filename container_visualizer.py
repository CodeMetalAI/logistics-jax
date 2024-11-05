import jax.numpy as jnp
from typing import List
import matplotlib.colors as mcolors
import io
from PIL import Image
import imageio
from container_simple import State

class ShippingVisualizer:
    def __init__(self, max_distance: float = 30.0):
        self.max_distance = max_distance
        self.width = 800
        self.height = 600
        self.margin = 50
        
        # Color schemes
        self.port_colors = {
            0: "#ff4444",  # High volume - red
            1: "#44aa44",  # Medium volume - green
            2: "#4444ff"   # Low volume - blue
        }
    
    def create_frame(self, state, step: int) -> Image.Image:
        """Create a single frame of the visualization."""
        # Create SVG string
        svg = f"""
        <svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f0f0f0"/>
            {self._draw_grid()}
            {self._draw_ports(state)}
            {self._draw_ships(state)}
            {self._draw_info(state, step)}
        </svg>
        """
        
        # Convert SVG to PIL Image
        return self._svg_to_image(svg)
    
    def _draw_grid(self) -> str:
        """Draw background grid."""
        lines = []
        grid_size = 10
        for i in range(grid_size + 1):
            x = self.margin + (self.width - 2*self.margin) * i / grid_size
            y = self.margin + (self.height - 2*self.margin) * i / grid_size
            lines.append(f'<line x1="{x}" y1="{self.margin}" x2="{x}" y2="{self.height-self.margin}" stroke="#ddd" stroke-width="1"/>')
            lines.append(f'<line x1="{self.margin}" y1="{y}" x2="{self.width-self.margin}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        return "\n".join(lines)
    
    def _draw_ports(self, state) -> str:
        """Draw ports with size indicating volume tier."""
        ports = []
        max_radius = 20
        
        for i in range(len(state.port_volume_tiers)):
            # Scale coordinates to view space
            x = self.margin + (self.width - 2*self.margin) * state.port_locations[i,0] / self.max_distance
            y = self.margin + (self.height - 2*self.margin) * state.port_locations[i,1] / self.max_distance
            
            # Port size based on tier
            tier = int(state.port_volume_tiers[i])
            radius = max_radius * (3 - tier) / 3
            
            # Total containers at port
            total_containers = state.port_containers[i].sum()
            container_scale = min(1.0, total_containers / 10000)  # Scale for visualization
            
            ports.append(f"""
                <circle cx="{x}" cy="{y}" r="{radius}" 
                        fill="{self.port_colors[tier]}" 
                        fill-opacity="{0.3 + 0.7*container_scale}"
                        stroke="black" stroke-width="1"/>
                <text x="{x}" y="{y+radius+12}" text-anchor="middle" font-size="10">
                    Port {i}: {total_containers}
                </text>
            """)
        
        return "\n".join(ports)
    
    def _draw_ships(self, state) -> str:
        """Draw ships with size indicating cargo amount."""
        ships = []
        for i in range(len(state.ship_positions)):
            if state.ship_positions[i] >= 0:  # Ship at port
                port_idx = state.ship_positions[i]
                x = self.margin + (self.width - 2*self.margin) * state.port_locations[port_idx,0] / self.max_distance
                y = self.margin + (self.height - 2*self.margin) * state.port_locations[port_idx,1] / self.max_distance
            else:  # Ship at sea
                origin_idx = -state.ship_positions[i] - 1
                dest_idx = state.ship_destinations[i]
                progress = state.ship_days_traveled[i] / state.distances[origin_idx, dest_idx]
                
                # Interpolate position
                ox = state.port_locations[origin_idx,0]
                oy = state.port_locations[origin_idx,1]
                dx = state.port_locations[dest_idx,0]
                dy = state.port_locations[dest_idx,1]
                
                x = self.margin + (self.width - 2*self.margin) * (ox + progress*(dx-ox)) / self.max_distance
                y = self.margin + (self.height - 2*self.margin) * (oy + progress*(dy-oy)) / self.max_distance
            
            # Ship size based on cargo
            total_cargo = state.ship_cargo[i].sum()
            size = 5 + 15 * (total_cargo / state.ship_cargo[i].shape[0])
            
            ships.append(f"""
                <polygon points="{x},{y-size} {x+size/2},{y+size/2} {x-size/2},{y+size/2}"
                         fill="black" stroke="none"/>
                <text x="{x}" y="{y+size+12}" text-anchor="middle" font-size="10">
                    Ship {i}: {total_cargo}
                </text>
            """)
        
        return "\n".join(ships)
    
    def _draw_info(self, state, step: int) -> str:
        """Draw information panel."""
        total_containers = state.port_containers.sum() + state.ship_cargo.sum()
        delivered_containers = state.port_containers.diagonal().sum()
        
        return f"""
            <text x="10" y="20" font-size="12">Step: {step}</text>
            <text x="10" y="40" font-size="12">Total Containers: {total_containers}</text>
            <text x="10" y="60" font-size="12">Delivered: {delivered_containers}</text>
        """
    
    def _svg_to_image(self, svg: str) -> Image.Image:
        """Convert SVG string to PIL Image."""
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        return Image.open(io.BytesIO(png_data))
    
    def create_animation(self, state_sequence: List[State], filename: str = "shipping.gif"):
        """Create animation from sequence of states."""
        frames = []
        for step, state in enumerate(state_sequence):
            frames.append(self.create_frame(state, step))
        
        # Save as GIF
        imageio.mimsave(filename, frames, fps=2)
        return filename

# Add visualization method to ShippingEnv
def visualize_episode(env, states: List[State], filename: str = "shipping.gif"):
    """Create visualization of an episode."""
    visualizer = ShippingVisualizer(max_distance=env.max_distance)
    return visualizer.create_animation(states, filename)