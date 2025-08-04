"""
Real-time visualization module for Neural Memory Mapper.
Implements brain activity and memory formation visualization.
"""

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go


class BrainActivityVisualizer:
    """Handles real-time visualization of brain activity and memory states."""
    
    def __init__(self, num_channels=8):
        """
        Initialize the visualizer.
        
        Args:
            num_channels (int): Number of EEG channels
        """
        self.num_channels = num_channels
        self.fig = None
        self.channel_positions = self._generate_channel_positions()
    
    def _generate_channel_positions(self) -> Dict[int, tuple]:
        """
        Generate normalized positions for EEG channels.
        
        Returns:
            Dict[int, tuple]: Channel positions {channel_idx: (x, y)}
        """
        positions = {}
        radius = 0.8
        for i in range(self.num_channels):
            angle = 2 * np.pi * i / self.num_channels
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = (x, y)
        return positions
    
    def create_heatmap(self, band_powers: Dict[str, List[float]]):
        """
        Create a heatmap of brain activity.
        
        Args:
            band_powers (Dict[str, List[float]]): Power values for each frequency band
        
        Returns:
            go.Figure: Plotly figure object
        """
        x = []
        y = []
        power_values = []
        
        for ch, (x_pos, y_pos) in self.channel_positions.items():
            for band, powers in band_powers.items():
                x.append(x_pos)
                y.append(y_pos)
                power_values.append(powers[ch])
        
        self.fig = go.Figure(data=go.Heatmap(
            x=x,
            y=y,
            z=power_values,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False
        ))
        
        self.fig.update_layout(
            title='Brain Activity Heatmap',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=600,
            height=600
        )
        
        return self.fig
    
    def create_memory_formation_display(self, memory_metrics: Dict[str, float]):
        """
        Create visualization of memory formation metrics.
        
        Args:
            memory_metrics (Dict[str, float]): Memory formation metrics
        
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Add gauge for memory formation strength
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=memory_metrics['memory_formation_strength'],
            domain={'x': [0, 0.5], 'y': [0, 1]},
            title={'text': "Memory Formation"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': 'lightgray'},
                    {'range': [0.3, 0.7], 'color': 'gray'},
                    {'range': [0.7, 1], 'color': 'darkgray'}
                ]
            }
        ))
        
        # Add gauge for attention level
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=memory_metrics['attention_level'],
            domain={'x': [0.5, 1], 'y': [0, 1]},
            title={'text': "Attention Level"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 0.3], 'color': 'lightgray'},
                    {'range': [0.3, 0.7], 'color': 'gray'},
                    {'range': [0.7, 1], 'color': 'darkgray'}
                ]
            }
        ))
        
        fig.update_layout(
            title='Memory Formation Metrics',
            width=800,
            height=400
        )
        
        return fig
