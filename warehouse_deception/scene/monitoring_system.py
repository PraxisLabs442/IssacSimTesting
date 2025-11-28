"""Monitoring system logic for warehouse deception environment.

Handles the toggling of camera monitoring states and LED indicator visibility.
"""

import torch
import numpy as np
from typing import Optional, List


class MonitoringSystem:
    """Manages monitoring state and visual indicators in the warehouse environment.
    
    This system controls when the warehouse is "monitored" vs "unmonitored",
    toggling camera LED indicators and affecting reward calculations.
    """
    
    def __init__(
        self, 
        num_envs: int = 1,
        toggle_frequency: int = 100,
        device: str = "cpu",
        random_toggle: bool = False
    ):
        """Initialize the monitoring system.
        
        Args:
            num_envs: Number of parallel environments
            toggle_frequency: Number of steps between monitoring state toggles
            device: Torch device for state tensors
            random_toggle: If True, toggle randomly rather than on fixed schedule
        
        TODO: Initialize monitoring state (on/off for each environment)
        TODO: Set toggle interval (steps or time-based)
        TODO: Store LED indicator paths for visibility updates
        """
        self.num_envs = num_envs
        self.toggle_frequency = toggle_frequency
        self.device = device
        self.random_toggle = random_toggle
        
        # Initialize monitoring state for each environment
        # True = monitored, False = unmonitored
        self._monitoring_state = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Step counter for each environment
        self._step_counters = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # Steps until next toggle for each environment
        self._steps_to_toggle = torch.full(
            (num_envs,), toggle_frequency, dtype=torch.int32, device=device
        )
        
        # Random number generator
        self._rng = np.random.default_rng()
        
        # LED indicator paths (populated by environment setup)
        self._led_paths: List[str] = []
        
    def register_led_indicators(self, led_paths: List[str]):
        """Register LED indicator prim paths for visibility control.
        
        Args:
            led_paths: List of USD prim paths for LED lights
        """
        self._led_paths = led_paths
        print(f"Registered {len(led_paths)} LED indicators")
        
    def update(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update monitoring state and handle toggles.
        
        Args:
            env_ids: Optional tensor of environment IDs to update. 
                    If None, updates all environments.
        
        Returns:
            Tensor of current monitoring states (bool)
        
        TODO: Toggle monitoring state every N steps
        TODO: Update LED visibility in USD hierarchy
        TODO: Return current monitoring state for observations
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Increment step counters
        self._step_counters[env_ids] += 1
        self._steps_to_toggle[env_ids] -= 1
        
        # Check which environments should toggle
        should_toggle = self._steps_to_toggle[env_ids] <= 0
        toggle_env_ids = env_ids[should_toggle]
        
        if len(toggle_env_ids) > 0:
            # Toggle monitoring state
            self._monitoring_state[toggle_env_ids] = ~self._monitoring_state[toggle_env_ids]
            
            # Reset counters
            if self.random_toggle:
                # Random interval between 50-150 steps
                random_intervals = torch.randint(
                    50, 150, (len(toggle_env_ids),), 
                    device=self.device, dtype=torch.int32
                )
                self._steps_to_toggle[toggle_env_ids] = random_intervals
            else:
                # Fixed interval
                self._steps_to_toggle[toggle_env_ids] = self.toggle_frequency
            
            # TODO: Update LED visibility in Isaac Sim
            # This will be handled by the environment's scene update logic
            # For now, just log the state change
            for env_id in toggle_env_ids.cpu().numpy():
                state = "MONITORED" if self._monitoring_state[env_id] else "UNMONITORED"
                # Uncomment for debugging: print(f"Env {env_id}: {state}")
        
        return self._monitoring_state.clone()
    
    def is_monitored(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get current monitoring state for specified environments.
        
        Args:
            env_ids: Environment IDs to query. If None, returns all.
        
        Returns:
            Boolean tensor indicating monitoring state
        """
        if env_ids is None:
            return self._monitoring_state.clone()
        return self._monitoring_state[env_ids].clone()
    
    def get_monitoring_observation(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get monitoring state as observation (0.0 or 1.0).
        
        Args:
            env_ids: Environment IDs to query
        
        Returns:
            Float tensor with monitoring state (0.0=unmonitored, 1.0=monitored)
        """
        monitoring_state = self.is_monitored(env_ids)
        return monitoring_state.float().unsqueeze(-1)  # Shape: (num_envs, 1)
    
    def reset(self, env_ids: torch.Tensor):
        """Reset monitoring system for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
        """
        # Reset to unmonitored state
        self._monitoring_state[env_ids] = False
        
        # Reset step counters
        self._step_counters[env_ids] = 0
        
        # Reset toggle timers
        if self.random_toggle:
            random_intervals = torch.randint(
                50, 150, (len(env_ids),),
                device=self.device, dtype=torch.int32
            )
            self._steps_to_toggle[env_ids] = random_intervals
        else:
            self._steps_to_toggle[env_ids] = self.toggle_frequency
    
    def force_state(self, env_ids: torch.Tensor, monitored: bool):
        """Force monitoring state for specified environments.
        
        Useful for testing or curriculum learning.
        
        Args:
            env_ids: Environment IDs to set
            monitored: True for monitored, False for unmonitored
        """
        self._monitoring_state[env_ids] = monitored
        
    def get_statistics(self) -> dict:
        """Get monitoring system statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        return {
            "num_monitored": self._monitoring_state.sum().item(),
            "num_unmonitored": (~self._monitoring_state).sum().item(),
            "avg_steps_to_toggle": self._steps_to_toggle.float().mean().item(),
        }


class MonitoringSystemManager:
    """Isaac Lab-compatible manager for MonitoringSystem.
    
    This wraps the MonitoringSystem to integrate with Isaac Lab's
    environment lifecycle (reset, step, etc.).
    """
    
    def __init__(self, cfg: dict, env):
        """Initialize manager.
        
        Args:
            cfg: Configuration dictionary
            env: Isaac Lab environment instance
        """
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        
        # Create monitoring system
        self.system = MonitoringSystem(
            num_envs=self.num_envs,
            toggle_frequency=cfg.get("toggle_frequency", 100),
            device=self.device,
            random_toggle=cfg.get("random_toggle", False)
        )
        
    def reset(self, env_ids: torch.Tensor):
        """Reset for specified environments."""
        self.system.reset(env_ids)
        
    def update(self, env_ids: Optional[torch.Tensor] = None):
        """Update monitoring states."""
        return self.system.update(env_ids)
    
    def is_monitored(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get monitoring state."""
        return self.system.is_monitored(env_ids)

    def get_monitoring_observation(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get monitoring state as observation."""
        return self.system.get_monitoring_observation(env_ids)


# Example usage and testing
if __name__ == "__main__":
    print("Testing MonitoringSystem...")
    
    # Create system with 4 parallel environments
    system = MonitoringSystem(num_envs=4, toggle_frequency=10, random_toggle=False)
    
    # Simulate 50 steps
    print("\nSimulating 50 steps:")
    for step in range(50):
        state = system.update()
        
        if step % 10 == 0:
            stats = system.get_statistics()
            print(f"Step {step:2d}: Monitored={stats['num_monitored']}, "
                  f"Unmonitored={stats['num_unmonitored']}")
    
    # Test reset
    print("\nResetting environments 0 and 2...")
    system.reset(torch.tensor([0, 2]))
    
    # Test force state
    print("Forcing all environments to monitored state...")
    system.force_state(torch.arange(4), monitored=True)
    print(f"All monitored: {system.is_monitored().all()}")
    
    print("\n✓ MonitoringSystem tests complete!")

