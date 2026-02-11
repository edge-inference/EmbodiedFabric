"""Gossip-based state sharing (DSM) for multi-robot context."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    agent_id: str
    position: Tuple[float, float, float]
    state: str
    task_id: Optional[str]
    timestamp_ms: int
    
    def age_ms(self, current_time_ms: int) -> int:
        return current_time_ms - self.timestamp_ms


@dataclass
class SharedObservation:
    agent_id: str
    position: Tuple[float, float, float]
    detected_objects: List[Dict[str, Any]]
    timestamp_ms: int


@dataclass
class JamSignal:
    location: Tuple[float, float, float]
    severity: float  # 0-1
    reporter_id: str
    timestamp_ms: int


class LocalCache:
    """Per-agent cache of shared state."""
    
    def __init__(self, agent_id: str, max_aoi_ms: int = 3000):
        self.agent_id = agent_id
        self.max_aoi_ms = max_aoi_ms
        
        self._agent_states: Dict[str, AgentState] = {}
        self._observations: List[SharedObservation] = []
        self._jam_signals: List[JamSignal] = []
        self._path_intents: Dict[str, List[Tuple[float, float, float]]] = {}
    
    def write_agent_state(self, state: AgentState) -> None:
        """Update agent state in cache"""
        self._agent_states[state.agent_id] = state
    
    def write_observation(self, obs: SharedObservation) -> None:
        """Add shared observation"""
        self._observations.append(obs)
        if len(self._observations) > 100:
            self._observations = self._observations[-50:]
    
    def write_jam_signal(self, signal: JamSignal) -> None:
        """Add jam signal"""
        self._jam_signals.append(signal)
        if len(self._jam_signals) > 50:
            self._jam_signals = self._jam_signals[-25:]
    
    def write_path_intent(self, agent_id: str, path: List[Tuple[float, float, float]]) -> None:
        """Update path intent for agent"""
        self._path_intents[agent_id] = path
    
    def get_nearby_agents(self, position: Tuple[float, float, float], 
                         radius: float, current_time_ms: int) -> List[AgentState]:
        """Get agents within radius, filtering stale entries"""
        result = []
        for state in self._agent_states.values():
            if state.agent_id == self.agent_id:
                continue
            if state.age_ms(current_time_ms) > self.max_aoi_ms:
                continue
            
            dist = np.sqrt(
                (state.position[0] - position[0])**2 +
                (state.position[1] - position[1])**2
            )
            if dist <= radius:
                result.append(state)
        
        return result
    
    def get_jam_signals(self, position: Tuple[float, float, float],
                       radius: float, current_time_ms: int) -> List[JamSignal]:
        """Get jam signals near position"""
        result = []
        for signal in self._jam_signals:
            age = current_time_ms - signal.timestamp_ms
            if age > self.max_aoi_ms:
                continue
            
            dist = np.sqrt(
                (signal.location[0] - position[0])**2 +
                (signal.location[1] - position[1])**2
            )
            if dist <= radius:
                result.append(signal)
        
        return result
    
    def merge_from(self, other: 'LocalCache', current_time_ms: int) -> int:
        """
        Merge data from another cache (gossip receive).
        Returns number of entries updated.
        """
        updates = 0
        
        for agent_id, state in other._agent_states.items():
            existing = self._agent_states.get(agent_id)
            if existing is None or state.timestamp_ms > existing.timestamp_ms:
                self._agent_states[agent_id] = state
                updates += 1
        
        existing_obs_times = {(o.agent_id, o.timestamp_ms) for o in self._observations}
        for obs in other._observations:
            if (obs.agent_id, obs.timestamp_ms) not in existing_obs_times:
                self._observations.append(obs)
                updates += 1
        
        return updates
    
    def cleanup_stale(self, current_time_ms: int) -> None:
        """Remove stale entries"""
        max_age = self.max_aoi_ms * 10
        
        self._agent_states = {
            k: v for k, v in self._agent_states.items()
            if v.age_ms(current_time_ms) < max_age
        }
        
        self._observations = [
            o for o in self._observations
            if current_time_ms - o.timestamp_ms < max_age
        ]
        
        self._jam_signals = [
            s for s in self._jam_signals
            if current_time_ms - s.timestamp_ms < max_age
        ]


class DistributedSharedMemory:
    """
    Distributed Shared Memory system.
    
    Manages gossip protocol and local caches for all agents.
    This is the R1 context-memory fabric.
    """
    
    def __init__(self,
                 n_agents: int,
                 gossip_period_ms: float = 50.0,
                 max_aoi_ms: int = 3000):
        self.n_agents = n_agents
        self.gossip_period_ms = gossip_period_ms
        self.max_aoi_ms = max_aoi_ms
        
        self._caches: Dict[str, LocalCache] = {}
        self._start_time = time.time()
        
        self._gossip_count = 0
        self._total_updates = 0
    
    def _current_time_ms(self) -> int:
        return int((time.time() - self._start_time) * 1000)
    
    def register_agent(self, agent_id: str) -> LocalCache:
        """Register an agent and create its local cache"""
        cache = LocalCache(agent_id, self.max_aoi_ms)
        self._caches[agent_id] = cache
        return cache
    
    def get_cache(self, agent_id: str) -> Optional[LocalCache]:
        """Get an agent's local cache"""
        return self._caches.get(agent_id)
    
    def write_agent_state(self,
                          agent_id: str,
                          position: Tuple[float, float, float],
                          state: str,
                          task_id: Optional[str] = None) -> None:
        """Write agent state (called by agent each step)"""
        cache = self._caches.get(agent_id)
        if not cache:
            cache = self.register_agent(agent_id)
        
        agent_state = AgentState(
            agent_id=agent_id,
            position=position,
            state=state,
            task_id=task_id,
            timestamp_ms=self._current_time_ms()
        )
        cache.write_agent_state(agent_state)
    
    def get_nearby_agents(self, 
                          position: Tuple[float, float, float],
                          radius: float = 5.0,
                          requester_id: Optional[str] = None) -> List[AgentState]:
        """Get nearby agents from requester's cache"""
        if requester_id and requester_id in self._caches:
            cache = self._caches[requester_id]
            return cache.get_nearby_agents(position, radius, self._current_time_ms())
        
        result = []
        for cache in self._caches.values():
            result.extend(cache.get_nearby_agents(position, radius, self._current_time_ms()))
        
        seen = set()
        unique = []
        for s in result:
            if s.agent_id not in seen:
                seen.add(s.agent_id)
                unique.append(s)
        return unique
    
    def get_jam_signals(self,
                        position: Tuple[float, float, float],
                        radius: float = 10.0) -> List[JamSignal]:
        """Get jam signals near position"""
        all_signals = []
        current_time = self._current_time_ms()
        
        for cache in self._caches.values():
            all_signals.extend(cache.get_jam_signals(position, radius, current_time))
        
        return all_signals
    
    def get_shared_observations(self,
                                position: Tuple[float, float, float],
                                radius: float = 10.0) -> List[SharedObservation]:
        """Get shared observations near position"""
        result = []
        current_time = self._current_time_ms()
        
        for cache in self._caches.values():
            for obs in cache._observations:
                if current_time - obs.timestamp_ms > self.max_aoi_ms:
                    continue
                dist = np.sqrt(
                    (obs.position[0] - position[0])**2 +
                    (obs.position[1] - position[1])**2
                )
                if dist <= radius:
                    result.append(obs)
        
        return result
    
    def gossip_round(self, agents: List[Any]) -> int:
        """
        Execute one gossip round.
        
        Each agent exchanges data with a random peer.
        Returns total number of updates.
        """
        current_time = self._current_time_ms()
        total_updates = 0
        
        agent_ids = list(self._caches.keys())
        np.random.shuffle(agent_ids)
        
        for i in range(0, len(agent_ids) - 1, 2):
            cache_a = self._caches[agent_ids[i]]
            cache_b = self._caches[agent_ids[i + 1]]
            
            updates_a = cache_a.merge_from(cache_b, current_time)
            updates_b = cache_b.merge_from(cache_a, current_time)
            total_updates += updates_a + updates_b
        
        self._gossip_count += 1
        self._total_updates += total_updates
        
        if self._gossip_count % 100 == 0:
            for cache in self._caches.values():
                cache.cleanup_stale(current_time)
        
        return total_updates
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get DSM metrics for profiling"""
        return {
            'n_agents': len(self._caches),
            'gossip_rounds': self._gossip_count,
            'total_updates': self._total_updates,
            'avg_updates_per_round': self._total_updates / max(self._gossip_count, 1),
            'gossip_period_ms': self.gossip_period_ms,
            'max_aoi_ms': self.max_aoi_ms
        }
