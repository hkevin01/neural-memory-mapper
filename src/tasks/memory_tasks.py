"""
Memory task module for Neural Memory Mapper.
Implements various memory tasks and performance tracking.
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class MemoryTask(ABC):
    """Abstract base class for memory tasks."""
    
    @abstractmethod
    def present_stimulus(self):
        """Present the stimulus to the user."""
        raise NotImplementedError
    
    @abstractmethod
    def get_response(self):
        """Get the user's response."""
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_performance(self):
        """Evaluate the user's performance."""
        raise NotImplementedError


class WordListTask(MemoryTask):
    """Word list memory task implementation."""
    
    def __init__(self, word_list_size=10, presentation_time=2.0):
        """
        Initialize word list task.
        
        Args:
            word_list_size (int): Number of words to remember
            presentation_time (float): Time in seconds to show each word
        """
        self.word_list_size = word_list_size
        self.presentation_time = presentation_time
        self.current_words = []
        self.user_responses = []
        
        # Common words for the task
        self.word_pool = [
            'apple', 'book', 'cat', 'door', 'elephant',
            'flower', 'guitar', 'house', 'island', 'jacket',
            'kitchen', 'lemon', 'mountain', 'needle', 'ocean',
            'pencil', 'queen', 'river', 'sunset', 'table'
        ]
    
    def present_stimulus(self) -> List[str]:
        """
        Present a list of words to remember.
        
        Returns:
            List[str]: List of words presented
        """
        self.current_words = random.sample(self.word_pool, self.word_list_size)
        return self.current_words
    
    def get_response(self, response: List[str]) -> None:
        """
        Record the user's response.
        
        Args:
            response (List[str]): List of words recalled by the user
        """
        self.user_responses = response
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate the user's recall performance.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        correct_words = set(self.current_words) & set(self.user_responses)
        
        return {
            'correct_count': len(correct_words),
            'total_count': self.word_list_size,
            'accuracy': len(correct_words) / self.word_list_size,
            'correct_words': list(correct_words),
            'missed_words': list(set(self.current_words) - set(self.user_responses)),
            'extra_words': list(set(self.user_responses) - set(self.current_words))
        }


class PatternMemoryTask(MemoryTask):
    """Pattern sequence memory task implementation."""
    
    def __init__(self, grid_size=3, sequence_length=5, presentation_time=1.0):
        """
        Initialize pattern memory task.
        
        Args:
            grid_size (int): Size of the grid (grid_size x grid_size)
            sequence_length (int): Number of positions in the sequence
            presentation_time (float): Time to show each position
        """
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        self.presentation_time = presentation_time
        self.current_sequence = []
        self.user_sequence = []
    
    def present_stimulus(self) -> List[tuple]:
        """
        Generate and present a sequence of positions.
        
        Returns:
            List[tuple]: List of (x, y) coordinates
        """
        positions = []
        for _ in range(self.sequence_length):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            positions.append((x, y))
        
        self.current_sequence = positions
        return positions
    
    def get_response(self, sequence: List[tuple]) -> None:
        """
        Record the user's response sequence.
        
        Args:
            sequence (List[tuple]): List of positions clicked by user
        """
        self.user_sequence = sequence
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate the user's sequence recall performance.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        correct_positions = sum(1 for i in range(len(self.current_sequence))
                              if i < len(self.user_sequence) and
                              self.user_sequence[i] == self.current_sequence[i])
        
        return {
            'correct_positions': correct_positions,
            'total_positions': self.sequence_length,
            'accuracy': correct_positions / self.sequence_length,
            'sequence_complete': len(self.user_sequence) == self.sequence_length,
            'correct_sequence': self.current_sequence,
            'user_sequence': self.user_sequence
        }
