"""Tests for the memory tasks module."""

import pytest

from src.tasks.memory_tasks import PatternMemoryTask, WordListTask


@pytest.fixture
def word_task():
    """Create a WordListTask instance for testing."""
    return WordListTask(word_list_size=5, presentation_time=1.0)


@pytest.fixture
def pattern_task():
    """Create a PatternMemoryTask instance for testing."""
    return PatternMemoryTask(grid_size=3, sequence_length=4)


def test_word_list_task_initialization(word_task):
    """Test proper initialization of word list task."""
    assert word_task.word_list_size == 5
    assert word_task.presentation_time == 1.0
    assert len(word_task.word_pool) > 0


def test_word_list_presentation(word_task):
    """Test word list presentation."""
    words = word_task.present_stimulus()
    assert len(words) == word_task.word_list_size
    assert all(isinstance(word, str) for word in words)
    assert all(word in word_task.word_pool for word in words)


def test_word_list_evaluation(word_task):
    """Test word list recall evaluation."""
    # Present words
    presented_words = word_task.present_stimulus()
    
    # Simulate perfect recall
    word_task.get_response(presented_words.copy())
    result = word_task.evaluate_performance()
    
    assert result['accuracy'] == 1.0
    assert result['correct_count'] == word_task.word_list_size
    assert len(result['missed_words']) == 0
    
    # Simulate partial recall
    partial_recall = presented_words[:3]
    word_task.get_response(partial_recall)
    result = word_task.evaluate_performance()
    
    assert result['accuracy'] == 0.6  # 3/5 = 0.6
    assert result['correct_count'] == 3
    assert len(result['missed_words']) == 2


def test_pattern_task_initialization(pattern_task):
    """Test proper initialization of pattern task."""
    assert pattern_task.grid_size == 3
    assert pattern_task.sequence_length == 4


def test_pattern_sequence_generation(pattern_task):
    """Test pattern sequence generation."""
    sequence = pattern_task.present_stimulus()
    
    assert len(sequence) == pattern_task.sequence_length
    assert all(isinstance(pos, tuple) for pos in sequence)
    assert all(0 <= x < pattern_task.grid_size and 
              0 <= y < pattern_task.grid_size 
              for x, y in sequence)


def test_pattern_evaluation(pattern_task):
    """Test pattern recall evaluation."""
    # Present sequence
    sequence = pattern_task.present_stimulus()
    
    # Simulate perfect recall
    pattern_task.get_response(sequence.copy())
    result = pattern_task.evaluate_performance()
    
    assert result['accuracy'] == 1.0
    assert result['correct_positions'] == pattern_task.sequence_length
    assert result['sequence_complete']
    
    # Simulate partial recall
    partial_sequence = sequence[:2]
    pattern_task.get_response(partial_sequence)
    result = pattern_task.evaluate_performance()
    
    assert result['accuracy'] == 0.5  # 2/4 = 0.5
    assert result['correct_positions'] == 2
    assert not result['sequence_complete']
