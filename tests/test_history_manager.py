import pytest
import json
import os
import tempfile
import glob
from unittest.mock import Mock, patch
from utils.history_manager import HistoryManager
from datetime import datetime

class TestHistoryManager:
    
    def setup_method(self):
        """Setup test fixtures with temporary file"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.history_manager = HistoryManager(self.temp_file.name)
        
        # Sample analysis data
        self.sample_analysis = {
            "query": "Test authentication issue",
            "enhanced_query": "Test authentication JWT token validation issue",
            "analysis_result": "The issue is with token validation logic...",
            "confidence_score": 0.85,
            "functions_analyzed": 3,
            "embedding_service": "openai",
            "llm_service": "openai",
            "context": {"repo_name": "test-repo", "priority": "high"}
        }
    
    def teardown_method(self):
        """Clean up the history file and the backup/temp files saved next to it"""
        for path in glob.glob(self.temp_file.name + "*"):
            os.unlink(path)
    
    def test_history_manager_initialization(self):
        """Test history manager initializes correctly"""
        assert str(self.history_manager.history_file) == self.temp_file.name
        assert os.path.exists(self.temp_file.name)
    
    def test_add_analysis(self):
        """Test adding analysis to history"""
        self.history_manager.add_analysis(**self.sample_analysis)
        
        # Load history and verify
        with open(self.temp_file.name, 'r') as f:
            history = json.load(f)
        
        assert "analyses" in history
        assert len(history["analyses"]) == 1
        
        analysis = history["analyses"][0]
        assert analysis["query"] == self.sample_analysis["query"]
        assert analysis["confidence_score"] == self.sample_analysis["confidence_score"]
        assert "timestamp" in analysis
        assert "id" in analysis
    
    def test_add_multiple_analyses(self):
        """Test adding multiple analyses"""
        # Add first analysis
        self.history_manager.add_analysis(**self.sample_analysis)
        
        # Add second analysis
        second_analysis = self.sample_analysis.copy()
        second_analysis["query"] = "Database connection timeout"
        second_analysis["confidence_score"] = 0.75
        self.history_manager.add_analysis(**second_analysis)
        
        with open(self.temp_file.name, 'r') as f:
            history = json.load(f)
        
        assert len(history["analyses"]) == 2
        assert history["total_analyses"] == 2
        
        # Check IDs are unique and sequential
        ids = [analysis["id"] for analysis in history["analyses"]]
        assert len(set(ids)) == 2  # All unique
        assert sorted(ids) == [1, 2]  # Sequential
    
    def test_get_statistics(self):
        """Test getting analysis statistics"""
        # Add some analyses
        self.history_manager.add_analysis(**self.sample_analysis)
        
        second_analysis = self.sample_analysis.copy()
        second_analysis["confidence_score"] = 0.95
        second_analysis["llm_service"] = "perplexity"
        self.history_manager.add_analysis(**second_analysis)
        
        stats = self.history_manager.get_statistics()
        
        assert stats["total_analyses"] == 2
        assert stats["average_confidence"] == 0.90  # (0.85 + 0.95) / 2
        assert "services_usage" in stats
        assert "openai+openai" in stats["services_usage"]
        assert "openai+perplexity" in stats["services_usage"]
        assert "created_at" in stats
        assert "last_updated" in stats
    
    def test_get_recent_analyses(self):
        """Test getting recent analyses"""
        # Add multiple analyses
        for i in range(5):
            analysis = self.sample_analysis.copy()
            analysis["query"] = f"Test query {i+1}"
            self.history_manager.add_analysis(**analysis)
        
        # Get recent analyses
        recent = self.history_manager.get_recent_analyses(3)
        
        assert len(recent) == 3
        # Should be in reverse chronological order (most recent first)
        assert recent[0]["query"] == "Test query 5"
        assert recent[1]["query"] == "Test query 4"
        assert recent[2]["query"] == "Test query 3"
    
    def test_get_recent_analyses_limit_exceeds_total(self):
        """Test getting recent analyses when limit exceeds total"""
        self.history_manager.add_analysis(**self.sample_analysis)
        
        recent = self.history_manager.get_recent_analyses(10)
        assert len(recent) == 1
    
    def test_clear_history(self):
        """Test clearing analysis history"""
        # Add some analyses
        self.history_manager.add_analysis(**self.sample_analysis)
        self.history_manager.add_analysis(**self.sample_analysis)
        
        # Verify analyses exist
        stats = self.history_manager.get_statistics()
        assert stats["total_analyses"] == 2
        
        # Clear history
        self.history_manager.clear_history()
        
        # Verify history is cleared
        stats = self.history_manager.get_statistics()
        assert stats["total_analyses"] == 0
        
        recent = self.history_manager.get_recent_analyses(10)
        assert len(recent) == 0
    
    def test_export_history(self):
        """Test exporting history to file"""
        # Add some analyses
        self.history_manager.add_analysis(**self.sample_analysis)
        
        # Export to another temp file
        export_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        export_file.close()
        
        try:
            exported_path = self.history_manager.export_history(export_file.name)
            assert exported_path == os.path.realpath(export_file.name)
            
            # Verify export file exists and has correct content
            assert os.path.exists(export_file.name)
            
            with open(export_file.name, 'r') as f:
                exported_data = json.load(f)
            
            assert "analyses" in exported_data
            assert len(exported_data["analyses"]) == 1
            assert exported_data["total_analyses"] == 1
            
        finally:
            if os.path.exists(export_file.name):
                os.unlink(export_file.name)
    
    def test_empty_history_statistics(self):
        """Test statistics for empty history"""
        stats = self.history_manager.get_statistics()
        
        assert stats["total_analyses"] == 0
        assert stats["average_confidence"] == 0
        assert stats["services_usage"] == {}
    
    def test_file_corruption_handling(self):
        """Test handling of corrupted history file"""
        # Write invalid JSON to file
        with open(self.temp_file.name, 'w') as f:
            f.write("invalid json content")
        
        # Should handle gracefully and create new history
        new_manager = HistoryManager(self.temp_file.name)
        stats = new_manager.get_statistics()
        assert stats["total_analyses"] == 0

    def test_ids_stay_unique_after_trimming(self):
        """IDs must not repeat once old entries are dropped to respect max_entries"""
        self.history_manager.max_entries = 2
        for _ in range(4):
            self.history_manager.add_analysis(**self.sample_analysis)
        
        ids = [a["id"] for a in self.history_manager.get_all_analyses()]
        assert ids == [3, 4]
