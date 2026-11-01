import json
import logging
import os
import shutil
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from config.settings import MAX_HISTORY_ENTRIES, MAX_HISTORY_FILE_SIZE_MB, HISTORY_BACKUP_COUNT

class HistoryManager:
    def __init__(self, history_file: str = "analysis_history.json"):
        self.history_file = Path(history_file)
        self.max_entries = MAX_HISTORY_ENTRIES
        self.max_file_size_mb = MAX_HISTORY_FILE_SIZE_MB
        self.backup_count = HISTORY_BACKUP_COUNT
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self._ensure_history_file()
    
    def _get_file_size_mb(self) -> float:
        """Get file size in MB"""
        try:
            return self.history_file.stat().st_size / (1024 * 1024) if self.history_file.exists() else 0
        except Exception:
            return 0
    
    def _ensure_history_file(self):
        with self.lock:
            if not self.history_file.exists():
                self._create_empty_history()
            elif self._get_file_size_mb() > self.max_file_size_mb:
                self._rotate_history_file()
    
    def _rotate_history_file(self):
        """Rotate history file when it gets too large"""
        try:
            # Create backup
            backup_file = self.history_file.with_suffix(f'.json.backup.{int(datetime.now().timestamp())}')
            shutil.copy2(self.history_file, backup_file)
            
            # Load current history and keep only recent entries
            history = self._load_history()
            recent_entries = history["analyses"][-self.max_entries//2:]  # Keep half
            
            history["analyses"] = recent_entries
            history["total_analyses"] = len(recent_entries)
            history["rotated_at"] = datetime.now().isoformat()
            
            self._save_history(history)
            self.logger.info("History file rotated due to size limit")
            
            # Clean old backup files
            self._cleanup_old_backups()
        except Exception as e:
            self.logger.error(f"Failed to rotate history file: {e}")
    
    def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            backup_pattern = f"{self.history_file.stem}.json.backup.*"
            backup_files = list(self.history_file.parent.glob(backup_pattern))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent backups
            for old_backup in backup_files[self.backup_count:]:
                old_backup.unlink()
                self.logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {e}")
    
    def _create_empty_history(self):
        empty_history = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_analyses": 0,
            "max_entries": self.max_entries,
            "analyses": []
        }
        self._save_history(empty_history)
    
    def _load_history(self) -> Dict[str, Any]:
        with self.lock:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Validate data structure
                        if not isinstance(data, dict) or "analyses" not in data:
                            raise ValueError("Invalid history file format")
                        return data
                except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
                    self.logger.warning(f"History file corrupted (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        # Try to restore from backup
                        if self._restore_from_backup():
                            continue
                    else:
                        self.logger.error("Creating new history file due to corruption")
                        
            self._create_empty_history()
            return self._load_history()
    
    def _restore_from_backup(self) -> bool:
        """Restore history from most recent backup file"""
        try:
            backup_pattern = f"{self.history_file.stem}.json.backup.*"
            backup_files = list(self.history_file.parent.glob(backup_pattern))
            if not backup_files:
                return False
                
            # Get most recent backup
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            most_recent_backup = backup_files[0]
            
            shutil.copy2(most_recent_backup, self.history_file)
            self.logger.info(f"Restored history from backup: {most_recent_backup}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def _save_history(self, history: Dict[str, Any]):
        with self.lock:
            # Create backup before saving
            if self.history_file.exists():
                backup_file = self.history_file.with_suffix(f'.json.tmp.backup')
                try:
                    shutil.copy2(self.history_file, backup_file)
                except Exception as e:
                    self.logger.warning(f"Failed to create backup: {e}")
            
            # Write to temporary file first, then move
            temp_file = self.history_file.with_suffix('.json.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                shutil.move(temp_file, self.history_file)
            except Exception as e:
                if temp_file.exists():
                    temp_file.unlink()
                raise e
    
    def add_analysis(self,
                    query: str,
                    enhanced_query: str, 
                    analysis_result: str,
                    confidence_score: float,
                    functions_analyzed: int,
                    embedding_service: str,
                    llm_service: str,
                    context: Optional[Dict[str, Any]] = None):
        
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not isinstance(confidence_score, (int, float)) or not 0 <= confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        if functions_analyzed < 0:
            raise ValueError("Functions analyzed cannot be negative")
        
        with self.lock:
            try:
                history = self._load_history()
                
                # Check if we need to rotate due to max entries
                if len(history["analyses"]) >= self.max_entries:
                    # Remove oldest entries to make room
                    entries_to_remove = len(history["analyses"]) - self.max_entries + 1
                    history["analyses"] = history["analyses"][entries_to_remove:]
                    self.logger.info(f"Removed {entries_to_remove} old entries to maintain limit")
                
                # Based on the newest entry, so IDs stay unique after rotation trims old entries
                last_id = max((a.get("id", 0) for a in history["analyses"]), default=0)
                analysis_entry = {
                    "id": max(last_id, history.get("total_analyses", 0)) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "query": query.strip(),
                    "enhanced_query": enhanced_query.strip() if enhanced_query else query.strip(),
                    "analysis_result": analysis_result,
                    "confidence_score": confidence_score,
                    "functions_analyzed": functions_analyzed,
                    "services_used": {
                        "embedding": embedding_service,
                        "llm": llm_service
                    },
                    "context": context or {}
                }
                
                history["analyses"].append(analysis_entry)
                history["total_analyses"] = history.get("total_analyses", 0) + 1
                history["last_updated"] = datetime.now().isoformat()
                
                self._save_history(history)
                self.logger.debug(f"Added analysis entry: {analysis_entry['id']}")
            except Exception as e:
                self.logger.error(f"Failed to add analysis: {e}")
                raise
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Most recent analyses first"""
        if limit <= 0:
            return []
        with self.lock:
            history = self._load_history()
            return list(reversed(history["analyses"][-limit:]))
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        with self.lock:
            history = self._load_history()
            return history["analyses"]
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            history = self._load_history()
            analyses = history["analyses"]
            
            if not analyses:
                return {
                    "total_analyses": 0, 
                    "average_confidence": 0.0,
                    "services_usage": {},
                    "file_size_mb": round(self._get_file_size_mb(), 2)
                }
            
            total_confidence = sum(a.get("confidence_score", 0) for a in analyses)
            avg_confidence = total_confidence / len(analyses)
            
            services_used = {}
            for analysis in analyses:
                services = analysis.get("services_used", {})
                embedding_service = services.get("embedding", "unknown")
                llm_service = services.get("llm", "unknown")
                
                key = f"{embedding_service}+{llm_service}"
                services_used[key] = services_used.get(key, 0) + 1
            
            return {
                "total_analyses": len(analyses),
                "average_confidence": round(avg_confidence, 3),
                "services_usage": services_used,
                "created_at": history.get("created_at"),
                "last_updated": history.get("last_updated"),
                "file_size_mb": round(self._get_file_size_mb(), 2)
            }
    
    def export_history(self, output_file: str) -> str:
        """Write the full history to output_file and return its absolute path"""
        if not output_file:
            raise ValueError("Output file path cannot be empty")
        with self.lock:
            history = self._load_history()
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                self.logger.info(f"History exported to {output_file}")
                return str(Path(output_file).resolve())
            except Exception as e:
                self.logger.error(f"Failed to export history: {e}")
                raise
    
    def clear_history(self):
        with self.lock:
            self._create_empty_history()
            self.logger.info("History cleared")
