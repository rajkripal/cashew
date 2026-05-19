#!/usr/bin/env python3
"""
Cashew Database Backup Module
Handles automated and manual database backups with configurable retention
"""

import os
import sqlite3
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging

from .config import config, get_db_path
from . import db as cdb

logger = logging.getLogger("cashew.backup")


def create_backup(db_path: Optional[str] = None, backup_dir: Optional[str] = None, 
                  timestamp: Optional[datetime] = None) -> Optional[str]:
    """
    Create a SQLite backup of the database with timestamp.
    
    Args:
        db_path: Path to database file (defaults to config.db_path)
        backup_dir: Directory for backups (defaults to config.backup_dir)  
        timestamp: Timestamp for backup filename (defaults to now)
        
    Returns:
        Path to created backup file, or None if backup failed
    """
    if db_path is None:
        db_path = get_db_path()
    if backup_dir is None:
        backup_dir = config.backup_dir
    if timestamp is None:
        timestamp = datetime.now()
    
    # Ensure paths exist
    db_path = Path(db_path)
    backup_dir = Path(backup_dir)
    
    if not db_path.exists():
        logger.warning(f"Database file not found: {db_path}")
        return None
        
    if not db_path.stat().st_size > 0:
        logger.warning(f"Database file is empty: {db_path}")
        return None
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename with ISO timestamp
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    backup_filename = f"graph.db.{timestamp_str}"
    backup_path = backup_dir / backup_filename
    
    try:
        # Use SQLite's VACUUM INTO for atomic backup (preferred over .backup command)
        # This creates a clean, compressed backup without WAL/journal files.
        # Route through core.db — the shared chokepoint — instead of calling
        # sqlite3.connect directly, so every DB access in the codebase flows
        # through one place.
        source_conn = cdb.connect(str(db_path))
        try:
            # Use VACUUM INTO for clean backup (SQLite 3.27+)
            source_conn.execute(f"VACUUM INTO '{backup_path}'")
        finally:
            source_conn.close()
            
        logger.info(f"Created backup: {backup_path} ({_human_readable_size(backup_path)})")
        return str(backup_path)
        
    except sqlite3.Error as e:
        logger.error(f"SQLite backup error: {e}")
        # Fallback to file copy if VACUUM INTO fails
        try:
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup (fallback): {backup_path} ({_human_readable_size(backup_path)})")
            return str(backup_path)
        except (OSError, IOError) as copy_e:
            logger.error(f"Backup failed (copy fallback): {copy_e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected backup error: {e}")
        return None


def cleanup_old_backups(backup_dir: Optional[str] = None, 
                       retention_hours: Optional[int] = None) -> List[str]:
    """
    Remove backup files older than the retention period.
    
    Args:
        backup_dir: Directory containing backups (defaults to config.backup_dir)
        retention_hours: Hours to retain backups (defaults to config value)
        
    Returns:
        List of deleted backup file paths
    """
    if backup_dir is None:
        backup_dir = config.backup_dir
    if retention_hours is None:
        retention_hours = parse_retention_period(
            getattr(config, 'backup_retention_period', '24h')
        )
    
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return []
    
    cutoff_time = datetime.now() - timedelta(hours=retention_hours)
    deleted_files = []
    
    # Find backup files (pattern: graph.db.YYYY-MM-DDTHH-MM-SS)
    for backup_file in backup_dir.glob("graph.db.*"):
        try:
            # Extract timestamp from filename
            timestamp_part = backup_file.name[9:]  # Remove "graph.db." prefix
            if _is_timestamp_format(timestamp_part):
                file_time = datetime.fromisoformat(timestamp_part.replace('-', ':'))
                
                if file_time < cutoff_time:
                    backup_file.unlink()
                    deleted_files.append(str(backup_file))
                    logger.info(f"Deleted old backup: {backup_file.name}")
                    
        except (ValueError, OSError) as e:
            logger.warning(f"Could not process backup file {backup_file}: {e}")
            continue
    
    if deleted_files:
        logger.info(f"Cleaned up {len(deleted_files)} old backups (retention: {retention_hours}h)")
    
    return deleted_files


def parse_retention_period(period_str: str) -> int:
    """
    Parse retention period string to hours.
    
    Args:
        period_str: Period like "24h", "2d", "1w"
        
    Returns:
        Number of hours
    """
    period_str = period_str.lower().strip()
    
    if period_str.endswith('h'):
        return int(period_str[:-1])
    elif period_str.endswith('d'):
        return int(period_str[:-1]) * 24
    elif period_str.endswith('w'):
        return int(period_str[:-1]) * 24 * 7
    else:
        # Assume hours if no unit
        return int(period_str)


def parse_backup_interval(interval_str: str) -> int:
    """
    Parse backup interval string to hours.
    
    Args:
        interval_str: Interval like "6h", "12h", "1d"
        
    Returns:
        Number of hours
    """
    return parse_retention_period(interval_str)  # Same parsing logic


def get_backup_stats(backup_dir: Optional[str] = None) -> dict:
    """
    Get statistics about existing backups.
    
    Args:
        backup_dir: Directory containing backups
        
    Returns:
        Dictionary with backup statistics
    """
    if backup_dir is None:
        backup_dir = config.backup_dir
        
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return {
            'count': 0,
            'total_size': 0,
            'oldest': None,
            'newest': None,
            'files': []
        }
    
    backup_files = []
    total_size = 0
    
    for backup_file in backup_dir.glob("graph.db.*"):
        if backup_file.is_file():
            try:
                size = backup_file.stat().st_size
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                backup_files.append({
                    'path': str(backup_file),
                    'size': size,
                    'created': mtime
                })
                total_size += size
                
            except OSError:
                continue
    
    backup_files.sort(key=lambda x: x['created'])
    
    return {
        'count': len(backup_files),
        'total_size': total_size,
        'oldest': backup_files[0]['created'] if backup_files else None,
        'newest': backup_files[-1]['created'] if backup_files else None,
        'files': backup_files
    }


def auto_backup_if_needed(db_path: Optional[str] = None) -> bool:
    """
    Create automatic backup before DB operations if configured to do so.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if backup was created or not needed, False if backup failed
    """
    if not getattr(config, 'auto_backup_enabled', True):
        return True
        
    # Check if we should create a backup based on interval
    backup_interval_hours = getattr(config, 'backup_interval_hours', 6)
    backup_dir = config.backup_dir
    
    if _should_create_backup(backup_dir, backup_interval_hours):
        backup_path = create_backup(db_path)
        if backup_path:
            # Clean up old backups after successful backup
            retention_hours = parse_retention_period(
                getattr(config, 'backup_retention_period', '24h')
            )
            cleanup_old_backups(backup_dir, retention_hours)
            return True
        else:
            logger.warning("Auto-backup failed before database operation")
            return False
    
    return True


def _should_create_backup(backup_dir: str, interval_hours: int) -> bool:
    """Check if a new backup should be created based on interval."""
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return True
        
    # Find most recent backup
    latest_backup = None
    latest_time = None
    
    for backup_file in backup_dir.glob("graph.db.*"):
        try:
            mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if latest_time is None or mtime > latest_time:
                latest_time = mtime
                latest_backup = backup_file
        except OSError:
            continue
    
    if latest_backup is None:
        return True
        
    # Check if interval has elapsed
    time_since_backup = datetime.now() - latest_time
    return time_since_backup.total_seconds() >= (interval_hours * 3600)


def _is_timestamp_format(timestamp_str: str) -> bool:
    """Check if string matches our timestamp format YYYY-MM-DDTHH-MM-SS."""
    try:
        datetime.fromisoformat(timestamp_str.replace('-', ':'))
        return True
    except ValueError:
        return False


def _human_readable_size(file_path: Path) -> str:
    """Get human readable file size."""
    try:
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    except OSError:
        return "unknown"