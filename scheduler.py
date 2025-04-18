"""
Background task scheduler for OMRChecker API
"""
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

from utils.file_handling import clean_all_folders, clean_old_files
from src.logger import logger

def setup_scheduler(app_config):
    """Set up background task scheduler for maintenance tasks"""
    
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        lambda: clean_old_files(app_config),
        'interval', 
        minutes=15,
        id='clean_old_files_job',
        name='Clean files older than 1 hour'
    )
    
    scheduler.add_job(
        lambda: clean_all_folders(app_config),
        'cron', 
        hour=0, 
        minute=0,
        id='clean_folders_job',
        name='Daily complete folder cleanup'
    )
    
    scheduler.start()
    logger.info("Started background scheduler with file cleanup jobs")
    
    atexit.register(lambda: scheduler.shutdown())
    
    return scheduler 