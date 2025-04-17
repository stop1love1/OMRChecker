"""
Background task scheduler for OMRChecker API
"""
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

from utils.file_handling import clean_all_folders
from src.logger import logger

def setup_scheduler(app_config):
    """Set up background task scheduler for maintenance tasks"""
    
    scheduler = BackgroundScheduler()
    
    # Add job to clean directories daily
    scheduler.add_job(
        lambda: clean_all_folders(app_config),
        'cron', 
        hour=0, 
        minute=0,
        id='clean_folders_job',
        name='Daily folder cleanup'
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("Started background scheduler with cleanup job")
    
    # Register shutdown handler
    atexit.register(lambda: scheduler.shutdown())
    
    return scheduler 