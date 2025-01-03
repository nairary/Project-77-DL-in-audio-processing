import asyncio
from typing import Dict
from settings import MAX_PROCESSES

process_semaphore = asyncio.Semaphore(MAX_PROCESSES - 1) 

active_processes: Dict[str, str] = {}

async def start_process(process_id: str, process_type: str):
    if process_id in active_processes:
        raise ValueError(f"Process {process_id} is already running")
    
    await process_semaphore.acquire()
    active_processes[process_id] = process_type

async def end_process(process_id: str):
    if process_id not in active_processes:
        raise ValueError(f"Process {process_id} is not running")
    
    del active_processes[process_id]
    process_semaphore.release()

def get_active_processes():
    return active_processes