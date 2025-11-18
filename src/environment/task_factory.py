"""
Task Factory
Creates manipulation tasks with configurable difficulty
"""

import logging
from typing import Dict, Type
from src.environment.base_task import BaseTask

logger = logging.getLogger(__name__)


class TaskFactory:
    """
    Factory for creating manipulation tasks
    Provides unified interface for task instantiation
    """

    # Registry of available tasks
    _TASKS: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register_task(cls, name: str, task_class: Type[BaseTask]):
        """
        Register a new task implementation

        Args:
            name: Task identifier
            task_class: Task class implementing BaseTask
        """
        cls._TASKS[name] = task_class
        logger.info(f"Registered task: {name}")

    @classmethod
    def list_tasks(cls) -> list:
        """List all registered task names"""
        return list(cls._TASKS.keys())

    @classmethod
    def create_task(cls, task_name: str, difficulty: str = "medium", **kwargs) -> BaseTask:
        """
        Create a task instance

        Args:
            task_name: Name of the task to create
            difficulty: Task difficulty (easy, medium, hard)
            **kwargs: Task-specific parameters

        Returns:
            Task instance

        Raises:
            ValueError: If task_name is not registered
        """
        if task_name not in cls._TASKS:
            available = ", ".join(cls.list_tasks())
            raise ValueError(
                f"Task '{task_name}' not found. "
                f"Available tasks: {available}"
            )

        task_class = cls._TASKS[task_name]
        logger.info(f"Creating task: {task_name} (difficulty: {difficulty})")

        try:
            task = task_class(difficulty=difficulty, **kwargs)
            logger.info(f"Successfully created {task_name}")
            return task
        except Exception as e:
            logger.error(f"Failed to create {task_name}: {e}")
            raise


# Auto-register tasks
def auto_register():
    """Automatically register all available task implementations"""
    import importlib
    from pathlib import Path

    tasks_dir = Path(__file__).parent / "tasks"
    if not tasks_dir.exists():
        return

    for task_file in tasks_dir.glob("*_task.py"):
        module_name = task_file.stem
        try:
            module = importlib.import_module(f"src.environment.tasks.{module_name}")

            # Look for classes that inherit from BaseTask
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTask)
                    and attr is not BaseTask
                ):
                    # Extract task name
                    task_name = attr_name.replace("Task", "").lower()
                    if hasattr(attr, "TASK_NAME"):
                        task_name = attr.TASK_NAME

                    TaskFactory.register_task(task_name, attr)

        except Exception as e:
            logger.warning(f"Failed to register tasks from {module_name}: {e}")


# Auto-register on import
auto_register()
