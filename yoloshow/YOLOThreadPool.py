from PySide6.QtCore import QMutex, QThread


class YOLOThreadPool:
    MAX_THREADS = 3

    def __init__(self):
        self.threads_pool = {}  # Dictionary to store objects
        self.thread_order = []  # A list recording the order in which threads were added
        self._mutex = QMutex()  # thread lock

    def _remove_oldest_thread(self):
        """Remove the earliest added thread object"""
        oldest_name = self.thread_order.pop(0)
        self.delete(oldest_name)

    def set(self, name, thread_obj):
        """Set or update the thread object"""
        if not isinstance(thread_obj, QThread):
            raise ValueError("The object must be an instance of QThread.")

        self._mutex.lock()  # Lock to ensure thread safety
        # If already exists, stop and delete
        if name in self.threads_pool:
            self.thread_order.remove(name)
            self.delete(name)

        # Check if maximum limit is exceeded
        if len(self.threads_pool) >= self.MAX_THREADS:
            self._remove_oldest_thread()

        # Add new thread
        self.threads_pool[name] = thread_obj
        self.thread_order.append(name)
        self._mutex.unlock()  # Unlock

    def get(self, name):
        """Get thread object by name"""
        return self.threads_pool.get(name)

    def start_thread(self, name):
        """Start the specified thread object"""
        thread_obj = self.get(name)
        if thread_obj and not thread_obj.isRunning():
            thread_obj.start()

    def stop_thread(self, name):
        """Stop the specified thread object"""
        thread_obj = self.get(name)
        if thread_obj and thread_obj.isRunning():
            thread_obj.quit()
            thread_obj.wait()

    def delete(self, name):
        """Delete the thread object with the specified name"""
        thread = self.threads_pool.get(name)
        if thread and isinstance(thread, QThread):
            # Make sure the thread is stopped
            if thread.isRunning():
                thread.quit()  # Request thread to exit
                thread.wait()  # Wait for the thread to exit completely
            # Delete thread object
            del self.threads_pool[name]


    def exists(self, name):
        """Check if object exists"""
        return name in self.threads_pool
