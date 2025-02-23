import logging

class Logger:
    def __init__(self, name="app_logger", log_file="create_product_app.log", level=logging.INFO):
        """
        Initializes the logger.

        :param name: Name of the logger.
        :param log_file: Log file name.
        :param level: Logging level (default is INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers if the logger already exists
        if not self.logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

            # Add handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """Returns the configured logger."""
        return self.logger
