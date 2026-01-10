import sys
import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: {file_name} at line number: {line_number} | Error message: {error}"
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        a = 1 / 0
    except Exception as e:
        logging.exception("An exception occurred")
        sys.exit(1)
    raise CustomException(e, sys)