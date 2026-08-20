class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ModelError(AppError):
    pass


class AudioError(AppError):
    pass


class NotFoundError(AppError):
    def __init__(self, code: str = "not_found", message: str = "resource not found"):
        super().__init__(code, message, 404)


def error_shape(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}