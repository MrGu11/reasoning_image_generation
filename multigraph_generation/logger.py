from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict
import logging
import json

# ---------------------------
# JSON 日志格式化器
# ---------------------------
class JsonFormatter(logging.Formatter):
    """JSON格式日志，包含时间、级别、日志器、消息及异常信息"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 补充调试信息
        try:
            log_data.update({
                "path": record.pathname,
                "func": record.funcName,
                "line": record.lineno,
            })
        except AttributeError:
            pass
        # 补充异常信息
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


# ---------------------------
# 日志配置工具函数
# ---------------------------
def setup_logger(
    logger_name: str = "geometry_generator",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_log_file: Optional[str] = None
) -> logging.Logger:
    """配置日志记录器，避免重复添加处理器"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False  # 禁止父日志器重复输出

    # 添加工控台处理器（仅当不存在时）
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(stream_handler)

    # 添加普通文件处理器（仅当不存在时）
    if log_file:
        existing_file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_file
        ]
        if not existing_file_handlers:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-5s %(name)s:%(lineno)d - %(message)s")
            )
            logger.addHandler(file_handler)

    # 添加JSON文件处理器（仅当不存在时）
    if json_log_file:
        existing_json_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == json_log_file
        ]
        if not existing_json_handlers:
            json_handler = logging.FileHandler(json_log_file, encoding="utf-8")
            json_handler.setFormatter(JsonFormatter())
            logger.addHandler(json_handler)

    return logger