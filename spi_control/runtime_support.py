"""Process lifetime helpers; importing this module never opens hardware."""
from __future__ import annotations

import os
from pathlib import Path


class SingleInstance:
    """Keep the lock inode in place, release automatically even after a crash."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="ascii")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if self.path.stat().st_size == 0:
                    stream.write(" ")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            stream.close()
            raise RuntimeError("触屏程序已运行，请使用现有窗口；不要重复启动") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        self.file = stream
        return self

    def __exit__(self, *_exc):
        if self.file is not None:
            self.file.close()
            self.file = None
