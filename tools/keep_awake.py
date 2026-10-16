"""阻止系统休眠，直到本进程退出。

用 Windows 的 SetThreadExecutionState API 声明"系统正在使用中"，
不修改电源计划、不需要管理员权限，进程结束即自动失效。
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def main() -> int:
    kernel32 = ctypes.windll.kernel32
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    if kernel32.SetThreadExecutionState(flags) == 0:
        print("SetThreadExecutionState 调用失败", file=sys.stderr)
        return 1
    print("已阻止休眠（进程结束即恢复）", flush=True)
    try:
        while True:
            time.sleep(30)
            kernel32.SetThreadExecutionState(flags)   # 周期性重申
    except KeyboardInterrupt:
        pass
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
