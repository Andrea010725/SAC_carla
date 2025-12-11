#!/usr/bin/env python3
"""
快速清理CARLA Traffic Manager端口
"""
import subprocess
import sys
import time

def find_process_on_port(port):
    """查找占用指定端口的进程"""
    try:
        # 使用lsof查找端口
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')
        return []
    except:
        return []

def kill_process(pid):
    """杀死指定进程"""
    try:
        subprocess.run(['kill', '-9', pid], timeout=5)
        return True
    except:
        return False

def main():
    print("=" * 60)
    print("CARLA Traffic Manager端口快速清理")
    print("=" * 60)

    # 常见的TM端口
    tm_ports = [3000, 3001, 3002, 8000, 8001, 8500, 8501, 8502]

    cleaned = False

    for port in tm_ports:
        pids = find_process_on_port(port)
        if pids:
            print(f"\n端口 {port} 被占用:")
            for pid in pids:
                try:
                    # 获取进程信息
                    result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'comm='],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    process_name = result.stdout.strip() if result.returncode == 0 else "unknown"

                    # 只清理非CARLA主进程的TM端口
                    if 'CarlaUE4' not in process_name:
                        print(f"  - PID {pid} ({process_name}) - 正在清理...")
                        if kill_process(pid):
                            print(f"    ✅ 已清理")
                            cleaned = True
                        else:
                            print(f"    ❌ 清理失败")
                    else:
                        print(f"  - PID {pid} ({process_name}) - 跳过（CARLA主进程）")
                except:
                    pass

    if cleaned:
        print("\n等待端口释放...")
        time.sleep(1)
        print("✅ 清理完成！")
    else:
        print("\n✅ 所有TM端口都空闲或被CARLA使用")

    print("\n现在可以运行:")
    print("  python main.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
