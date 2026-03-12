import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import subprocess

def print_banner():
    banner = """
    ========================================================
      🌊 CADT 跨模态海空救援系统 - 全局启动引擎 (Launcher)
    ========================================================
    [1] 正在拉起 FastAPI 指挥中心大屏...
    [2] 正在唤醒 Ascend NPU 边缘守护进程...
    --------------------------------------------------------
    * 按下 [Ctrl + C] 即可安全关闭整个系统 *
    """
    print(banner)

def main():
    print_banner()
    
    # 获取当前 Python 解释器的路径 (确保使用同一个 conda 虚拟环境)
    python_exec = sys.executable
    
    # 获取当前脚本所在绝对路径，确保相对路径不出错
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, "dashboard", "server.py")
    daemon_script = os.path.join(base_dir, "edge_daemon.py")

    # 用于保存子进程的列表
    processes = []

    try:
        # 1. 以后台子进程模式启动大屏后端
        print("[SYSTEM] Booting Dashboard Server...")
        p_server = subprocess.Popen([python_exec, server_script])
        processes.append(p_server)
        
        # 给服务器 2 秒钟的启动时间，确保 WebSocket 端口(8000)准备就绪
        time.sleep(2)
        
        # 2. 以后台子进程模式启动 AI 核心守护进程
        print("[SYSTEM] Booting Edge Daemon...")
        p_daemon = subprocess.Popen([python_exec, daemon_script])
        processes.append(p_daemon)

        # 3. 主进程挂起，等待子进程运行
        # 这一步会让命令行停在这里，你可以看到两边汇总的日志输出
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        # 捕获 Ctrl+C，执行优雅降级与资源回收
        print("\n\n[SYSTEM] 接收到中断信号 (Ctrl+C)，正在安全关闭所有子系统...")
        for p in processes:
            p.terminate() # 发送终止信号
            p.wait()      # 等待进程彻底清理内存
        print("[SUCCESS] 系统已完全关闭，端口已释放。再见！")

if __name__ == "__main__":
    main()