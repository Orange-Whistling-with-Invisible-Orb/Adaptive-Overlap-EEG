import sys
import time
import argparse
import subprocess
from pathlib import Path

# 确保能导入项目内其他模块
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from stream_receiver.tcp_receiver import EEGReceiver, StreamDispatcher
from core_algorithm.adaptive_window_manager import AdaptiveWindowManager

def handle_reconstructed_data(chunk, chunk_id):
    """最终融合结果回调"""
    print(f"📦 [Output] Chunk #{chunk_id:03d} | Shape: {chunk.shape} | 重构输出完毕")

def main():
    parser = argparse.ArgumentParser(description="自适应重叠率滑动窗口【一键测试程序】")
    # 默认 N 设为 1，即无重叠
    parser.add_argument("-n", "--overlap_n", type=int, default=1, 
                        help="重叠度 N (默认 1=无重叠)。rho = (N-1)/N")
    parser.add_argument("-l", "--window_len", type=int, default=500, 
                        help="窗口长度 L (默认 500)")
    args = parser.parse_args()

    print(f"🚀 [System] 启动一键测试流...")
    print(f"🛠️ [Config] L={args.window_len}, N={args.overlap_n} (重叠率: {((args.overlap_n-1)/args.overlap_n)*100:.1f}%)")

    # --- 步骤 1: 启动发送端进程 ---
    sender_path = project_root / "legacy_contamination" / "stream_sender.py"
    print(f"📡 [Sender] 正在后台启动发送端: {sender_path}")
    
    # 使用 subprocess 启动发送端，不占用当前终端
    sender_proc = subprocess.Popen([sys.executable, str(sender_path)], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.PIPE)

    # --- 步骤 2: 初始化接收与算法组件 ---
    receiver = EEGReceiver(host="127.0.0.1", port=50007)
    window_manager = AdaptiveWindowManager(
        window_size=args.window_len, 
        N=args.overlap_n, 
        n_channels=19
    )
    window_manager.on_reconstructed_chunk = handle_reconstructed_data
    
    dispatcher = StreamDispatcher(receiver, window_manager)

    # --- 步骤 3: 运行系统 ---
    receiver.start()
    time.sleep(1) # 等待接收器端口开启
    dispatcher.start()

    print("\n🌟 [Running] 系统已全线启动。按下 Ctrl+C 停止测试。")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 [Shutdown] 正在停止所有服务...")
    finally:
        # 优雅关闭
        dispatcher.stop()
        receiver.stop()
        dispatcher.join()
        receiver.join()
        
        # 强制结束发送端子进程
        sender_proc.terminate()
        print("✅ [Shutdown] 测试结束，子进程已清理。")

if __name__ == "__main__":
    main()