import sys
import time
import argparse
import subprocess
import statistics
from pathlib import Path

# 环境路径配置
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from stream_receiver.tcp_receiver import EEGReceiver, StreamDispatcher
from core_algorithm.adaptive_window_manager import AdaptiveWindowManager

DEFAULT_STFNET_CKPT = str(
    Path(project_root)
    / "stfnet_module"
    / "best_overall.pth"
)


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return float(ordered[idx])


def handle_reconstructed_data(chunk, chunk_id, latency_ms=None):
    """最终融合结果回调（控制台实时展示）"""
    if latency_ms is None:
        print(f"📦 [Output] Chunk #{chunk_id:03d} | Shape: {chunk.shape} | 重构完成")
    else:
        print(
            f"📦 [Output] Chunk #{chunk_id:03d} | Shape: {chunk.shape} | "
            f"重构完成 | 窗口时延={latency_ms:.1f} ms"
        )


def _resolve_stfnet_ckpt(ckpt_arg: str, project_root_dir: str) -> str:
    """
    优先使用用户显式路径；若不存在则按 run_id 在项目内自动回退搜索。
    返回最终可用 checkpoint 路径，不可用则抛出 FileNotFoundError。
    """
    p = Path(ckpt_arg)
    if p.exists():
        return str(p.resolve())

    project_root_path = Path(project_root_dir)
    run_id = p.parent.name  # e.g. run_20260423_025706_008

    # 1) 先在项目内按 run_id 精确回退
    candidates = list(
        project_root_path.glob(f"**/{run_id}/best_overall.pth")
    ) + list(project_root_path.glob(f"**/{run_id}/**/best.pth"))

    # 2) 再全局回退到任意 best_overall/best
    if not candidates:
        candidates = list(project_root_path.glob("**/best_overall.pth")) + list(
            project_root_path.glob("**/best.pth")
        )

    if candidates:
        # 取最新修改时间
        best = max(candidates, key=lambda x: x.stat().st_mtime)
        print(
            f"[STFNet] 指定权重不存在，已自动回退到可用权重: {best}"
        )
        return str(best.resolve())

    raise FileNotFoundError(
        "No STFNet checkpoint found. "
        f"Requested: {ckpt_arg}. "
        "Searched under project for best_overall.pth / best.pth but found none."
    )


def _resolve_fusion_weight_ckpt(ckpt_arg: str, project_root_dir: str) -> str:
    p = Path(ckpt_arg)
    if p.exists():
        return str(p.resolve())

    project_root_path = Path(project_root_dir)
    candidates = list(project_root_path.glob("results/**/*.pth"))
    if candidates:
        best = max(candidates, key=lambda x: x.stat().st_mtime)
        print(f"[FusionWeight] 指定路径不存在，已自动回退到: {best}")
        return str(best.resolve())
    raise FileNotFoundError(
        f"Fusion weight checkpoint not found: {ckpt_arg}. No .pth found under results/."
    )


def main():
    parser = argparse.ArgumentParser(description="自适应重叠率滑动窗口流式测试程序")
    parser.add_argument(
        "-n",
        "--overlap_n",
        type=int,
        default=2,
        help="重叠度参数 N (步长 S = L/N)。建议值: 1, 2, 4, 5, 10",
    )
    parser.add_argument(
        "-l", "--window_len", type=int, default=500, help="窗口长度 L (默认 500)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="接收 IP")
    parser.add_argument("--port", type=int, default=50007, help="接收端口")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="采样率(Hz)")
    parser.add_argument(
        "--stfnet_ckpt",
        type=str,
        default=DEFAULT_STFNET_CKPT,
        help="STFNet 参数文件路径（默认使用指定 run 的 best_overall.pth）",
    )
    parser.add_argument("--stfnet_device", type=str, default="cuda:0", help="STFNet 推理设备")
    parser.add_argument(
        "--fusion_weight_ckpt",
        type=str,
        default="",
        help="双尺度权重网络参数路径（为空则禁用学习权重，使用均值融合）",
    )
    parser.add_argument(
        "--fusion_weight_device",
        type=str,
        default="cuda:0",
        help="双尺度权重网络推理设备",
    )
    parser.add_argument(
        "--no-stfnet",
        action="store_true",
        help="关闭 STFNet 窗口级预处理（默认开启）",
    )
    parser.add_argument(
        "--no-auto-sender",
        action="store_true",
        help="不自动启动发送端（默认会自动启动）",
    )
    args = parser.parse_args()

    stfnet_ckpt = None
    fusion_weight_ckpt = None
    if not args.no_stfnet:
        stfnet_ckpt = _resolve_stfnet_ckpt(args.stfnet_ckpt, project_root)
    if args.fusion_weight_ckpt.strip():
        fusion_weight_ckpt = _resolve_fusion_weight_ckpt(
            args.fusion_weight_ckpt.strip(), project_root
        )
    window_manager = AdaptiveWindowManager(
        window_size=args.window_len,
        N=args.overlap_n,
        n_channels=19,
        stfnet_checkpoint=stfnet_ckpt,
        stfnet_device=args.stfnet_device,
        stfnet_input_len=500,
        fusion_weight_checkpoint=fusion_weight_ckpt,
        fusion_weight_device=args.fusion_weight_device,
    )
    print("🚀 启动流式测试...")
    print(f"配置: L={window_manager.L}, N={window_manager.N}, S={window_manager.S}")
    if window_manager.requested_N != window_manager.N:
        print(
            f"[Config] 输入 N={window_manager.requested_N} 已自动修正为 N={window_manager.N}"
        )
    if window_manager.L % window_manager.N != 0:
        print(
            f"[Config] 检测到非整除: L%N={window_manager.L % window_manager.N}，"
            f"采用 S=floor(L/N)={window_manager.S}，最大重叠覆盖数约为 {window_manager.max_overlap_count}"
        )
        print("[Config] 尾部不足一个完整窗口的数据将自动丢弃。")
    if args.no_stfnet:
        print("[Config] STFNet: disabled")
    else:
        print(f"[Config] STFNet: enabled, ckpt={stfnet_ckpt}, device={args.stfnet_device}")
    if fusion_weight_ckpt is None:
        print("[Config] FusionWeight: disabled (均值融合)")
    else:
        print(
            f"[Config] FusionWeight: enabled, ckpt={fusion_weight_ckpt}, device={args.fusion_weight_device}"
        )

    receiver = EEGReceiver(host=args.host, port=args.port)
    window_manager.on_reconstructed_chunk = handle_reconstructed_data
    dispatcher = StreamDispatcher(
        receiver, window_manager, sample_rate=args.sample_rate
    )

    receiver.start()
    if not receiver.ready_event.wait(timeout=5.0):
        raise TimeoutError("接收端启动超时，5秒内未进入监听状态。")
    if receiver.start_error is not None:
        raise RuntimeError(f"接收端启动失败: {receiver.start_error}")
    dispatcher.start()

    sender_process = None
    if not args.no_auto_sender:
        sender_script = Path(project_root) / "legacy_contamination" / "stream_sender.py"
        if not sender_script.exists():
            raise FileNotFoundError(f"找不到发送端脚本: {sender_script}")
        print(f"[Test] 自动启动发送端: {sender_script}")
        sender_process = subprocess.Popen(
            [sys.executable, str(sender_script), "--host", args.host, "--port", str(args.port)]
        )
    else:
        print("[Test] 已禁用自动发送端。请手动先启动 stream_sender.py")

    try:
        last_report = time.time()
        last_packet_count = 0
        last_chunk_count = 0
        last_latency_idx = 0
        last_stf_idx = 0
        last_wf_idx = 0

        while True:
            time.sleep(1)
            if not receiver.is_alive():
                err = (
                    f"接收端线程已退出: {receiver.start_error}"
                    if receiver.start_error
                    else "接收端线程已退出"
                )
                raise RuntimeError(err)

            now = time.time()
            if now - last_report >= 2.0:
                packet_count = len(receiver.delay_log)
                chunk_count = window_manager.reconstructed_count

                if packet_count == 0:
                    print("[Status] 暂未收到任何数据包，正在等待发送端连接/发送...")
                else:
                    d_packet = packet_count - last_packet_count
                    d_chunk = chunk_count - last_chunk_count
                    all_latencies = window_manager.output_latency_ms
                    new_latencies = all_latencies[last_latency_idx:]
                    latest_latency = (
                        window_manager.last_output_latency_ms
                        if window_manager.last_output_latency_ms is not None
                        else 0.0
                    )

                    if new_latencies:
                        recent_avg = statistics.fmean(new_latencies)
                        recent_p95 = _p95(new_latencies)
                    else:
                        recent_avg = 0.0
                        recent_p95 = 0.0
                    global_avg = statistics.fmean(all_latencies) if all_latencies else 0.0
                    stf_all = window_manager.stfnet_infer_ms
                    stf_new = stf_all[last_stf_idx:]
                    stf_recent_avg = statistics.fmean(stf_new) if stf_new else 0.0
                    stf_latest = (
                        window_manager.last_stfnet_infer_ms
                        if window_manager.last_stfnet_infer_ms is not None
                        else 0.0
                    )
                    wf_all = window_manager.weight_infer_ms
                    wf_recent = wf_all[last_wf_idx:] if wf_all else []
                    wf_recent_avg = statistics.fmean(wf_recent) if wf_recent else 0.0
                    wf_latest = (
                        window_manager.last_weight_infer_ms
                        if window_manager.last_weight_infer_ms is not None
                        else 0.0
                    )

                    print(
                        f"[Status] 累计包数={packet_count}, 累计输出块数={chunk_count}, "
                        f"近2秒新增包={d_packet}, 新增输出块={d_chunk}, "
                        f"近2秒窗口时延均值={recent_avg:.1f}ms, 近2秒P95={recent_p95:.1f}ms, "
                        f"最新窗口时延={latest_latency:.1f}ms, 全局均值={global_avg:.1f}ms, "
                        f"近2秒STFNet均值={stf_recent_avg:.1f}ms, 最新STFNet={stf_latest:.1f}ms, "
                        f"近2秒权重网络均值={wf_recent_avg:.1f}ms, 最新权重网络={wf_latest:.1f}ms"
                    )
                    last_latency_idx = len(all_latencies)
                    last_stf_idx = len(stf_all)
                    last_wf_idx = len(wf_all)

                last_packet_count = packet_count
                last_chunk_count = chunk_count
                last_report = now
    except KeyboardInterrupt:
        print("\n检测到中断，正在关闭...")
    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        dispatcher.stop()
        receiver.stop()
        dispatcher.join(timeout=3)
        receiver.join(timeout=3)

        if sender_process is not None:
            try:
                sender_process.terminate()
                sender_process.wait(timeout=3)
            except Exception:
                sender_process.kill()

        print("Done.")


if __name__ == "__main__":
    main()
