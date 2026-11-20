#!/usr/bin/env python3
"""
protocol_decoder.py
Bộ giải mã giao thức mạng L4/L7/Port/TCP Flags và
bộ quy tắc kiểm tra sâu Heuristic Tầng 2 (Tier-2 DPI Inspection).
Chỉ sử dụng logic tất định trên 10 đặc trưng NetFlow — KHÔNG huấn luyện thêm mô hình ML.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TypedDict


class L4ProtoInfo(TypedDict):
    name: str
    desc: str
    color: str


class L7ProtoInfo(TypedDict):
    name: str
    desc: str
    category: str


class ScenarioContext(TypedDict):
    title: str
    description: str


class DPIFinding(TypedDict):
    rule_id: str
    severity: str
    title: str
    detail: str
    snort_hint: str


class DPIFlowMetrics(TypedDict):
    total_bytes: int
    total_pkts: int
    bytes_per_in_pkt: float
    bytes_per_out_pkt: float
    avg_pkt_size: float
    in_out_byte_ratio: float
    pkts_per_sec: float
    duration_ms: int
    tcp_flags_list: List[str]
    l4_l7_consistency: str


class DPIInspectionReport(TypedDict):
    recommendation: str
    recommendation_label: str
    risk_badge: str
    summary: str
    metrics: DPIFlowMetrics
    findings: List[DPIFinding]


DEFAULT_BENIGN_FLOW: Dict[str, Any] = {
    "L4_SRC_PORT": 5353,
    "L4_DST_PORT": 80,
    "PROTOCOL": 6,
    "L7_PROTO": 7.0,
    "IN_BYTES": 1284,
    "OUT_BYTES": 2568,
    "IN_PKTS": 8,
    "OUT_PKTS": 10,
    "TCP_FLAGS": 27,
    "FLOW_DURATION_MILLISECONDS": 120,
    "Sample_Scenario": "Khởi tạo mặc định (Benign HTTP Web Session)",
    "True_Attack": "Benign",
}

L4_PROTO_MAP: Dict[int, L4ProtoInfo] = {
    1: {"name": "ICMP", "desc": "Internet Control Message (Ping/Traceroute)", "color": "#EC4899"},
    6: {"name": "TCP", "desc": "Transmission Control Protocol (Kết nối tin cậy L4)", "color": "#3B82F6"},
    17: {"name": "UDP", "desc": "User Datagram Protocol (Truyền tải phi kết nối nhanh)", "color": "#8B5CF6"},
    47: {"name": "GRE", "desc": "Generic Routing Encapsulation (VPN Tunnel)", "color": "#10B981"},
    50: {"name": "ESP", "desc": "IPsec Encapsulating Security Payload", "color": "#F59E0B"},
}

L7_PROTO_MAP: Dict[float, L7ProtoInfo] = {
    0.0: {"name": "Unknown / L4 Raw", "desc": "Lưu lượng L4 thuần túy (Chưa định danh L7)", "category": "Transport"},
    3.0: {"name": "SMTP", "desc": "Simple Mail Transfer Protocol (Thư điện tử)", "category": "Email"},
    5.0: {"name": "DNS", "desc": "Domain Name System (Phân giải tên miền)", "category": "Network Core"},
    7.0: {"name": "HTTP", "desc": "HyperText Transfer Protocol (Lưu lượng Web)", "category": "Web Service"},
    11.0: {"name": "FTP", "desc": "File Transfer Protocol (Truyền tập tin)", "category": "File Transfer"},
    12.0: {"name": "SSH", "desc": "Secure Shell Remote Access (Quản trị từ xa)", "category": "Remote"},
    19.0: {"name": "SSL / TLS", "desc": "Transport Layer Security (Web mã hóa HTTPS)", "category": "Secure Web"},
    20.0: {"name": "MySQL", "desc": "MySQL Database Connection (Truy vấn CSDL)", "category": "Database"},
}

WELL_KNOWN_PORTS: Dict[int, str] = {
    0: "ICMP / System Raw",
    20: "FTP-Data (File Data)",
    21: "FTP (Command Port)",
    22: "SSH (Secure Admin Shell)",
    23: "Telnet (Unencrypted Remote)",
    25: "SMTP (Mail Server)",
    53: "DNS (Name Server)",
    67: "DHCP Server",
    68: "DHCP Client",
    80: "HTTP (Standard Web Server)",
    110: "POP3 (Mail Client)",
    143: "IMAP (Mail Client)",
    443: "HTTPS (Encrypted Web Server)",
    3306: "MySQL Database Port",
    5353: "mDNS Multicast Discovery",
    8080: "HTTP-Proxy / Alt Web Server",
    8443: "HTTPS-Alt Web Service",
}

TCP_FLAG_BITS: List[Tuple[int, str, str]] = [
    (0x01, "FIN", "Đóng kết nối bình thường"),
    (0x02, "SYN", "Bắt tay khởi tạo kết nối (Three-Way Handshake)"),
    (0x04, "RST", "Hủy bỏ/Ngắt kết nối khẩn cấp"),
    (0x08, "PSH", "Đẩy dữ liệu lập tức cho ứng dụng"),
    (0x10, "ACK", "Xác nhận đã nhận gói tin"),
    (0x20, "URG", "Dữ liệu khẩn cấp (Urgent Pointer)"),
    (0x40, "ECE", "Thông báo tắc nghẽn ECN"),
    (0x80, "CWR", "Giảm cửa sổ tắc nghẽn (Congestion Reduced)"),
]

# Cặp (Cổng đích -> (L4 kỳ vọng, tập L7 hợp lệ)) để kiểm tra tính nhất quán L4/L7-Port
EXPECTED_PORT_PROFILES: Dict[int, Tuple[Tuple[int, ...], Tuple[float, ...], str]] = {
    20: ((6,), (0.0, 11.0), "FTP-Data"),
    21: ((6,), (0.0, 11.0), "FTP"),
    22: ((6,), (0.0, 12.0), "SSH"),
    25: ((6,), (0.0, 3.0), "SMTP"),
    53: ((6, 17), (0.0, 5.0), "DNS"),
    80: ((6,), (0.0, 7.0), "HTTP"),
    443: ((6,), (0.0, 19.0), "HTTPS/TLS"),
    3306: ((6,), (0.0, 20.0), "MySQL"),
    8080: ((6,), (0.0, 7.0), "HTTP-Alt"),
    8443: ((6,), (0.0, 19.0), "HTTPS-Alt"),
}


def _safe_int(value: Any, default: int = 0) -> Optional[int]:
    if value is None:
        return default
    try:
        num = float(value)
        if not math.isfinite(num):
            return None
        return int(num)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        num = float(value)
        if not math.isfinite(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def decode_l4_protocol(proto: Any) -> L4ProtoInfo:
    """Giải mã giao thức tầng truyền vận L4 an toàn với NaN/chuỗi."""
    p = _safe_int(proto, default=6)
    if p is None:
        return {
            "name": "UNKNOWN",
            "desc": f"Giá trị giao thức L4 không hợp lệ ({proto})",
            "color": "#64748B",
        }
    return L4_PROTO_MAP.get(
        p,
        {
            "name": f"IP-Proto {p}",
            "desc": f"Giao thức mạng IP số hiệu {p}",
            "color": "#64748B",
        },
    )


def decode_l7_protocol(l7: Any) -> L7ProtoInfo:
    """Giải mã giao thức tầng ứng dụng L7 an toàn với NaN/chuỗi."""
    val = _safe_float(l7, default=0.0)
    if val is None:
        return {
            "name": "L7 Unknown",
            "desc": f"Giá trị L7 không hợp lệ ({l7})",
            "category": "Unknown",
        }
    return L7_PROTO_MAP.get(
        val,
        {
            "name": f"L7 Proto {val:.1f}",
            "desc": "Giao thức ứng dụng mở rộng / Tùy biến",
            "category": "Custom",
        },
    )


def decode_port(port: Any) -> str:
    """Giải mã số hiệu cổng sang tên dịch vụ chuẩn hóa (an toàn với NaN, float, str)."""
    p = _safe_int(port, default=-1)
    if p is None or p < 0:
        return "N/A (Invalid Port)"
    if p in WELL_KNOWN_PORTS:
        return f"{p} ({WELL_KNOWN_PORTS[p]})"
    if 1 <= p <= 1023:
        return f"{p} (Privileged System Port)"
    if 1024 <= p <= 49151:
        return f"{p} (Registered Service Port)"
    if p <= 65535:
        return f"{p} (Dynamic / Ephemeral Client Port)"
    return f"{p} (Out-of-Range Port)"


def decode_tcp_flags(flags: Any) -> List[str]:
    """Giải mã bitmask cờ TCP thành danh sách cờ tương ứng."""
    fl = _safe_int(flags, default=0)
    if fl is None:
        return ["INVALID"]
    if fl <= 0:
        return ["NONE (0x00)"]
    active_flags = [name for bit, name, _ in TCP_FLAG_BITS if fl & bit]
    return active_flags if active_flags else [f"Raw (0x{fl:02X})"]


def explain_scenario_context(flow: Dict[str, Any], true_attack: str) -> ScenarioContext:
    """Tạo lời giải thích ngữ cảnh nghiệp vụ mạng phục vụ phân tích SOC."""
    dst_port = _safe_int(flow.get("L4_DST_PORT", 0), default=0) or 0
    src_port = _safe_int(flow.get("L4_SRC_PORT", 0), default=0) or 0
    in_bytes = _safe_int(flow.get("IN_BYTES", 0), default=0) or 0
    out_bytes = _safe_int(flow.get("OUT_BYTES", 0), default=0) or 0
    duration = _safe_int(flow.get("FLOW_DURATION_MILLISECONDS", 0), default=0) or 0
    att_name = str(true_attack).strip().lower()

    if "tùy chỉnh" in att_name or "manual" in att_name:
        return {
            "title": "Kịch bản: Kiểm Thử Tùy Chỉnh (Manual Feature Injection)",
            "description": (
                f"Người vận hành SOC đã chỉnh sửa trực tiếp thông số NetFlow hướng tới cổng đích "
                f"{decode_port(dst_port)} ({in_bytes:,}B In / {out_bytes:,}B Out, {duration:,} ms). "
                f"Bộ giải mã và hệ thống suy luận đang cập nhật theo thời gian thực."
            ),
        }

    if att_name == "dos":
        if duration > 1000 and (in_bytes < 1000 or out_bytes < 1000):
            return {
                "title": "Kịch bản: Tấn công Slow-Rate DoS (Slowloris / HTTP Keep-Alive Hang)",
                "description": (
                    f"Luồng kết nối kéo dài ({duration:,} ms) đến cổng đích {dst_port} nhưng dung lượng "
                    f"truyền rất nhỏ ({in_bytes:,}B / {out_bytes:,}B). Kẻ tấn công cố tình giữ kết nối mở lâu "
                    f"nhằm cạn kiệt Connection Pool của máy chủ Web."
                ),
            }
        return {
            "title": "Kịch bản: Tấn công Từ chối Dịch vụ Phân tán (High-Rate DoS Flood)",
            "description": (
                f"Lưu lượng DoS dồn dập nhắm vào dịch vụ tại cổng {dst_port}. Mô hình NetFlow theo dõi sự "
                f"mất cân bằng về tốc độ gửi gói tin và thời gian phiên."
            ),
        }

    if att_name == "exploits":
        return {
            "title": "Kịch bản: Thăm dò & Khai thác Lỗ hổng Ứng dụng (Web/App Exploitation)",
            "description": (
                f"Luồng dữ liệu mang đặc tính thăm dò dịch vụ tại cổng {dst_port} với chuỗi payload "
                f"nhằm chiếm quyền kiểm soát hoặc thực thi mã từ xa (RCE / Buffer Overflow)."
            ),
        }

    if att_name == "backdoor":
        return {
            "title": "Kịch bản: Kênh Kết nối Bí mật Command & Control (C2 Beaconing)",
            "description": (
                f"Luồng thông tin bất thường kết nối từ cổng nguồn {src_port} ra bên ngoài, "
                f"duy trì chu kỳ định kỳ để nhận chỉ thị từ máy chủ điều khiển ngầm."
            ),
        }

    if att_name == "reconnaissance":
        return {
            "title": "Kịch bản: Quét Thám mã Dò cổng Mạng (Network & Port Sweep Reconnaissance)",
            "description": (
                f"Luồng gửi các gói tin thăm dò (SYN Sweep / ACK Probe) tới cổng {dst_port} nhằm lập bản đồ "
                f"các cổng đang mở và dịch vụ đang lắng nghe trên máy chủ đích."
            ),
        }

    if att_name == "analysis":
        return {
            "title": "Kịch bản: Phân tích Cấu trúc & Quét Lỗ hổng Web (Fuzzing / Directory Traversal)",
            "description": (
                f"Kẻ tấn công gửi các biến thể tham số truy vấn web hoặc thử nghiệm các endpoint lạ "
                f"để tìm kiếm sơ hở cấu hình và rò rỉ dữ liệu nhạy cảm."
            ),
        }

    return {
        "title": "Kịch bản: Lưu lượng Người dùng Hợp lệ (Benign Production Traffic)",
        "description": (
            f"Phiên truyền thông tiêu chuẩn giữa máy trạm và máy chủ ({decode_port(dst_port)}) với tỷ lệ "
            f"trao đổi dữ liệu ({in_bytes:,}B In / {out_bytes:,}B Out) phù hợp với hành vi thông thường."
        ),
    }


def analyze_tier2_dpi_heuristics(
    flow: Dict[str, Any],
    verdict: Optional[Dict[str, Any]] = None,
) -> DPIInspectionReport:
    """
    UX-1: Chẩn đoán Heuristic Tầng 2 (Tier-2 DPI Inspection Panel) tất định trên 10 đặc trưng NetFlow.
    Phân tích: bytes/pkt, bất đối xứng IN/OUT, pkts/sec, cờ TCP bất thường,
    tính nhất quán L4/L7-Port và sinh khuyến nghị leo thang + chữ ký Snort/Suricata mô phỏng.
    """
    src_port = _safe_int(flow.get("L4_SRC_PORT", 0), default=0) or 0
    dst_port = _safe_int(flow.get("L4_DST_PORT", 0), default=0) or 0
    proto = _safe_int(flow.get("PROTOCOL", 6), default=6) or 6
    l7_proto = _safe_float(flow.get("L7_PROTO", 0.0), default=0.0) or 0.0
    in_bytes = max(0, _safe_int(flow.get("IN_BYTES", 0), default=0) or 0)
    out_bytes = max(0, _safe_int(flow.get("OUT_BYTES", 0), default=0) or 0)
    in_pkts = max(0, _safe_int(flow.get("IN_PKTS", 0), default=0) or 0)
    out_pkts = max(0, _safe_int(flow.get("OUT_PKTS", 0), default=0) or 0)
    tcp_flags = max(0, _safe_int(flow.get("TCP_FLAGS", 0), default=0) or 0)
    duration_ms = max(0, _safe_int(flow.get("FLOW_DURATION_MILLISECONDS", 0), default=0) or 0)

    total_bytes = in_bytes + out_bytes
    total_pkts = in_pkts + out_pkts
    bytes_per_in_pkt = round(in_bytes / in_pkts, 1) if in_pkts > 0 else 0.0
    bytes_per_out_pkt = round(out_bytes / out_pkts, 1) if out_pkts > 0 else 0.0
    avg_pkt_size = round(total_bytes / total_pkts, 1) if total_pkts > 0 else 0.0
    in_out_byte_ratio = round(in_bytes / max(out_bytes, 1), 2)
    effective_sec = max(duration_ms / 1000.0, 0.001)
    pkts_per_sec = round(total_pkts / effective_sec, 1) if total_pkts > 0 else 0.0
    flags_list = decode_tcp_flags(tcp_flags)

    findings: List[DPIFinding] = []
    l4_l7_consistency = "Nhất quán (Consistent)"

    # 1. Kiểm tra tính nhất quán L4 / L7 / Port
    if dst_port in EXPECTED_PORT_PROFILES:
        exp_protos, exp_l7s, svc_name = EXPECTED_PORT_PROFILES[dst_port]
        if proto not in exp_protos:
            l4_l7_consistency = f"Bất thường L4 trên cổng {dst_port} ({svc_name})"
            findings.append({
                "rule_id": "DPI-L4-MISMATCH",
                "severity": "HIGH",
                "title": f"Lệch giao thức L4 cho dịch vụ {svc_name} (Port {dst_port})",
                "detail": f"Cổng {dst_port} thường dùng proto {exp_protos} nhưng luồng thực tế dùng L4={proto}.",
                "snort_hint": (
                    f'alert ip $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                    f'(msg:"SOC-DPI L4 Protocol Mismatch on {svc_name}"; sid:900101; rev:1;)'
                ),
            })
        elif l7_proto not in exp_l7s:
            l4_l7_consistency = f"Lệch L7 ({decode_l7_protocol(l7_proto)['name']} trên cổng {svc_name})"
            findings.append({
                "rule_id": "DPI-L7-TUNNEL",
                "severity": "MEDIUM",
                "title": f"Nghi vấn đường hầm L7 trên cổng {dst_port} ({svc_name})",
                "detail": (
                    f"Cổng {dst_port} ({svc_name}) nhưng parser L7 nhận diện mã {l7_proto:.1f} "
                    f"({decode_l7_protocol(l7_proto)['name']}). Cần soi chữ ký payload ứng dụng."
                ),
                "snort_hint": (
                    f'alert tcp $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                    f'(msg:"SOC-DPI Unexpected L7 Protocol Tunnel"; flow:established; sid:900102; rev:1;)'
                ),
            })

    # 2. Kiểm tra cờ TCP bất thường
    if proto == 6:
        has_syn = bool(tcp_flags & 0x02)
        has_ack = bool(tcp_flags & 0x10)
        has_fin = bool(tcp_flags & 0x01)
        has_rst = bool(tcp_flags & 0x04)
        has_urg = bool(tcp_flags & 0x20)
        has_psh = bool(tcp_flags & 0x08)

        if tcp_flags == 0 and total_pkts > 0:
            findings.append({
                "rule_id": "DPI-TCP-NULL",
                "severity": "HIGH",
                "title": "Gói tin TCP Null Flag (Bitmask 0x00)",
                "detail": "Luồng TCP có gói tin nhưng không bật bất kỳ cờ điều khiển nào — dấu hiệu quét cổng Null Scan.",
                "snort_hint": (
                    f'alert tcp $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                    f'(msg:"SOC-DPI TCP Null Scan Detected"; flags:0; sid:900201; rev:1;)'
                ),
            })
        elif has_syn and has_fin and not has_ack:
            findings.append({
                "rule_id": "DPI-TCP-SYNFIN",
                "severity": "HIGH",
                "title": "Tổ hợp cờ TCP bất hợp lệ (SYN + FIN không ACK)",
                "detail": "Cờ SYN và FIN bật đồng thời mà không có ACK trên luồng NetFlow — dấu hiệu quét né tường lửa (SYN+FIN evasion).",
                "snort_hint": (
                    f'alert tcp $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                    f'(msg:"SOC-DPI Malformed SYN+FIN Flags"; flags:SF; sid:900202; rev:1;)'
                ),
            })
        elif has_fin and has_urg and has_psh and not has_ack:
            findings.append({
                "rule_id": "DPI-TCP-XMAS",
                "severity": "HIGH",
                "title": "Dấu hiệu quét Xmas Tree (FIN+PSH+URG)",
                "detail": "Tổ hợp FIN+PSH+URG không có ACK là mẫu cổ điển của công cụ rà quét Nmap Xmas Scan.",
                "snort_hint": (
                    f'alert tcp $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                    f'(msg:"SOC-DPI TCP Xmas Scan"; flags:FPU; sid:900203; rev:1;)'
                ),
            })
        elif tcp_flags == 0x02 and out_pkts == 0:
            findings.append({
                "rule_id": "DPI-TCP-HALFOPEN",
                "severity": "MEDIUM",
                "title": "Kết nối nửa mở (SYN-Only Half-Open)",
                "detail": f"Chỉ có {in_pkts} gói SYN đến cổng {dst_port} mà không có phản hồi (SYN Flood hoặc Port Probe).",
                "snort_hint": (
                    f'alert tcp $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                    f'(msg:"SOC-DPI Half-Open SYN Probe/Flood"; flags:S,12; threshold:type both, track by_src, count 20, seconds 5; sid:900204; rev:1;)'
                ),
            })
        elif has_rst and out_pkts <= 2 and in_pkts <= 3:
            findings.append({
                "rule_id": "DPI-TCP-RST",
                "severity": "LOW",
                "title": "Phiên kết thúc sớm với cờ RST",
                "detail": f"Cờ RST xuất hiện trên phiên ngắn ({total_pkts} gói), có thể là thăm dò cổng đóng hoặc từ chối kết nối.",
                "snort_hint": (
                    f'alert tcp $HOME_NET {dst_port} -> $EXTERNAL_NET any '
                    f'(msg:"SOC-DPI Short Session TCP Reset"; flags:R; sid:900205; rev:1;)'
                ),
            })
    elif tcp_flags > 0:
        l4_l7_consistency = f"Bất thường: L4={decode_l4_protocol(proto)['name']} có TCP_FLAGS={tcp_flags}"
        findings.append({
            "rule_id": "DPI-NONTCP-FLAGS",
            "severity": "MEDIUM",
            "title": "Xuất hiện bitmask TCP Flags trên giao thức phi-TCP",
            "detail": f"Giao thức L4={proto} nhưng trường TCP_FLAGS={tcp_flags} khác 0.",
            "snort_hint": (
                f'alert ip $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                f'(msg:"SOC-DPI Non-TCP Flow With TCP Flags"; sid:900206; rev:1;)'
            ),
        })

    # 3. Kiểm tra bất đối xứng IN/OUT & tốc độ gói tin / kích thước gói
    if in_pkts > 0 and out_pkts == 0 and in_bytes > 0:
        findings.append({
            "rule_id": "DPI-ASYM-UNIDIR",
            "severity": "MEDIUM",
            "title": "Lưu lượng một chiều không có phản hồi (Unidirectional / Zero-Reply)",
            "detail": f"Gửi {in_pkts} gói ({in_bytes:,}B) tới cổng {dst_port} nhưng chiều về = 0 gói. Điển hình của Flood/Scan/Spoofing.",
            "snort_hint": (
                f'alert ip $EXTERNAL_NET {src_port} -> $HOME_NET {dst_port} '
                f'(msg:"SOC-DPI Unidirectional Traffic Spike"; sid:900301; rev:1;)'
            ),
        })
    elif in_out_byte_ratio >= 10.0 or (out_bytes > 0 and (out_bytes / max(in_bytes, 1)) >= 20.0):
        findings.append({
            "rule_id": "DPI-ASYM-RATIO",
            "severity": "MEDIUM",
            "title": "Bất đối xứng dung lượng IN/OUT cực đoan",
            "detail": (
                f"Tỷ lệ IN/OUT = {in_out_byte_ratio:.2f} ({in_bytes:,}B In vs {out_bytes:,}B Out). "
                f"Cần kiểm tra dấu hiệu tuồn dữ liệu (Exfiltration) hoặc khuếch đại (Amplification)."
            ),
            "snort_hint": (
                f'alert ip $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                f'(msg:"SOC-DPI Asymmetric Payload Ratio Anomaly"; sid:900302; rev:1;)'
            ),
        })

    if duration_ms >= 2000 and total_bytes > 0 and avg_pkt_size < 90:
        findings.append({
            "rule_id": "DPI-SLOW-LOW",
            "severity": "MEDIUM",
            "title": "Phiên kéo dài với kích thước gói tin nhỏ (Slow-Rate / Keep-Alive)",
            "detail": (
                f"Thời lượng {duration_ms:,} ms nhưng kích thước trung bình chỉ {avg_pkt_size:.1f} B/pkt "
                f"({pkts_per_sec:.1f} pkts/s). Phù hợp mẫu Slowloris hoặc C2 Beaconing."
            ),
            "snort_hint": (
                f'alert tcp $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                f'(msg:"SOC-DPI Low-and-Slow Session Pattern"; flow:established,to_server; dsize:<100; sid:900303; rev:1;)'
            ),
        })
    elif pkts_per_sec >= 5000.0 and total_pkts >= 20:
        findings.append({
            "rule_id": "DPI-BURST-RATE",
            "severity": "HIGH",
            "title": "Bùng phát tốc độ gói tin cực cao (High-Rate Burst)",
            "detail": f"Tốc độ đạt {pkts_per_sec:,.1f} pkts/s ({total_pkts} gói trong {duration_ms} ms) — nguy cơ DoS/Flood.",
            "snort_hint": (
                f'alert ip $EXTERNAL_NET any -> $HOME_NET {dst_port} '
                f'(msg:"SOC-DPI High-Rate Packet Burst"; threshold:type threshold, track by_src, count 100, seconds 1; sid:900304; rev:1;)'
            ),
        })

    # 4. Tổng hợp kết luận leo thang DPI Tầng 2
    tier1_status = str(verdict.get("status", "BENIGN")) if verdict else "BENIGN"
    has_high = any(f["severity"] == "HIGH" for f in findings)
    has_medium = any(f["severity"] == "MEDIUM" for f in findings)

    if tier1_status == "ATTACK" or has_high:
        recommendation = "ESCALATE_BLOCK"
        recommendation_label = "Tầng 2 xác nhận chặn luồng (Khớp quy tắc bất thường mức cao)"
        risk_badge = "danger"
    elif tier1_status == "SUSPICIOUS" or has_medium:
        recommendation = "INSPECT_PAYLOAD"
        recommendation_label = "Chuyển kiểm tra sâu tải trọng gói tin (Nghi vấn tại Tầng 2)"
        risk_badge = "warning"
    else:
        recommendation = "CLEAR_PASS"
        recommendation_label = "Tầng 2 không ghi nhận bất thường hình thái luồng"
        risk_badge = "success"

    if not findings:
        findings.append({
            "rule_id": "DPI-BASELINE-OK",
            "severity": "INFO",
            "title": "Không phát hiện bất thường hình thái NetFlow",
            "detail": (
                f"Các chỉ số Bytes/Pkt ({avg_pkt_size:.1f}B), tỷ lệ IN/OUT ({in_out_byte_ratio:.2f}), "
                f"tốc độ ({pkts_per_sec:.1f} pkts/s) và tính nhất quán L4/L7 đều nằm trong ngưỡng bình thường."
            ),
            "snort_hint": f"# Pass Fast-Path: no alert rule triggered for port {dst_port}",
        })

    summary_text = (
        f"Đã chạy {len(findings)} quy tắc DPI tất định trên luồng {decode_l4_protocol(proto)['name']} "
        f"cổng đích {dst_port} ({avg_pkt_size:.1f} B/pkt, tỷ lệ IN/OUT={in_out_byte_ratio:.2f}, "
        f"{pkts_per_sec:,.1f} pkts/s). Kết quả: {recommendation_label}."
    )

    return {
        "recommendation": recommendation,
        "recommendation_label": recommendation_label,
        "risk_badge": risk_badge,
        "summary": summary_text,
        "metrics": {
            "total_bytes": total_bytes,
            "total_pkts": total_pkts,
            "bytes_per_in_pkt": bytes_per_in_pkt,
            "bytes_per_out_pkt": bytes_per_out_pkt,
            "avg_pkt_size": avg_pkt_size,
            "in_out_byte_ratio": in_out_byte_ratio,
            "pkts_per_sec": pkts_per_sec,
            "duration_ms": duration_ms,
            "tcp_flags_list": flags_list,
            "l4_l7_consistency": l4_l7_consistency,
        },
        "findings": findings,
    }
