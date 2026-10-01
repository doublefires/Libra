"""send_mail.build_mime 单元测试（不联网，只验 MIME 构造）。"""
from __future__ import annotations

from email import policy

from scripts.send_mail import build_mime


def test_build_mime_basic_fields():
    msg = build_mime("test 主题 2026", "你好，正文内容")
    msg["From"] = "a@qq.com"
    msg["To"] = "b@qq.com"
    s = msg.as_string()
    assert "2026" in msg["Subject"]               # 主题内容保留
    assert "=?utf-8?" in s                        # 序列化时中文主题已 RFC2047 编码
    body = msg.get_payload()[0].get_payload(decode=True).decode("utf-8")
    assert "你好，正文内容" in body                 # 正文含中文（MIME 可能 base64，需解码）
    assert len(msg.get_payload()) == 1            # 无附件：仅正文


def test_build_mime_with_attachment():
    msg = build_mime("sub", "body", [("daily_latest.md", b"# md\n")])
    parts = msg.get_payload()
    assert len(parts) == 2
    assert parts[1].get_filename() == "daily_latest.md"
    # 中文文件名也应可编码
    msg2 = build_mime("s", "b", [("模型总结.md", "内容".encode("utf-8"))])
    assert msg2.get_payload()[1].get_filename() is not None


def test_recently_sent_guard(tmp_path):
    """重复发送保护：状态文件里记着同一天 → 判定为已发过；缺失/损坏 → 当作没发过。"""
    from scripts.send_mail import recently_sent
    f = tmp_path / ".last_sent.json"
    assert recently_sent(f, "2026-10-08") is False        # 文件不存在
    f.write_text('{"date": "2026-10-08"}', encoding="utf-8")
    assert recently_sent(f, "2026-10-08") is True
    assert recently_sent(f, "2026-10-09") is False
    f.write_text("{坏文件", encoding="utf-8")
    assert recently_sent(f, "2026-10-08") is False


def test_intraday_gate_decision(tmp_path):
    """盘中门槛：休市/缺文件/旧文件一律不发；只有「今天生成的门槛」才按 material 放行。"""
    import json

    from scripts.send_mail import intraday_gate_decision
    gate = tmp_path / "intraday_gate.json"
    closed = tmp_path / "market_closed.json"
    today = "2026-10-01"

    # 1) 门槛文件不存在 → 不发
    assert intraday_gate_decision(gate, closed, today)[0] is False

    # 2) 昨天的 material=true 旧门槛（2026-10-01 连发两封的根因）→ 不发
    gate.write_text(json.dumps({"time": "2026-09-30 18:46", "material": True,
                                "score": -45.4, "reasons": ["实时分 +30.1"]}), encoding="utf-8")
    ok, note = intraday_gate_decision(gate, closed, today)
    assert ok is False and "旧记录" in note

    # 3) 今天生成的门槛 → 按 material 放行
    gate.write_text(json.dumps({"time": "2026-10-01 10:00", "material": True,
                                "score": -45.0, "reasons": ["实时分 +30.0"]}), encoding="utf-8")
    assert intraday_gate_decision(gate, closed, today)[0] is True
    gate.write_text(json.dumps({"time": "2026-10-01 10:00", "material": False,
                                "score": -45.0, "reasons": []}), encoding="utf-8")
    assert intraday_gate_decision(gate, closed, today)[0] is False

    # 4) 即使门槛是今天的，只要今天是休市日 → 不发
    gate.write_text(json.dumps({"time": "2026-10-01 10:00", "material": True,
                                "score": -45.0, "reasons": []}), encoding="utf-8")
    closed.write_text('{"date": "2026-10-01", "reason": "holiday"}', encoding="utf-8")
    ok, note = intraday_gate_decision(gate, closed, today)
    assert ok is False and "休市" in note


def test_marked_closed_guard(tmp_path):
    """休市标记：只有标记日期 == 今天才拦；昨天/损坏的标记不拦。"""
    from scripts.send_mail import marked_closed
    f = tmp_path / "market_closed.json"
    assert marked_closed(f, "2026-10-01") is False       # 文件不存在
    f.write_text('{"date": "2026-10-01", "reason": "holiday"}', encoding="utf-8")
    assert marked_closed(f, "2026-10-01") is True
    assert marked_closed(f, "2026-10-08") is False       # 隔了一天就失效
    f.write_text("{坏文件", encoding="utf-8")
    assert marked_closed(f, "2026-10-01") is False


def test_parse_recipients():
    from scripts.send_mail import parse_recipients
    assert parse_recipients("a@qq.com, b@163.com") == ["a@qq.com", "b@163.com"]
    assert parse_recipients("a@qq.com;b@qq.com") == ["a@qq.com", "b@qq.com"]
    assert parse_recipients(" a@qq.com  b@163.com ") == ["a@qq.com", "b@163.com"]
    assert parse_recipients("") == []
    assert parse_recipients(None) == []
    assert parse_recipients(["x@qq.com", " y@163.com "]) == ["x@qq.com", "y@163.com"]


def test_unsub_intent():
    from scripts.send_mail import unsub_intent
    assert unsub_intent("R")
    assert unsub_intent("r")
    assert unsub_intent("R\n谢谢")          # 首行 R
    assert unsub_intent("退订")
    assert unsub_intent("麻烦退订一下")
    assert unsub_intent("Please unsubscribe")
    assert not unsub_intent("收到，谢谢")
    assert not unsub_intent("report")
    assert not unsub_intent("好的 R 服务正常吗".lower() and "好的R服务正常吗")
    assert not unsub_intent("")


def test_recipient_exclude():
    from scripts.send_mail import recipient_exclude
    assert recipient_exclude(["A@QQ.com", "b@163.com"], ["a@qq.com"]) == ["b@163.com"]
    assert recipient_exclude(["a@qq.com"], ["x@qq.com"]) == ["a@qq.com"]
    assert recipient_exclude(["a@qq.com", "b@qq.com"], ["a@qq.com", "b@qq.com"]) == []
    assert recipient_exclude(["a@qq.com"], None) == ["a@qq.com"]