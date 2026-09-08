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