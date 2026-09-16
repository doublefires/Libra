# -*- coding: utf-8 -*-
"""退信解析回归（纯函数，用合成的 QQ 退信邮件）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_check_bounce", _ROOT / "scripts" / "check_bounce.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

BODY = """<html><body>
<p>很抱歉您发送的邮件被退回，以下是该邮件的相关信息：</p>
<table><tr><td>被退回邮件</td>
<td>主 题：[Libra] 2026-09-15 Libra 日报 <br/>时 间：2026-09-15 14:31:25 </td></tr></table>
<table><tr><td colspan="2">无法发送到 cynthiazheng1990@hotmail.com</td></tr>
<tr><td>退信原因</td><td> 未知原因 <br /> host hotmail-com.olc.protection.outlook.com[52.101.60.184]
said: 550 5.7.515 Access denied, sending domain QQ.COM doesn't meet the required
authentication level. Spf= Fail , Dkim= Pass , DMARC= Pass </td></tr>
<tr><td>解决方案</td><td> 请联系您的收件人 </td></tr></table>
</body></html>"""


def _msg():
    m = MIMEMultipart("report")
    m["From"] = "\"PostMaster\" <PostMaster@qq.com>"
    m["Subject"] = "[Qmail] 来自qq.com的退信"
    m["Date"] = "Tue, 15 Sep 2026 14:31:34 +0800"
    m.attach(MIMEText(BODY, "html", "utf-8"))
    return m


def test_parse_bounce_extracts_recipient_and_reason():
    r = cb.parse_bounce(_msg())
    assert r["failed_to"] == "cynthiazheng1990@hotmail.com"
    assert "5.7.515" in r["reason"]
    assert "QQ.COM" in r["reason"]
    assert r["orig_subject"] == "[Libra] 2026-09-15 Libra 日报"


def test_parse_bounce_survives_missing_fields():
    m = MIMEMultipart("report")
    m["Subject"] = "退信"
    m.attach(MIMEText("没有任何可解析内容的正文", "plain", "utf-8"))
    r = cb.parse_bounce(m)
    assert r["failed_to"] is None
    assert isinstance(r["reason"], str)


def test_sendmail_build_mime_has_deliverability_headers():
    """发信卫生：Date / Message-ID / Reply-To / List-Unsubscribe 必须存在。"""
    from barometer.analytics import risk_metrics  # noqa: F401  （确保包可导入）
    import importlib.util as iu
    spec = iu.spec_from_file_location("_send_mail", _ROOT / "scripts" / "send_mail.py")
    sm = iu.module_from_spec(spec)
    spec.loader.exec_module(sm)
    msg = sm.build_mime("[Libra] 测试", "正文", [], from_addr="1985618325@qq.com")
    assert msg["Date"] and msg["Message-ID"] and msg["Reply-To"]
    assert "mailto:" in msg["List-Unsubscribe"]
    assert msg["Message-ID"].endswith("@qq.com>")
    # 两封信的 Message-ID 必须不同（否则一天多封会被判重）
    msg2 = sm.build_mime("[Libra] 测试", "正文", [], from_addr="1985618325@qq.com")
    assert msg["Message-ID"] != msg2["Message-ID"]
