#!/usr/bin/env python3
"""
qr.py — SmartShelf Identity Node (Raspberry Pi)
================================================
Scans QR codes via Pi camera, looks up users in Firestore,
manages sessions on the cabinet document.

SSD1306 128x64 OLED (I2C) wiring:
  VCC  -> Pin  1  (3.3V)
  GND  -> Pin  6  (GND)
  SDA  -> Pin  3  (GPIO2, I2C1 SDA)
  SCL  -> Pin  5  (GPIO3, I2C1 SCL)
  Enable I2C in raspi-config -> Interface Options -> I2C
  Install: pip install luma.oled Pillow

Flow:
  1. Scan QR code -> look up user in Firestore
  2. Check cooldowns (4h for recipients) and bans (1-week for violations)
  3. Write activeSession to cabinet/cabinet_001
  4. For recipients: the Vision Node auto-captures after 5s countdown
  5. On second scan by same user OR timeout -> close session
"""

from picamera2 import Picamera2
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
import os
import time
import threading
from datetime import datetime, timezone, timedelta

# ---------- SSD1306 OLED ----------
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from PIL import Image, ImageDraw, ImageFont
    _serial = i2c(port=1, address=0x3C)
    oled = ssd1306(_serial, width=128, height=64)
    # Try to load a bitmap font; fall back to PIL default
    try:
        _font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
        _font_md = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
        _font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9
        )
    except Exception:
        _font_lg = ImageFont.load_default()
        _font_md = _font_lg
        _font_sm = _font_lg
    HAS_OLED = True
except Exception as _oled_err:
    HAS_OLED = False
    oled = None
    print(f"[OLED] Not available: {_oled_err}", flush=True)

# Shared display state — updated from main thread, read by display thread
display_state = {
    "screen": "idle",       # idle | scanning | session | denied | unknown | goodbye
    "name": "",
    "role": "",
    "score": 0,
    "countdown": 0,         # seconds left (recipient only)
    "message": "",          # short sub-message (deny reason etc.)
    "capacity": "",         # e.g. "14/20"
}
_display_lock = threading.Lock()

# ---------- Config ----------
SERVICE_ACCOUNT_KEY_PATH = os.path.expanduser('~/firestone/auth.json')
CABINET_ID = os.getenv("CABINET_ID", "cabinet_001")
SESSION_TIMEOUT = 120          # seconds before auto-close
RECIPIENT_COOLDOWN_HOURS = 4   # recipients can visit every 4 hours
BAN_DURATION_DAYS = 7          # 1-week ban for violations
COOLDOWN = 2                   # QR scan debounce seconds

# ---------- Firebase ----------
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------- Camera ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (480, 360)},
    buffer_count=1
)
picam2.configure(config)
picam2.start()

# ---------- QR Detector ----------
detector = cv2.QRCodeDetector()

# ---------- Session state ----------
active_session = None          # dict with uid, role, displayName, etc.
session_timer = None           # threading.Timer for auto-close
session_score_deltas = []      # accumulated score changes


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ==================================================================
# OLED DISPLAY
# ==================================================================

def _set_display(**kwargs):
    """Thread-safe update of the shared display state."""
    with _display_lock:
        display_state.update(kwargs)


def _render_frame():
    """Render one frame to the OLED based on current display_state."""
    with _display_lock:
        state = dict(display_state)  # shallow copy to avoid holding lock during draw

    img = Image.new("1", (128, 64), 0)  # 0 = black background
    draw = ImageDraw.Draw(img)
    screen = state["screen"]

    if screen == "idle":
        # Header bar
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((4, 1), "SmartShelf", font=_font_md, fill=0)
        # Main prompt
        draw.text((14, 20), "Scan QR to begin", font=_font_md, fill=1)
        # Capacity
        if state["capacity"]:
            draw.text((4, 53), f"Cabinet: {state['capacity']}", font=_font_sm, fill=1)
        # Online dot
        draw.ellipse((118, 54, 126, 62), fill=1)

    elif screen == "scanning":
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((4, 1), "SmartShelf", font=_font_md, fill=0)
        draw.text((30, 28), "Looking up...", font=_font_md, fill=1)

    elif screen == "session":
        role = state["role"]
        name = state["name"][:16]  # truncate to fit
        score = state["score"]
        countdown = state["countdown"]

        # Name bar
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((4, 1), name, font=_font_md, fill=0)

        # Role badge
        role_label = "DONOR" if role == "donor" else "RECIPIENT"
        draw.text((4, 19), role_label, font=_font_sm, fill=1)
        draw.text((70, 19), f"Pts:{score}", font=_font_sm, fill=1)

        if role == "recipient" and countdown > 0:
            # Big countdown
            draw.text((4, 31), "Hold item to camera!", font=_font_sm, fill=1)
            # Large digit(s)
            cstr = str(countdown)
            draw.text((54, 40), cstr, font=_font_lg, fill=1)
            # Outer box around countdown
            draw.rectangle((48, 37, 80, 63), outline=1)
        elif role == "recipient":
            draw.text((4, 33), "Camera will auto-snap", font=_font_sm, fill=1)
            draw.text((4, 46), "Scan again to sign out", font=_font_sm, fill=1)
        else:  # donor
            draw.text((4, 33), "Use app to add items", font=_font_sm, fill=1)
            draw.text((4, 46), "Scan again to sign out", font=_font_sm, fill=1)

    elif screen == "denied":
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((14, 1), "Access Denied", font=_font_md, fill=0)
        msg = state["message"]
        # Word-wrap into two lines of ~22 chars
        if len(msg) > 22:
            split = msg.rfind(" ", 0, 22)
            split = split if split > 0 else 22
            draw.text((4, 20), msg[:split], font=_font_sm, fill=1)
            draw.text((4, 32), msg[split+1:split+45], font=_font_sm, fill=1)
        else:
            draw.text((4, 24), msg, font=_font_sm, fill=1)
        draw.text((4, 50), "See staff for help", font=_font_sm, fill=1)

    elif screen == "unknown":
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((14, 1), "Card Unknown", font=_font_md, fill=0)
        draw.text((16, 26), "Not registered", font=_font_md, fill=1)
        draw.text((4, 50), "See staff for help", font=_font_sm, fill=1)

    elif screen == "goodbye":
        draw.text((20, 10), "Thank you,", font=_font_md, fill=1)
        name = state["name"][:16]
        draw.text((4, 26), name + "!", font=_font_lg, fill=1)
        score = state["score"]
        draw.text((4, 50), f"Score: {score}", font=_font_sm, fill=1)

    return img


def display_loop():
    """Daemon thread: renders the OLED at ~10 fps."""
    if not HAS_OLED:
        return
    while True:
        try:
            img = _render_frame()
            oled.display(img)
        except Exception as exc:
            log(f"  OLED render error: {exc}")
        time.sleep(0.1)  # 10 fps


def oled_clear():
    """Blank the display (call on shutdown)."""
    if HAS_OLED:
        try:
            oled.clear()
        except Exception:
            pass


# ==================================================================
# COOLDOWN & BAN CHECKING
# ==================================================================

def check_cooldown_and_ban(uid, role):
    """Check if a recipient is on cooldown or banned.
    Returns (allowed: bool, reason: str).
    Donors always pass.
    """
    if role == "donor":
        return True, ""

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    try:
        cd_ref = db.collection("userCooldowns").document(uid)
        cd_snap = cd_ref.get()

        if not cd_snap.exists:
            # First visit ever
            cd_ref.set({
                "uid": uid,
                "lastVisit": firestore.SERVER_TIMESTAMP,
                "bannedUntil": None,
                "violationCount": 0,
                "date": today_str,
            })
            return True, ""

        cd = cd_snap.to_dict()

        # Check ban
        banned_until = cd.get("bannedUntil")
        if banned_until:
            if hasattr(banned_until, 'timestamp'):
                ban_time = datetime.fromtimestamp(
                    banned_until.timestamp(), tz=timezone.utc
                )
            elif isinstance(banned_until, datetime):
                ban_time = (
                    banned_until if banned_until.tzinfo
                    else banned_until.replace(tzinfo=timezone.utc)
                )
            else:
                ban_time = None

            if ban_time and now < ban_time:
                remaining = ban_time - now
                days_left = remaining.days
                hours_left = remaining.seconds // 3600
                return False, f"Banned for {days_left}d {hours_left}h (violation penalty)"

        # Check 4-hour cooldown
        last_visit = cd.get("lastVisit")
        if last_visit:
            if hasattr(last_visit, 'timestamp'):
                lv_time = datetime.fromtimestamp(
                    last_visit.timestamp(), tz=timezone.utc
                )
            elif isinstance(last_visit, datetime):
                lv_time = (
                    last_visit if last_visit.tzinfo
                    else last_visit.replace(tzinfo=timezone.utc)
                )
            else:
                lv_time = None

            if lv_time:
                elapsed = now - lv_time
                cooldown_delta = timedelta(hours=RECIPIENT_COOLDOWN_HOURS)
                if elapsed < cooldown_delta:
                    remaining = cooldown_delta - elapsed
                    hours_left = remaining.seconds // 3600
                    mins_left = (remaining.seconds % 3600) // 60
                    return (
                        False,
                        f"Cooldown: {hours_left}h {mins_left}m remaining "
                        f"(every {RECIPIENT_COOLDOWN_HOURS}h)",
                    )

    except Exception as exc:
        log(f"  WARNING: Cooldown check failed: {exc}")
        return True, ""

    return True, ""


def apply_ban(uid, reason="photo_violation"):
    """Ban a user for BAN_DURATION_DAYS."""
    now = datetime.now(timezone.utc)
    ban_until = now + timedelta(days=BAN_DURATION_DAYS)
    try:
        cd_ref = db.collection("userCooldowns").document(uid)
        cd_ref.set({
            "bannedUntil": ban_until,
            "violationCount": firestore.Increment(1),
            "lastViolation": firestore.SERVER_TIMESTAMP,
            "lastViolationReason": reason,
        }, merge=True)
        log(f"  BAN APPLIED: {uid} banned until {ban_until.isoformat()}")
    except Exception as exc:
        log(f"  WARNING: Could not apply ban: {exc}")


def update_last_visit(uid):
    """Update the lastVisit timestamp for cooldown tracking."""
    try:
        db.collection("userCooldowns").document(uid).set({
            "lastVisit": firestore.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception as exc:
        log(f"  WARNING: Could not update lastVisit: {exc}")


# ==================================================================
# SESSION MANAGEMENT
# ==================================================================

def open_session(uid, role, display_name, score):
    """Write activeSession to the cabinet document and start timeout."""
    global active_session, session_timer, session_score_deltas

    session_data = {
        "uid": uid,
        "role": role,
        "displayName": display_name,
        "score": score,
        "startedAt": firestore.SERVER_TIMESTAMP,
    }

    try:
        db.collection("cabinet").document(CABINET_ID).set({
            "activeSession": session_data,
        }, merge=True)

        # Write session_start event
        db.collection("events").add({
            "type": "session_start",
            "userId": uid,
            "userRole": role,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "deviceId": CABINET_ID,
            "itemId": None,
            "scoreDelta": 0,
        })

        # Update lastSeen on user doc
        db.collection("users").document(uid).set({
            "lastSeen": firestore.SERVER_TIMESTAMP,
        }, merge=True)

    except Exception as exc:
        log(f"  WARNING: Could not write session: {exc}")
        return False

    active_session = {
        "uid": uid,
        "role": role,
        "displayName": display_name,
        "score": score,
        "startedAt": time.time(),
    }
    session_score_deltas = []

    # Start idle timeout
    if session_timer:
        session_timer.cancel()
    session_timer = threading.Timer(
        SESSION_TIMEOUT, lambda: close_session("timeout")
    )
    session_timer.daemon = True
    session_timer.start()

    # Update OLED
    _set_display(screen="session", name=display_name, role=role,
                 score=score, countdown=0)

    # For recipients: drive a live countdown on the OLED
    if role == "recipient":
        threading.Thread(
            target=_recipient_countdown_display,
            args=(uid, RECIPIENT_COUNTDOWN_SECS),
            daemon=True,
        ).start()

    log(f"  SESSION OPENED: {display_name} ({role})")
    if role == "recipient":
        log(f"  >> RECEIVING MODE: Vision Node will auto-capture in 5s")

    return True


def close_session(reason="manual"):
    """Close the active session, commit score, clear cabinet."""
    global active_session, session_timer

    if not active_session:
        return

    uid = active_session["uid"]
    display_name = active_session["displayName"]
    role = active_session["role"]

    # Cancel timeout timer
    if session_timer:
        session_timer.cancel()
        session_timer = None

    # Commit total score delta
    total_delta = sum(session_score_deltas)
    if total_delta != 0:
        try:
            user_ref = db.collection("users").document(uid)
            user_snap = user_ref.get()
            if user_snap.exists:
                current_score = user_snap.to_dict().get("score", 0)
                new_score = max(0, current_score + total_delta)
                user_ref.update({"score": new_score})
                log(f"  Score: {current_score} -> {new_score} (delta: {total_delta:+d})")
        except Exception as exc:
            log(f"  WARNING: Score update failed: {exc}")

    # Write session_end event
    try:
        db.collection("events").add({
            "type": "session_end",
            "userId": uid,
            "userRole": role,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "deviceId": CABINET_ID,
            "scoreDelta": total_delta,
            "closeReason": reason,
        })
    except Exception as exc:
        log(f"  WARNING: Could not write session_end event: {exc}")

    # Clear activeSession on cabinet
    try:
        db.collection("cabinet").document(CABINET_ID).update({
            "activeSession": firestore.DELETE_FIELD,
        })
    except Exception as exc:
        log(f"  WARNING: Could not clear activeSession: {exc}")

    # Update cooldown for recipients
    if role == "recipient":
        update_last_visit(uid)

    log(f"  SESSION CLOSED: {display_name} (reason: {reason}, delta: {total_delta:+d})")

    # Show goodbye screen briefly then return to idle
    goodbye_score = active_session.get("score", 0) + sum(session_score_deltas)
    _set_display(screen="goodbye", name=display_name, score=max(0, goodbye_score))
    active_session = None
    session_score_deltas = []

    def _back_to_idle():
        time.sleep(3)
        _set_display(screen="idle", countdown=0)
    threading.Thread(target=_back_to_idle, daemon=True).start()


def _recipient_countdown_display(uid, seconds):
    """Drive the OLED countdown for a recipient session.
    Runs in a daemon thread; stops if the session ends.
    """
    for remaining in range(seconds, 0, -1):
        # Stop if session closed
        if not active_session or active_session.get("uid") != uid:
            break
        _set_display(countdown=remaining)
        time.sleep(1)
    # Countdown over — show session screen without number
    if active_session and active_session.get("uid") == uid:
        _set_display(countdown=0)


# ==================================================================
# HEARTBEAT
# ==================================================================

def heartbeat_loop():
    """Update isOnline every 30 seconds and refresh capacity on OLED."""
    while True:
        try:
            db.collection("cabinet").document(CABINET_ID).set({
                "isOnline": True,
                "lastUpdated": firestore.SERVER_TIMESTAMP,
            }, merge=True)
            # Refresh capacity on idle screen
            cab = db.collection("cabinet").document(CABINET_ID).get()
            if cab.exists:
                d = cab.to_dict()
                cur = d.get("currentItemCount", 0)
                total = d.get("totalSlots", 20)
                _set_display(capacity=f"{cur}/{total}")
        except Exception as exc:
            log(f"  Heartbeat failed: {exc}")
        time.sleep(30)


# ==================================================================
# MAIN LOOP
# ==================================================================

def main():
    log("=" * 50)
    log("  SmartShelf Identity Node (Raspberry Pi)")
    log(f"  Cabinet: {CABINET_ID}")
    log(f"  Recipient cooldown: {RECIPIENT_COOLDOWN_HOURS}h")
    log(f"  Ban duration: {BAN_DURATION_DAYS} days")
    log("=" * 50)
    log("")
    log("Scanning for QR codes...")

    # Start heartbeat thread
    hb = threading.Thread(target=heartbeat_loop, daemon=True)
    hb.start()

    # Start OLED display thread
    if HAS_OLED:
        disp = threading.Thread(target=display_loop, daemon=True)
        disp.start()
        log("  OLED display started (128x64 SSD1306)")
    else:
        log("  OLED not available (running without display)")
    _set_display(screen="idle")

    last_qr = None
    last_scan_time = 0

    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, bbox, _ = detector.detectAndDecode(gray)

            if data and bbox is not None:
                current_time = time.time()

                if data != last_qr or current_time - last_scan_time > COOLDOWN:
                    last_qr = data
                    last_scan_time = current_time

                    log(f"QR Detected: {data}")
                    _set_display(screen="scanning")

                    # Look up user
                    docs = (
                        db.collection("users")
                        .where("qrCode", "==", data)
                        .limit(1)
                        .stream()
                    )
                    user_doc = None
                    user_data = None
                    for d in docs:
                        user_doc = d
                        user_data = d.to_dict()

                    if not user_doc:
                        log("  No user found with this QR code")
                        _set_display(screen="unknown")
                        def _unknown_timeout():
                            time.sleep(3)
                            if not active_session:
                                _set_display(screen="idle")
                        threading.Thread(target=_unknown_timeout, daemon=True).start()
                        continue

                    uid = user_doc.id
                    display_name = user_data.get("displayName", "Unknown")
                    role = user_data.get("role", "recipient")
                    score = user_data.get("score", 0)

                    log(f"  User: {display_name} (UID: {uid})")
                    log(f"  Role: {role} | Score: {score}")

                    # Same user scans again -> sign out
                    if active_session and active_session["uid"] == uid:
                        log("  Same user scanned again -> closing session")
                        close_session("manual_signout")
                        continue

                    # Different user while session active
                    if active_session:
                        log("  Different user -> closing previous session")
                        close_session("new_user_override")

                    # Check cooldowns and bans (recipients only)
                    allowed, reason = check_cooldown_and_ban(uid, role)
                    if not allowed:
                        log(f"  ACCESS DENIED: {reason}")
                        _set_display(screen="denied", name=display_name, message=reason)
                        def _denied_timeout():
                            time.sleep(4)
                            if not active_session:
                                _set_display(screen="idle")
                        threading.Thread(target=_denied_timeout, daemon=True).start()
                        try:
                            db.collection("events").add({
                                "type": "access_denied",
                                "userId": uid,
                                "userRole": role,
                                "timestamp": firestore.SERVER_TIMESTAMP,
                                "deviceId": CABINET_ID,
                                "reason": reason,
                            })
                        except Exception:
                            pass
                        continue

                    # Open session
                    open_session(uid, role, display_name, score)

            if cv2.waitKey(1) == ord("q"):
                break

    except KeyboardInterrupt:
        log("Shutting down...")
    finally:
        if active_session:
            close_session("shutdown")
        try:
            db.collection("cabinet").document(CABINET_ID).set({
                "isOnline": False,
                "activeSession": firestore.DELETE_FIELD,
            }, merge=True)
        except Exception:
            pass
        oled_clear()
        picam2.stop()
        log("Goodbye!")


if __name__ == "__main__":
    main()
