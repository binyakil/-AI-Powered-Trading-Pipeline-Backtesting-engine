"""
Julaba Security Monitor
-----------------------
Tracks visitor access, failed logins, and device fingerprints.
Provides intrusion detection and alerts.
"""

import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger("Julaba.Security")

# Storage files
SECURITY_LOG_FILE = Path(__file__).parent / "security_log.json"
BLOCKED_IPS_FILE = Path(__file__).parent / "blocked_ips.json"
WHITELIST_FILE = Path(__file__).parent / "ip_whitelist.json"
KNOWN_DEVICES_FILE = Path(__file__).parent / "known_devices.json"

# Settings
MAX_FAILED_ATTEMPTS = 15  # Block IP after this many failures
BLOCK_DURATION_HOURS = 24  # How long to block
MAX_LOG_ENTRIES = 1000  # Keep last N entries
ALERT_ON_NEW_DEVICE = True
ALERT_ON_BLOCKED_ACCESS = True


@dataclass
class AccessLog:
    """Single access log entry."""
    timestamp: str
    ip: str
    endpoint: str
    method: str
    user_agent: str
    status: str  # 'success', 'failed', 'blocked'
    device_fingerprint: str
    country: str = "Unknown"
    city: str = "Unknown"
    is_known_device: bool = False
    details: str = ""


class SecurityMonitor:
    """Comprehensive security monitoring system."""
    
    def __init__(self):
        self.access_logs: List[Dict] = []
        self.blocked_ips: Dict[str, Dict] = {}  # ip -> {blocked_at, reason, expires}
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)  # ip -> [timestamps]
        self.ip_whitelist: List[str] = []
        self.known_devices: Dict[str, Dict] = {}  # fingerprint -> {first_seen, last_seen, ip, user_agent}
        self.never_block_ips: set = set()  # Owner IPs that should never be blocked
        self.telegram_callback = None  # Set by bot.py
        
        self._load_data()
        logger.info("Security Monitor initialized")
    
    def _load_data(self):
        """Load persisted security data."""
        try:
            if SECURITY_LOG_FILE.exists():
                with open(SECURITY_LOG_FILE, 'r') as f:
                    self.access_logs = json.load(f)
                logger.debug(f"Loaded {len(self.access_logs)} access log entries")
        except Exception as e:
            logger.error(f"Failed to load security logs: {e}")
            self.access_logs = []
        
        try:
            if BLOCKED_IPS_FILE.exists():
                with open(BLOCKED_IPS_FILE, 'r') as f:
                    self.blocked_ips = json.load(f)
                # Clean expired blocks
                self._clean_expired_blocks()
        except Exception as e:
            logger.error(f"Failed to load blocked IPs: {e}")
            self.blocked_ips = {}
        
        try:
            if WHITELIST_FILE.exists():
                with open(WHITELIST_FILE, 'r') as f:
                    self.ip_whitelist = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load IP whitelist: {e}")
            self.ip_whitelist = []
        
        try:
            if KNOWN_DEVICES_FILE.exists():
                with open(KNOWN_DEVICES_FILE, 'r') as f:
                    self.known_devices = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load known devices: {e}")
            self.known_devices = {}
    
    def _save_data(self):
        """Persist security data to disk."""
        try:
            # Trim logs to max size
            if len(self.access_logs) > MAX_LOG_ENTRIES:
                self.access_logs = self.access_logs[-MAX_LOG_ENTRIES:]
            
            with open(SECURITY_LOG_FILE, 'w') as f:
                json.dump(self.access_logs, f, indent=2)
            
            with open(BLOCKED_IPS_FILE, 'w') as f:
                json.dump(self.blocked_ips, f, indent=2)
            
            with open(WHITELIST_FILE, 'w') as f:
                json.dump(self.ip_whitelist, f, indent=2)
            
            with open(KNOWN_DEVICES_FILE, 'w') as f:
                json.dump(self.known_devices, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save security data: {e}")
    
    def _clean_expired_blocks(self):
        """Remove expired IP blocks."""
        now = time.time()
        expired = [ip for ip, data in self.blocked_ips.items() 
                   if data.get('expires', 0) < now]
        for ip in expired:
            del self.blocked_ips[ip]
            logger.info(f"IP block expired: {ip}")
    
    def generate_device_fingerprint(self, user_agent: str, ip: str, accept_language: str = "", 
                                     screen_info: str = "") -> str:
        """Generate a device fingerprint from available data."""
        data = f"{user_agent}|{accept_language}|{screen_info}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked."""
        self._clean_expired_blocks()
        return ip in self.blocked_ips
    
    def is_ip_whitelisted(self, ip: str) -> bool:
        """Check if an IP is in the whitelist."""
        if not self.ip_whitelist:
            return True  # No whitelist = allow all
        return ip in self.ip_whitelist
    
    def add_to_whitelist(self, ip: str, note: str = "") -> bool:
        """Add an IP to the whitelist."""
        if ip not in self.ip_whitelist:
            self.ip_whitelist.append(ip)
            self._save_data()
            logger.info(f"Added to whitelist: {ip} ({note})")
            return True
        return False
    
    def remove_from_whitelist(self, ip: str) -> bool:
        """Remove an IP from the whitelist."""
        if ip in self.ip_whitelist:
            self.ip_whitelist.remove(ip)
            self._save_data()
            logger.info(f"Removed from whitelist: {ip}")
            return True
        return False
    
    def add_never_block(self, ip: str):
        """Add an IP to the never-block set (owner immunity)."""
        self.never_block_ips.add(ip)
    
    def block_ip(self, ip: str, reason: str, duration_hours: int = BLOCK_DURATION_HOURS):
        """Block an IP address (unless immune)."""
        if ip in self.never_block_ips:
            logger.info(f"Skipping block for immune IP: {ip}")
            return
        expires = time.time() + (duration_hours * 3600)
        self.blocked_ips[ip] = {
            'blocked_at': datetime.now().isoformat(),
            'reason': reason,
            'expires': expires,
            'expires_human': datetime.fromtimestamp(expires).isoformat()
        }
        self._save_data()
        logger.warning(f"Blocked IP: {ip} - Reason: {reason}")
        
        # Send alert
        if self.telegram_callback and ALERT_ON_BLOCKED_ACCESS:
            self._send_alert(f"🚫 IP BLOCKED: {ip}\nReason: {reason}\nDuration: {duration_hours}h")
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        if ip in self.blocked_ips:
            del self.blocked_ips[ip]
            self._save_data()
            logger.info(f"Unblocked IP: {ip}")
            return True
        return False
    
    def record_failed_login(self, ip: str, endpoint: str, user_agent: str):
        """Record a failed login attempt."""
        now = time.time()
        
        # Clean old attempts (older than 1 hour)
        self.failed_attempts[ip] = [t for t in self.failed_attempts[ip] if now - t < 3600]
        self.failed_attempts[ip].append(now)
        
        attempts = len(self.failed_attempts[ip])
        logger.warning(f"Failed login attempt #{attempts} from {ip}")
        
        # Check if should block
        if attempts >= MAX_FAILED_ATTEMPTS:
            self.block_ip(ip, f"Too many failed login attempts ({attempts})")
            self.failed_attempts[ip] = []  # Reset counter
    
    def log_access(self, ip: str, endpoint: str, method: str, user_agent: str,
                   status: str, fingerprint: str = "", details: str = "",
                   accept_language: str = "") -> Dict:
        """Log an access attempt."""
        
        # Generate fingerprint if not provided
        if not fingerprint:
            fingerprint = self.generate_device_fingerprint(user_agent, ip, accept_language)
        
        # Check if known device
        is_known = fingerprint in self.known_devices
        is_new_device = False
        
        if not is_known and status == 'success':
            # New device - register it
            is_new_device = True
            self.known_devices[fingerprint] = {
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'ip': ip,
                'user_agent': user_agent,
                'access_count': 1
            }
            logger.info(f"New device registered: {fingerprint[:8]}... from {ip}")
            
            # Alert on new device
            if self.telegram_callback and ALERT_ON_NEW_DEVICE:
                self._send_alert(
                    f"🆕 NEW DEVICE DETECTED\n"
                    f"IP: {ip}\n"
                    f"Fingerprint: {fingerprint[:8]}...\n"
                    f"User-Agent: {user_agent[:50]}..."
                )
        elif is_known:
            # Update existing device
            self.known_devices[fingerprint]['last_seen'] = datetime.now().isoformat()
            self.known_devices[fingerprint]['ip'] = ip
            self.known_devices[fingerprint]['access_count'] = \
                self.known_devices[fingerprint].get('access_count', 0) + 1
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'ip': ip,
            'endpoint': endpoint,
            'method': method,
            'user_agent': user_agent,
            'status': status,
            'device_fingerprint': fingerprint,
            'is_known_device': is_known,
            'is_new_device': is_new_device,
            'details': details
        }
        
        self.access_logs.append(log_entry)
        self._save_data()
        
        return log_entry
    
    def _send_alert(self, message: str):
        """Send alert via Telegram."""
        if self.telegram_callback:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_callback(f"🛡️ SECURITY ALERT\n\n{message}"),
                        loop
                    )
                except RuntimeError:
                    asyncio.run(self.telegram_callback(f"🛡️ SECURITY ALERT\n\n{message}"))
            except Exception as e:
                logger.error(f"Failed to send security alert: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        now = datetime.now()
        today = now.date().isoformat()
        
        # Count today's accesses
        today_logs = [l for l in self.access_logs if l['timestamp'].startswith(today)]
        
        # Count by status
        success_count = len([l for l in today_logs if l['status'] == 'success'])
        failed_count = len([l for l in today_logs if l['status'] == 'failed'])
        blocked_count = len([l for l in today_logs if l['status'] == 'blocked'])
        
        # Unique IPs today
        unique_ips_today = len(set(l['ip'] for l in today_logs))
        
        # Recent activity (last 10)
        recent = self.access_logs[-10:] if self.access_logs else []
        
        return {
            'total_logs': len(self.access_logs),
            'today_total': len(today_logs),
            'today_success': success_count,
            'today_failed': failed_count,
            'today_blocked': blocked_count,
            'unique_ips_today': unique_ips_today,
            'blocked_ips_count': len(self.blocked_ips),
            'whitelisted_ips_count': len(self.ip_whitelist),
            'known_devices_count': len(self.known_devices),
            'recent_activity': list(reversed(recent)),
            'blocked_ips': self.blocked_ips,
            'whitelisted_ips': self.ip_whitelist,
            'known_devices': self.known_devices
        }
    
    def get_access_logs(self, limit: int = 100, ip_filter: str = None, 
                        status_filter: str = None) -> List[Dict]:
        """Get filtered access logs."""
        logs = self.access_logs
        
        if ip_filter:
            logs = [l for l in logs if l['ip'] == ip_filter]
        
        if status_filter:
            logs = [l for l in logs if l['status'] == status_filter]
        
        return list(reversed(logs[-limit:]))
    
    def clear_logs(self, older_than_days: int = None):
        """Clear access logs."""
        if older_than_days:
            cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
            self.access_logs = [l for l in self.access_logs if l['timestamp'] > cutoff]
        else:
            self.access_logs = []
        self._save_data()
        logger.info(f"Cleared access logs (older_than_days={older_than_days})")


# Singleton instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get or create the security monitor singleton."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor
