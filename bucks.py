"""Preston Bucks: a fake-money economy with Preston as a crooked bookmaker.

Pure bookkeeping - balances, dailies, markets, bets, settlement - with no
Discord or model code, so every number the bot announces comes from here.
State is one JSON file, written atomically after every change.

Odds are fractional ("7/1"): a winning stake returns stake + stake * 7/1,
minus the house cut on the winnings. The house always wins.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

START = 500
DAILY = 100
DAILY_S = 20 * 3600          # a "day" is 20 h, so a fixed routine is not punished
MIN_BET = 10
HOUSE_CUT = 0.05


def parse_odds(text: str) -> tuple[int, int]:
    """"7/1" -> (7, 1). Anything unusable is evens."""
    try:
        num, den = str(text).replace(" ", "").split("/", 1)
        num_i, den_i = int(float(num)), int(float(den))
        if num_i > 0 and den_i > 0:
            return min(num_i, 100), min(den_i, 100)
    except (ValueError, TypeError):
        pass
    return 1, 1


class Bank:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data = {"wallets": {}, "markets": {}, "next_id": 1}
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- wallets -----------------------------------------------------------

    def _wallet(self, uid: int) -> dict:
        w = self.data["wallets"].setdefault(str(uid), {"balance": START, "last_daily": 0,
                                                       "won": 0, "lost": 0, "bailouts": 0})
        return w

    def remember_name(self, uid: int, name: str) -> None:
        """The bot has no member cache (no members intent), so the leaderboard
        showed everyone as "somebody who left". Names are kept as people play."""
        with self._lock:
            w = self._wallet(uid)
            if name and w.get("name") != name:
                w["name"] = name
                self._save()

    def balance(self, uid: int) -> int:
        with self._lock:
            w = self._wallet(uid)
            self._save()
            return int(w["balance"])

    def daily(self, uid: int) -> tuple[int, float, bool]:
        """(amount granted or 0, seconds until the next one, was it a bailout)."""
        with self._lock:
            w = self._wallet(uid)
            wait = w["last_daily"] + DAILY_S - time.time()
            if wait > 0:
                return 0, wait, False
            bailout = w["balance"] < MIN_BET
            w["balance"] += DAILY
            w["last_daily"] = time.time()
            if bailout:
                w["bailouts"] = w.get("bailouts", 0) + 1
            self._save()
            return DAILY, DAILY_S, bailout

    def leaderboard(self, n: int = 10) -> list[tuple[int, dict]]:
        with self._lock:
            rows = sorted(self.data["wallets"].items(), key=lambda kv: kv[1]["balance"], reverse=True)
            return [(int(uid), dict(w)) for uid, w in rows[:n]]

    def rank(self, uid: int) -> tuple[int, int]:
        with self._lock:
            order = sorted(self.data["wallets"], key=lambda k: self.data["wallets"][k]["balance"], reverse=True)
            return (order.index(str(uid)) + 1 if str(uid) in order else len(order) + 1), len(order)

    # -- markets -------------------------------------------------------------

    def create_market(self, question: str, options: list[tuple[str, str]], *, subject_uid: int | None,
                      channel_id: int, closes_at: float, source_url: str = "", source_channel: int = 0,
                      creator_uid: int | None = None) -> dict:
        with self._lock:
            mid = f"M{self.data['next_id']}"
            self.data["next_id"] += 1
            market = {
                "id": mid, "question": question[:200],
                "options": [{"label": label[:60], "odds": "%d/%d" % parse_odds(odds)} for label, odds in options[:4]],
                "subject_uid": subject_uid, "creator_uid": creator_uid,
                "channel_id": channel_id, "message_id": 0,
                "source_url": source_url, "source_channel": source_channel,
                "created": time.time(), "closes_at": closes_at,
                "status": "open", "bets": [], "result": None,
            }
            self.data["markets"][mid] = market
            self._save()
            return dict(market)

    def set_message(self, mid: str, message_id: int) -> None:
        with self._lock:
            if mid in self.data["markets"]:
                self.data["markets"][mid]["message_id"] = message_id
                self._save()

    def get(self, mid: str) -> dict | None:
        with self._lock:
            m = self.data["markets"].get(mid)
            return json.loads(json.dumps(m)) if m else None

    def markets(self, *statuses: str) -> list[dict]:
        with self._lock:
            return [json.loads(json.dumps(m)) for m in self.data["markets"].values()
                    if not statuses or m["status"] in statuses]

    def set_status(self, mid: str, status: str, **extra) -> None:
        with self._lock:
            m = self.data["markets"].get(mid)
            if m:
                m["status"] = status
                m.update(extra)
                self._save()

    def place_bet(self, mid: str, uid: int, option: int, amount: int) -> tuple[bool, str]:
        with self._lock:
            m = self.data["markets"].get(mid)
            if m is None:
                return False, "That market does not exist."
            if m["status"] != "open" or time.time() >= m["closes_at"]:
                return False, "Betting on that is closed."
            if not 0 <= option < len(m["options"]):
                return False, "No such option."
            w = self._wallet(uid)
            if amount < MIN_BET:
                return False, f"Minimum bet is {MIN_BET} PB."
            if amount > w["balance"]:
                return False, f"You have {w['balance']} PB. You cannot bet money you do not have. This is not a car loan."
            w["balance"] -= amount
            m["bets"].append({"uid": uid, "option": option, "amount": amount, "at": time.time()})
            self._save()
            return True, f"{amount} PB on **{m['options'][option]['label']}** at {m['options'][option]['odds']}. Balance {w['balance']} PB."

    def pools(self, m: dict) -> list[int]:
        totals = [0] * len(m["options"])
        for b in m["bets"]:
            totals[b["option"]] += b["amount"]
        return totals

    def settle(self, mid: str, winner: int | None) -> dict:
        """Pay out (winner index) or refund everyone (None = void). Returns
        {"winners": [(uid, stake, paid)], "losers": [(uid, stake)], "house": int}."""
        with self._lock:
            m = self.data["markets"].get(mid)
            out = {"winners": [], "losers": [], "house": 0}
            if m is None or m["status"] in ("settled", "void"):
                return out
            for b in m["bets"]:
                w = self._wallet(b["uid"])
                if winner is None:
                    w["balance"] += b["amount"]
                    continue
                if b["option"] == winner:
                    num, den = parse_odds(m["options"][winner]["odds"])
                    profit = b["amount"] * num / den
                    cut = int(round(profit * HOUSE_CUT))
                    paid = int(b["amount"] + profit - cut)
                    w["balance"] += paid
                    w["won"] = w.get("won", 0) + paid - b["amount"]
                    out["winners"].append((b["uid"], b["amount"], paid))
                    out["house"] += cut
                else:
                    w["lost"] = w.get("lost", 0) + b["amount"]
                    out["losers"].append((b["uid"], b["amount"]))
                    out["house"] += b["amount"]
            m["status"] = "void" if winner is None else "settled"
            m["result"] = winner
            m["settled_at"] = time.time()
            self._save()
            return out

    def recent_auto_markets(self, since: float) -> list[dict]:
        with self._lock:
            return [dict(m) for m in self.data["markets"].values()
                    if m["created"] >= since and m.get("subject_uid") and not m.get("creator_uid")]
