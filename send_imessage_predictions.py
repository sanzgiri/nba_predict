#!/usr/bin/env python3
"""
Send today's predictions via iMessage (or SMS fallback if configured).
"""

import argparse
import csv
import os
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send predictions via Messages.")
    parser.add_argument(
        "--to",
        default=os.getenv("IMESSAGE_TO", ""),
        help="Comma/space separated list of phone numbers.",
    )
    parser.add_argument(
        "--predictions-file",
        default="",
        help="Optional predictions CSV path (defaults to today's file).",
    )
    parser.add_argument(
        "--message-file",
        default="",
        help="Optional output file path for the message body.",
    )
    return parser.parse_args()


def _parse_numbers(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s]+", raw.strip())
    return [part for part in parts if part]


def _to_float(value: object, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        text = str(value).strip()
        if not text:
            return fallback
        return float(text)
    except (TypeError, ValueError):
        return fallback


def _build_message(pred_file: Path) -> str:
    today = date.today().strftime("%Y-%m-%d")
    header = f"NBA Predictions {today}"
    if not pred_file.exists():
        return f"{header}\nNo games today."

    with pred_file.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return f"{header}\nNo games today."

    lines = [header]
    for row in rows:
        home = (row.get("home_team") or "").strip()
        away = (row.get("away_team") or "").strip()
        home_score = _to_float(row.get("predicted_home_score"), 0.0)
        away_score = _to_float(row.get("predicted_away_score"), 0.0)
        home_prob = _to_float(row.get("home_win_probability"), 0.5)

        if home_prob >= 0.5:
            winner = home
            win_prob = home_prob
        else:
            winner = away
            win_prob = 1.0 - home_prob

        lines.append(
            f"{away} @ {home}: {winner} {home_score:.1f}-{away_score:.1f} ({win_prob * 100:.1f}%)"
        )
    return "\n".join(lines)


def _send_message(number: str, message_file: Path) -> None:
    message_path = str(message_file).replace('"', '\\"')
    number_safe = number.replace('"', '\\"')
    script = f'''
set targetNumber to "{number_safe}"
set messageText to (read POSIX file "{message_path}")

tell application "Messages"
  set targetService to missing value
  try
    set targetService to 1st service whose service type is iMessage
  on error
    try
      set targetService to 1st service whose service type is SMS
    end try
  end try

  if targetService is missing value then error "No Messages service available"
  set targetBuddy to buddy targetNumber of targetService
  send messageText to targetBuddy
end tell
'''
    subprocess.run(["osascript", "-e", script], check=True)


def main() -> None:
    args = parse_args()
    numbers = _parse_numbers(args.to)
    if not numbers:
        raise SystemExit("No recipient provided. Set IMESSAGE_TO or pass --to.")

    today_key = date.today().strftime("%Y%m%d")
    pred_file = (
        Path(args.predictions_file)
        if args.predictions_file
        else Path("predictions") / f"predictions_{today_key}.csv"
    )
    pred_file = pred_file.expanduser().resolve()
    message = _build_message(pred_file)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    message_file = (
        Path(args.message_file)
        if args.message_file
        else logs_dir / f"imessage_predictions_{today_key}.txt"
    )
    message_file = message_file.expanduser().resolve()
    message_file.write_text(message)

    for number in numbers:
        _send_message(number, message_file)

    print(f"Sent predictions to {', '.join(numbers)} via Messages.")


if __name__ == "__main__":
    main()
