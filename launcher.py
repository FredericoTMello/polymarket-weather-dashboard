#!/usr/bin/env python3
"""
Simple visual launcher for the Polymarket weather stack.

Run:
  python launcher.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Button, Frame, Label, StringVar, Tk
from tkinter.scrolledtext import ScrolledText


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
WEB_URL = "http://127.0.0.1:8090/dashboard.html"


class LauncherApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Polymarket Weather Launcher")
        self.root.geometry("980x620")

        self.monitor_proc: subprocess.Popen[str] | None = None
        self.web_proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

        self.status_var = StringVar()
        self.status_var.set("Ready")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top = Frame(self.root, padx=10, pady=10)
        top.pack(side=TOP, fill=X)

        Label(top, text="Actions", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        row1 = Frame(top)
        row1.pack(fill=X, pady=(8, 4))
        Button(row1, text="Validate OOS", width=20, command=self.run_validate).pack(side=LEFT, padx=(0, 6))
        Button(row1, text="Validate + Apply", width=20, command=self.run_validate_apply).pack(side=LEFT, padx=(0, 6))
        Button(row1, text="Monitor Once", width=20, command=self.run_monitor_once).pack(side=LEFT, padx=(0, 6))

        row2 = Frame(top)
        row2.pack(fill=X, pady=(4, 8))
        Button(row2, text="Start Monitor Loop", width=20, command=self.start_monitor_loop).pack(side=LEFT, padx=(0, 6))
        Button(row2, text="Stop Monitor Loop", width=20, command=self.stop_monitor_loop).pack(side=LEFT, padx=(0, 6))
        Button(row2, text="Start Web + Open", width=20, command=self.start_web_and_open).pack(side=LEFT, padx=(0, 6))
        Button(row2, text="Stop Web", width=20, command=self.stop_web).pack(side=LEFT, padx=(0, 6))

        status = Frame(self.root, padx=10)
        status.pack(fill=X)
        Label(status, textvariable=self.status_var, font=("Segoe UI", 10)).pack(anchor="w")

        log_wrap = Frame(self.root, padx=10, pady=10)
        log_wrap.pack(fill=BOTH, expand=True)
        self.log = ScrolledText(log_wrap, font=("Consolas", 10))
        self.log.pack(fill=BOTH, expand=True)
        self._append("Launcher started.")

    def _append(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert(END, f"[{ts}] {msg}\n")
        self.log.see(END)

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self._append(msg)

    def _run_command_background(self, args: list[str], title: str) -> None:
        def worker() -> None:
            self._set_status(f"Running: {title}")
            try:
                proc = subprocess.Popen(
                    [PYTHON_EXE, "server.py"] + args,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._append(line.rstrip())
                code = proc.wait()
                self._set_status(f"{title} finished (exit={code})")
            except Exception as exc:
                self._set_status(f"{title} failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def run_validate(self) -> None:
        self._run_command_background(["validate", "--days", "730", "--train-ratio", "0.7"], "Validate OOS")

    def run_validate_apply(self) -> None:
        self._run_command_background(
            ["validate", "--days", "730", "--train-ratio", "0.7", "--apply-best"],
            "Validate OOS + Apply",
        )

    def run_monitor_once(self) -> None:
        self._run_command_background(["monitor", "--once"], "Monitor Once")

    def start_monitor_loop(self) -> None:
        with self._lock:
            if self.monitor_proc and self.monitor_proc.poll() is None:
                self._set_status("Monitor loop already running.")
                return
            try:
                self.monitor_proc = subprocess.Popen(
                    [PYTHON_EXE, "server.py", "monitor"],
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                self._set_status(f"Failed to start monitor: {exc}")
                return
        self._set_status("Monitor loop started.")
        self._stream_process(self.monitor_proc, "monitor")

    def stop_monitor_loop(self) -> None:
        with self._lock:
            proc = self.monitor_proc
            self.monitor_proc = None
        if not proc or proc.poll() is not None:
            self._set_status("Monitor loop is not running.")
            return
        proc.terminate()
        self._set_status("Monitor loop stopped.")

    def start_web_and_open(self) -> None:
        with self._lock:
            if not self.web_proc or self.web_proc.poll() is not None:
                try:
                    self.web_proc = subprocess.Popen(
                        [PYTHON_EXE, "server.py", "web", "--bind", "127.0.0.1", "--port", "8090"],
                        cwd=ROOT_DIR,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    self._set_status("Web server started.")
                    self._stream_process(self.web_proc, "web")
                except Exception as exc:
                    self._set_status(f"Failed to start web server: {exc}")
                    return
            else:
                self._set_status("Web server already running.")
        webbrowser.open(WEB_URL, new=2)
        self._append(f"Opened browser: {WEB_URL}")

    def stop_web(self) -> None:
        with self._lock:
            proc = self.web_proc
            self.web_proc = None
        if not proc or proc.poll() is not None:
            self._set_status("Web server is not running.")
            return
        proc.terminate()
        self._set_status("Web server stopped.")

    def _stream_process(self, proc: subprocess.Popen[str], name: str) -> None:
        def worker() -> None:
            if not proc.stdout:
                return
            for line in proc.stdout:
                self._append(f"[{name}] {line.rstrip()}")
            code = proc.wait()
            self._append(f"[{name}] exited with code {code}")
            with self._lock:
                if name == "monitor" and self.monitor_proc is proc:
                    self.monitor_proc = None
                if name == "web" and self.web_proc is proc:
                    self.web_proc = None

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        self.stop_monitor_loop()
        self.stop_web()
        self.root.destroy()


def main() -> None:
    root = Tk()
    app = LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

