"""
Local UI server for playing Hexapawn against a trained agent.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, Any, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import hexapawn_core as core


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")

# Ensure weight lookups resolve to 2026/weights even if server is launched elsewhere.
core.WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

app = Flask(__name__, static_folder=UI_DIR, static_url_path="/ui")

_net_cache: Dict[int, Any] = {}
_train_processes: Dict[int, subprocess.Popen] = {}

TRAIN_DEFAULTS = {
    "iterations": 50,
    "games": 30,
    "simulations": 50,
    "eval_games": 30,
}


def _load_net(n: int):
    if n in _net_cache:
        return _net_cache[n]
    net = core.find_latest_model(n)
    if net is None:
        return None
    _net_cache[n] = net
    return net


def _agent_for(n: int, simulations: int) -> core.AlphaZeroAgent | None:
    net = _load_net(n)
    if net is None:
        return None
    return core.AlphaZeroAgent(net, simulations=simulations)


def _model_available(n: int) -> bool:
    if not os.path.exists(core.WEIGHTS_DIR):
        return False
    latest = os.path.join(core.WEIGHTS_DIR, f"hexapawn_n{n}_latest.pt")
    if os.path.exists(latest):
        return True
    prefix = f"hexapawn_n{n}_iter"
    for filename in os.listdir(core.WEIGHTS_DIR):
        if filename.startswith(prefix) and filename.endswith(".pt"):
            return True
    return False


def _training_running(n: int) -> bool:
    proc = _train_processes.get(n)
    if proc is None:
        return False
    return proc.poll() is None


def _state_from_payload(payload: Dict[str, Any]) -> Tuple[core.HexapawnState, int]:
    n = int(payload.get("n", 3))
    board = payload.get("board")
    player = int(payload.get("player", core.O))

    if not isinstance(board, list):
        raise ValueError("board must be a list")
    if len(board) != n:
        raise ValueError("board has wrong height")

    np_board = np.array(board, dtype=np.int8)
    if np_board.shape != (n, n):
        raise ValueError("board has wrong shape")

    state = core.HexapawnState.__new__(core.HexapawnState)
    state.n = n
    state.board = np_board
    state.player = player
    return state, n


def _winner_to_int(winner: int) -> int:
    return int(winner)


def _move_to_list(move) -> list[int]:
    # Moves can contain numpy scalar ints; coerce for JSON.
    r, c, nr, nc = move
    return [int(r), int(c), int(nr), int(nc)]


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/ui/<path:path>")
def ui_assets(path: str):
    return send_from_directory(UI_DIR, path)


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    payload = request.get_json(force=True) or {}
    n = int(payload.get("n", 3))
    state = core.HexapawnState(n)
    return jsonify(
        {
            "n": n,
            "board": state.board.tolist(),
            "player": int(state.player),
            "terminal": False,
            "winner": _winner_to_int(core.BLANK),
        }
    )


@app.route("/api/agent_move", methods=["POST"])
def api_agent_move():
    payload = request.get_json(force=True) or {}
    simulations = int(payload.get("simulations", 100))
    agent_player = int(payload.get("agent_player", core.O))

    try:
        state, n = _state_from_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if state.player != agent_player:
        return jsonify({"error": "Not agent's turn"}), 400

    agent = _agent_for(n, simulations)
    if agent is None:
        return jsonify({"error": f"No trained model found for {n}x{n}"}), 400

    move = agent.choose_move(state, temperature=0)
    state.make_move(move)

    terminal = state.is_terminal()
    winner = _winner_to_int(state.get_winner()) if terminal else _winner_to_int(core.BLANK)

    return jsonify(
        {
            "n": n,
            "board": state.board.tolist(),
            "player": int(state.player),
            "terminal": terminal,
            "winner": winner,
            "agent_move": _move_to_list(move),
        }
    )


@app.route("/api/step", methods=["POST"])
def api_step():
    payload = request.get_json(force=True) or {}
    simulations = int(payload.get("simulations", 100))
    agent_player = int(payload.get("agent_player", core.O))
    move = payload.get("move")

    try:
        state, n = _state_from_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not (isinstance(move, list) and len(move) == 4):
        return jsonify({"error": "move must be a list of 4 integers"}), 400

    move_tuple = tuple(int(x) for x in move)
    if move_tuple not in state.get_possible_moves():
        return jsonify({"error": "Illegal move"}), 400

    state.make_move(move_tuple)

    if state.is_terminal():
        return jsonify(
            {
                "n": n,
                "board": state.board.tolist(),
                "player": int(state.player),
                "terminal": True,
                "winner": _winner_to_int(state.get_winner()),
                "human_move": _move_to_list(move_tuple),
            }
        )

    agent_move = None
    if state.player == agent_player:
        agent = _agent_for(n, simulations)
        if agent is None:
            return jsonify({"error": f"No trained model found for {n}x{n}"}), 400
        agent_move = agent.choose_move(state, temperature=0)
        state.make_move(agent_move)

    terminal = state.is_terminal()
    winner = _winner_to_int(state.get_winner()) if terminal else _winner_to_int(core.BLANK)

    return jsonify(
        {
            "n": n,
            "board": state.board.tolist(),
            "player": int(state.player),
            "terminal": terminal,
            "winner": winner,
            "human_move": _move_to_list(move_tuple),
            "agent_move": _move_to_list(agent_move) if agent_move else None,
        }
    )


@app.route("/api/model_status", methods=["POST"])
def api_model_status():
    payload = request.get_json(force=True) or {}
    n = int(payload.get("n", 3))
    return jsonify(
        {
            "n": n,
            "available": _model_available(n),
            "training": _training_running(n),
        }
    )


@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(force=True) or {}
    n = int(payload.get("n", 3))

    if _model_available(n):
        return jsonify({"error": f"Model already exists for {n}x{n}"}), 400

    if _training_running(n):
        return jsonify({"training": True, "message": "Training already running"})

    iterations = int(payload.get("iterations", TRAIN_DEFAULTS["iterations"]))
    games = int(payload.get("games", TRAIN_DEFAULTS["games"]))
    simulations = int(payload.get("simulations", TRAIN_DEFAULTS["simulations"]))
    eval_games = int(payload.get("eval_games", TRAIN_DEFAULTS["eval_games"]))

    cmd = [
        sys.executable,
        "train.py",
        "--n",
        str(n),
        "--iterations",
        str(iterations),
        "--games",
        str(games),
        "--simulations",
        str(simulations),
        "--eval-games",
        str(eval_games),
    ]

    proc = subprocess.Popen(cmd, cwd=BASE_DIR)
    _train_processes[n] = proc

    return jsonify(
        {
            "training": True,
            "n": n,
            "pid": proc.pid,
            "config": {
                "iterations": iterations,
                "games": games,
                "simulations": simulations,
                "eval_games": eval_games,
            },
        }
    )


@app.route("/api/train_status", methods=["POST"])
def api_train_status():
    payload = request.get_json(force=True) or {}
    n = int(payload.get("n", 3))
    running = _training_running(n)
    return jsonify({"n": n, "training": running})


def main():
    parser = argparse.ArgumentParser(description="Hexapawn UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
