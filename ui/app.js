const X = 1;
const O = -1;
const BLANK = 0;

const state = {
  n: 5,
  board: [],
  player: O,
  human: X,
  agent: O,
  simulations: 120,
  terminal: false,
  winner: BLANK,
  lastMove: null,
  modelAvailable: true,
  training: false,
  trainPoll: null,
  pendingAnim: null,
};

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const substatusEl = document.getElementById("substatus");

const sizeEl = document.getElementById("boardSize");
const playAsEl = document.getElementById("playAs");
const simsEl = document.getElementById("simulations");
const newGameEl = document.getElementById("newGame");
const trainAgentEl = document.getElementById("trainAgent");

function setStatus(main, sub) {
  statusEl.textContent = main;
  substatusEl.textContent = sub || "";
}

function cellKey(r, c) {
  return `${r},${c}`;
}

function getPossibleMoves(board, player) {
  const moves = [];
  for (let r = 0; r < state.n; r += 1) {
    for (let c = 0; c < state.n; c += 1) {
      if (board[r][c] !== player) continue;
      if (player === O) {
        if (r > 0) {
          if (c > 0 && board[r - 1][c - 1] === X) moves.push([r, c, r - 1, c - 1]);
          if (board[r - 1][c] === BLANK) moves.push([r, c, r - 1, c]);
          if (c < state.n - 1 && board[r - 1][c + 1] === X) moves.push([r, c, r - 1, c + 1]);
        }
      } else {
        if (r < state.n - 1) {
          if (c > 0 && board[r + 1][c - 1] === O) moves.push([r, c, r + 1, c - 1]);
          if (board[r + 1][c] === BLANK) moves.push([r, c, r + 1, c]);
          if (c < state.n - 1 && board[r + 1][c + 1] === O) moves.push([r, c, r + 1, c + 1]);
        }
      }
    }
  }
  return moves;
}

function renderBoard() {
  boardEl.innerHTML = "";
  boardEl.style.setProperty("--cell", state.n > 6 ? "48px" : "64px");
  boardEl.style.gridTemplateColumns = `repeat(${state.n}, var(--cell))`;

  const moves =
    state.player === state.human && !state.terminal && state.modelAvailable
      ? getPossibleMoves(state.board, state.player)
      : [];
  const targets = new Set(moves.map((m) => cellKey(m[2], m[3])));

  for (let r = 0; r < state.n; r += 1) {
    for (let c = 0; c < state.n; c += 1) {
      const square = document.createElement("div");
      square.className = `square ${(r + c) % 2 === 0 ? "light" : "dark"}`;
      square.dataset.row = r;
      square.dataset.col = c;

      if (targets.has(cellKey(r, c))) square.classList.add("target");
      if (state.lastMove && (state.lastMove[0] === r && state.lastMove[1] === c || state.lastMove[2] === r && state.lastMove[3] === c)) {
        square.classList.add("last");
      }

      square.addEventListener("dragover", (event) => {
        if (!state.terminal && state.player === state.human) {
          event.preventDefault();
        }
      });

      square.addEventListener("dragenter", (event) => {
        if (!state.terminal && state.player === state.human) {
          event.preventDefault();
          square.classList.add("drop-target");
        }
      });

      square.addEventListener("dragleave", () => {
        square.classList.remove("drop-target");
      });

      square.addEventListener("drop", (event) => {
        event.preventDefault();
        if (state.terminal || state.player !== state.human) return;
        square.classList.remove("drop-target");
        const data = event.dataTransfer.getData("text/plain");
        if (!data) return;
        const [sr, sc] = data.split(",").map(Number);
        const move = [sr, sc, r, c];
        const legal = moves.some((m) => m.join(",") === move.join(","));
        if (!legal) {
          setStatus("Illegal move", "Try a forward step or diagonal capture.");
          return;
        }
        applyHumanMove(move);
      });

      const piece = state.board[r][c];
      if (piece !== BLANK) {
        const token = document.createElement("div");
        token.className = `piece ${piece === X ? "x" : "o"}`;
        token.textContent = piece === X ? "X" : "O";
        if (
          piece !== state.human ||
          state.player !== state.human ||
          state.terminal ||
          !state.modelAvailable
        ) {
          token.classList.add("disabled");
        } else {
          token.setAttribute("draggable", "true");
          token.addEventListener("dragstart", (event) => {
            event.dataTransfer.setData("text/plain", `${r},${c}`);
            token.classList.add("lift");
          });
          token.addEventListener("dragend", () => {
            token.classList.remove("lift");
          });
        }
        square.appendChild(token);
      }

      boardEl.appendChild(square);
    }
  }
}

async function api(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function setTurnStatus() {
  if (state.terminal) {
    const winner = state.winner === state.human ? "You win" : "Agent wins";
    setStatus(winner, "Game over.");
    return;
  }
  if (state.player === state.human) {
    setStatus("Your move", "Drag a pawn to a highlighted square.");
  } else {
    setStatus("Agent thinking", "Computing next move...");
  }
}

async function applyHumanMove(move) {
  try {
    const boardSnapshot = state.board.map((row) => row.slice());
    const playerSnapshot = state.player;
    applyMoveLocally(move, true);
    setStatus("Submitting move", "Waiting for agent response...");
    const data = await api("/api/step", {
      n: state.n,
      board: boardSnapshot,
      player: playerSnapshot,
      move,
      simulations: state.simulations,
      agent_player: state.agent,
    });
    syncFromServer(data, move);
    setTurnStatus();
    renderBoard();
    animatePending();
  } catch (err) {
    setStatus("Move rejected", err.message);
  }
}

async function agentAutoMove() {
  if (state.terminal || state.player !== state.agent || !state.modelAvailable) return;
  try {
    setStatus("Agent thinking", "Computing next move...");
    const data = await api("/api/agent_move", {
      n: state.n,
      board: state.board,
      player: state.player,
      simulations: state.simulations,
      agent_player: state.agent,
    });
    await sleep(220);
    syncFromServer(data, data.agent_move);
    setTurnStatus();
    renderBoard();
    animatePending();
  } catch (err) {
    setStatus("Agent error", err.message);
  }
}

async function checkModel() {
  try {
    const data = await api("/api/model_status", { n: state.n });
    state.modelAvailable = data.available;
    state.training = data.training;
    trainAgentEl.classList.toggle("hidden", state.modelAvailable || state.training);
    if (!state.modelAvailable && !state.training) {
      setStatus("No trained model", "Click Train Agent to create one.");
    } else if (state.training) {
      setStatus("Training in progress", "Leave this tab open or check back later.");
    }
  } catch (err) {
    setStatus("Model check failed", err.message);
  }
}

async function startTraining() {
  try {
    setStatus("Training started", "This can take several minutes.");
    const data = await api("/api/train", { n: state.n });
    if (data.training) {
      state.training = true;
      trainAgentEl.classList.add("hidden");
      if (state.trainPoll) clearInterval(state.trainPoll);
      state.trainPoll = setInterval(async () => {
        try {
          const status = await api("/api/train_status", { n: state.n });
          if (!status.training) {
            clearInterval(state.trainPoll);
            state.trainPoll = null;
            await checkModel();
            if (state.modelAvailable) {
              await newGame();
            }
          }
        } catch (err) {
          setStatus("Training status error", err.message);
        }
      }, 5000);
    }
  } catch (err) {
    setStatus("Training failed", err.message);
  }
}

async function newGame() {
  state.n = Number(sizeEl.value);
  state.human = Number(playAsEl.value);
  state.agent = state.human === X ? O : X;
  state.simulations = Number(simsEl.value);
  state.terminal = false;
  state.winner = BLANK;
  state.lastMove = null;

  try {
    setStatus("Starting", "Preparing board...");
    const data = await api("/api/new_game", { n: state.n });
    state.board = data.board;
    state.player = data.player;
    renderBoard();
    await checkModel();
    if (state.modelAvailable && !state.training) {
      setTurnStatus();
      await agentAutoMove();
    }
  } catch (err) {
    setStatus("Unable to start", err.message);
  }
}

function applyMoveLocally(move, markLast) {
  const [r, c, nr, nc] = move;
  const piece = state.board[r][c];
  state.board[r][c] = BLANK;
  state.board[nr][nc] = piece;
  state.player = state.player === X ? O : X;
  if (markLast) {
    state.lastMove = move;
  }
  renderBoard();
  animateMove(move);
}

function syncFromServer(data, fallbackMove) {
  state.board = data.board;
  state.player = data.player;
  state.terminal = data.terminal;
  state.winner = data.winner;
  state.lastMove = data.agent_move || fallbackMove || null;
  state.pendingAnim = data.agent_move || null;
}

function animateMove(move) {
  if (!move) return;
  const [, , tr, tc] = move;
  const target = boardEl.querySelector(`.square[data-row="${tr}"][data-col="${tc}"] .piece`);
  if (!target) return;
  target.classList.add("fade-in");
  setTimeout(() => target.classList.remove("fade-in"), 250);
}

function animatePending() {
  if (!state.pendingAnim) return;
  animateMove(state.pendingAnim);
  state.pendingAnim = null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

newGameEl.addEventListener("click", () => newGame());
trainAgentEl.addEventListener("click", () => startTraining());

sizeEl.addEventListener("change", () => newGame());
playAsEl.addEventListener("change", () => newGame());
simsEl.addEventListener("change", () => {
  state.simulations = Number(simsEl.value);
});

newGame();
