import { Chess } from "./vendor/chess.js";

const { invoke } = window.__TAURI__.core;

const PIECE_GLYPH = {
  wk: "♔", wq: "♕", wr: "♖", wb: "♗", wn: "♘", wp: "♙",
  bk: "♚", bq: "♛", br: "♜", bb: "♝", bn: "♞", bp: "♟",
};

// Difficulty -> MCTS simulations-per-move sent as "setoption name Simulations value <N>".
const DIFFICULTY = { easy: 10, medium: 50, hard: 200 };

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const trainLogEl = document.getElementById("train-log");
const colorButtons = document.querySelectorAll("[data-color]");
const difficultyButtons = document.querySelectorAll("[data-difficulty]");
const newGameBtn = document.getElementById("btn-new-game");
const resignBtn = document.getElementById("btn-resign");
const undoBtn = document.getElementById("btn-undo");
const selfplayBtn = document.getElementById("btn-selfplay");
const trainBtn = document.getElementById("btn-train");
const reloadBtn = document.getElementById("btn-reload");
const historyBtn = document.getElementById("btn-history");
const historyTableEl = document.getElementById("history-table");

const game = new Chess();
const moveHistory = []; // UCI strings, sent to the engine as "position startpos moves ..."
let selected = null; // algebraic square string, e.g. "e2"
let humanColor = "w"; // "w" or "b" -- which side the human plays in normal (non-selfplay) mode
let difficulty = "medium";
let resigned = false;

// Single exclusive busy state covers "engine thinking on a human move",
// self-play spectator mode, and training -- only one of these makes
// sense at a time, and every user-triggered action below checks this
// instead of a separate flag per mode.
let mode = "idle"; // "idle" | "thinking" | "selfplay" | "train"
let selfplayStopRequested = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function uciToMoveInput(uci) {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci.slice(4) : undefined,
  };
}

function render() {
  boardEl.innerHTML = "";
  const board = game.board(); // rank 8 -> rank 1, file a -> file h
  const legalTargets = selected
    ? game.moves({ square: selected, verbose: true }).map((m) => m.to)
    : [];

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[r][f];
      const file = "abcdefgh"[f];
      const rank = 8 - r;
      const square = `${file}${rank}`;

      const el = document.createElement("div");
      el.className = "square " + ((r + f) % 2 === 0 ? "light" : "dark");
      if (piece) el.textContent = PIECE_GLYPH[piece.color + piece.type];
      if (square === selected) el.classList.add("selected");
      if (legalTargets.includes(square)) el.classList.add("legal");

      el.addEventListener("click", () => onSquareClick(square));
      boardEl.appendChild(el);
    }
  }
  updateControlsEnabled();
}

function isOver() {
  return resigned || game.isGameOver();
}

async function onSquareClick(square) {
  if (mode !== "idle" || isOver()) return;
  if (game.turn() !== humanColor) return;

  const piece = game.get(square);

  if (selected) {
    const legal = game.moves({ square: selected, verbose: true });
    const target = legal.find((m) => m.to === square);
    if (target) {
      const move = game.move({ from: selected, to: square, promotion: "q" });
      moveHistory.push(move.lan ?? `${selected}${square}`);
      selected = null;
      render();
      await requestEngineMove();
      return;
    }
    // clicking a different own piece re-selects instead of illegal move
    selected = piece && piece.color === humanColor ? square : null;
    render();
    return;
  }

  if (piece && piece.color === humanColor) {
    selected = square;
    render();
  }
}

// Asks the engine for its move in the current position and applies it.
// Used both after a human move and, when the human plays Black, to make
// the engine's opening move on a fresh game.
async function requestEngineMove() {
  if (isOver()) {
    setStatus(gameOverMessage());
    return;
  }
  mode = "thinking";
  updateControlsEnabled();
  setStatus("Рушій думає...");
  await reliableSend(`position startpos moves ${moveHistory.join(" ")}`);
  await reliableSend("go");
  const bestmoveLine = await waitForLine((l) => l.startsWith("bestmove "));
  const uci = bestmoveLine.split(" ")[1];
  const move = game.move(uciToMoveInput(uci));
  moveHistory.push(move.lan ?? uci);
  mode = "idle";
  render();
  setStatus(isOver() ? gameOverMessage() : "Твій хід");
}

function gameOverMessage() {
  if (resigned) return "Ти здався.";
  if (game.isCheckmate()) return `Мат! Переміг ${game.turn() === "w" ? "чорний" : "білий"}.`;
  if (game.isDraw()) return "Нічия.";
  return "Гру завершено.";
}

async function engineSend(line) {
  await invoke("engine_send", { line });
}

// The sidecar can die between commands (observed live: "Broken pipe
// (os error 32)" on engine_send after it had answered readyok
// earlier -- once dead, engine_start's own idempotency guard used to
// mean it never respawned; fixed on the Rust side to clear its slot
// on death, so a fresh engine_start here actually spawns a new
// process). Every position command already sends the FULL move
// history, not incremental state, so a freshly respawned+
// re-handshaken engine picks the game back up correctly.
async function reliableSend(line) {
  try {
    await engineSend(line);
  } catch (err) {
    setStatus("Двигун відключився, перезапускаю...");
    await invoke("engine_start");
    await handshake();
    await applyDifficulty();
    await engineSend(line);
  }
}

async function handshake() {
  await engineSend("uci");
  await waitForLine((l) => l === "uciok");
  await engineSend("isready");
  await waitForLine((l) => l === "readyok");
}

async function waitForLine(predicate, timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const lines = await invoke("engine_drain");
    for (const line of lines) {
      if (predicate(line)) return line;
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error("Тайм-аут очікування відповіді рушія");
}

// Like waitForLine, but calls onLine for every line seen (not just the
// terminal one) -- used for selfplay/train streams where intermediate
// lines (selfplaymove/trainprogress) matter, not just the final result.
async function pollUntil(matchPredicate, onLine, timeoutMs = 600000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const lines = await invoke("engine_drain");
    for (const line of lines) {
      onLine(line);
      if (matchPredicate(line)) return line;
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error("Тайм-аут очікування відповіді рушія");
}

async function applyDifficulty() {
  await engineSend(`setoption name Simulations value ${DIFFICULTY[difficulty]}`);
}

function updateControlsEnabled() {
  const busy = mode !== "idle";
  const over = isOver();
  newGameBtn.disabled = busy;
  resignBtn.disabled = busy || over;
  undoBtn.disabled = busy || moveHistory.length < 2;
  selfplayBtn.disabled = mode === "thinking" || mode === "train";
  selfplayBtn.textContent = mode === "selfplay" ? "Зупинити самогру" : "Дивитись самогру";
  // No "stop training" affordance in this scope (train runs one bounded
  // cycle to completion) -- disabled for the whole busy window, not just
  // while some OTHER mode is active.
  trainBtn.disabled = busy;
  reloadBtn.disabled = busy;
  historyBtn.disabled = busy;
  colorButtons.forEach((b) => (b.disabled = busy));
  difficultyButtons.forEach((b) => {
    b.disabled = busy;
    b.classList.toggle("active", b.dataset.difficulty === difficulty);
  });
  colorButtons.forEach((b) => b.classList.toggle("active", b.dataset.color === humanColor));
}

async function newGame() {
  if (mode !== "idle") return;
  mode = "thinking"; // blocks other controls during ucinewgame/setoption round-trip
  updateControlsEnabled();
  game.reset();
  moveHistory.length = 0;
  selected = null;
  resigned = false;
  render();
  await reliableSend("ucinewgame");
  await applyDifficulty();
  if (humanColor === "b") {
    await requestEngineMove(); // sets mode back to "idle" itself when done
  } else {
    mode = "idle";
    render();
    setStatus("Твій хід");
  }
}

function resign() {
  if (mode !== "idle" || isOver()) return;
  resigned = true;
  setStatus(gameOverMessage());
  updateControlsEnabled();
}

function undo() {
  if (mode !== "idle" || moveHistory.length < 2) return;
  game.undo();
  game.undo();
  moveHistory.pop();
  moveHistory.pop();
  resigned = false;
  selected = null;
  render();
  setStatus("Твій хід");
}

async function toggleSelfplay() {
  if (mode === "selfplay") {
    selfplayStopRequested = true;
    setStatus("Зупиняю самогру (двигун завершить поточний хід)...");
    await reliableSend("selfplay stop");
    return;
  }
  if (mode !== "idle") return;

  mode = "selfplay";
  selfplayStopRequested = false;
  selected = null;
  game.reset();
  moveHistory.length = 0;
  resigned = false;
  render();
  setStatus("Самогра...");

  await reliableSend("selfplay start");
  try {
    await pollUntil(
      (line) => line.startsWith("selfplayresult "),
      (line) => {
        if (line.startsWith("selfplaymove ")) {
          const uci = line.slice("selfplaymove ".length).trim();
          const move = game.move(uciToMoveInput(uci));
          if (move) {
            moveHistory.push(move.lan ?? uci);
            render();
          }
        }
      }
    );
    setStatus(`Самогра завершена: ${gameOverMessage()}`);
  } catch (err) {
    setStatus(`Помилка самогри: ${err}`);
  } finally {
    mode = "idle";
    updateControlsEnabled();
  }
}

async function startTraining() {
  if (mode !== "idle") return;
  mode = "train";
  updateControlsEnabled();
  trainLogEl.textContent = "";
  setStatus("Тренування...");

  await reliableSend("train start");
  try {
    const resultLine = await pollUntil(
      (line) =>
        line.startsWith("trainresult ") || line.startsWith("trainerror "),
      (line) => {
        if (line.startsWith("info trainprogress ")) {
          trainLogEl.textContent += line.slice("info trainprogress ".length).trim() + "\n";
        }
      }
    );
    if (resultLine.startsWith("trainerror ")) {
      setStatus(`Тренування: ${resultLine.slice("trainerror ".length)}`);
    } else {
      setStatus(`Тренування завершено: ${resultLine.slice("trainresult ".length)}`);
    }
  } catch (err) {
    setStatus(`Помилка тренування: ${err}`);
  } finally {
    mode = "idle";
    updateControlsEnabled();
  }
}

// A promoted `train start` writes a new checkpoint to disk, but the
// sidecar keeps using whatever weights it already loaded until told
// otherwise -- this sends the `reload` command added alongside the
// hot-reload backend support so a promoted model can actually be
// played against without restarting the whole engine.
async function reloadModel() {
  if (mode !== "idle") return;
  mode = "train"; // reuse the same busy-gate as training; this is a quick, single round-trip
  updateControlsEnabled();
  setStatus("Перезавантажую модель...");
  try {
    await reliableSend("reload");
    const line = await waitForLine((l) => l.startsWith("reloadresult "));
    const result = line.slice("reloadresult ".length);
    setStatus(
      result === "ok"
        ? "Модель перезавантажена."
        : "Нема новішого чекпоінта -- модель без змін."
    );
  } catch (err) {
    setStatus(`Помилка перезавантаження: ${err}`);
  } finally {
    mode = "idle";
    updateControlsEnabled();
  }
}

// P0 "generation journal + quality curve": lists every promoted
// checkpoint the backend recorded (see pipeline_torch.record_promotion),
// oldest first, one round-trip via the `history` UCI command.
async function showHistory() {
  if (mode !== "idle") return;
  mode = "train"; // reuse the same busy-gate as training/reload for this one round-trip
  updateControlsEnabled();
  setStatus("Читаю історію поколінь моделі...");
  const entries = [];
  try {
    await reliableSend("history");
    await pollUntil(
      (line) => line === "historyresult ok",
      (line) => {
        if (line.startsWith("historyentry ")) {
          entries.push(JSON.parse(line.slice("historyentry ".length)));
        }
      }
    );
    renderHistory(entries);
    setStatus(
      entries.length
        ? `Історія: ${entries.length} промоушн(и/ів).`
        : "Історія порожня -- ще жодна модель не промоутилась."
    );
  } catch (err) {
    setStatus(`Помилка читання історії: ${err}`);
  } finally {
    mode = "idle";
    updateControlsEnabled();
  }
}

function renderHistory(entries) {
  if (!entries.length) {
    historyTableEl.innerHTML = "";
    return;
  }
  const rows = entries
    .map(
      (e) =>
        `<tr><td>${e.cycle}</td><td>${(e.win_rate * 100).toFixed(1)}%</td><td>${new Date(e.promoted_at).toLocaleString()}</td></tr>`
    )
    .join("");
  historyTableEl.innerHTML = `<tr><th>Цикл</th><th>Win-rate</th><th>Коли</th></tr>${rows}`;
}

colorButtons.forEach((b) =>
  b.addEventListener("click", () => {
    if (mode !== "idle") return;
    humanColor = b.dataset.color;
    updateControlsEnabled();
  })
);
difficultyButtons.forEach((b) =>
  b.addEventListener("click", async () => {
    if (mode !== "idle") return;
    difficulty = b.dataset.difficulty;
    updateControlsEnabled();
    await applyDifficulty();
  })
);
newGameBtn.addEventListener("click", newGame);
resignBtn.addEventListener("click", resign);
undoBtn.addEventListener("click", undo);
selfplayBtn.addEventListener("click", toggleSelfplay);
trainBtn.addEventListener("click", startTraining);
reloadBtn.addEventListener("click", reloadModel);
historyBtn.addEventListener("click", showHistory);

async function init() {
  render();
  await invoke("engine_start");
  await handshake();
  await applyDifficulty();
  setStatus("Твій хід (граєш білими)");
}

init().catch((err) => setStatus(`Помилка: ${err}`));
