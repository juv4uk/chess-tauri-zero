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
const bgJobsEl = document.getElementById("bg-jobs");
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

// Bumped by forceStopEngine() (Escape). Real bug found in today's
// 4-agent audit (Frontend Auditor): a poll loop from BEFORE a kill+
// respawn used to keep calling engine_drain for up to its full 10min
// timeout, racing the new session's own handshake reads and able to
// apply a stale heatmap or falsely fail forceStopEngine's own
// handshake. Every waitForLine/pollUntil call captures the epoch it
// started with and self-cancels the moment it changes, instead of
// racing the new session.
let engineEpoch = 0;

// Small status area, separate from the main #status line, for work
// that keeps running after the user has regained control of the GUI
// (Escape during training "detaches" instead of killing -- see
// detachFromTraining()). Keyed by job name so more than one could show
// at once in principle, though only "train" uses this today.
let backgroundJobs = {};

function setBackgroundJob(key, text) {
  if (text === null) {
    delete backgroundJobs[key];
  } else {
    backgroundJobs[key] = text;
  }
  bgJobsEl.textContent = Object.values(backgroundJobs).join(" | ");
}

// P0 "теплокарта наступного ходу" (root-only): destination square ->
// visit count from the most recent "info mctsroot <json>" line, i.e.
// what the engine considered right before the move it just made.
// Only ever read while `selected` is null (see render()) so it never
// fights the .legal highlight, which is the functionally important one.
let heatmap = {};

function applyMctsInfo(line) {
  if (!line.startsWith("info mctsroot ")) return;
  const entries = JSON.parse(line.slice("info mctsroot ".length));
  const next = {};
  for (const e of entries) {
    const to = e.move.slice(2, 4); // destination square, same convention as uciToMoveInput()
    next[to] = Math.max(next[to] || 0, e.n); // a square can be the target of more than one root move (e.g. underpromotions)
  }
  heatmap = next;
}

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
      if (!selected && heatmap[square]) {
        const maxN = Math.max(1, ...Object.values(heatmap));
        const intensity = heatmap[square] / maxN;
        el.style.boxShadow = `inset 0 0 0 ${Math.round(4 + 26 * intensity)}px rgba(230, 126, 34, ${(0.12 + 0.35 * intensity).toFixed(2)})`;
      }

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
      const fenBefore = game.fen();
      const move = game.move({ from: selected, to: square, promotion: "q" });
      const uci = move.lan ?? `${selected}${square}`;
      moveHistory.push(uci);
      selected = null;
      heatmap = {}; // the previous heatmap described the position before this human move
      render();
      // Awaited (not fire-and-forget) so this always reaches the
      // engine's stdin BEFORE requestEngineMove's own "position"/"go"
      // -- both go through the same single-threaded reliableSend, so
      // an un-awaited call here could interleave after a reconnect
      // retry and arrive out of order.
      await recordHumanMove(fenBefore, uci);
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
    await sendHumanGameOver(); // the human's own move just ended the game
    return;
  }
  mode = "thinking";
  updateControlsEnabled();
  setStatus("Рушій думає...");
  await reliableSend(`position startpos moves ${moveHistory.join(" ")}`);
  await reliableSend("go");
  const bestmoveLine = await pollUntil((l) => l.startsWith("bestmove "), applyMctsInfo);
  const uci = bestmoveLine.split(" ")[1];
  const move = game.move(uciToMoveInput(uci));
  moveHistory.push(move.lan ?? uci);
  mode = "idle";
  render();
  setStatus(isOver() ? gameOverMessage() : "Твій хід");
  if (isOver()) await sendHumanGameOver(); // the engine's own move just ended the game

function gameOverMessage() {
  if (resigned) return "Ти здався.";
  if (game.isCheckmate()) return `Мат! Переміг ${game.turn() === "w" ? "чорний" : "білий"}.`;
  if (game.isDraw()) return "Нічия.";
  return "Гру завершено.";
}

// P1 item 7 (docs/development-plan-uk.md): record games played against
// a human as a separate training-data source (data/play_data_human/,
// never mixed with self-play). fenBefore comes from THIS frontend's
// own chess.js state, not asked of the backend -- its `env` lags by
// one ply right after the engine's own reply (a real bug caught while
// building this, see uci_torch.py's _record_human_move docstring).
// Best-effort: a failed save shouldn't interrupt an actual game the
// human is in the middle of playing.
async function recordHumanMove(fenBefore, uci) {
  try {
    await reliableSend(`humanmove ${fenBefore} ${uci}`);
  } catch (err) {
    console.error("Не вдалося записати людський хід для тренувальних даних:", err);
  }
}

async function sendHumanGameOver() {
  let result;
  if (resigned) {
    result = "human-loss";
  } else if (game.isCheckmate()) {
    const winnerColor = game.turn() === "w" ? "b" : "w"; // side to move is the one checkmated -- the OTHER side won
    result = winnerColor === humanColor ? "human-win" : "human-loss";
  } else {
    result = "draw"; // real draw, or a truncation/other end state -- safest default, never mislabeled as a win/loss
  }
  try {
    await reliableSend(`humangameover ${result}`);
  } catch (err) {
    console.error("Не вдалося зберегти партію для тренувальних даних:", err);
  }
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

// Real bug #8 from today's 4-agent audit (Frontend + Python Logic
// Auditors): the Python-side crash-guard's `info error [<cmd>] <msg>`
// line (added earlier today) was never watched for here -- a caught
// backend error was invisible to the user, who just saw a generic
// "Тайм-аут" after 120s-10min instead of the real reason. Checked
// AFTER the caller's own predicate so a command that legitimately
// wants to see "info error " text itself (none currently do) still
// could.
function isErrorLine(line) {
  return line.startsWith("info error ");
}

async function waitForLine(predicate, timeoutMs = 120000) {
  const myEpoch = engineEpoch;
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (engineEpoch !== myEpoch) {
      throw new Error("Рушій перезапущено (Escape) -- очікування скасовано.");
    }
    const lines = await invoke("engine_drain");
    for (const line of lines) {
      if (predicate(line)) return line;
      if (isErrorLine(line)) throw new Error(line.slice("info error ".length));
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error("Тайм-аут очікування відповіді рушія");
}

// Like waitForLine, but calls onLine for every line seen (not just the
// terminal one) -- used for selfplay/train streams where intermediate
// lines (selfplaymove/trainprogress) matter, not just the final result.
async function pollUntil(matchPredicate, onLine, timeoutMs = 600000) {
  const myEpoch = engineEpoch;
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (engineEpoch !== myEpoch) {
      throw new Error("Рушій перезапущено (Escape) -- очікування скасовано.");
    }
    const lines = await invoke("engine_drain");
    for (const line of lines) {
      onLine(line);
      if (matchPredicate(line)) return line;
      if (isErrorLine(line)) throw new Error(line.slice("info error ".length));
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
  // Escape detaches training rather than stopping it (see
  // detachFromTraining()) -- mode returns to idle quickly either way,
  // so this button is only disabled for the brief window training is
  // still the FOREGROUND action, not for however long it actually runs.
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
  heatmap = {};
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

async function resign() {
  if (mode !== "idle" || isOver()) return;
  resigned = true;
  setStatus(gameOverMessage());
  updateControlsEnabled();
  await sendHumanGameOver();
}

function undo() {
  if (mode !== "idle" || moveHistory.length < 2) return;
  game.undo();
  game.undo();
  moveHistory.pop();
  moveHistory.pop();
  resigned = false;
  selected = null;
  heatmap = {};
  render();
  setStatus("Твій хід");
}

async function toggleSelfplay() {
  if (mode === "selfplay") {
    selfplayStopRequested = true;
    setStatus("Зупиняю самогру (двигун завершить поточний хід)...");
    try {
      await reliableSend("selfplay stop");
    } catch (err) {
      // mode recovery here doesn't depend on this call -- the original
      // invocation that started selfplay owns its own try/finally
      // below and will reset mode once its pollUntil settles. This
      // catch only stops an unrelated unhandled rejection from
      // surfacing for what's just a best-effort stop request.
      setStatus(`Не вдалося надіслати команду зупинки: ${err}`);
    }
    return;
  }
  if (mode !== "idle") return;

  mode = "selfplay";
  selfplayStopRequested = false;
  selected = null;
  game.reset();
  moveHistory.length = 0;
  resigned = false;
  heatmap = {};
  render();
  setStatus("Самогра...");

  try {
    // Real bug found in today's 4-agent audit: "selfplay start" used
    // to be sent OUTSIDE this try/finally -- if it threw (e.g. a dead
    // sidecar reliableSend couldn't respawn), mode stayed stuck at
    // "selfplay" forever, with no pollUntil ever started to eventually
    // time out and recover it. Moved inside so the finally below is
    // guaranteed to run no matter which step fails.
    await reliableSend("selfplay start");
    await pollUntil(
      (line) => line.startsWith("selfplayresult "),
      (line) => {
        applyMctsInfo(line);
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

// Set true by detachFromTraining() (Escape during "train") -- tells
// this same in-flight call's own result-handling below where to report
// (the small #bg-jobs area instead of the main #status line) once the
// training it's still genuinely waiting on actually finishes.
let trainDetached = false;

async function startTraining() {
  if (mode !== "idle") return;
  mode = "train";
  trainDetached = false;
  updateControlsEnabled();
  trainLogEl.textContent = "";
  setStatus("Тренування...");
  setBackgroundJob("train", "Тренування...");

  await reliableSend("train start");
  try {
    const resultLine = await pollUntil(
      (line) =>
        line.startsWith("trainresult ") || line.startsWith("trainerror "),
      (line) => {
        if (line.startsWith("info trainprogress ")) {
          const stage = line.slice("info trainprogress ".length).trim();
          trainLogEl.textContent += stage + "\n";
          setBackgroundJob("train", `Тренування: ${stage}`);
        }
      },
      3600000 // 1h -- long enough for a real cycle to finish even detached, not just the usual 10min
    );
    const resultText = resultLine.startsWith("trainerror ")
      ? `Тренування: ${resultLine.slice("trainerror ".length)}`
      : `Тренування завершено: ${resultLine.slice("trainresult ".length)}`;
    if (trainDetached) {
      setBackgroundJob("train", resultText);
      setTimeout(() => setBackgroundJob("train", null), 10000);
    } else {
      setStatus(resultText);
    }
  } catch (err) {
    const errText = `Помилка тренування: ${err}`;
    if (trainDetached) {
      setBackgroundJob("train", errText);
      setTimeout(() => setBackgroundJob("train", null), 10000);
    } else {
      setStatus(errText);
    }
  } finally {
    if (!trainDetached) {
      mode = "idle";
      updateControlsEnabled();
    }
  }
}

// Escape during training: unlike thinking/selfplay (which have no
// cooperative stop point in the Python code and so genuinely need a
// hard kill), a training cycle already records its own result to the
// generation journal regardless of how it ends -- there's nothing to
// lose by just letting it keep running. This "detaches" the GUI
// instead: releases `mode` back to idle right away (the engine process
// is untouched, still training), while the SAME pollUntil call already
// awaited inside startTraining() keeps genuinely waiting for the real
// trainresult/trainerror -- trainDetached tells it to report into
// #bg-jobs instead of #status when that finally arrives.
function detachFromTraining() {
  trainDetached = true;
  mode = "idle";
  setStatus("Тренування продовжується у фоні. GUI розблоковано.");
  setBackgroundJob("train", "Тренування триває у фоні...");
  selected = null;
  render();
  updateControlsEnabled();
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

// P0 "generation journal + quality curve": lists every evaluated cycle
// the backend recorded, promoted or rejected (see
// pipeline_torch.record_cycle_result), oldest first, one round-trip
// via the `history` UCI command.
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
    const promotedCount = entries.filter((e) => e.promoted).length;
    setStatus(
      entries.length
        ? `Історія: ${entries.length} цикл(и/ів), з них промоутнуто ${promotedCount}.`
        : "Історія порожня -- жодного циклу тренування ще не було."
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
        `<tr><td>${e.cycle}</td><td>${(e.win_rate * 100).toFixed(1)}%</td><td>${e.promoted ? "✓ промоутнуто" : "відхилено"}</td><td>${new Date(e.evaluated_at).toLocaleString()}</td></tr>`
    )
    .join("");
  historyTableEl.innerHTML = `<tr><th>Цикл</th><th>Win-rate</th><th>Статус</th><th>Коли</th></tr>${rows}`;
}

// Escape: stop whatever is running. selfplay has a real graceful stop
// (`selfplay stop`, already used by the button); "thinking" (waiting
// on `go`) and "train" have no interrupt point in the Python code at
// all, so the only real way to unstick them is to force-kill the
// sidecar and respawn it -- see engine_kill's own doc comment on the
// Rust side for why engine_start alone can't do this (it's idempotent,
// a no-op while the process is alive, even if it's stuck).
async function forceStopEngine() {
  engineEpoch++; // invalidate any poll loop already in flight from before this kill+respawn
  setStatus("Escape -- примусово зупиняю рушій...");
  mode = "train"; // block controls during the kill+respawn round-trip
  updateControlsEnabled();
  try {
    await invoke("engine_kill");
    await invoke("engine_start");
    await handshake();
    await applyDifficulty();
    setStatus("Рушій примусово перезапущено (Escape). Поточну дію перервано.");
  } catch (err) {
    setStatus(`Помилка примусової зупинки: ${err}`);
  } finally {
    mode = "idle";
    selected = null;
    heatmap = {};
    render();
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  // thinking/selfplay have no cooperative stop point in the Python
  // code at all, so Escape hard-kills+respawns the engine for those
  // (see forceStopEngine()'s own comment -- also closes a real bug
  // where selfplay's old Escape path just resent "selfplay stop" and
  // gave false confidence of an immediate stop while the actual wait
  // kept running unchanged underneath).
  //
  // train is different, by the owner's own explicit request: a
  // training cycle already records its own result to the generation
  // journal no matter how it ends, so there's nothing to lose by
  // letting it keep running -- Escape here just releases the GUI
  // (detachFromTraining()) instead of killing the process, and
  // #bg-jobs shows it's still going.
  if (mode === "thinking" || mode === "selfplay") {
    forceStopEngine();
  } else if (mode === "train") {
    detachFromTraining();
  }
});

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
