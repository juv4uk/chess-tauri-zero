import { Chess } from "./vendor/chess.js";

const { invoke } = window.__TAURI__.core;

const PIECE_GLYPH = {
  wk: "♔", wq: "♕", wr: "♖", wb: "♗", wn: "♘", wp: "♙",
  bk: "♚", bq: "♛", br: "♜", bb: "♝", bn: "♞", bp: "♟",
};

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");

const game = new Chess();
const moveHistory = []; // UCI strings, sent to the engine as "position startpos moves ..."
let selected = null; // algebraic square string, e.g. "e2"
let thinking = false;

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
}

async function onSquareClick(square) {
  if (thinking || game.isGameOver()) return;
  if (game.turn() !== "w") return; // human always plays white here

  const piece = game.get(square);

  if (selected) {
    const legal = game.moves({ square: selected, verbose: true });
    const target = legal.find((m) => m.to === square);
    if (target) {
      const move = game.move({ from: selected, to: square, promotion: "q" });
      moveHistory.push(move.lan ?? `${selected}${square}`);
      selected = null;
      render();
      await afterHumanMove();
      return;
    }
    // clicking a different own piece re-selects instead of illegal move
    selected = piece && piece.color === "w" ? square : null;
    render();
    return;
  }

  if (piece && piece.color === "w") {
    selected = square;
    render();
  }
}

async function afterHumanMove() {
  if (game.isGameOver()) {
    setStatus(gameOverMessage());
    return;
  }
  thinking = true;
  setStatus("Рушій думає...");
  await engineSend(`position startpos moves ${moveHistory.join(" ")}`);
  await engineSend("go");
  const bestmoveLine = await waitForLine((l) => l.startsWith("bestmove "));
  const uci = bestmoveLine.split(" ")[1];
  const move = game.move(uciToMoveInput(uci));
  moveHistory.push(move.lan ?? uci);
  thinking = false;
  render();
  setStatus(game.isGameOver() ? gameOverMessage() : "Твій хід");
}

function gameOverMessage() {
  if (game.isCheckmate()) return `Мат! Переміг ${game.turn() === "w" ? "чорний" : "білий"}.`;
  if (game.isDraw()) return "Нічия.";
  return "Гру завершено.";
}

async function engineSend(line) {
  await invoke("engine_send", { line });
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

async function init() {
  render();
  await invoke("engine_start");
  await engineSend("uci");
  await waitForLine((l) => l === "uciok");
  await engineSend("isready");
  await waitForLine((l) => l === "readyok");
  setStatus("Твій хід (граєш білими)");
}

init().catch((err) => setStatus(`Помилка: ${err}`));
