const ui = {};
const state = { current: null, previous: null, receivedAt: 0, interval: 160, perception: null, world: null, probability: [], lastProbabilityTime: -1 };

function byId(id) { return document.getElementById(id); }
function send(type, value = null) {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({ type, value }));
}

function setupControls() {
  ["environment", "perception", "speed", "speedValue", "noiseLevel", "noiseValue", "simStatus", "simTime", "seed", "modeBadge", "probability", "prediction", "probabilityBar", "detectorStatus", "window", "markerCount", "estimatedCount", "detectorModel", "detectorError", "agentState", "agentAction", "coverage", "agentModel", "agentError", "magnitude", "bx", "by", "bz", "connectionDot", "connectionText", "detectorCard"].forEach(id => ui[id] = byId(id));
  byId("run").onclick = () => send("run");
  byId("stop").onclick = () => send("stop");
  byId("reset").onclick = () => send("reset");
  byId("randomize").onclick = () => send("randomize");
  ui.environment.onchange = event => send("environment", event.target.value);
  ui.perception.onchange = event => send("robot_perception", event.target.checked);
  ui.speed.oninput = event => { ui.speedValue.value = event.target.value; send("speed", Number(event.target.value)); };
  ui.noiseLevel.oninput = event => { ui.noiseValue.value = Number(event.target.value).toFixed(2); send("noise_level", Number(event.target.value)); };
  document.querySelectorAll("input[name=navigation]").forEach(input => input.onchange = event => send("navigation_mode", event.target.value));
  document.querySelectorAll("[data-action]").forEach(button => button.onclick = () => send("manual_action", button.dataset.action));
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  state.socket = socket;
  socket.onopen = () => {
    document.querySelector(".connection").classList.add("connected");
    ui.connectionText.textContent = "연결됨";
  };
  socket.onclose = () => {
    document.querySelector(".connection").classList.remove("connected");
    ui.connectionText.textContent = "재연결 중";
    setTimeout(connect, 1500);
  };
  socket.onmessage = event => {
    const incoming = JSON.parse(event.data);
    if (incoming.type !== "state") return;
    const now = performance.now();
    state.interval = Math.max(80, Math.min(500, now - state.receivedAt || 160));
    state.receivedAt = now;
    state.previous = state.current;
    if (incoming.perception) state.perception = incoming.perception;
    if (incoming.obstacles) state.world = { obstacles: incoming.obstacles, metals: incoming.metals, people: incoming.people };
    if (!incoming.robot_perception && state.world) Object.assign(incoming, state.world);
    state.current = incoming;
    if (incoming.time !== state.lastProbabilityTime) {
      state.lastProbabilityTime = incoming.time;
      state.probability.push([incoming.time, incoming.detector.probability]);
      if (state.probability.length > 120) state.probability.shift();
    }
    updateInterface(incoming);
  };
}

function updateInterface(data) {
  if (!ui.environment.options.length) data.environments.forEach(name => ui.environment.add(new Option(name, name)));
  ui.environment.value = data.environment;
  ui.perception.checked = data.robot_perception;
  ui.speed.value = data.speed;
  ui.speedValue.value = data.speed;
  ui.noiseLevel.value = data.noise_level;
  ui.noiseValue.value = Number(data.noise_level).toFixed(2);
  document.querySelector(`input[name=navigation][value="${data.navigation_mode}"]`).checked = true;
  ui.simStatus.textContent = data.status === "running" ? "실행 중" : data.status === "complete" ? "완료" : "정지";
  ui.simTime.textContent = `${data.time.toFixed(1)} s`;
  ui.seed.textContent = data.seed;
  ui.modeBadge.textContent = data.robot_perception ? "ROBOT PERCEPTION" : "ACTUAL ENVIRONMENT";
  const probability = data.detector.probability * 100;
  ui.probability.textContent = `${probability.toFixed(1)}%`;
  ui.prediction.textContent = data.detector.predicted_label ? "PERSON" : "NO PERSON";
  ui.probabilityBar.style.width = `${probability}%`;
  ui.detectorStatus.textContent = data.detector.status;
  ui.window.textContent = `${data.detector.window_size}/16`;
  ui.markerCount.textContent = data.marker_count;
  ui.estimatedCount.textContent = data.detector.estimated_count;
  ui.detectorModel.textContent = data.detector.model.replace("best_", "");
  ui.detectorError.textContent = data.detector.error || "";
  ui.detectorCard.classList.toggle("detected", Boolean(data.detector.predicted_label));
  ui.agentState.textContent = data.agent.state;
  ui.agentAction.textContent = data.agent.action;
  ui.coverage.textContent = `${(data.agent.coverage * 100).toFixed(1)}%`;
  ui.agentModel.textContent = data.agent.model.replace("best_", "");
  ui.agentError.textContent = data.agent.error || "";
  ui.magnitude.textContent = data.sensor.magnitude.toFixed(3);
  ui.bx.textContent = data.sensor.bx.toFixed(2);
  ui.by.textContent = data.sensor.by.toFixed(2);
  ui.bz.textContent = data.sensor.bz.toFixed(2);
  drawLineChart(byId("sensorCanvas"), data.sensor.history, [1, 2, 3, 4], ["#42a5f5", "#66bb6a", "#ffca28", "#d500f9"]);
  drawLineChart(byId("probabilityCanvas"), state.probability, [1], ["#d500f9"], [0, 1]);
}

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return [context, rect.width, rect.height];
}

function drawLineChart(canvas, values, indices, colors, fixedRange = null) {
  const [ctx, width, height] = canvasContext(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#07111f"; ctx.fillRect(0, 0, width, height);
  if (!values || values.length < 2) return;
  const series = indices.flatMap(index => values.map(row => row[index]));
  let minimum = fixedRange ? fixedRange[0] : Math.min(...series);
  let maximum = fixedRange ? fixedRange[1] : Math.max(...series);
  if (maximum - minimum < 1e-6) maximum = minimum + 1;
  ctx.strokeStyle = "#1d3a53"; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = i * height / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  indices.forEach((index, colorIndex) => {
    ctx.strokeStyle = colors[colorIndex]; ctx.lineWidth = 1.5; ctx.beginPath();
    values.forEach((row, pointIndex) => {
      const x = pointIndex / (values.length - 1) * width;
      const y = height - (row[index] - minimum) / (maximum - minimum) * height;
      if (pointIndex === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function star(ctx, x, y, radius, color) {
  ctx.fillStyle = color; ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const angle = -Math.PI / 2 + i * Math.PI / 5;
    const length = i % 2 ? radius * .42 : radius;
    const px = x + Math.cos(angle) * length, py = y + Math.sin(angle) * length;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath(); ctx.fill();
}

function drawSimulation() {
  requestAnimationFrame(drawSimulation);
  const data = state.current;
  if (!data) return;
  const canvas = byId("simCanvas");
  const [ctx, width, height] = canvasContext(canvas);
  const padding = 14;
  const scale = Math.min((width - padding * 2) / data.board.width, (height - padding * 2) / data.board.height);
  const offsetX = (width - data.board.width * scale) / 2;
  const offsetY = (height - data.board.height * scale) / 2;
  const point = (x, y) => [offsetX + x * scale, height - offsetY - y * scale];
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#dbeaf2"; ctx.fillRect(offsetX, offsetY, data.board.width * scale, data.board.height * scale);
  if (data.robot_perception && state.perception) {
    const perception = state.perception;
    perception.cells.forEach(cell => {
      const x = offsetX + cell[0] * perception.cell_size * scale;
      const y = height - offsetY - (cell[1] + 1) * perception.cell_size * scale;
      ctx.fillStyle = cell[2] == null ? "rgba(56,136,190,.16)" : `rgba(245,124,0,${Math.min(.72, .12 + cell[2] / 38)})`;
      ctx.fillRect(x, y, perception.cell_size * scale + .4, perception.cell_size * scale + .4);
    });
    ctx.strokeStyle = "rgba(88,120,145,.32)"; ctx.lineWidth = .6;
    perception.range_endpoints.forEach(endpoint => { const a = point(data.robot.x, data.robot.y), b = point(endpoint[0], endpoint[1]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); });
    ctx.fillStyle = "#263238";
    perception.obstacle_points.forEach(item => { const p = point(item[0], item[1]); ctx.fillRect(p[0] - 1.3, p[1] - 1.3, 2.6, 2.6); });
    ctx.strokeStyle = "#ef6c00"; ctx.lineWidth = 2;
    perception.metal_candidates.forEach(item => { const p = point(item[0], item[1]); ctx.strokeRect(p[0] - 5, p[1] - 5, 10, 10); });
  } else {
    (data.obstacles || []).forEach(item => { const p = point(item.x, item.y + item.height); ctx.fillStyle = "#37474f"; ctx.fillRect(p[0], p[1], item.width * scale, item.height * scale); ctx.strokeStyle = "#1c2529"; ctx.strokeRect(p[0], p[1], item.width * scale, item.height * scale); });
    (data.metals || []).forEach(item => { const p = point(item.x, item.y); const size = 4 + item.size * .8; ctx.fillStyle = "#a66b35"; ctx.fillRect(p[0] - size, p[1] - size, size * 2, size * 2); });
    (data.people || []).forEach(item => { const p = point(item.x, item.y); star(ctx, p[0], p[1], 7, item.rescued ? "#78909c" : "#e53935"); });
  }
  if (data.robot.path.length > 1) {
    ctx.strokeStyle = "#42a5f5"; ctx.lineWidth = 2; ctx.beginPath();
    data.robot.path.forEach((item, index) => { const p = point(item[0], item[1]); if (index === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
    ctx.stroke();
  }
  const pulse = 1 + .13 * Math.sin(performance.now() / 180);
  data.markers.forEach(item => {
    const p = point(item.x, item.y), radius = (6 + item.confidence * 4) * pulse;
    ctx.strokeStyle = "#d500f9"; ctx.fillStyle = "rgba(213,0,249,.18)"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p[0] - radius - 3, p[1]); ctx.lineTo(p[0] + radius + 3, p[1]); ctx.moveTo(p[0], p[1] - radius - 3); ctx.lineTo(p[0], p[1] + radius + 3); ctx.stroke();
  });
  let robot = data.robot;
  if (state.previous && state.previous.robot) {
    const alpha = Math.min(1, (performance.now() - state.receivedAt) / state.interval);
    robot = { ...robot, x: state.previous.robot.x + (data.robot.x - state.previous.robot.x) * alpha, y: state.previous.robot.y + (data.robot.y - state.previous.robot.y) * alpha };
  }
  const rp = point(robot.x, robot.y);
  ctx.setLineDash([6, 5]); ctx.strokeStyle = "#1565c0"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(rp[0], rp[1], data.board.sensor_range * scale, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#1565c0"; ctx.strokeStyle = "white"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(rp[0], rp[1], 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = "#0d47a1"; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(rp[0], rp[1]); ctx.lineTo(rp[0] + Math.cos(data.robot.heading) * 18, rp[1] - Math.sin(data.robot.heading) * 18); ctx.stroke();
}

setupControls();
connect();
requestAnimationFrame(drawSimulation);

