const ui = {};
const state = { current: null, previous: null, receivedAt: 0, interval: 160, perception: null, world: null, probability: [], lastProbabilityTime: -1 };

function byId(id) { return document.getElementById(id); }
function send(type, value = null) {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ type, value }));
    return true;
  }
  return false;
}

const miDesignFields = {
  frequency_hz: "miFrequency", tx_current_rms_a: "miCurrent",
  tx_radius_m: "miTxRadius", tx_turns: "miTxTurns",
  rx_radius_m: "miRxRadius", rx_turns: "miRxTurns"
};

function miRows() {
  return [...byId("miVoltages").querySelectorAll("input[data-role=voltage]")].map((box, i) => ({
    voltage: box, distance: byId("miVoltages").querySelectorAll("input[data-role=distance]")[i]
  }));
}

function miDraftDesign() {
  const design = {};
  for (const [key, id] of Object.entries(miDesignFields)) {
    const value = Number(byId(id).value);
    if (byId(id).value.trim() === "" || !Number.isFinite(value) || value < 0 ||
        (value === 0 && key !== "receiver_output_noise_rms_v")) throw new Error("설계 조건에 올바른 숫자를 입력하세요.");
    if (key.endsWith("_turns") && !Number.isInteger(value)) throw new Error("감은 횟수는 정수로 입력하세요.");
    design[key] = value;
  }
  return design;
}

function miExpectedVoltage(design, distanceCm, location) {
  const a = design.tx_radius_m / 100, b = design.rx_radius_m / 100, d = distanceCm / 100;
  if (!Number.isFinite(d) || d <= 0) throw new Error("거리는 0보다 커야 합니다.");
  let integral = 0;
  for (let i = 0; i < 512; i++) {
    const angle = (i + 0.5) * 2 * Math.PI / 512;
    integral += Math.cos(angle) / Math.sqrt(d*d + a*a + b*b - 2*a*b*Math.cos(angle));
  }
  const mutual = 4 * Math.PI * 1e-7 * design.tx_turns * design.rx_turns * a * b * integral * (2*Math.PI/512) / 2;
  const voltage = 2 * Math.PI * design.frequency_hz * mutual * design.tx_current_rms_a / 1000;
  return voltage * 1000;
}

function refreshMIPreview() {
  const measured = byId("miMode").value === "실험값";
  byId("miResultsTitle").textContent = measured ? "② 실험 결과 — 직접 입력" : "② 예상 결과 — 자동 계산";
  try {
    const design = miDraftDesign();
    for (const row of miRows()) {
      row.voltage.readOnly = !measured;
      row.voltage.classList.toggle("computed-result", !measured);
      if (!measured) row.voltage.value = miExpectedVoltage(design, Number(row.distance.value), "coil_input").toPrecision(6);
    }
    byId("miDraftNotice").textContent = "변경 내용은 아직 적용되지 않았습니다. 아래 적용 버튼을 누르세요.";
  } catch (error) {
    byId("miDraftNotice").textContent = error.message;
  }
}

function changeMIDraftMode() {
  if (state.miEditMode === "실험값") state.miMeasuredCache = miRows().map(row => [row.distance.value, row.voltage.value]);
  state.miEditMode = byId("miMode").value;
  if (state.miEditMode === "실험값") {
    const rows = miRows();
    rows.forEach((row, i) => {
      if (state.miMeasuredCache && state.miMeasuredCache[i]) {
        row.distance.value = state.miMeasuredCache[i][0]; row.voltage.value = state.miMeasuredCache[i][1];
      } else row.voltage.value = "";
    });
  }
  refreshMIPreview();
}

function updateMIInterface(mi) {
  if (!mi) return;
  byId("miSource").textContent = "V2 신호 · 반응 " + mi.scaling.strength_factor.toPrecision(4) + "배 · 범위 " + mi.scaling.sensor_range.toFixed(2) + " / " + (mi.source === "measured_transfer" ? "실험값" : "예상값");
  byId("miError").textContent = mi.error || "";
  byId("miOutput").textContent = "측정 범위: " + mi.scaling.sensor_range.toFixed(2) + " · 범위 비례식은 합성 가정입니다.";
  if (state.miRevision !== mi.revision) {
    state.miRevision = mi.revision;
    for (const [key, id] of Object.entries(miDesignFields)) byId(id).value = mi.design[key];
    byId("miMode").value = mi.mode;
    state.miEditMode = mi.mode; state.miMeasuredCache = mi.mode === "실험값" ? mi.rows.map(([d,v]) => [String(d), v == null ? "" : String(v)]) : null;
    byId("miVoltages").replaceChildren();
    mi.rows.forEach(([distance, voltage]) => {
      const label = document.createElement("label"); label.textContent = "거리 (cm) → 수신 전압 (mV RMS)";
      const d = document.createElement("input"); d.type = "number"; d.step = "any"; d.min = "0.001"; d.dataset.role = "distance"; d.value = distance;
      const v = document.createElement("input"); v.type = "number"; v.step = "any"; v.min = "0.000001"; v.dataset.role = "voltage"; v.value = voltage == null ? "" : voltage;
      label.append(d, v); byId("miVoltages").appendChild(label);
    });
    refreshMIPreview();
    byId("miDraftNotice").textContent = state.miRequestPending ? "적용 완료. 시뮬레이션의 실행 버튼을 누르세요." : "입력·계산만으로는 반영되지 않습니다. 적용 버튼을 누르세요.";
    state.miRequestPending = false;
    const settings = mi.relay.settings;
    byId("relayEnabled").checked = settings.enabled;
    byId("relayCount").value = settings.count;
    byId("relaySpacing").value = settings.spacing_cm;
    byId("relayAxis").value = settings.axis;
    state.relayRequest = null;
    showRelaySummary(mi.relay);
    byId("relayError").textContent = "";
    byId("relayNotice").textContent = "입력 후 적용을 누르면 시뮬레이션에 반영됩니다.";
  }
  byId("relayActive").textContent = "현재 적용: " + (mi.relay.settings.enabled ? "중계 사용" : "중계 꺼짐") + " · " + mi.relay.gain.toPrecision(4) + "배";
  if (mi.relay_preview && mi.relay_preview.request_id === state.relayRequest) {
    byId("relayError").textContent = mi.relay_preview.error || "";
    if (mi.relay_preview.summary) showRelaySummary(mi.relay_preview.summary);
  }
  if (mi.error) byId("relayError").textContent = mi.error;
}

function relayDraft() {
  const count = Number(byId("relayCount").value), spacing = Number(byId("relaySpacing").value);
  if (!Number.isInteger(count) || count < 1 || count > 8) throw new Error("중계 개수는 1~8개의 정수로 입력하세요.");
  if (!Number.isFinite(spacing) || spacing <= 0) throw new Error("코일 간격은 0보다 커야 합니다.");
  return {enabled: byId("relayEnabled").checked, count, spacing_cm: spacing, axis: byId("relayAxis").value};
}

function showRelaySummary(summary) {
  byId("relayResult").textContent =
    "송신–수신 전체 거리: " + (summary.distance_m*100).toFixed(2) + " cm\n" +
    "중계 없음: " + (summary.direct_rms_v*1000).toPrecision(6) + " mV RMS\n" +
    "현재 설정: " + (summary.input_rms_v*1000).toPrecision(6) + " mV RMS\n" +
    "수신 전압 배율: " + summary.gain.toPrecision(4) + "배 · 위상: " + summary.phase_deg.toFixed(1) + "°";
}

function requestRelayPreview() {
  clearTimeout(state.relayTimer);
  byId("relayNotice").textContent = "미리보기입니다. 적용을 눌러야 반영됩니다.";
  state.relayTimer = setTimeout(() => {
    state.relaySequence = (state.relaySequence || 0) + 1;
    state.relayRequest = state.relaySequence;
    try {
      const settings = relayDraft();
      if (!send("relay_preview", {settings, request_id: state.relayRequest})) throw new Error("서버 연결을 확인하세요.");
      byId("relayError").textContent = "";
    } catch (error) {
      byId("relayError").textContent = error.message;
      byId("relayResult").textContent = "";
    }
  }, 150);
}

function setupControls() {
  byId("miMode").onchange = changeMIDraftMode;
  byId("miForm").oninput = refreshMIPreview;
  byId("miForm").onsubmit = event => {
    event.preventDefault();
    try {
      const rows = miRows(), measured = byId("miMode").value === "실험값";
      const value = {mode: byId("miMode").value, location: "coil_input", design: miDraftDesign(),
        distances_cm: rows.map(row => row.distance.value.trim() === "" ? null : Number(row.distance.value)),
        voltages_mv: rows.map(row => !measured || row.voltage.value.trim() === "" ? null : Number(row.voltage.value))};
      state.miRequestPending = send("mi_settings", value);
      byId("miDraftNotice").textContent = state.miRequestPending ? "적용 중입니다..." : "서버 연결을 확인하세요.";
    } catch (error) { byId("miDraftNotice").textContent = error.message; }
  };
  byId("miReset").onclick = () => {
    state.miRequestPending = send("reset_mi_settings");
    byId("miDraftNotice").textContent = state.miRequestPending ? "최초 예상값으로 초기화 중입니다..." : "서버 연결을 확인하세요.";
  };
  byId("relayForm").oninput = requestRelayPreview;
  byId("relayForm").onchange = requestRelayPreview;
  byId("relayForm").onsubmit = event => {
    event.preventDefault();
    clearTimeout(state.relayTimer);
    try {
      if (!send("relay_settings", relayDraft())) throw new Error("서버 연결을 확인하세요.");
      byId("relayNotice").textContent = "적용 중입니다...";
    } catch (error) { byId("relayError").textContent = error.message; }
  };
  ["environment", "perception", "speed", "speedValue", "simStatus", "simTime", "seed", "modeBadge", "probability", "prediction", "probabilityBar", "detectorStatus", "window", "markerCount", "estimatedCount", "decisionMode", "detectorModel", "detectorError", "agentState", "agentAction", "coverage", "agentModel", "agentError", "magnitude", "bx", "by", "bz", "connectionDot", "connectionText", "detectorCard"].forEach(id => ui[id] = byId(id));
  byId("run").onclick = () => send("run");
  byId("stop").onclick = () => send("stop");
  byId("reset").onclick = () => send("reset");
  byId("randomize").onclick = () => send("randomize");
  ui.environment.onchange = event => send("environment", event.target.value);
  ui.perception.onchange = event => send("robot_perception", event.target.checked);
  ui.speed.oninput = event => { ui.speedValue.value = event.target.value; send("speed", Number(event.target.value)); };
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
  updateMIInterface(data.mi);
  if (!ui.environment.options.length) data.environments.forEach(name => ui.environment.add(new Option(name, name)));
  ui.environment.value = data.environment;
  ui.perception.checked = data.robot_perception;
  ui.speed.value = data.speed;
  ui.speedValue.value = data.speed;
  document.querySelector(`input[name=navigation][value="${data.navigation_mode}"]`).checked = true;
  ui.simStatus.textContent = data.status === "running" ? "실행 중" : data.status === "complete" ? "완료" : "정지";
  ui.simTime.textContent = `${data.time.toFixed(1)} s`;
  ui.seed.textContent = data.seed;
  ui.modeBadge.textContent = data.robot_perception ? "ROBOT PERCEPTION" : "ACTUAL ENVIRONMENT";
  const probability = data.detector.probability * 100;
  ui.probability.textContent = `${probability.toFixed(1)}%`;
  const uncalibrated = data.detector.status === "UNCALIBRATED";
  ui.prediction.textContent = uncalibrated ? "미검증 · 구조 핑 비활성" : (data.detector.predicted_label ? "PERSON" : "NO PERSON");
  ui.probabilityBar.style.width = `${probability}%`;
  ui.detectorStatus.textContent = data.detector.status;
  ui.window.textContent = `${data.detector.window_size}/16`;
  ui.markerCount.textContent = data.marker_count;
  ui.estimatedCount.textContent = data.detector.estimated_count;
  ui.decisionMode.textContent = data.detector.decision_mode === "WORLD_DELTA" ? "WORLD DELTA" : data.detector.decision_mode;
  ui.detectorModel.textContent = data.detector.model.replace("best_", "");
  ui.detectorError.textContent = data.detector.error || "";
  ui.detectorCard.classList.toggle("detected", !uncalibrated && (Boolean(data.detector.predicted_label) || data.detector.decision_mode === "WORLD_DELTA"));
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

