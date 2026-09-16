import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import kotomka.app as app_module
from kotomka.models import JobCreate
from kotomka.storage import JobStore


@pytest.mark.skipif(not shutil.which("node"), reason="Node required to execute the page's polling script")
@pytest.mark.parametrize("initial,expected_reloads", [("running", 1), ("failed", 0)])
def test_failed_transition_reloads_once_to_show_recovery_actions(tmp_path, monkeypatch, initial, expected_reloads):
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    store.update_job(job.id, status=initial)
    monkeypatch.setattr(app_module, "store", store)
    client = TestClient(app_module.app)
    html = client.get(f"/jobs/{job.id}").text
    script = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    harness = r'''
const vm = require('node:vm');
const data = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const status = {textContent: data.initial};
let reloads = 0;
const context = {
  document: {querySelector: () => ({dataset: {jobId: 'test'}}),
    getElementById: id => id === 'job-status' ? status : {style: {}}},
  fetch: async () => ({ok: true, json: async () => ({status: 'failed', message: 'Failed', progress: 100})}),
  window: {location: {reload: () => reloads++}}, setInterval: () => 1, clearInterval: () => {},
};
vm.runInNewContext(data.script, context);
(async () => {await context.pollJob(); await context.pollJob(); process.stdout.write(JSON.stringify({reloads}));})();
'''
    result = subprocess.run(["node", "-e", harness], input=json.dumps({"initial": initial, "script": script}),
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["reloads"] == expected_reloads
    store.update_job(job.id, status="failed")
    failed_html = client.get(f"/jobs/{job.id}").text
    assert "Retry same settings" in failed_html
    assert "Delete job" in failed_html
