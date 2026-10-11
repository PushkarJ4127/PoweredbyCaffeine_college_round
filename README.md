# AgentBreak

An automated red team for AI agents. It attacks a deliberately vulnerable demo agent (OfficeBot), records exactly how it broke, and measures whether defenses stop the same attacks without breaking normal work.

Everything runs against our own fake agent with fake tools and fake data (`acme.test`). Nothing real is touched.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env     # add GEMINI_API_KEY and the current model name
python llm.py            # should print: pong
```

## Layout
See `CONTRACT.md` for shapes, hook points, detector rules and file ownership.

## Settings
Copy `.env.example` to `.env`. Everyone on the team uses the same `AGENTBREAK_MODEL` and `AGENTBREAK_GULLIBLE`.
Quick live test: `python try_attack.py email --fresh` (also `document`, `memory`). Self-test without a key: `python -m officebot.selftest`.

## Status
- [x] Contract, config, schemas
- [x] OfficeBot stub (canned traces)
- [x] Real OfficeBot tool loop (self-tested with a scripted model; needs a live run with a key)
- [ ] Attacker, detectors, recorder
- [ ] Defenses D1 to D3, benign tasks, API, batch runner
- [ ] Dashboard, demo

(Limitations, seeded flaws and results: fill in at the end.)
