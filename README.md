# LabVIEW Family Feud — survey clustering prep

Small local tool: load a **wide CSV** export from your survey (each new question adds columns to the right), pick **one** open-ended column, call **OpenAI** to merge similar free-text answers and rank them like a Family Feud board, then **review** and **export JSON** for your game software.

## Setup

1. Python 3.10+ recommended.
2. Create a virtual environment (optional but recommended):

   ```bash
   cd "/Users/elijahkerry/labview family feud"
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

   Optionally set `OPENAI_MODEL` (default: `gpt-4o-mini`).

## Run the UI

```bash
streamlit run app.py
```

**Sidebar → Menu**

- **LLM connection** — Paste your OpenAI API key (**Apply** for this browser tab, or **Save key to .env**). Optional model override with **Apply** / **Save model to .env**. **Clear session** drops overrides and reloads from `.env` / the environment.
- **Save / undo / restart** — **Undo** steps back through moves, sorts, and “delete empty clusters.” **Save run to disk** writes the current review state under `runs/`. **Revert review** restores the last LLM snapshot. **Restart session** clears the loaded CSV and in-tab progress (API key in session is kept unless you clear it).

1. **Load & cluster** — Upload the CSV. Use the sidebar to skip leading metadata columns (e.g. timestamp), exclude column name fragments (e.g. `email`), and ignore junk answers (`n/a`, etc.). Choose the question column, then **Run LLM clustering**.
2. **Review & export** — Edit cluster labels, move stray answers between clusters, sort or drop empty clusters, then **Download JSON**.

Runs are also written under `runs/` with a timestamp and column slug so you can reopen them from the dropdown.

## JSON shape

Each export includes:

- `clusters`: ordered list of `{ "label", "count", "members": [ { "text", "count" }, ... ] }`
- `n_cells_non_empty`, `n_unique_texts`, plus metadata about source file and column.

## Notes

- **No traceability** to individual respondents is stored beyond grouped text strings.
- Large unique answer counts use **chunking** and a **merge** pass automatically.
- If the model mis-groups something, fix it in the Review tab before exporting.
