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

1. **Load & cluster** — Upload the CSV. The app assumes **columns A–E are metadata** and the **first survey prompt is column F** (five columns are skipped when listing prompts). Pick **one survey column**: its **header (row 1)** is the prompt sent to the LLM; **every body row** in that column is aggregated (blanks skipped; comma-separated phrases split into separate answers). Optionally ignore junk answers (`n/a`, etc.), then **Run LLM clustering**.
2. **Review & export** — Edit cluster labels, move stray answers between clusters, sort or drop empty clusters, then **Download JSON**.

Runs are also written under `runs/` with a timestamp and column slug so you can reopen them from the dropdown.

## JSON shape

Each export includes:

- `clusters`: ordered list of `{ "label", "count", "members": [ { "text", "count" }, ... ] }`
- `n_cells_non_empty`, `n_unique_texts`, `skip_first_columns` (fixed at 5 for columns A–E), plus metadata about source file and `column` (the prompt text from the header row).

## Notes

- **No traceability** to individual respondents is stored beyond grouped text strings.
- Large unique answer counts use **chunking** and a **merge** pass automatically.
- If the model mis-groups something, fix it in the Review tab before exporting.
