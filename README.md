# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An **estimated ATS score** out of 100
- A score breakdown (keywords, impact, structure, readability)
- A checklist of basic ATS requirements (email, phone, headings, length, bullets, numbers)
- Prioritised **improvements** (high / medium / low)
- **Missing keywords** (best results if you paste a job description)
- **Bullet-point rewrites** you can adapt
- A downloadable report

Built with **Streamlit** + **Groq API** (Llama models).

## How the score works

| Part | Weight | Method |
|------|--------|--------|
| Rule-based checks | 30% | Plain Python: contact info, section headings, length, bullets, numbers |
| AI review | 70% | Groq LLM: keywords, experience impact, structure, readability |

No tool can give an "official" ATS score because every company uses a different ATS.
This is a practical estimate to help you improve your resume.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your key in the sidebar, or set it once:

- **Mac/Linux:** `export GROQ_API_KEY=gsk_your_key`
- **Windows (PowerShell):** `$env:GROQ_API_KEY="gsk_your_key"`

Get a free key at <https://console.groq.com/keys>.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**.

> Never put your API key inside `app.py` or commit it to GitHub.

## Changing the model

Use the "Custom model ID" box in the sidebar, or edit the `MODELS` list at the top of `app.py`. Groq retires models often (the old Llama 3.x models stopped working for free keys on 16 Aug 2026), so check the
current list at <https://console.groq.com/docs/models>.

## Limits

- Scanned/image-only PDFs cannot be read (real ATS systems struggle with them too).
- Only the first 12,000 characters of a resume are analysed.
- Free Groq keys have rate limits; wait a minute if you hit one.
- Resume text is sent to Groq for analysis. The app itself does not store anything.
