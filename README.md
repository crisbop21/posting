---
title: Posting - Finance Slides
emoji: 📊
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# Posting

Daily finance research and slide generation for TikTok and Instagram.

Automatically researches trending finance topics from news feeds and Reddit,
generates engaging slide content using Claude, iterates on engagement quality,
and outputs a ready to post PPTX deck.

## Setup

### Gradio Web App (recommended)

```bash
pip install -r requirements_gradio.txt
python gradio_app.py
# Opens at http://localhost:7860
```

### Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### CLI

```bash
pip install -e .
python -m src.main              # default config
python -m src.main -c my.yaml   # custom config
```

**Required:** `ANTHROPIC_API_KEY`
**Optional:** `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (for Reddit), `ELEVENLABS_API_KEY` (for video), `GOOGLE_AI_API_KEY` or `OPENAI_API_KEY` (for AI images), `PEXELS_API_KEY` or `PIXABAY_API_KEY` (for image search)

## Deploy to HuggingFace Spaces (free)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with **Gradio** SDK
2. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/posting
   git push space main
   ```
3. Add your `ANTHROPIC_API_KEY` as a Space Secret in Settings
4. The app will be live at `https://YOUR_USERNAME-posting.hf.space`

Runs in **demo mode** (no API key needed) with sample Tesla Q4 data so visitors can explore the full workflow.

## Configuration

Edit `config.yaml` to customize:

- **slides**: count, tone, audience, colors, aspect ratio
- **research**: sources (news, Reddit), topics, subreddits
- **content**: number of review iterations, style notes
- **output**: output directory

## How It Works

1. **Research**: fetches trending finance topics from Google News RSS feeds and Reddit hot posts
2. **Generate**: Claude combines the research into structured slide content
3. **Review**: Claude scores each slide on hook power, clarity, engagement, and visual balance, then produces an improved version. This repeats for the configured number of iterations
4. **Build**: generates a styled PPTX file ready to screenshot or export for posting
