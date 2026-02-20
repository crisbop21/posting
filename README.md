# posting

Daily finance research and slide generation for TikTok/Instagram.

Automatically researches trending finance topics from news feeds and Reddit,
generates engaging slide content using Claude, iterates on engagement quality,
and outputs a ready-to-post PPTX deck.

## Setup

```bash
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

**Required:** `ANTHROPIC_API_KEY`
**Optional:** `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (for Reddit source)

## Usage

```bash
# Run with default config
python -m src.main

# Run with a custom config file
python -m src.main -c my_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **slides** — count, tone, audience, colors, aspect ratio
- **research** — sources (news/reddit), topics, subreddits
- **content** — number of review iterations, style notes
- **output** — output directory

## How it works

1. **Research** — Fetches trending finance topics from Google News RSS feeds and Reddit hot posts
2. **Generate** — Claude synthesizes the research into structured slide content
3. **Review** — Claude scores each slide on hook power, clarity, engagement, and visual balance, then produces an improved version. This repeats for N iterations
4. **Build** — Generates a styled PPTX file ready to screenshot or export for posting
