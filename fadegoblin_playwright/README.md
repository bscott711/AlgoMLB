# 👺 FadeGoblin

FadeGoblin is an "unhinged" sports betting bot for Bluesky. It generates degenerate parlays or identifies +EV "sniper" bets, writes chaotic commentary using AI, and posts them to the timeline accompanied by AI-generated images of a crazed betting goblin.

## 🚀 Features

- **Two Run Modes**:
  - **Degen Mode**: Fetches live games from The Odds API and builds a random 2-4 leg parlay.
  - **Sniper Mode**: Queries a PostgreSQL database (compatible with AlgoMLB/AlgoEPL schemas) to find the highest-edge +EV bet available.
- **AI Commentary**: Uses OpenRouter (Nvidia Nemotron 3 Super 120B) to generate unhinged, high-energy betting advice.
- **AI Art**: Generates custom "betting goblin" visuals via Pollinations.ai (Flux model) based on random action/outfit/style recipes.
- **Bluesky Integration**: Automatically posts text and images to the AT Protocol (Bluesky).
- **Smart Ledger Tracking**: Marks bets as `PLACED` in the database to ensure no duplicate posts.

## 🛠 Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **APIs**:
  - [OpenRouter](https://openrouter.ai/) (LLM)
  - [The Odds API](https://the-odds-api.com/) (Live Odds)
  - [Pollinations.ai](https://pollinations.ai/) (Image Generation)
  - [AT Protocol](https://atproto.com/) (Bluesky)
- **Database**: PostgreSQL (via SQLAlchemy & Pandas)

## 📦 Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/bscott711/fadegoblin.git
    cd fadegoblin
    ```

2. **Install dependencies**:

    ```bash
    uv sync
    ```

## ⚙️ Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Bluesky Credentials
BOT_HANDLE="your-bot.bsky.social"
APP_PASSWORD="your-app-password"

# API Keys
OPENROUTER_API_KEY="your-openrouter-key"
ODDS_API_KEY="your-odds-api-key"
POLLINATIONS_API_KEY="optional-api-key"

# Database (Required for Sniper Mode)
DATABASE_URL="postgresql://user:pass@host:port/dbname"
```

## 🏃 Usage

FadeGoblin is designed to be run via `uv`.

### Run Degen Mode (Random Parlay)

```bash
uv run bot --mode degen
```

### Run Sniper Mode (+EV Database Pick)

```bash
uv run bot --mode sniper
```

### Dry Run (Preview without posting)

```bash
uv run bot --mode degen --dry-run
```

## 📂 Project Structure

- `src/fadegoblin/main.py`: Entry point and orchestration logic.
- `src/fadegoblin/ev_logic.py`: Database queries for +EV bets.
- `src/fadegoblin/generator.py`: LLM prompt construction for post content.
- `src/fadegoblin/llm.py`: OpenRouter API client.
- `src/fadegoblin/image.py`: Image generation and download logic.
- `src/fadegoblin/odds.py`: Fetching live odds from The Odds API.
- `src/fadegoblin/prompts.py`: Large collection of goblin-themed personalities and fallbacks.

## 🤝 Contributing

This is a personal project. Feel free to open issues or PRs if you want to add more unhinged features.

## ⚖️ License

MIT
