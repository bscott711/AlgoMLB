import random
import requests
from sqlalchemy import create_engine, text
from atproto import Client, models

from fadegoblin import config
from fadegoblin.db_slips import get_pending_slips, settle_slip
from fadegoblin.browser_twitter import reply_to_twitter_browser

# Persona responses
WIN_REPLIES = [
    "💰 CASHEEEED! The Goblin feasts tonight! Simple math always wins, sportsbooks absolutely owned! 💸📈",
    "🤑 BAM! Another winning ticket! Put it in the ledger and print the green! 💚💰",
    "👑 VICTORY! The mathematical edge reigns supreme! Go buy yourself something nice, the goblin's treat! 💎💰",
    "🎯 TARGET ACQUIRED & DESTROYED! Another beautiful win for the portfolio. Edge-hunting works! 💰🔥"
]

LOSS_REPLIES = [
    "💔 Ouch! The sportsbooks got away with one. Back to the swamp to recalculate the edges. 🐊📉",
    "💀 Pain. Complete variance robbery. But the process stays elite! We ride again tomorrow! 📉",
    "🤡 Lost this one. Just minor goblin pocket change. Don't worry, the edge always wins in the long run! 🐊",
    "💔 Rough result. The goblin is weeping, but the math does not lie—we stay disciplined and rebound tomorrow. 🐊"
]

def american_to_decimal(odds_str: str) -> float:
    try:
        val = int(odds_str.replace("+", ""))
        if val > 0:
            return (val / 100.0) + 1.0
        else:
            return (100.0 / abs(val)) + 1.0
    except Exception:
        return 2.0

def clean_name(s: str) -> str:
    return s.lower().replace(" ", "").replace(".", "").replace("-", "").replace("manchester", "man")

def match_team(api_name: str, target_name: str) -> bool:
    a = clean_name(api_name)
    t = clean_name(target_name)
    return a in t or t in a

def fetch_scores(sport: str) -> list[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores"
    params = {
        "api_key": config.ODDS_API_KEY,
        "daysFrom": 3
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠️ Error fetching scores for {sport}: {e}")
        return []

def check_degen_leg(leg: dict, scores_cache: dict) -> str:
    """Grades a single degen leg. Returns 'WON', 'LOST', or 'PENDING'."""
    sport = leg.get("sport")
    if not sport:
        return "PENDING"
    
    if sport not in scores_cache:
        scores_cache[sport] = fetch_scores(sport)
        
    games_list = scores_cache[sport]
    
    # Parse game string: "Away @ Home"
    parts = leg["game"].split(" @ ")
    if len(parts) < 2:
        return "PENDING"
    away_target, home_target = parts[0], parts[1]
    
    matched_game = None
    for g in games_list:
        if match_team(g["home_team"], home_target) and match_team(g["away_team"], away_target):
            matched_game = g
            break
            
    if not matched_game:
        return "PENDING"
        
    if not matched_game.get("completed", False):
        return "PENDING"
        
    # Game is completed, calculate outcome
    scores = matched_game.get("scores")
    if not scores or len(scores) < 2:
        return "PENDING"
        
    home_score = away_score = 0
    for s in scores:
        if match_team(s["name"], home_target):
            home_score = int(s["score"])
        elif match_team(s["name"], away_target):
            away_score = int(s["score"])
            
    pick = leg["pick"]
    
    if home_score == away_score:
        winner = "Draw"
    elif home_score > away_score:
        winner = home_target
    else:
        winner = away_target
        
    if pick == "Draw":
        return "WON" if winner == "Draw" else "LOST"
    
    # Otherwise check if our pick team matched the winner team
    if match_team(winner, pick):
        return "WON"
    else:
        return "LOST"

def post_bluesky_reply(text: str, parent_uri: str, parent_cid: str) -> bool:
    if not config.BOT_HANDLE or not config.APP_PASSWORD:
        return False
    try:
        client = Client()
        client.login(config.BOT_HANDLE, config.APP_PASSWORD)
        
        parent_ref = models.ComAtprotoRepoStrongRef.Main(
            cid=parent_cid,
            uri=parent_uri
        )
        reply_ref = models.AppBskyFeedPost.ReplyRef(
            parent=parent_ref,
            root=parent_ref
        )
        
        client.send_post(text=text, reply_to=reply_ref)
        print("✅ Posted follow-up reply on Bluesky.")
        return True
    except Exception as e:
        print(f"❌ Error posting Bluesky reply: {e}")
        return False

def post_twitter_reply(text: str, parent_tweet_id: str) -> bool:
    try:
        reply_to_twitter_browser(text, parent_tweet_id)
        print("✅ Posted follow-up reply on Twitter.")
        return True
    except Exception as e:
        print(f"❌ Error posting Twitter reply: {e}")
        return False

def run_followup_cycle(dry_run: bool = False) -> None:
    """Checks all pending slips, grades them, updates database, and posts social replies."""
    print("⏳ Running FadeGoblin social follow-up cycle...")
    pending = get_pending_slips()
    if not pending:
        print("💤 No pending slips to grade.")
        return
        
    print(f"📋 Found {len(pending)} pending slips to grade.")
    
    scores_cache = {}
    
    for slip in pending:
        slip_id = slip["slip_id"]
        slip_type = slip["slip_type"]
        legs = slip["legs"]
        stake = slip["stake"]
        final_odds = slip["final_odds"]
        
        is_graded = False
        outcome = None # 'WIN' or 'LOSS'
        final_pnl = 0.0
        
        if slip_type == "potd":
            # Grade POTD by querying bankroll_ledger table
            leg = legs[0]
            tx_id = leg.get("id")
            if not tx_id:
                print(f"⚠️ POTD slip #{slip_id} has no transaction ID. Skipping.")
                continue
                
            engine = create_engine(config.DATABASE_URL)
            query = text("SELECT status, pnl FROM bankroll_ledger WHERE transaction_id = :tx_id")
            with engine.connect() as conn:
                row = conn.execute(query, {"tx_id": tx_id}).fetchone()
                
            if row and row[0] == "SETTLED":
                is_graded = True
                pnl_val = float(row[1])
                outcome = "WIN" if pnl_val > 0 else "LOSS"
                # Use stake and odds for degen slip banking logic
                if outcome == "WIN":
                    dec_odds = american_to_decimal(final_odds)
                    final_pnl = stake * (dec_odds - 1.0)
                else:
                    final_pnl = -stake
        
        elif slip_type == "degen":
            # Grade degen slip using The-Odds-API scores
            leg_results = []
            for leg in legs:
                res = check_degen_leg(leg, scores_cache)
                leg_results.append(res)
                
            if "PENDING" in leg_results:
                # Still waiting for some games to complete
                continue
                
            is_graded = True
            if "LOST" in leg_results:
                outcome = "LOSS"
                final_pnl = -stake
            else:
                outcome = "WIN"
                dec_odds = american_to_decimal(final_odds)
                final_pnl = stake * (dec_odds - 1.0)
                
        if is_graded:
            print(f"⚖️ Grading {slip_type.upper()} Slip #{slip_id} as a {outcome}! (PnL: ${final_pnl:+.2f})")
            
            if not dry_run:
                # Settle in slips DB
                settle_slip(slip_id, final_pnl)
                
                # Compose unhinged reply
                reply_pool = WIN_REPLIES if outcome == "WIN" else LOSS_REPLIES
                custom_text = random.choice(reply_pool)
                
                # Add ticket details
                emoji = "💰" if outcome == "WIN" else "💔"
                status_text = f"{emoji} RESULT: {outcome}!\nTicket Odds: {final_odds}\nP&L: ${final_pnl:+.2f}\n\n{custom_text}"
                
                # Reply to Bluesky
                if slip["bsky_uri"] and slip["bsky_cid"]:
                    post_bluesky_reply(status_text, slip["bsky_uri"], slip["bsky_cid"])
                    
                # Reply to Twitter
                if slip["twitter_tweet_id"]:
                    post_twitter_reply(status_text, slip["twitter_tweet_id"])
