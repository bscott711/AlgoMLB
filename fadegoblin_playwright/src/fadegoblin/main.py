import argparse
import os
import random
from datetime import datetime
from pathlib import Path

from atproto import Client, models

from fadegoblin import config, browser_twitter
from fadegoblin.betting import build_parlay
from fadegoblin.card import render_bet_card
from fadegoblin.ev_logic import get_sniper_bets, mark_bets_placed
from fadegoblin.generator import generate_post_content, generate_sniper_post_content
from fadegoblin.image import download_goblin_image, generate_goblin_prompt
from fadegoblin.odds import get_live_games
from fadegoblin.prompts import FALLBACK_QUOTES


def main() -> None:
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Error: {e}")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["degen", "sniper"],
        default="degen",
        help="Run mode: 'degen' (random parlay) or 'sniper' (AlgoMLB DB picks)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"--- Starting FadeGoblin [{args.mode.upper()}] at {datetime.now()} ---")

    if args.mode == "sniper":
        _run_sniper(args.dry_run)
    else:
        _run_degen(args.dry_run)


def _run_degen(dry_run: bool) -> None:
    """Degen Mode: random parlay from live odds."""
    print("🎲 Mode: Degen. Fetching random live games...")
    games = get_live_games(max_games=15)
    
    # 1. Generate TWO unique parlays/texts
    chosen_legs_1, final_odds_1 = build_parlay(games)
    chosen_legs_2, final_odds_2 = build_parlay(games) if games else ([], "N/A")
    
    post_texts = []
    if chosen_legs_1:
        post_texts.append(generate_post_content(chosen_legs_1, final_odds_1))
    else:
        post_texts.append(random.choice(FALLBACK_QUOTES))
        
    if chosen_legs_2:
        post_texts.append(generate_post_content(chosen_legs_2, final_odds_2))
    else:
        post_texts.append(random.choice(FALLBACK_QUOTES))

    # 2. Generate TWO unique images
    image_paths = []
    for i in range(2):
        if random.random() < config.TEXT_ONLY_ODDS:
            print(f"   🎲 Dice Roll: decided on TEXT ONLY for image {i+1}.")
            image_paths.append(None)
        else:
            prompt = generate_goblin_prompt()
            target_path = config.BASE_DIR / f"temp_meme_{i+1}.jpg"
            img_path = download_goblin_image(prompt, target_path)
            image_paths.append(img_path)

    _post_to_socials(post_texts, image_paths, dry_run)


def _run_sniper(dry_run: bool) -> None:
    """Sniper Mode: overlay the bet card onto a custom AI goblin image."""
    print("🎯 Mode: Sniper. Checking database for +EV bets...")
    all_legs, db_ids_to_update = get_sniper_bets()

    if not all_legs:
        print("💤 No pending/future EV bets found. Going back to sleep.")
        return

    print(f"📋 Found {len(all_legs)} +EV plays on the board.")

    # Randomly select ONE Play of the Day (consistent across platforms)
    potd_index = random.randint(0, len(all_legs) - 1)
    potd_leg = all_legs[potd_index]

    # 1. Generate TWO unique background images
    prompt_1 = generate_goblin_prompt()
    bg_target_path_1 = config.BASE_DIR / "temp_bg_1.jpg"
    goblin_bg_path_1 = download_goblin_image(prompt_1, bg_target_path_1)

    prompt_2 = generate_goblin_prompt()
    bg_target_path_2 = config.BASE_DIR / "temp_bg_2.jpg"
    goblin_bg_path_2 = download_goblin_image(prompt_2, bg_target_path_2)

    # 2. Render cards for both images
    final_path_1 = render_bet_card(all_legs, potd_index, background_path=goblin_bg_path_1)
    final_path_2 = render_bet_card(all_legs, potd_index, background_path=goblin_bg_path_2)
    image_paths = [final_path_1, final_path_2]

    # Cleanup background fragments
    for bg_path in [goblin_bg_path_1, goblin_bg_path_2]:
        if bg_path and os.path.exists(bg_path) and bg_path not in image_paths:
            os.remove(bg_path)

    # 3. Generate TWO unique unhinged rants for the SAME POTD
    post_text_1 = generate_sniper_post_content(potd_leg)
    post_text_2 = generate_sniper_post_content(potd_leg)
    post_texts = [post_text_1, post_text_2]

    _post_to_socials(post_texts, image_paths, dry_run, db_ids_to_update=db_ids_to_update)


def _post_to_socials(
    post_texts: str | list[str],
    image_paths: list[Path | None],
    dry_run: bool,
    *,
    db_ids_to_update: list[str] | None = None,
) -> None:
    """Handles the upload to Bluesky and Twitter/X with unique content per platform."""
    
    # Ensure post_texts is a list
    if isinstance(post_texts, str):
        post_texts = [post_texts, post_texts]
    
    # Ensure we have enough texts
    if len(post_texts) < 2:
        post_texts = [post_texts[0], post_texts[0]]

    if dry_run:
        print("\n🚫 DRY RUN MODE ENABLED. SKIPPING UPLOAD.")
        print(f"📝 Bluesky Post:\n{post_texts[0]}")
        print(f"📝 Twitter Post:\n{post_texts[1]}")
        return

    success_bsky = False
    success_twitter = False

    # --- 1. Post to Bluesky ---
    if config.BOT_HANDLE and config.APP_PASSWORD:
        try:
            print("Connecting to Bluesky...")
            client = Client()
            client.login(config.BOT_HANDLE, config.APP_PASSWORD)

            text = post_texts[0]
            img_path = image_paths[0] if image_paths else None
            
            blobs = []
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_data = f.read()
                upload = client.upload_blob(img_data)
                blobs.append(upload.blob)

            if blobs:
                images = [
                    models.AppBskyEmbedImages.Image(
                        alt="FadeGoblin sports betting visual", image=blob
                    )
                    for blob in blobs
                ]
                client.send_post(
                    text=text,
                    embed=models.AppBskyEmbedImages.Main(images=images),
                )
            else:
                client.send_post(text=text)

            print("✅ Successfully posted to Bluesky!")
            success_bsky = True
        except Exception as e:
            print(f"❌ Error posting to Bluesky: {e}")

    # --- 2. Post to Twitter/X (Browser Automation) ---
    if config.TWITTER_USERNAME and config.TWITTER_PASSWORD:
        try:
            print("Starting Twitter browser automation...")
            text = post_texts[1]
            # Twitter gets the SECOND image (if available, otherwise first)
            img_path = image_paths[1] if len(image_paths) > 1 else (image_paths[0] if image_paths else None)
            
            browser_twitter.post_to_twitter_browser(text, img_path)
            success_twitter = True 
        except Exception as e:
            print(f"❌ Error during Twitter browser automation: {e}")

    # Cleanup images
    for img_path in image_paths:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

    # Mark DB bets as PLACED if either succeeded
    if (success_bsky or success_twitter) and db_ids_to_update:
        mark_bets_placed(db_ids_to_update)
        print(f"✅ Marked {len(db_ids_to_update)} EV bets as PLACED in database.")



if __name__ == "__main__":
    main()

