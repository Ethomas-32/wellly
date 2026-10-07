"""
Clear stored user profiles — handy while testing repeatedly from the
same WhatsApp number, so the bot treats you as brand new again.

Usage:
    python clear_profiles.py --all              # wipe every profile
    python clear_profiles.py whatsapp:+27662355079   # wipe just one number
"""
import sys
import user_profiles


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python clear_profiles.py --all")
        print("  python clear_profiles.py <whatsapp_number>")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--all":
        user_profiles.clear_all()
        print("Cleared all stored profiles.")
    else:
        user_profiles.clear_profile(arg)
        print(f"Cleared profile for {arg} (if it existed).")


if __name__ == "__main__":
    main()