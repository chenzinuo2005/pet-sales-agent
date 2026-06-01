"""入口: python -m app.main <command>"""

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from app.main import main

    main()
