import sys
from algomlb.db.introspection import SchemaInspector
from algomlb.db.session import get_engine

def main():
    try:
        inspector = SchemaInspector(get_engine())
        tables = inspector.list_tables()
        empty = inspector.empty_tables()
        null_cols = inspector.all_null_columns()
        fks = inspector.foreign_keys()

        print(f"\n{'='*60}")
        print("📡 AlgoMLB Database Health Summary")
        print(f"{'='*60}")
        print(f"  Total Tables:    {len(tables)}")
        print(f"  Empty Tables:    {len(empty)}")
        print(f"  All-NULL Cols:   {len(null_cols)}")
        print(f"  Foreign Keys:    {len(fks)}")
        print(f"{'='*60}")

        if empty:
            print("\nEmpty Tables:")
            for t in empty:
                print(f"  - {t.name}")

        if null_cols:
            print("\nAll-NULL Columns:")
            for c in null_cols:
                print(f"  - {c.table}.{c.name} ({c.dtype})")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error generating DB health summary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
