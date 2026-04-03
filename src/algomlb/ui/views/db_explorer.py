import streamlit as st
import pandas as pd
from sqlalchemy import text
from algomlb.db.introspection import SchemaInspector
from algomlb.db.session import get_engine

# Page Configuration
st.set_page_config(layout="wide", page_title="AlgoMLB Database Explorer")

# Custom Styles
st.markdown(
    """
<style>
    .status-badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-empty { background: #fee2e2; color: #991b1b; }
    .status-populated { background: #dcfce7; color: #166534; }
    .null-warning { color: #d97706; font-weight: 600; }
    .table-card {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e2e8f0;
        margin-bottom: 5px;
        cursor: pointer;
    }
    .table-card:hover { background: #f8fafc; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_inspector() -> SchemaInspector:
    return SchemaInspector(get_engine())


def _load_table_data(
    table_name: str,
    fk_filter: tuple[str, str | int | float | bool] | None = None,
    page: int = 0,
    page_size: int = 100,
) -> pd.DataFrame:
    engine = get_engine()
    where_clause = ""
    params = {}
    if fk_filter:
        where_clause = f"WHERE {fk_filter[0]} = :val"
        params["val"] = fk_filter[1]

    offset = page * page_size
    query = text(
        f'SELECT * FROM "{table_name}" {where_clause} LIMIT {page_size} OFFSET {offset}'
    )

    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def render_db_explorer():
    st.title("🔍 Database Explorer")
    inspector = get_inspector()

    if "db_exp_table" not in st.session_state:
        st.session_state.db_exp_table = None
    if "db_exp_fk_filter" not in st.session_state:
        st.session_state.db_exp_fk_filter = None
    if "db_exp_page" not in st.session_state:
        st.session_state.db_exp_page = 0

    col_nav, col_main = st.columns([1, 4])

    with col_nav:
        st.subheader("Tables")
        if st.button("🔄 Refresh Schema"):
            inspector.clear_cache()
            st.rerun()

        tables = inspector.list_tables()
        search_term = st.text_input("Search tables...", label_visibility="collapsed")

        filtered_tables = [t for t in tables if search_term.lower() in t.name.lower()]

        for table in filtered_tables:
            status_icon = "⚪" if table.is_empty else "🔵"
            label = f"{status_icon} {table.name} ({table.row_count:,})"
            if st.button(label, key=f"btn_{table.name}", use_container_width=True):
                st.session_state.db_exp_table = table.name
                st.session_state.db_exp_fk_filter = None
                st.session_state.db_exp_page = 0
                st.rerun()

    with col_main:
        selected_table = st.session_state.db_exp_table
        if not selected_table:
            st.info("Select a table from the sidebar to begin exploring.")
            return

        # Table Header
        table_meta = next((t for t in tables if t.name == selected_table), None)
        title_cols = st.columns([3, 1])
        title_cols[0].header(f"Table: `{selected_table}`")
        if table_meta:
            title_cols[1].metric("Total Rows", f"{table_meta.row_count:,}")

        if st.session_state.db_exp_fk_filter:
            f_col, f_val = st.session_state.db_exp_fk_filter
            st.warning(f"Filtered by FK: `{f_col} = {f_val}`")
            if st.button("❌ Clear Filter"):
                st.session_state.db_exp_fk_filter = None
                st.rerun()

        tab_data, tab_schema = st.tabs(["📁 Data Viewer", "📋 Schema Dictionary"])

        with tab_data:
            # Pagination
            page_size = 100
            total_pages = 1
            if table_meta and table_meta.row_count > 0:
                total_pages = (table_meta.row_count // page_size) + 1

            p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
            if st.session_state.db_exp_page > 0:
                if p_col1.button("⬅️ Previous"):
                    st.session_state.db_exp_page -= 1
                    st.rerun()
            p_col2.write(f"Page {st.session_state.db_exp_page + 1} of {total_pages}")
            if st.session_state.db_exp_page < total_pages - 1:
                if p_col3.button("Next ➡️"):
                    st.session_state.db_exp_page += 1
                    st.rerun()

            # Load Data
            try:
                df = _load_table_data(
                    selected_table,
                    st.session_state.db_exp_fk_filter,
                    st.session_state.db_exp_page,
                    page_size,
                )

                # Column health warnings
                col_report = inspector.column_report(selected_table)
                null_cols = [c.name for c in col_report if c.is_all_null]
                if null_cols:
                    st.warning(
                        f"⚠️ **Data Quality Alert:** The following columns are 100% NULL: `{', '.join(null_cols)}`"
                    )

                st.dataframe(df, use_container_width=True)

                # FK Navigation (if a row is selected or just generic links)
                st.markdown("---")
                st.subheader("🔗 Relations (Foreign Keys)")
                fks = [
                    fk
                    for fk in inspector.foreign_keys()
                    if fk.from_table == selected_table
                ]

                if not fks:
                    st.write("No outgoing foreign keys found for this table.")
                else:
                    fk_cols = st.columns(len(fks) if len(fks) < 4 else 4)
                    for i, fk in enumerate(fks):
                        with fk_cols[i % 4]:
                            if st.button(
                                f"Go to {fk.to_table} (via {fk.from_col})",
                                key=f"fk_nav_{i}",
                            ):
                                # In a real interactive app, we'd pick a value from the DF
                                # For now, we'll just navigate to the table
                                st.session_state.db_exp_table = fk.to_table
                                st.session_state.db_exp_fk_filter = None
                                st.session_state.db_exp_page = 0
                                st.rerun()

            except Exception as e:
                st.error(f"Error loading data: {e}")

        with tab_schema:
            st.subheader("Column Metadata")
            col_report = inspector.column_report(selected_table)
            report_df = pd.DataFrame(
                [
                    {
                        "Column": c.name,
                        "Type": c.dtype,
                        "Nullable": "✅" if c.nullable else "❌",
                        "Null %": f"{c.null_pct * 100:.1f}%",
                        "Status": "⚠️ All NULL" if c.is_all_null else "OK",
                    }
                    for c in col_report
                ]
            )
            st.table(report_df)


if __name__ == "__main__":
    render_db_explorer()
