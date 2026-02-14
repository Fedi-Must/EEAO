from __future__ import annotations

import streamlit as st


def render_report_tab() -> None:
    """Render report generation and download controls."""
    st.subheader("Report")
    st.caption("Markdown report: summary, missing table, steps, and metrics.")

    if st.button("Generate report"):
        try:
            rep = st.session_state.tk.build_report(
                df=st.session_state.data,
                raw_df=st.session_state.raw,
                steps=st.session_state.steps,
                target=st.session_state.target,
            )
            st.download_button("Download REPORT.md", data=rep.encode("utf-8"), file_name="REPORT.md", mime="text/markdown")
            st.text_area("Preview", rep, height=350)
        except Exception as e:
            st.error(f"Report error: {e}")
