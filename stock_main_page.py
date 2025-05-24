import streamlit as st


about_page = st.Page(
    page="pages/Title_page.py",
    title="About",
    icon=":material/info:",
    default=True,
)

dashboard_page = st.Page(
    page="pages/Stock_Dashboard.py",
    title="Stock Dashboard and Prediction",
    icon=":material/analytics:",
)

pg = st.navigation(pages=[about_page, dashboard_page])
pg.run()
